#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  validate_multiple_openvla_oft_lora_checkpoints.sh [options] /path/to/checkpoint_or_lora_adapter [...]
  validate_multiple_openvla_oft_lora_checkpoints.sh [options] --list checkpoints.txt

Options:
  --base-model DIR        Validate an unmodified base OpenVLA-OFT model in addition
                          to any LoRA adapters. DIR must contain config.json.
  --list FILE              Read checkpoint/LoRA adapter paths from FILE, one path per line.
                           Empty lines and lines starting with # are ignored.
  --log-dir DIR            Folder for all validation logs.
                           Default: validation_logs/multi-validate-YYYYmmdd-HHMMSS
  --target-rollouts N      Episodes per checkpoint. Default: 300.
  --continue-on-error      Continue with the next checkpoint if one validation fails.
  -h, --help               Show this help.

Everything after -- is passed to the single-checkpoint validation wrapper as Hydra overrides.

Common environment knobs passed through:
  SFT_MODEL_PATH, DATASET_NAME, LIBERO_ROOT, NUM_GPUS, VAL_BATCH_SIZE, WANDB_MODE,
  WANDB_API_KEY, CKPT_PATH, ALIGN_PATH, NUM_TRIALS_PER_TASK

Examples:
  NUM_GPUS=2 VAL_BATCH_SIZE=2 WANDB_MODE=disabled \
    bash examples/validate_multiple_openvla_oft_lora_checkpoints.sh \
      /path/to/checkpoint_10 \
      /path/to/checkpoint_20

  NUM_GPUS=2 VAL_BATCH_SIZE=2 \
    bash examples/validate_multiple_openvla_oft_lora_checkpoints.sh \
      --base-model /path/to/base_openvla_oft_model \
      --list checkpoints.txt \
      -- reward.subgoal.enabled=False
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALIDATE_ONE="$SCRIPT_DIR/validate_openvla_oft_rl_libero_lora_checkpoint.sh"

CHECKPOINT_LIST_FILE=""
LOG_DIR=""
TARGET_ROLLOUTS_PER_CHECKPOINT="300"
CONTINUE_ON_ERROR="false"
CHECKPOINTS=()
BASE_MODELS=()
EXTRA_OVERRIDES=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --list)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --list" >&2
                exit 1
            fi
            CHECKPOINT_LIST_FILE="$2"
            shift 2
            ;;
        --base-model)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --base-model" >&2
                exit 1
            fi
            BASE_MODELS+=("$2")
            shift 2
            ;;
        --log-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --log-dir" >&2
                exit 1
            fi
            LOG_DIR="$2"
            shift 2
            ;;
        --target-rollouts)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --target-rollouts" >&2
                exit 1
            fi
            TARGET_ROLLOUTS_PER_CHECKPOINT="$2"
            shift 2
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR="true"
            shift
            ;;
        --)
            shift
            EXTRA_OVERRIDES=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            CHECKPOINTS+=("$1")
            shift
            ;;
    esac
done

if [ -n "$CHECKPOINT_LIST_FILE" ]; then
    if [ ! -f "$CHECKPOINT_LIST_FILE" ]; then
        echo "Missing checkpoint list file: $CHECKPOINT_LIST_FILE" >&2
        exit 1
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [ -z "$line" ] || [[ "$line" == \#* ]]; then
            continue
        fi
        CHECKPOINTS+=("$line")
    done < "$CHECKPOINT_LIST_FILE"
fi

if [ "${#CHECKPOINTS[@]}" -eq 0 ] && [ "${#BASE_MODELS[@]}" -eq 0 ]; then
    echo "Provide at least one LoRA adapter checkpoint path or --base-model DIR." >&2
    usage
    exit 1
fi

if [ ! -x "$VALIDATE_ONE" ] && [ ! -f "$VALIDATE_ONE" ]; then
    echo "Missing validation wrapper: $VALIDATE_ONE" >&2
    exit 1
fi

if ! [[ "$TARGET_ROLLOUTS_PER_CHECKPOINT" =~ ^[0-9]+$ ]] || [ "$TARGET_ROLLOUTS_PER_CHECKPOINT" -le 0 ]; then
    echo "--target-rollouts must be a positive integer, got: $TARGET_ROLLOUTS_PER_CHECKPOINT" >&2
    exit 1
fi

if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$REPO_ROOT/validation_logs/multi-validate-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

SUMMARY_FILE="$LOG_DIR/summary.tsv"
COMMANDS_FILE="$LOG_DIR/commands.txt"
: > "$COMMANDS_FILE"
printf "index\tstatus\tcheckpoint\texperiment\tlog\taccuracy\tnum_rollouts\tsupported_rollouts\tunsupported_rollouts\treach_object_success\tgrasp_object_success\tlift_object_success\tmove_to_target_success\tplace_or_success_success\n" > "$SUMMARY_FILE"

sanitize_name() {
    local raw="$1"
    raw="${raw%/}"
    raw="${raw//\//_}"
    raw="${raw//[^A-Za-z0-9._-]/_}"
    printf '%s' "$raw"
}

resolve_lora_adapter_path() {
    local path="${1%/}"
    local candidate
    for candidate in \
        "$path" \
        "$path/actor/lora_adapter" \
        "$path/lora_adapter" \
        "$path/actor"; do
        if [ -f "$candidate/adapter_config.json" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    echo "Could not find PEFT adapter under: $1" >&2
    echo "Tried: $path, $path/actor/lora_adapter, $path/lora_adapter, $path/actor" >&2
    return 1
}

extract_metrics() {
    local log_file="$1"
    python - "$log_file" <<'PY'
import ast
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
keys = [
    "test_score/all",
    "test_score/num_rollouts",
    "test_subgoal/supported_rollouts",
    "test_subgoal/unsupported_rollouts",
    "test_subgoal_phase_success/reach_object",
    "test_subgoal_phase_success/grasp_object",
    "test_subgoal_phase_success/lift_object",
    "test_subgoal_phase_success/move_to_target",
    "test_subgoal_phase_success/place_or_success",
]
values = {key: "" for key in keys}

for match in re.finditer(r"(?:Initial|Final)? validation metrics:\s*(\{.*?\})", text, re.S):
    raw = match.group(1)
    try:
        metrics = ast.literal_eval(raw)
    except Exception:
        continue
    for key in keys:
        values[key] = metrics.get(f"val/{key}", metrics.get(key, values[key]))

for key in keys:
    if values[key] != "":
        continue
    matches = re.findall(
        rf"['\"](?:val/)?{re.escape(key)}['\"]\s*:\s*([-+0-9.eE]+)",
        text,
    )
    if matches:
        values[key] = matches[-1]

print("\t".join(str(values[key]) for key in keys))
PY
}

echo "Validating ${#CHECKPOINTS[@]} checkpoint(s)"
if [ "${#BASE_MODELS[@]}" -gt 0 ]; then
    echo "Validating ${#BASE_MODELS[@]} base model(s)"
fi
echo "Episodes per checkpoint: $TARGET_ROLLOUTS_PER_CHECKPOINT"
echo "Logs: $LOG_DIR"

validate_base_model() {
    local base_model_path="$1"
    local experiment_name="$2"
    local log_file="$3"

    if [ ! -f "$base_model_path/config.json" ]; then
        echo "Base model must contain config.json: $base_model_path" >&2
        return 1
    fi

    SFT_MODEL_PATH="$base_model_path" \
    TARGET_ROLLOUTS="$TARGET_ROLLOUTS_PER_CHECKPOINT" \
    SAVE_VIDEO="${SAVE_VIDEO:-False}" \
    bash "$SCRIPT_DIR/run_openvla_oft_rl_libero_lora_dense.sh" \
        actor_rollout_ref.model.lora_adapter_path=null \
        actor_rollout_ref.rollout.experiment_name="$experiment_name" \
        trainer.experiment_name="$experiment_name" \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.test_freq=-1 \
        trainer.validation.target_rollouts="$TARGET_ROLLOUTS_PER_CHECKPOINT" \
        trainer.validation.start_index="${START_INDEX:-0}" \
        trainer.validation.save_video="${SAVE_VIDEO:-False}" \
        trainer.validation.video_every_n_calls=1 \
        trainer.validation.video_max_episodes="${VIDEO_MAX_EPISODES:-10}" \
        trainer.validation.video_per_task_limit="${VIDEO_PER_TASK_LIMIT:-1}" \
        trainer.validation.video_frame_stride="${VIDEO_FRAME_STRIDE:-1}" \
        data.val_batch_size="${VAL_BATCH_SIZE:-${NUM_GPUS:-2}}" \
        trainer.wandb_mode="${WANDB_MODE:-disabled}" \
        "${EXTRA_OVERRIDES[@]}"
}

display_idx=0
for base_model in "${BASE_MODELS[@]}"; do
    display_idx=$((display_idx + 1))
    base_model="${base_model%/}"
    safe_name="$(sanitize_name "base_$(basename "$base_model")")"
    experiment_name="multi-val-${display_idx}-${safe_name}"
    log_file="$LOG_DIR/${display_idx}_${safe_name}.log"
    status="ok"

    echo
    echo "[$display_idx/$(( ${#BASE_MODELS[@]} + ${#CHECKPOINTS[@]} ))] Validating base model: $base_model"
    echo "Log file: $log_file"

    {
        printf '[%s] ' "$(date -Is)"
        printf 'SFT_MODEL_PATH=%q TARGET_ROLLOUTS=%q SAVE_VIDEO=%q bash %q' \
            "$base_model" "$TARGET_ROLLOUTS_PER_CHECKPOINT" "${SAVE_VIDEO:-False}" \
            "$SCRIPT_DIR/run_openvla_oft_rl_libero_lora_dense.sh"
        printf ' %q' actor_rollout_ref.model.lora_adapter_path=null "${EXTRA_OVERRIDES[@]}"
        printf '\n'
    } >> "$COMMANDS_FILE"

    set +e
    (cd "$REPO_ROOT" && validate_base_model "$base_model" "$experiment_name" "$log_file") 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [ "$exit_code" -ne 0 ]; then
        status="failed:$exit_code"
        metrics=("" "" "" "" "" "" "" "" "")
        echo "Validation failed for base model $base_model with exit code $exit_code" >&2
        if [ "$CONTINUE_ON_ERROR" != "true" ]; then
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$display_idx" "$status" "$base_model" "$experiment_name" "$log_file" "${metrics[@]}" >> "$SUMMARY_FILE"
            exit "$exit_code"
        fi
    else
        IFS=$'\t' read -r accuracy num_rollouts supported_rollouts unsupported_rollouts \
            reach_object_success grasp_object_success lift_object_success \
            move_to_target_success place_or_success_success < <(extract_metrics "$log_file")
        metrics=(
            "$accuracy" "$num_rollouts" "$supported_rollouts" "$unsupported_rollouts"
            "$reach_object_success" "$grasp_object_success" "$lift_object_success"
            "$move_to_target_success" "$place_or_success_success"
        )
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$display_idx" "$status" "$base_model" "$experiment_name" "$log_file" "${metrics[@]}" >> "$SUMMARY_FILE"
    echo "Finished base model $base_model: status=$status accuracy=${accuracy:-unknown} rollouts=${num_rollouts:-unknown}"
done

for idx in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$idx]}"
    lora_adapter_path="$(resolve_lora_adapter_path "$checkpoint")"
    display_idx=$((display_idx + 1))
    parent="$(basename "$(dirname "${lora_adapter_path%/}")")"
    leaf="$(basename "${lora_adapter_path%/}")"
    safe_name="$(sanitize_name "${parent}_${leaf}")"
    experiment_name="multi-val-${display_idx}-${safe_name}"
    log_file="$LOG_DIR/${display_idx}_${safe_name}.log"
    status="ok"

    echo
    echo "[$display_idx/$(( ${#BASE_MODELS[@]} + ${#CHECKPOINTS[@]} ))] Validating: $checkpoint"
    echo "Resolved LoRA adapter: $lora_adapter_path"
    echo "Log file: $log_file"

    {
        printf '[%s] ' "$(date -Is)"
        printf 'TARGET_ROLLOUTS=%q SAVE_VIDEO=%q bash %q %q %q' \
            "$TARGET_ROLLOUTS_PER_CHECKPOINT" "${SAVE_VIDEO:-False}" "$VALIDATE_ONE" "$lora_adapter_path" "$experiment_name"
        printf ' %q' "${EXTRA_OVERRIDES[@]}"
        printf '\n'
    } >> "$COMMANDS_FILE"

    set +e
    TARGET_ROLLOUTS="$TARGET_ROLLOUTS_PER_CHECKPOINT" \
    SAVE_VIDEO="${SAVE_VIDEO:-False}" \
    bash "$VALIDATE_ONE" "$lora_adapter_path" "$experiment_name" "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [ "$exit_code" -ne 0 ]; then
        status="failed:$exit_code"
        metrics=("" "" "" "" "" "" "" "" "")
        echo "Validation failed for $checkpoint with exit code $exit_code" >&2
        if [ "$CONTINUE_ON_ERROR" != "true" ]; then
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$display_idx" "$status" "$checkpoint" "$experiment_name" "$log_file" "${metrics[@]}" >> "$SUMMARY_FILE"
            exit "$exit_code"
        fi
    else
        IFS=$'\t' read -r accuracy num_rollouts supported_rollouts unsupported_rollouts \
            reach_object_success grasp_object_success lift_object_success \
            move_to_target_success place_or_success_success < <(extract_metrics "$log_file")
        metrics=(
            "$accuracy" "$num_rollouts" "$supported_rollouts" "$unsupported_rollouts"
            "$reach_object_success" "$grasp_object_success" "$lift_object_success"
            "$move_to_target_success" "$place_or_success_success"
        )
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$display_idx" "$status" "$checkpoint" "$experiment_name" "$log_file" "${metrics[@]}" >> "$SUMMARY_FILE"
    echo "Finished $checkpoint: status=$status accuracy=${accuracy:-unknown} rollouts=${num_rollouts:-unknown}"
done

echo
echo "All requested validations finished."
echo "Summary: $SUMMARY_FILE"
column -t -s $'\t' "$SUMMARY_FILE" || cat "$SUMMARY_FILE"
