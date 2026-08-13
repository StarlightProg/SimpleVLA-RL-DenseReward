#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  validate_multiple_openvla_oft_lora_checkpoints.sh [options] /path/to/checkpoint_or_lora_adapter [...]
  validate_multiple_openvla_oft_lora_checkpoints.sh [options] --list checkpoints.txt

Options:
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

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "Provide at least one LoRA adapter checkpoint path." >&2
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
printf "index\tstatus\tcheckpoint\texperiment\tlog\taccuracy\tnum_rollouts\n" > "$SUMMARY_FILE"

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
accuracy = ""
num_rollouts = ""

for match in re.finditer(r"(?:Initial|Final)? validation metrics:\s*(\{.*?\})", text, re.S):
    raw = match.group(1)
    try:
        metrics = ast.literal_eval(raw)
    except Exception:
        continue
    accuracy = metrics.get("val/test_score/all", metrics.get("test_score/all", accuracy))
    num_rollouts = metrics.get(
        "val/test_score/num_rollouts",
        metrics.get("test_score/num_rollouts", num_rollouts),
    )

if accuracy == "":
    matches = re.findall(r"['\"](?:val/)?test_score/all['\"]\s*:\s*([-+0-9.eE]+)", text)
    if matches:
        accuracy = matches[-1]
if num_rollouts == "":
    matches = re.findall(r"['\"](?:val/)?test_score/num_rollouts['\"]\s*:\s*([-+0-9.eE]+)", text)
    if matches:
        num_rollouts = matches[-1]

print(f"{accuracy}\t{num_rollouts}")
PY
}

echo "Validating ${#CHECKPOINTS[@]} checkpoint(s)"
echo "Episodes per checkpoint: $TARGET_ROLLOUTS_PER_CHECKPOINT"
echo "Logs: $LOG_DIR"

for idx in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$idx]}"
    lora_adapter_path="$(resolve_lora_adapter_path "$checkpoint")"
    display_idx=$((idx + 1))
    parent="$(basename "$(dirname "${lora_adapter_path%/}")")"
    leaf="$(basename "${lora_adapter_path%/}")"
    safe_name="$(sanitize_name "${parent}_${leaf}")"
    experiment_name="multi-val-${display_idx}-${safe_name}"
    log_file="$LOG_DIR/${display_idx}_${safe_name}.log"
    status="ok"

    echo
    echo "[$display_idx/${#CHECKPOINTS[@]}] Validating: $checkpoint"
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
        accuracy=""
        num_rollouts=""
        echo "Validation failed for $checkpoint with exit code $exit_code" >&2
        if [ "$CONTINUE_ON_ERROR" != "true" ]; then
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$display_idx" "$status" "$checkpoint" "$experiment_name" "$log_file" "$accuracy" "$num_rollouts" >> "$SUMMARY_FILE"
            exit "$exit_code"
        fi
    else
        IFS=$'\t' read -r accuracy num_rollouts < <(extract_metrics "$log_file")
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$display_idx" "$status" "$checkpoint" "$experiment_name" "$log_file" "$accuracy" "$num_rollouts" >> "$SUMMARY_FILE"
    echo "Finished $checkpoint: status=$status accuracy=${accuracy:-unknown} rollouts=${num_rollouts:-unknown}"
done

echo
echo "All requested validations finished."
echo "Summary: $SUMMARY_FILE"
column -t -s $'\t' "$SUMMARY_FILE" || cat "$SUMMARY_FILE"
