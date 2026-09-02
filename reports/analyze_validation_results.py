#!/usr/bin/env python3
"""Audit SimpleVLA-RL validation artifacts and emit reproducible statistics.

The script intentionally treats raw validation log lines as the authority for
checkpoint evaluations. The supplied summary CSV is cross-checked against those
logs rather than trusted without verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OVERALL_RE = re.compile(r"val/test_score/libero_spatial:(\d+(?:\.\d+)?)")
TASK_RE = re.compile(r"val/test_score_task/libero_spatial_task_(\d+):(\d+(?:\.\d+)?)")
CI_RE = re.compile(r"val/test_score/all_ci95_halfwidth:(\d+(?:\.\d+)?)")
N_RE = re.compile(r"val/test_score/num_rollouts:(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class ValidationRecord:
    file: str
    overall: float
    tasks: tuple[float, ...]
    ci95_halfwidth: float
    num_rollouts: int


def _last_metric_line(text: str) -> str:
    lines = [ANSI_RE.sub("", line) for line in text.splitlines()]
    candidates = [line for line in lines if "val/test_score/libero_spatial:" in line]
    if not candidates:
        raise ValueError("no aggregate LIBERO-Spatial validation metric found")
    return candidates[-1]


def parse_log(path: Path) -> ValidationRecord:
    line = _last_metric_line(path.read_text(encoding="utf-8", errors="replace"))
    overall_match = OVERALL_RE.search(line)
    ci_match = CI_RE.search(line)
    n_match = N_RE.search(line)
    task_pairs = [(int(task), float(value)) for task, value in TASK_RE.findall(line)]
    if overall_match is None or ci_match is None or n_match is None:
        raise ValueError(f"incomplete final metric line in {path.name}")
    if sorted(task for task, _ in task_pairs) != list(range(10)):
        raise ValueError(f"expected exactly tasks 0..9 in {path.name}, got {task_pairs}")
    task_map = dict(task_pairs)
    return ValidationRecord(
        file=path.name,
        overall=float(overall_match.group(1)),
        tasks=tuple(task_map[i] for i in range(10)),
        ci95_halfwidth=float(ci_match.group(1)),
        num_rollouts=int(float(n_match.group(1))),
    )


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_initial(path: Path) -> tuple[float, tuple[float, ...]]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values[row["metric"]] = float(row["value"])
            except (KeyError, ValueError):
                continue
    overall = values["val/test_score/all"]
    tasks = tuple(values[f"val/test_score_task/libero_spatial_task_{i}"] for i in range(10))
    return overall, tasks


def normal_diff(success_a: float, n_a: int, success_b: float, n_b: int) -> dict[str, float]:
    """Unpaired normal approximation; paired episode outcomes are unavailable."""
    diff = success_a - success_b
    se = math.sqrt(success_a * (1 - success_a) / n_a + success_b * (1 - success_b) / n_b)
    z = diff / se if se else math.inf
    return {
        "difference": diff,
        "standard_error": se,
        "ci95_low": diff - 1.96 * se,
        "ci95_high": diff + 1.96 * se,
        "z": z,
        "two_sided_p_approx": math.erfc(abs(z) / math.sqrt(2)),
    }


def audit(root: Path) -> dict:
    plots = root / "plots"
    summary_rows = read_summary(plots / "validation_metrics_summary.csv")
    initial, initial_tasks = read_initial(plots / "initial_default_checkpoint_distribution.csv")

    raw_main = {path.name: parse_log(path) for path in sorted(root.glob("*.log"))}
    mismatches: list[str] = []
    by_ablation: dict[str, list[dict]] = {}
    for row in summary_rows:
        file_name = row["file"]
        if file_name not in raw_main:
            mismatches.append(f"summary references missing log: {file_name}")
            continue
        record = raw_main[file_name]
        csv_overall = float(row["success_rate"])
        csv_tasks = tuple(float(row[f"libero_spatial_task_{i}"]) for i in range(10))
        if not math.isclose(csv_overall, record.overall, abs_tol=5e-4):
            mismatches.append(f"overall mismatch for {file_name}: CSV={csv_overall}, log={record.overall}")
        for task_id, (csv_value, log_value) in enumerate(zip(csv_tasks, record.tasks)):
            if not math.isclose(csv_value, log_value, abs_tol=5e-4):
                mismatches.append(
                    f"task {task_id} mismatch for {file_name}: CSV={csv_value}, log={log_value}"
                )
        by_ablation.setdefault(row["ablation"], []).append(
            {
                "step": int(row["step"]),
                "overall": record.overall,
                "tasks": record.tasks,
                "ci95_halfwidth": record.ci95_halfwidth,
                "num_rollouts": record.num_rollouts,
                "file": record.file,
            }
        )

    summaries: dict[str, dict] = {}
    for name, rows in by_ablation.items():
        rows.sort(key=lambda row: row["step"])
        best = max(rows, key=lambda row: row["overall"])
        summaries[name] = {
            "first": rows[0],
            "best": best,
            "last": rows[-1],
            "best_gain_over_initial": best["overall"] - initial,
            "best_task_change_from_initial": [
                value - baseline for value, baseline in zip(best["tasks"], initial_tasks)
            ],
        }

    terminal = by_ablation["Terminal only (coef=5)"]
    dense = by_ablation["Dense 0.05/0.05 + terminal 5"]
    terminal_by_step = {row["step"]: row for row in terminal}
    aligned = [
        {
            "step": row["step"],
            "dense_minus_terminal": row["overall"] - terminal_by_step[row["step"]]["overall"],
        }
        for row in dense
        if row["step"] in terminal_by_step
    ]
    aligned_mean = sum(row["dense_minus_terminal"] for row in aligned) / len(aligned)

    best_terminal = summaries["Terminal only (coef=5)"]["best"]
    best_dense = summaries["Dense 0.05/0.05 + terminal 5"]["best"]
    dynamic_dir = root / "trainings_with_dynamic_clip"
    dynamic = [asdict(parse_log(path)) for path in sorted(dynamic_dir.glob("*.log"))]

    malformed_dynamic_summary = False
    dynamic_summary_path = dynamic_dir / "summary.tsv"
    if dynamic_summary_path.exists():
        with dynamic_summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        malformed_dynamic_summary = bool(rows) and all(
            row.get("num_rollouts", "") == "" and float(row.get("accuracy", "nan")) >= 1 for row in rows
        )

    return {
        "initial": {"overall": initial, "tasks": initial_tasks, "provenance": "standalone CSV"},
        "main_log_count": len(raw_main),
        "summary_row_count": len(summary_rows),
        "summary_vs_raw_mismatches": mismatches,
        "ablations": summaries,
        "aligned_dense_vs_terminal": aligned,
        "aligned_dense_minus_terminal_mean": aligned_mean,
        "best_dense_vs_best_terminal_unpaired_normal": normal_diff(
            best_dense["overall"], best_dense["num_rollouts"],
            best_terminal["overall"], best_terminal["num_rollouts"],
        ),
        "best_dense_vs_initial_unpaired_normal": normal_diff(
            best_dense["overall"], best_dense["num_rollouts"], initial, 300
        ),
        "dynamic": dynamic,
        "dynamic_summary_tsv_malformed": malformed_dynamic_summary,
    }


def self_test() -> None:
    sample = (
        "\x1b[36m(pid=1)\x1b[0m step:0 - val/test_score/libero_spatial:0.600 - "
        + " - ".join(
            f"val/test_score_task/libero_spatial_task_{i}:{i / 10:.3f}" for i in range(10)
        )
        + " - val/test_score/all_ci95_halfwidth:0.050 - val/test_score/num_rollouts:300.000"
    )
    assert "val/test_score/libero_spatial:0.600" in _last_metric_line(sample)
    try:
        _last_metric_line("no validation here")
    except ValueError:
        pass
    else:
        raise AssertionError("missing metrics must be rejected")
    result = normal_diff(0.6, 300, 0.6, 300)
    assert result["difference"] == 0.0 and result["two_sided_p_approx"] == 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, help="extracted validation_results directory")
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: ok")
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    result = audit(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
