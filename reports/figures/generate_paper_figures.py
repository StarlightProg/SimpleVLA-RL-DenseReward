#!/usr/bin/env python3
"""Generate publication-ready SimpleVLA-RL validation figures as SVG/PDF/PNG.

The script uses only the Python standard library for SVG construction. If
`rsvg-convert` is available, it also exports vector PDF and 2400-pixel PNG.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


INK = "#172033"
MUTED = "#5F6B7A"
GRID = "#DDE3EA"
BASELINE = "#6B7280"
BLUE = "#2F6BBD"
ORANGE = "#D97706"
PURPLE = "#7C5CC4"
TEAL = "#238B8E"
GOLD = "#B58B00"
PALE_GOLD = "#FFF8DD"
WHITE = "#FFFFFF"
FONT = "Arial, Helvetica, sans-serif"


METHOD_STYLE = {
    "Terminal only (coef=5)": (BLUE, "circle", "solid"),
    "Dense 0.05/0.05 + terminal 5": (ORANGE, "square", "solid"),
    "Dense clipped + terminal 5": (PURPLE, "triangle", "dash"),
    "Dense 0.05/0.05 + terminal 1": (TEAL, "diamond", "dash"),
    "Phase 0.2 + terminal 1": (GOLD, "cross", "dash"),
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, width: int, height: int, title: str, description: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{esc(title)}</title>",
            f"<desc id=\"desc\">{esc(description)}</desc>",
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]

    def line(self, x1, y1, x2, y2, *, stroke=INK, width=2, dash=None, opacity=1.0):
        attrs = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{attrs}/>'
        )

    def rect(self, x, y, w, h, *, fill="none", stroke="none", width=1, rx=0, opacity=1.0):
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" rx="{rx}" opacity="{opacity}"/>'
        )

    def circle(self, x, y, r, *, fill=WHITE, stroke=INK, width=2, opacity=1.0):
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    def polygon(self, points, *, fill=WHITE, stroke=INK, width=2):
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polygon points="{point_text}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def polyline(self, points, *, stroke=INK, width=3, dash=None, opacity=1.0):
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        attrs = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{point_text}" fill="none" stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{attrs}/>'
        )

    def text(self, x, y, value, *, size=18, fill=INK, anchor="start", weight=400,
             rotate=None, family=FONT, style="normal"):
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}"{transform}>'
            f'{esc(value)}</text>'
        )

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def marker(svg: Svg, x: float, y: float, shape: str, color: str, size: float = 9,
           *, fill: str | None = None):
    fill = color if fill is None else fill
    if shape == "circle":
        svg.circle(x, y, size, fill=fill, stroke=color, width=3)
    elif shape == "square":
        svg.rect(x - size, y - size, 2 * size, 2 * size, fill=fill, stroke=color, width=3, rx=1)
    elif shape == "triangle":
        svg.polygon([(x, y - size - 1), (x - size, y + size), (x + size, y + size)],
                    fill=fill, stroke=color, width=3)
    elif shape == "diamond":
        svg.polygon([(x, y - size - 1), (x - size, y), (x, y + size + 1), (x + size, y)],
                    fill=fill, stroke=color, width=3)
    elif shape == "cross":
        svg.line(x - size, y - size, x + size, y + size, stroke=color, width=4)
        svg.line(x - size, y + size, x + size, y - size, stroke=color, width=4)
    else:
        raise ValueError(f"unsupported marker: {shape}")


def title_block(svg: Svg, title: str, subtitle: str, panel: str | None = None):
    if panel:
        svg.text(54, 60, panel, size=31, weight=700)
        title_x = 98
    else:
        title_x = 70
    svg.text(title_x, 60, title, size=32, weight=700)
    svg.text(title_x, 98, subtitle, size=18, fill=MUTED)


def linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max <= domain_min:
        raise ValueError("invalid scale domain")
    return range_min + (value - domain_min) / (domain_max - domain_min) * (range_max - range_min)


def percent(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def load_main(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "method": row["method"],
                "step": int(row["step"]),
                "success": float(row["success"]),
                "ci": float(row["ci95_halfwidth"]),
                "n": int(row["num_rollouts"]),
                "tasks": [float(row[f"task_{i}"]) for i in range(10)],
            }
            if parsed["method"] not in METHOD_STYLE:
                raise ValueError(f"unknown method {parsed['method']}")
            if not (0 <= parsed["success"] <= 1 and 0 <= parsed["ci"] <= 1):
                raise ValueError(f"invalid rate in {row}")
            if len(parsed["tasks"]) != 10 or any(not 0 <= value <= 1 for value in parsed["tasks"]):
                raise ValueError(f"invalid task vector in {row}")
            rows.append(parsed)
    if len(rows) != 16:
        raise ValueError(f"expected 16 audited main rows, got {len(rows)}")
    return rows


def draw_xy_axes(svg: Svg, *, left: float, top: float, right: float, bottom: float,
                 x_min: float, x_max: float, x_ticks, y_min: float, y_max: float, y_ticks,
                 x_label: str, y_label: str):
    x = lambda value: linear(value, x_min, x_max, left, right)
    y = lambda value: linear(value, y_min, y_max, bottom, top)
    for value in y_ticks:
        py = y(value)
        svg.line(left, py, right, py, stroke=GRID, width=1.5)
        svg.text(left - 16, py + 6, percent(value, 0), size=16, fill=MUTED, anchor="end")
    for value in x_ticks:
        px = x(value)
        svg.line(px, bottom, px, bottom + 7, stroke=INK, width=1.5)
        svg.text(px, bottom + 31, str(value), size=16, fill=MUTED, anchor="middle")
    svg.line(left, top, left, bottom, stroke=INK, width=2)
    svg.line(left, bottom, right, bottom, stroke=INK, width=2)
    svg.text((left + right) / 2, bottom + 72, x_label, size=19, anchor="middle")
    svg.text(left - 92, (top + bottom) / 2, y_label, size=19, anchor="middle", rotate=-90)
    return x, y


def draw_vertical_error(svg: Svg, x: float, y_low: float, y_high: float, color: str):
    svg.line(x, y_low, x, y_high, stroke=color, width=2, opacity=0.62)
    svg.line(x - 7, y_low, x + 7, y_low, stroke=color, width=2, opacity=0.62)
    svg.line(x - 7, y_high, x + 7, y_high, stroke=color, width=2, opacity=0.62)


def draw_horizontal_error(svg: Svg, x_low: float, x_high: float, y: float, color: str):
    svg.line(x_low, y, x_high, y, stroke=color, width=3, opacity=0.65)
    svg.line(x_low, y - 8, x_low, y + 8, stroke=color, width=2, opacity=0.65)
    svg.line(x_high, y - 8, x_high, y + 8, stroke=color, width=2, opacity=0.65)


def figure_primary(rows, output: Path):
    wanted = {"Terminal only (coef=5)", "Dense 0.05/0.05 + terminal 5"}
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] in wanted:
            grouped[row["method"]].append(row)
    svg = Svg(1600, 1000, "LIBERO-Spatial validation success: primary reward comparison",
              "Terminal-only and dense-labelled checkpoint success with logged 95 percent confidence intervals.")
    title_block(svg, "LIBERO-Spatial validation success",
                "Primary reward comparison · 300 rollouts/checkpoint · logged 95% confidence intervals")
    left, top, right, bottom = 155, 170, 1490, 815
    x, y = draw_xy_axes(
        svg, left=left, top=top, right=right, bottom=bottom,
        x_min=0, x_max=180, x_ticks=[0, 40, 80, 120, 160],
        y_min=0.50, y_max=0.72, y_ticks=[0.50, 0.54, 0.58, 0.62, 0.66, 0.70],
        x_label="Training checkpoint step", y_label="Validation success",
    )
    baseline_y = y(0.58)
    svg.line(left, baseline_y, right, baseline_y, stroke=BASELINE, width=2, dash="9 8")
    svg.rect(right - 164, baseline_y - 31, 154, 25, fill=WHITE, opacity=0.92)
    svg.text(right - 12, baseline_y - 12, "Initial 58.0%", size=16, fill=BASELINE, anchor="end", weight=600)

    for method in ["Terminal only (coef=5)", "Dense 0.05/0.05 + terminal 5"]:
        color, shape, line_style = METHOD_STYLE[method]
        points = sorted(grouped[method], key=lambda row: row["step"])
        coords = [(x(row["step"]), y(row["success"])) for row in points]
        svg.polyline(coords, stroke=color, width=4, dash="10 7" if line_style == "dash" else None)
        for row, (px, py) in zip(points, coords):
            draw_vertical_error(svg, px, y(row["success"] - row["ci"]), y(row["success"] + row["ci"]), color)
            marker(svg, px, py, shape, color, size=9)
            svg.text(px, py - 19, percent(row["success"]), size=15, fill=color, anchor="middle", weight=700)

    legend_y = 132
    for idx, method in enumerate(["Terminal only (coef=5)", "Dense 0.05/0.05 + terminal 5"]):
        color, shape, _ = METHOD_STYLE[method]
        lx = 610 + idx * 410
        svg.line(lx, legend_y, lx + 45, legend_y, stroke=color, width=4)
        marker(svg, lx + 22, legend_y, shape, color, size=7)
        svg.text(lx + 58, legend_y + 6, method, size=17)

    svg.text(left, 932,
             "Focused y-axis. Error bars describe checkpoint-evaluation uncertainty, not training-seed variability.",
             size=16, fill=MUTED)
    output.write_text(svg.finish(), encoding="utf-8")


def figure_all_ablations(rows, output: Path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    svg = Svg(1800, 1080, "LIBERO-Spatial validation success across reward ablations",
              "Five reward configurations evaluated at sparse training checkpoints with logged confidence intervals.")
    title_block(svg, "LIBERO-Spatial validation success across reward ablations",
                "Measured checkpoints only · 300 rollouts/checkpoint · logged 95% confidence intervals")
    left, top, right, bottom = 155, 190, 1285, 875
    x, y = draw_xy_axes(
        svg, left=left, top=top, right=right, bottom=bottom,
        x_min=0, x_max=180, x_ticks=[0, 40, 80, 120, 160],
        y_min=0.50, y_max=0.72, y_ticks=[0.50, 0.54, 0.58, 0.62, 0.66, 0.70],
        x_label="Training checkpoint step", y_label="Validation success",
    )
    baseline_y = y(0.58)
    svg.line(left, baseline_y, right, baseline_y, stroke=BASELINE, width=2, dash="9 8")
    svg.text(right - 10, baseline_y - 12, "Initial 58.0%", size=15, fill=BASELINE, anchor="end", weight=600)

    method_order = list(METHOD_STYLE)
    for method in method_order:
        color, shape, line_style = METHOD_STYLE[method]
        points = sorted(grouped[method], key=lambda row: row["step"])
        coords = [(x(row["step"]), y(row["success"])) for row in points]
        svg.polyline(coords, stroke=color, width=3.5, dash="10 7" if line_style == "dash" else None)
        for row, (px, py) in zip(points, coords):
            draw_vertical_error(svg, px, y(row["success"] - row["ci"]), y(row["success"] + row["ci"]), color)
            marker(svg, px, py, shape, color, size=8)

    legend_x, legend_y = 1345, 235
    svg.text(legend_x, 185, "Reward configuration", size=19, weight=700)
    for idx, method in enumerate(method_order):
        color, shape, line_style = METHOD_STYLE[method]
        py = legend_y + idx * 88
        svg.line(legend_x, py, legend_x + 52, py, stroke=color, width=4,
                 dash="10 7" if line_style == "dash" else None)
        marker(svg, legend_x + 25, py, shape, color, size=7)
        label = method.replace("Terminal only (coef=5)", "Terminal only, coef. 5")
        svg.text(legend_x + 68, py + 6, label, size=16)
    svg.text(legend_x, legend_y + 5 * 88 + 22, "Lines connect sparse observed", size=15, fill=MUTED)
    svg.text(legend_x, legend_y + 5 * 88 + 45, "checkpoints; they are not", size=15, fill=MUTED)
    svg.text(legend_x, legend_y + 5 * 88 + 68, "interpolated measurements.", size=15, fill=MUTED)
    svg.text(left, 1000,
             "Focused y-axis. Reward names follow the archive labels; exact training configurations require provenance verification.",
             size=16, fill=MUTED)
    output.write_text(svg.finish(), encoding="utf-8")


def figure_best(rows, output: Path):
    best = {}
    for row in rows:
        if row["method"] not in best or row["success"] > best[row["method"]]["success"]:
            best[row["method"]] = row
    order = sorted(best, key=lambda method: best[method]["success"], reverse=True)
    svg = Svg(1650, 940, "Best observed validation checkpoint by reward configuration",
              "Best checkpoint success and logged 95 percent confidence intervals for five configurations.")
    title_block(svg, "Best observed validation checkpoint by reward configuration",
                "Descriptive best-of-run comparison · 300 rollouts/checkpoint · logged 95% confidence intervals")
    left, right, top, bottom = 600, 1510, 190, 735
    x_min, x_max = 0.50, 0.72
    x = lambda value: linear(value, x_min, x_max, left, right)
    for tick in [0.50, 0.54, 0.58, 0.62, 0.66, 0.70]:
        px = x(tick)
        svg.line(px, top, px, bottom, stroke=GRID, width=1.5)
        svg.text(px, bottom + 34, percent(tick, 0), size=16, fill=MUTED, anchor="middle")
    svg.line(left, bottom, right, bottom, stroke=INK, width=2)
    baseline_x = x(0.58)
    svg.line(baseline_x, top, baseline_x, bottom, stroke=BASELINE, width=2, dash="9 8")
    svg.text(baseline_x, top - 18, "Initial 58.0%", size=15, fill=BASELINE, anchor="middle", weight=600)

    gap = (bottom - top) / len(order)
    for idx, method in enumerate(order):
        row = best[method]
        color, shape, _ = METHOD_STYLE[method]
        py = top + gap * (idx + 0.5)
        if idx % 2 == 0:
            svg.rect(45, py - gap * 0.43, 1465, gap * 0.86, fill="#F8FAFC")
        label = method.replace("Terminal only (coef=5)", "Terminal only, coefficient 5")
        svg.text(left - 30, py + 7, label, size=17, anchor="end", weight=600)
        draw_horizontal_error(svg, x(row["success"] - row["ci"]), x(row["success"] + row["ci"]), py, color)
        marker(svg, x(row["success"]), py, shape, color, size=10)
        svg.text(min(x(row["success"] + row["ci"]) + 18, right - 5), py + 7,
                 f"{percent(row['success'])}  ·  step {row['step']}", size=16, fill=color, weight=700)
    svg.text((left + right) / 2, bottom + 82, "Validation success", size=19, anchor="middle")
    svg.text(70, 868,
             "Best checkpoint selection is post hoc; intervals describe evaluation uncertainty only, not training-run variability.",
             size=16, fill=MUTED)
    output.write_text(svg.finish(), encoding="utf-8")


def figure_tasks(rows, output: Path):
    def best_row(method):
        candidates = [row for row in rows if row["method"] == method]
        return max(candidates, key=lambda row: row["success"])

    terminal = best_row("Terminal only (coef=5)")
    dense = best_row("Dense 0.05/0.05 + terminal 5")
    initial = [0.367, 0.100, 0.867, 0.800, 0.767, 0.033, 0.767, 0.633, 0.767, 0.700]
    svg = Svg(1650, 1080, "Per-task validation success at selected checkpoints",
              "Initial, best terminal-only, and best dense-labelled success for ten LIBERO-Spatial tasks.")
    title_block(svg, "Per-task validation success at selected checkpoints",
                "Initial vs best terminal-only (step 170) vs best dense-labelled (step 160) · 30 trials/task")
    left, right, top, bottom = 230, 1530, 190, 900
    x = lambda value: linear(value, 0, 1, left, right)
    row_gap = (bottom - top) / 10
    for task in [1, 5]:
        py = top + row_gap * (task + 0.5)
        svg.rect(left - 120, py - row_gap * 0.44, right - left + 120, row_gap * 0.88, fill=PALE_GOLD)
    for tick in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        px = x(tick)
        svg.line(px, top, px, bottom, stroke=GRID, width=1.5)
        svg.text(px, bottom + 34, percent(tick, 0), size=16, fill=MUTED, anchor="middle")
    svg.line(left, bottom, right, bottom, stroke=INK, width=2)
    for task in range(10):
        py = top + row_gap * (task + 0.5)
        svg.text(left - 28, py + 7, f"Task {task}", size=17, anchor="end", weight=600)
        svg.line(x(terminal["tasks"][task]), py, x(dense["tasks"][task]), py,
                 stroke="#AEB8C4", width=4)
        marker(svg, x(initial[task]), py, "circle", BASELINE, size=7, fill=WHITE)
        marker(svg, x(terminal["tasks"][task]), py, "circle", BLUE, size=8)
        marker(svg, x(dense["tasks"][task]), py, "square", ORANGE, size=8)
    svg.text((left + right) / 2, bottom + 82, "Task success", size=19, anchor="middle")

    legend_y = 137
    marker(svg, 470, legend_y, "circle", BASELINE, size=7, fill=WHITE)
    svg.text(488, legend_y + 6, "Initial", size=16)
    marker(svg, 660, legend_y, "circle", BLUE, size=8)
    svg.text(680, legend_y + 6, "Terminal only best", size=16)
    marker(svg, 945, legend_y, "square", ORANGE, size=8)
    svg.text(966, legend_y + 6, "Dense-labelled best", size=16)
    svg.rect(1230, legend_y - 12, 28, 20, fill=PALE_GOLD)
    svg.text(1270, legend_y + 6, "Persistent low-success tasks", size=16)
    svg.text(70, 1015,
             "Per-task estimates have 3.33-point resolution (n=30). No task-level uncertainty bars are shown to avoid clutter.",
             size=16, fill=MUTED)
    output.write_text(svg.finish(), encoding="utf-8")


def figure_auxiliary(audit_path: Path, output: Path):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = {row["file"]: row for row in audit["dynamic"]}
    specs = [
        ("Terminal · step 19", "1_termianal_global_step_19_lora_adapter.log", BLUE, "circle"),
        ("Terminal · step 59", "2_terminal_global_step_59_lora_adapter.log", BLUE, "circle"),
        ("Dense 0.1/0.1 · step 19", "1_dense_01_01_global_step_19_lora_adapter.log", ORANGE, "square"),
        ("Dense 0.1 · step 79", "3_dense_01_global_step_79_lora_adapter.log", ORANGE, "square"),
        ("Phase 0.2 · step 19", "2_phase_02_global_step_19_lora_adapter.log", GOLD, "cross"),
        ("Phase 0.2 · step 79", "2_phase_02_79_lora_adapter.log", GOLD, "cross"),
        ("Dense clip 0.3/0.5 · step 79", "1_dense_clip_03_05_step_79_lora_adapter.log", PURPLE, "triangle"),
    ]
    missing = [file for _, file, _, _ in specs if file not in rows]
    if missing:
        raise ValueError(f"missing auxiliary records: {missing}")
    svg = Svg(1600, 940, "Auxiliary dynamic-filtering checkpoint evaluations",
              "Seven non-matched checkpoint evaluations with logged confidence intervals and rollout counts.")
    title_block(svg, "Auxiliary dynamic-filtering checkpoint evaluations",
                "Diagnostic only · checkpoint steps and validation denominators differ")
    left, right, top, bottom = 540, 1480, 180, 760
    x_min, x_max = 0.50, 0.70
    x = lambda value: linear(value, x_min, x_max, left, right)
    for tick in [0.50, 0.54, 0.58, 0.62, 0.66, 0.70]:
        px = x(tick)
        svg.line(px, top, px, bottom, stroke=GRID, width=1.5)
        svg.text(px, bottom + 34, percent(tick, 0), size=16, fill=MUTED, anchor="middle")
    svg.line(left, bottom, right, bottom, stroke=INK, width=2)
    gap = (bottom - top) / len(specs)
    for idx, (label, file, color, shape) in enumerate(specs):
        row = rows[file]
        py = top + gap * (idx + 0.5)
        if idx % 2 == 0:
            svg.rect(65, py - gap * 0.43, 1415, gap * 0.86, fill="#F8FAFC")
        svg.text(left - 28, py + 7, label, size=17, anchor="end", weight=600)
        draw_horizontal_error(svg, x(row["overall"] - row["ci95_halfwidth"]),
                              x(row["overall"] + row["ci95_halfwidth"]), py, color)
        marker(svg, x(row["overall"]), py, shape, color, size=9)
        svg.text(min(x(row["overall"] + row["ci95_halfwidth"]) + 16, right - 10), py + 7,
                 f"{percent(row['overall'])} · n={row['num_rollouts']}", size=16, fill=color, weight=700)
    svg.text((left + right) / 2, bottom + 82, "Validation success", size=19, anchor="middle")
    svg.text(70, 868,
             "Do not use this figure as a matched reward comparison; it documents auxiliary checkpoint behavior only.",
             size=16, fill=MUTED)
    output.write_text(svg.finish(), encoding="utf-8")


def export(svg_path: Path):
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return
    subprocess.run([converter, "-f", "pdf", "-o", str(svg_path.with_suffix(".pdf")), str(svg_path)], check=True)
    # 2400 px on the long edge is suitable for common two-column paper layouts.
    subprocess.run([converter, "-w", "2400", "-o", str(svg_path.with_suffix(".png")), str(svg_path)], check=True)


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here / "paper_figure_data.csv")
    parser.add_argument("--audit", type=Path, default=here.parent / "validation_audit.json")
    parser.add_argument("--output-dir", type=Path, default=here / "paper")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_main(args.data)
    figures = [
        ("fig1_primary_reward_comparison.svg", lambda path: figure_primary(rows, path)),
        ("fig2_all_reward_ablations.svg", lambda path: figure_all_ablations(rows, path)),
        ("fig3_best_checkpoint_intervals.svg", lambda path: figure_best(rows, path)),
        ("fig4_per_task_comparison.svg", lambda path: figure_tasks(rows, path)),
        ("figS1_auxiliary_dynamic_runs.svg", lambda path: figure_auxiliary(args.audit, path)),
    ]
    for filename, build in figures:
        path = args.output_dir / filename
        build(path)
        export(path)
        print(path)


if __name__ == "__main__":
    main()
