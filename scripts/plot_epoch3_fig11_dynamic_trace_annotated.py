#!/usr/bin/env python3
"""Create annotated full and fallback-zoom views of the saved epoch-3 trace.

This script is read-only with respect to the trace and result artifacts. It
writes only new PNG files into the explicitly selected output directory.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "runs/current_valid_baseline/epoch3_baseline_validation_20260919/fig11/dynamic_trace.csv"
DEFAULT_OUT_DIR = TRACE.parent / "regime_annotated_20260923"

SEGMENTS = [
    (0, 39, "20 ns", "#dbeafe"),
    (40, 79, "80 ns", "#ffedd5"),
    (80, 119, "10 ns", "#dcfce7"),
    (120, 159, "40 ns", "#fef3c7"),
    (160, 199, "60 ns", "#fce7f3"),
    (200, 239, "120 ns", "#ede9fe"),
]
CANDIDATE_NGS = [1, 4, 8, 16, 32, 64, 128]
BOUNDARIES = [40, 80, 120, 160, 200]


def read_trace(trace_path: Path) -> list[dict[str, str]]:
    with trace_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def boolean(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() == "true"


def add_regime_background(ax: plt.Axes, x_min: float, x_max: float) -> None:
    for start, end, label, color in SEGMENTS:
        if end < x_min or start > x_max:
            continue
        left = max(start - 0.5, x_min)
        right = min(end + 0.5, x_max)
        ax.axvspan(left, right, color=color, alpha=0.42, zorder=0)
        midpoint = (start + end) / 2.0
        if x_min <= midpoint <= x_max:
            ax.text(
                midpoint,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                color="#243047",
                zorder=5,
            )
    for boundary in BOUNDARIES:
        if x_min <= boundary <= x_max:
            ax.axvline(boundary, color="#475569", linestyle="--", linewidth=1.1, alpha=0.75, zorder=1)


def style_axis(ax: plt.Axes, x_min: float, x_max: float) -> None:
    add_regime_background(ax, x_min, x_max)
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(False)


def make_figure(rows: list[dict[str, str]], fallback_step: int, x_max: float, title: str, output: Path) -> None:
    x = np.asarray([int(row["time_step"]) for row in rows], dtype=int)
    ng = np.asarray([int(row["current_ng"]) for row in rows], dtype=int)
    nmse = np.asarray([float(row["nmse_db"]) for row in rows], dtype=float)
    finite_nmse = nmse[np.isfinite(nmse)]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
    ax_ng, ax_nmse = axes
    for ax in axes:
        style_axis(ax, 0, x_max)

    ax_ng.step(x, ng, where="post", color="#1f4e79", linewidth=2.4, zorder=3, label="current Ng")
    ax_ng.set_ylabel("Ng", fontsize=14)
    ax_ng.set_yticks(CANDIDATE_NGS)
    ax_ng.set_ylim(0.5, 140)
    ax_ng.tick_params(labelsize=11)

    ax_nmse.plot(x, nmse, color="#111827", linewidth=2.0, marker="o", markersize=2.8, zorder=3, label="omitted NMSE")
    ax_nmse.set_ylabel("Omitted NMSE [dB]", fontsize=14)
    ax_nmse.set_xlabel("Time step", fontsize=14)
    ax_nmse.tick_params(labelsize=11)
    if finite_nmse.size:
        y_min = float(np.min(finite_nmse))
        y_max = float(np.max(finite_nmse))
        pad = max(1.0, 0.08 * (y_max - y_min))
        ax_nmse.set_ylim(y_min - pad, y_max + pad)

    for ax in axes:
        ax.axvline(fallback_step, color="#dc2626", linestyle="-", linewidth=2.0, alpha=0.9, zorder=4)

    ax_ng.annotate(
        "False trigger at 10 ns → Ng=1 fallback",
        xy=(fallback_step, 64),
        xytext=(min(fallback_step - 25, x_max - 20), 112),
        fontsize=12,
        color="#b91c1c",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.8},
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#fca5a5"},
        zorder=6,
    )

    if fallback_step + 1 <= x_max:
        y_text = float(np.max(finite_nmse) - 0.12 * (np.max(finite_nmse) - np.min(finite_nmse))) if finite_nmse.size else 0.0
        ax_nmse.text(
            fallback_step + 1.2,
            y_text,
            "Ng=1: no omitted subcarriers,\nNMSE undefined",
            fontsize=12,
            color="#b91c1c",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#fca5a5"},
            zorder=6,
        )

    # Keep the legend away from the first regime label at the top center.
    ax_ng.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax_nmse.legend(loc="lower left", fontsize=11, framealpha=0.9)
    if x_max <= fallback_step + 5:
        ax_nmse.set_xticks(np.arange(0, x_max + 1, 10))
    else:
        ax_nmse.set_xticks(np.arange(0, int(x_max) + 1, 20))
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Background bands show the saved channel-regime sequence; dashed lines mark regime boundaries.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.955))
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=TRACE, help="Saved dynamic_trace.csv (read-only).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="New directory for the two PNG outputs; existing files are never replaced.",
    )
    args = parser.parse_args()
    trace_path = args.trace if args.trace.is_absolute() else ROOT / args.trace
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_out = output_dir / "epoch3_dynamic_trace_full.png"
    zoom_out = output_dir / "epoch3_dynamic_trace_false_trigger_zoom.png"
    if full_out.exists() or zoom_out.exists():
        raise FileExistsError(f"Refusing to overwrite existing visualization in {output_dir}")

    rows = read_trace(trace_path)
    fallback_rows = [row for row in rows if boolean(row, "full_feedback_fallback")]
    if not fallback_rows:
        raise RuntimeError("No full_feedback_fallback=True row found in saved trace.")
    fallback_step = int(fallback_rows[0]["time_step"])
    fallback_regime = fallback_rows[0]["regime"]
    if fallback_regime != "10 ns":
        raise RuntimeError(f"Expected false trigger in 10 ns, found {fallback_regime} at step {fallback_step}.")
    if not any(int(row["time_step"]) == fallback_step + 1 and row["nmse_db"].lower() == "nan" for row in rows):
        raise RuntimeError("Trace does not show undefined NMSE immediately after fallback.")

    first_ng1_step = next(int(row["time_step"]) for row in rows if int(row["current_ng"]) == 1)
    make_figure(rows, fallback_step, 239, "Epoch3 Fig.11-style dynamic trace — full sequence (0–239)", full_out)
    make_figure(rows, fallback_step, fallback_step + 5, "Epoch3 Fig.11-style dynamic trace — false-trigger zoom", zoom_out)
    print(f"trace={trace_path}")
    print(f"false_trigger_step={fallback_step}")
    print(f"first_ng1_step={first_ng1_step}")
    print(f"false_trigger_regime={fallback_regime}")
    print(f"zoom_xlim=0..{fallback_step + 5}")
    print(f"full_png={full_out}")
    print(f"zoom_png={zoom_out}")


if __name__ == "__main__":
    main()
