#!/usr/bin/env python3
"""Plot the saved Ng-conditioned q99 Fig.11-style dynamic trace.

The trace and result files are read-only inputs.  This script writes only new
PNG files to an explicitly selected output directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "runs/current_valid_baseline/epoch3_ng_conditioned_threshold_fig11_20260919_final/dynamic_trace.csv"
DEFAULT_OUT = ROOT / "runs/current_valid_baseline/epoch3_ng_conditioned_threshold_fig11_20260919_final/q99_annotated_20260923"

SEGMENTS = [
    (0, 39, "20 ns", "#dbeafe"),
    (40, 79, "80 ns", "#ffedd5"),
    (80, 119, "10 ns", "#dcfce7"),
    (120, 159, "40 ns", "#fef3c7"),
    (160, 199, "60 ns", "#fce7f3"),
    (200, 239, "120 ns", "#ede9fe"),
]
BOUNDARIES = [40, 80, 120, 160, 200]
NG_TICKS = [1, 4, 8, 16, 32, 64, 128]


def add_regime_background(ax: plt.Axes, x_min: float, x_max: float) -> None:
    for start, end, label, color in SEGMENTS:
        if end < x_min or start > x_max:
            continue
        left = max(start - 0.5, x_min)
        right = min(end + 0.5, x_max)
        ax.axvspan(left, right, color=color, alpha=0.42, zorder=0)
        midpoint = (start + end) / 2.0
        if x_min <= midpoint <= x_max:
            ax.text(midpoint, 0.98, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=12, fontweight="bold",
                    color="#243047", zorder=5)
    for boundary in BOUNDARIES:
        if x_min <= boundary <= x_max:
            ax.axvline(boundary, color="#475569", linestyle="--", linewidth=1.1,
                       alpha=0.75, zorder=1)


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def make_plot(df: pd.DataFrame, fallback_step: int, x_max: int, title: str, out: Path) -> None:
    x = pd.to_numeric(df["time_step"], errors="coerce")
    ng = pd.to_numeric(df["current_ng"], errors="coerce")
    epi = finite(df["epistemic"])
    tau = finite(df["tau_ng"])
    nmse = finite(df["nmse_db"])
    # Threshold is not a meaningful observed threshold after Ng=1 fallback,
    # because the omitted set is empty and epistemic/NMSE are undefined.
    tau = tau.where(ng != 1)
    visible = (x >= 0) & (x <= x_max)
    x, ng, epi, tau, nmse = x[visible], ng[visible], epi[visible], tau[visible], nmse[visible]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [0.9, 1.35, 1.0]})
    ax_ng, ax_epi, ax_nmse = axes
    for ax in axes:
        add_regime_background(ax, 0, x_max)
        ax.set_xlim(0, x_max)
        ax.grid(alpha=0.24, zorder=0)
        ax.set_axisbelow(False)
        ax.tick_params(labelsize=11)
        ax.axvline(fallback_step, color="#dc2626", linestyle="-", linewidth=2.0,
                   alpha=0.9, zorder=4)

    ax_ng.step(x, ng, where="post", color="#1f4e79", linewidth=2.4,
               label="Selected/current Ng", zorder=3)
    ax_ng.set_ylabel("Selected Ng", fontsize=14)
    ax_ng.set_yticks(NG_TICKS)
    ax_ng.set_ylim(0.5, 140)
    ax_ng.legend(loc="lower left", fontsize=11, framealpha=0.92)

    ax_epi.plot(x, epi, color="#7c3aed", linewidth=2.1, marker="o", markersize=2.6,
                label="Epistemic", zorder=3)
    ax_epi.plot(x, tau, color="#b45309", linestyle="--", linewidth=1.8,
                label="Ng-conditioned q99 threshold", zorder=3)
    ax_epi.set_yscale("log", base=10, subs=[])
    ax_epi.minorticks_off()
    ax_epi.set_ylabel("Epistemic / threshold (log scale)", fontsize=14)
    ax_epi.legend(loc="upper left", fontsize=10, framealpha=0.92)
    trigger = df[(df["ood_trigger"].astype(str).str.lower() == "true") &
                 (pd.to_numeric(df["time_step"], errors="coerce") <= x_max)]
    for _, row in trigger.iterrows():
        tx = int(row["time_step"])
        ty = float(row["epistemic"])
        ax_epi.scatter([tx], [ty], color="#dc2626", s=52, zorder=6, label="OOD trigger")
    handles, labels = ax_epi.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax_epi.legend(dedup.values(), dedup.keys(), loc="upper left", fontsize=10, framealpha=0.92)

    ax_nmse.plot(x, nmse, color="#111827", linewidth=2.0, marker="o", markersize=2.6,
                 label="Omitted NMSE", zorder=3)
    ax_nmse.set_ylabel("Omitted NMSE [dB]", fontsize=14)
    ax_nmse.set_xlabel("Streaming time step", fontsize=14)
    ax_nmse.legend(loc="lower left", fontsize=10, framealpha=0.92)
    vals = nmse[np.isfinite(nmse)]
    if len(vals):
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(1.0, 0.08 * (hi - lo))
        ax_nmse.set_ylim(lo - pad, hi + pad)

    ax_ng.annotate("False trigger at 10 ns → Ng=1 fallback",
                   xy=(fallback_step, 32),
                   xytext=(max(3, fallback_step - 27), 112),
                   fontsize=12, color="#b91c1c", fontweight="bold",
                   arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.8},
                   bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#fca5a5"},
                   zorder=7)
    if fallback_step + 1 <= x_max:
        ax_nmse.text(fallback_step + 1.2, 0.90,
                     "Ng=1: no omitted subcarriers,\nNMSE undefined",
                     transform=ax_nmse.transAxes, fontsize=11, color="#b91c1c",
                     ha="left", va="top",
                     bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#fca5a5"},
                     zorder=7)

    if x_max <= fallback_step + 5:
        ax_nmse.set_xticks(np.arange(0, x_max + 1, 10))
    else:
        ax_nmse.set_xticks(np.arange(0, x_max + 1, 20))
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.006,
             "Shaded bands: saved channel-regime sequence. Dashed lines: regime boundaries. "
             "Threshold is Ng-conditioned ID-only q99; values after Ng=1 fallback are omitted.",
             ha="center", fontsize=10, color="#475569")
    # Explicit spacing is more stable for the large top annotations than
    # tight_layout, and avoids an unnecessarily expensive log-axis layout pass.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.09, hspace=0.18)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    trace = args.trace if args.trace.is_absolute() else ROOT / args.trace
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / "epoch3_ng_conditioned_q99_dynamic_trace_full.png"
    zoom = out_dir / "epoch3_ng_conditioned_q99_dynamic_trace_false_trigger_zoom.png"
    if full.exists() or zoom.exists():
        raise FileExistsError(f"Refusing to overwrite existing files in {out_dir}")
    df = pd.read_csv(trace)
    required = {"time_step", "current_ng", "epistemic", "tau_ng", "nmse_db", "ood_trigger"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    trigger_rows = df[df["ood_trigger"].astype(str).str.lower() == "true"]
    if trigger_rows.empty:
        raise RuntimeError("No trigger row found in the saved q99 trace.")
    first = trigger_rows.iloc[0]
    fallback_step = int(first["time_step"])
    if str(first["regime"]) != "10 ns":
        raise RuntimeError(f"Expected 10 ns trigger, found {first['regime']} at {fallback_step}")
    make_plot(df, fallback_step, 239,
              "Epoch3 Fig.11-style dynamic trace — Ng-conditioned q99 threshold (full)", full)
    make_plot(df, fallback_step, fallback_step + 5,
              "Epoch3 Fig.11-style dynamic trace — Ng-conditioned q99 threshold (zoom)", zoom)
    print(f"trace={trace}")
    print(f"first_trigger_step={fallback_step}")
    print(f"trigger_regime={first['regime']}")
    print(f"full_png={full}")
    print(f"zoom_png={zoom}")


if __name__ == "__main__":
    main()
