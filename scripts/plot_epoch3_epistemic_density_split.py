#!/usr/bin/env python3
"""Plot the saved epoch-3 Fig.8 score without modifying source artifacts.

The source score is the saved ``fig8_score`` column, i.e. the Eq.(13)
aggregate used by the existing static Fig.8 evaluator.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/current_valid_baseline/epoch3_baseline_validation_20260919/static_fig8_fig9/static_per_sample.csv"
OUT = SOURCE.parent / "fig8_epistemic_density_split.png"
STATS_OUT = SOURCE.parent / "fig8_epistemic_density_split_stats.txt"

REGIMES = [
    ("ID-Easy 20 ns", "#1f77b4"),
    ("ID-Hard 80 ns", "#ff7f0e"),
    ("OOD-Near 120 ns", "#2ca02c"),
    ("OOD-Far 1 ms", "#d62728"),
]


def load_scores() -> dict[str, np.ndarray]:
    values = {name: [] for name, _ in REGIMES}
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["regime"]].append(float(row["fig8_score"]))
    return {name: np.asarray(v, dtype=float) for name, v in values.items()}


def stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "finite": int(finite.size),
        "nonfinite": int(values.size - finite.size),
        "positive_finite": int(np.sum(finite > 0)),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "median": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
        "q90": float(np.quantile(finite, 0.90)) if finite.size else float("nan"),
        "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "q99": float(np.quantile(finite, 0.99)) if finite.size else float("nan"),
        "q995": float(np.quantile(finite, 0.995)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
    }


def fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    scores = load_scores()
    summary = {name: stats(values) for name, values in scores.items()}

    for name, _ in REGIMES:
        s = summary[name]
        print(
            f"{name}: n={s['n']}, finite={s['finite']}, non-finite={s['nonfinite']}, "
            f"positive_finite={s['positive_finite']}, min={fmt(s['min'])}, "
            f"median={fmt(s['median'])}, q90={fmt(s['q90'])}, q95={fmt(s['q95'])}, "
            f"q99={fmt(s['q99'])}, q99.5={fmt(s['q995'])}, max={fmt(s['max'])}"
        )

    id_near = np.concatenate([scores[name][np.isfinite(scores[name])] for name in [
        "ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns"
    ]])
    left_q = float(np.quantile(id_near, 0.995))
    left_xmax = left_q * 1.05
    far = scores["OOD-Far 1 ms"]
    far_finite = far[np.isfinite(far)]
    far_q99 = summary["OOD-Far 1 ms"]["q99"]
    far_max = summary["OOD-Far 1 ms"]["max"]
    print(f"left_cutoff_q99.5_finite_ID_Near={fmt(left_q)}")
    print(f"left_visual_xmax={fmt(left_xmax)}")
    print(f"Far max/q99={fmt(far_max / far_q99)}")
    print(f"Far count > q99.5={int(np.sum(far_finite > summary['OOD-Far 1 ms']['q995']))}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    bins_left = np.linspace(0.0, left_xmax, 46)
    for name, color in REGIMES[:3]:
        values = scores[name]
        finite = values[np.isfinite(values)]
        visible = finite[finite <= left_xmax]
        ax_left.hist(
            visible,
            bins=bins_left,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=f"{name} (median={summary[name]['median']:.3g})",
        )
        ax_left.axvline(summary[name]["median"], color=color, linestyle="--", linewidth=1.1, alpha=0.9)
    ax_left.set_title("ID / Near-OOD expanded view")
    ax_left.set_xlabel("Raw Epistemic score (Fig.8 Eq.13; finite values)")
    ax_left.set_ylabel("Density of visible finite scores")
    ax_left.set_xlim(0.0, left_xmax)
    ax_left.grid(alpha=0.25)
    ax_left.legend(fontsize=8, loc="upper right")
    ax_left.text(
        0.02,
        0.02,
        "Visual cutoff = pooled ID/Near finite q99.5\n"
        f"q99.5={left_q:.3g}; values above cutoff are not deleted,\nonly excluded from this panel's view.",
        transform=ax_left.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    epsilon = 1e-12
    all_positive = all(np.all(values[np.isfinite(values)] > 0) for values in scores.values())
    if not all_positive:
        raise ValueError("Non-positive finite Fig.8 score found; log10(score + epsilon) is not valid.")
    transformed = {name: np.log10(values[np.isfinite(values)] + epsilon) for name, values in scores.items()}
    all_log = np.concatenate(list(transformed.values()))
    bins_right = np.linspace(float(np.min(all_log)), float(np.max(all_log)), 61)
    for name, color in REGIMES:
        vals = transformed[name]
        median_log = float(np.log10(summary[name]["median"] + epsilon))
        ax_right.hist(
            vals,
            bins=bins_right,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=f"{name} (median={summary[name]['median']:.3g})",
        )
        ax_right.axvline(median_log, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
    ax_right.set_title("All regimes on log-transformed x-axis")
    ax_right.set_xlabel(r"log10(raw Epistemic score + $\epsilon$), $\epsilon=10^{-12}$")
    ax_right.set_ylabel("Density")
    ax_right.grid(alpha=0.25)
    ax_right.legend(fontsize=8, loc="upper left")
    ax_right.text(
        0.02,
        0.02,
        "All finite scores shown; no score clipping.\n"
        "Transform: log10(score + 1e-12).",
        transform=ax_right.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    fig.suptitle("Epoch3 Epistemic score distribution — common-bin histograms", fontsize=14)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        f"source={SOURCE}",
        "score_column=fig8_score (saved Fig.8 Eq.(13) aggregate; raw score)",
        "plot_type=common-bin histogram density; no KDE",
        "colors=20ns blue, 80ns orange, 120ns green, 1ms red",
        f"left_cutoff=pooled finite ID-Easy/ID-Hard/OOD-Near q99.5={left_q:.12g}",
        f"left_visual_xmax={left_xmax:.12g}",
        "left_values_above_cutoff=retained_in_source_but_excluded_only_from_left_panel_view",
        "right_transform=log10(score + epsilon)",
        f"epsilon={epsilon:.12g}",
        f"all_finite_scores_positive={all_positive}",
        f"far_max_over_q99={far_max / far_q99:.12g}",
        "",
    ]
    for name, _ in REGIMES:
        s = summary[name]
        lines.append(
            f"{name}: n={s['n']}, finite={s['finite']}, nonfinite={s['nonfinite']}, "
            f"min={s['min']:.12g}, median={s['median']:.12g}, q90={s['q90']:.12g}, "
            f"q95={s['q95']:.12g}, q99={s['q99']:.12g}, q99.5={s['q995']:.12g}, max={s['max']:.12g}"
        )
    STATS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_png={OUT}")
    print(f"saved_stats={STATS_OUT}")


if __name__ == "__main__":
    main()
