#!/usr/bin/env python3
"""Replot saved canonical 100k×1 sample-level Fig.8 epistemic scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/fig8_eval_retry/per_sample.csv"
RESULTS = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/fig8_eval_retry/results.json"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]
COLORS = {"ID-Easy 20 ns": "#1f77b4", "ID-Hard 80 ns": "#ff7f0e",
          "OOD-Near 120 ns": "#2ca02c", "OOD-Far 1 ms": "#d62728"}
PAPER_XLIM = (-37.5, -22.5)


def histogram_density(samples: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return common-bin density normalized against every source sample."""
    counts, returned_edges = np.histogram(samples, bins=edges)
    return counts / (len(samples) * np.diff(returned_edges)), returned_edges


def regime_summary(db_values: np.ndarray, xmin: float, xmax: float,
                   linear_values: np.ndarray | None = None) -> dict:
    x = np.asarray(db_values, dtype=np.float64)
    below = x < xmin
    above = x > xmax
    inside = ~(below | above)
    row = {
        "count": int(x.size), "mean_db": float(x.mean()), "median_db": float(np.median(x)),
        "std_db": float(x.std()), "min_db": float(x.min()), "max_db": float(x.max()),
        "q01_db": float(np.quantile(x, .01)), "q99_db": float(np.quantile(x, .99)),
        "below_range_count": int(below.sum()), "below_range_fraction": float(below.mean()),
        "above_range_count": int(above.sum()), "above_range_fraction": float(above.mean()),
        "inside_range_count": int(inside.sum()), "inside_range_fraction": float(inside.mean()),
    }
    if linear_values is not None:
        row["linear_mean"] = float(np.mean(linear_values))
    return row


def auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Mann–Whitney AUROC with half credit for ties."""
    n, p = np.asarray(negative), np.asarray(positive)
    # Scores are only 10k per regime; this chunked comparison avoids large NxM allocation.
    wins = ties = 0
    for start in range(0, len(p), 256):
        chunk = p[start:start + 256, None]
        wins += int(np.count_nonzero(chunk > n[None, :]))
        ties += int(np.count_nonzero(chunk == n[None, :]))
    return float((wins + .5 * ties) / (len(n) * len(p)))


def read_sample_scores(path: Path) -> tuple[dict[str, dict[str, list[float]]], list[dict]]:
    values = {regime: {"linear": [], "db": []} for regime in REGIMES}
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            regime = row["regime"]
            if regime not in values:
                continue
            linear = float(row["epistemic"])
            db = 10.0 * np.log10(linear)
            values[regime]["linear"].append(linear)
            values[regime]["db"].append(float(db))
            rows.append({"sample_id": int(row["sample"]), "regime": regime,
                         "U_epi_linear": linear, "U_epi_dB": float(db)})
    for regime in REGIMES:
        values[regime] = {key: np.asarray(v, dtype=np.float64) for key, v in values[regime].items()}
    return values, rows


def render(values: dict, edges: np.ndarray, output: Path, xlim: tuple[float, float] | None) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.45), facecolor="white")
    ax.set_facecolor("white")
    for regime in REGIMES:
        density, bins = histogram_density(values[regime]["db"], edges)
        ax.stairs(density, bins, fill=True, baseline=0, color=COLORS[regime],
                  alpha=0.48, linewidth=0.9, edgecolor=COLORS[regime], label=regime)
    ax.set_xlabel("Epistemic Uncertainty (dB)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_ylim(0.0, 0.4)
    ax.set_yticks(np.arange(0.0, 0.41, 0.1))
    if xlim is None:
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    else:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.arange(-37.5, -22.49, 2.5))
    ax.grid(True, linestyle="--", linewidth=0.55, color="#bdbdbd", alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8.5, colors="black")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#555555",
              framealpha=0.92, fontsize=8.0, borderpad=0.45, handlelength=1.5)
    fig.tight_layout(pad=0.8)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    if not SOURCE.is_file() or not RESULTS.is_file():
        raise FileNotFoundError("canonical sample scores/results are missing")
    out.mkdir(parents=True, exist_ok=True)
    values, rows = read_sample_scores(SOURCE)
    prior = json.loads(RESULTS.read_text(encoding="utf-8"))
    expected_means = {item["regime"]: float(item["mean_db"]) for item in prior["distribution"]}
    expected_auc = prior["auroc"]

    # Fixed common 0.25 dB bins spanning all original samples; no viewport truncation
    # or per-regime renormalization. Bin width is not specified in the paper.
    all_db = np.concatenate([values[r]["db"] for r in REGIMES])
    bin_width = 0.25
    edge_start = np.floor(all_db.min() / bin_width) * bin_width - bin_width
    edge_stop = np.ceil(all_db.max() / bin_width) * bin_width + bin_width
    edges = np.arange(edge_start, edge_stop + bin_width * .5, bin_width)

    summaries = {}
    for regime in REGIMES:
        summaries[regime] = regime_summary(values[regime]["db"], *PAPER_XLIM,
                                           values[regime]["linear"])
    pooled_id = np.concatenate([values[REGIMES[0]]["db"], values[REGIMES[1]]["db"]])
    pooled_ood = np.concatenate([values[REGIMES[2]]["db"], values[REGIMES[3]]["db"]])
    aurocs = {
        "id_vs_ood_pooled": auc(pooled_id, pooled_ood),
        "id_vs_ood_near": auc(pooled_id, values[REGIMES[2]]["db"]),
        "id_vs_ood_far": auc(pooled_id, values[REGIMES[3]]["db"]),
        "id_hard_80_vs_ood_near_120": auc(values[REGIMES[1]]["db"], values[REGIMES[2]]["db"]),
    }
    prior_key_map = {"id_vs_ood_pooled": "id_vs_ood_pooled", "id_vs_ood_near": "id_vs_near",
                     "id_vs_ood_far": "id_vs_far", "id_hard_80_vs_ood_near_120": "id_hard_80_vs_near_120"}
    checks = {regime: abs(summaries[regime]["mean_db"] - expected_means[regime]) for regime in REGIMES}
    auc_checks = {key: abs(aurocs[key] - expected_auc[prior_key_map[key]]) for key in aurocs}
    means_match = all(v < 1e-9 for v in checks.values())
    auc_match = all(v < 1e-12 for v in auc_checks.values())

    with (out / "fig8_sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "regime", "U_epi_linear", "U_epi_dB"])
        writer.writeheader()
        writer.writerows(rows)

    if means_match and auc_match:
        render(values, edges, out / "fig8_paper_style_axis", PAPER_XLIM)
        render(values, edges, out / "fig8_full_range", None)
    summary = {
        "purpose": "Replot canonical 100k×1 sample-level epistemic distributions in paper Fig.8 visual scale",
        "checkpoint": prior["checkpoint"], "source_sample_scores": str(SOURCE.relative_to(ROOT)),
        "sample_count_per_regime": {k: v["count"] for k, v in summaries.items()},
        "score_calculation": "existing corrected Eq.(7)/(8)/(12)/(13) sample-level U_epi; samplewise 10*log10(U_epi)",
        "dB_conversion": "IMPLEMENTATION-ASSUMPTION: 10*log10 per sample; no offset/rescaling",
        "density": {"type": "filled common-bin histogram", "bin_width_db": bin_width,
                    "edges_cover_full_sample_range": [float(edges[0]), float(edges[-1])],
                    "normalization": "count/(all samples * bin width); same bins and normalization for all regimes; no clipping renormalization",
                    "paper_bin_specification": "not published; 0.25 dB is IMPLEMENTATION-ASSUMPTION"},
        "paper_axis": {"xlim": list(PAPER_XLIM), "ylim": [0.0, 0.4], "xticks": list(np.arange(-37.5, -22.49, 2.5)), "yticks": list(np.arange(0.0, .41, .1))},
        "colors": COLORS, "summary_by_regime": summaries, "auroc": aurocs,
        "validation": {"against_existing_dB_means": checks, "against_existing_AUROC": auc_checks,
                       "means_match": means_match, "auroc_match": auc_match,
                       "plot_generated": bool(means_match and auc_match)},
        "figures": ["fig8_paper_style_axis.png", "fig8_paper_style_axis.pdf", "fig8_paper_style_axis.svg", "fig8_full_range.png"],
    }
    (out / "fig8_plot_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "means_match": means_match, "auroc_match": auc_match,
                      "auroc": aurocs, "summary_by_regime": summaries, "plots_generated": means_match and auc_match}, indent=2), flush=True)


if __name__ == "__main__":
    main()
