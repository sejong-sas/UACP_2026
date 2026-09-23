#!/usr/bin/env python3
"""Paper-style plotting from the validated canonical Fig.9 audit CSVs."""
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
AUDIT = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig9_audit_20260915_final"
NOMINALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
ANCHOR_GRID = [0.5, 0.8, 0.9, 0.95, 0.99]
EXPECTED = {
    (0.50, "ID"): 0.5131, (0.50, "OOD"): 0.2174,
    (0.80, "ID"): 0.8126, (0.80, "OOD"): 0.3858,
    (0.90, "ID"): 0.9091, (0.90, "OOD"): 0.4675,
    (0.95, "ID"): 0.9561, (0.95, "OOD"): 0.5262,
    (0.99, "ID"): 0.9921, (0.99, "OOD"): 0.6114,
}


def calibration_mae(nominals: list[float], empirical: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(empirical) - np.asarray(nominals))))


def combine_pool_rows(rows: list[dict], nominal: float) -> dict:
    at = {row["pool"]: row for row in rows if np.isclose(float(row["nominal"]), nominal, atol=1e-10)}
    required = {"ID", "OOD", "OOD-Near", "OOD-Far"}
    if not required.issubset(at):
        raise ValueError(f"missing pools at nominal={nominal}: {required - at.keys()}")
    return {
        "nominal": float(nominal), "ID_empirical": float(at["ID"]["empirical"]),
        "OOD_empirical": float(at["OOD"]["empirical"]),
        "OOD_Near_empirical": float(at["OOD-Near"]["empirical"]),
        "OOD_Far_empirical": float(at["OOD-Far"]["empirical"]),
        "ID_count": int(at["ID"]["count"]), "OOD_count": int(at["OOD"]["count"]),
        "OOD_Near_count": int(at["OOD-Near"]["count"]),
        "OOD_Far_count": int(at["OOD-Far"]["count"]),
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["nominal"] = float(row["nominal"])
        row["empirical"] = float(row["empirical"])
        row["count"] = int(row["count"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ticks = np.arange(0.0, 1.01, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Nominal Coverage", fontsize=10)
    ax.set_ylabel("Empirical Coverage", fontsize=10)
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.55, color="#bdbdbd", alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8.5, colors="black")


def render_main(rows: list[dict], out: Path, ce: dict) -> None:
    x = [r["nominal"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.25), facecolor="white")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4d4d4d", linewidth=1.4, label="Ideal", zorder=1)
    ax.plot(x, [r["ID_empirical"] for r in rows], color="#1f77b4", linestyle="-",
            marker="^", markersize=4.2, linewidth=1.55, label=f"ID (CE={ce['ID_plot_grid']:.4f})", zorder=3)
    ax.plot(x, [r["OOD_empirical"] for r in rows], color="#ff7f0e", linestyle="-.",
            marker="o", markersize=4.0, linewidth=1.55, label=f"OOD (CE={ce['OOD_plot_grid']:.4f})", zorder=2)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#555555",
              framealpha=0.96, fontsize=8.5, borderpad=0.45, handlelength=1.8)
    fig.tight_layout(pad=0.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_diagnostic(rows: list[dict], out: Path, ce: dict) -> None:
    x = [r["nominal"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.25), facecolor="white")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4d4d4d", linewidth=1.4, label="Ideal", zorder=1)
    ax.plot(x, [r["ID_empirical"] for r in rows], color="#1f77b4", linestyle="-",
            marker="^", markersize=4.2, linewidth=1.55, label=f"ID (CE={ce['ID_plot_grid']:.4f})", zorder=4)
    ax.plot(x, [r["OOD_Near_empirical"] for r in rows], color="#2ca02c", linestyle="--",
            marker="s", markersize=3.8, linewidth=1.45, label=f"OOD-Near (CE={ce['OOD_Near_plot_grid']:.4f})", zorder=3)
    ax.plot(x, [r["OOD_Far_empirical"] for r in rows], color="#d62728", linestyle=":",
            marker="D", markersize=3.6, linewidth=1.55, label=f"OOD-Far (CE={ce['OOD_Far_plot_grid']:.4f})", zorder=2)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#555555",
              framealpha=0.96, fontsize=7.7, borderpad=0.4, handlelength=1.8)
    ax.set_title("Current-baseline diagnostic (not paper Fig. 9 reproduction)", fontsize=8.4, pad=5)
    fig.tight_layout(pad=0.8)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    curve_path = AUDIT / "calibration_curve.csv"
    error_path = AUDIT / "calibration_error.csv"
    results_path = AUDIT / "results.json"
    if not all(p.is_file() for p in (curve_path, error_path, results_path)):
        raise FileNotFoundError("canonical Fig.9 audit artifacts are incomplete")
    out.mkdir(parents=True, exist_ok=True)
    source = read_csv(curve_path)
    with error_path.open(newline="", encoding="utf-8") as f:
        source_errors = list(csv.DictReader(f))
    audit = json.loads(results_path.read_text(encoding="utf-8"))
    actual_nominals = sorted({r["nominal"] for r in source})
    if len(actual_nominals) != len(NOMINALS) or not np.allclose(actual_nominals, NOMINALS, atol=1e-10):
        raise ValueError(f"audit nominal grid differs from requested plotting grid: {actual_nominals}")
    rows = [combine_pool_rows(source, nominal) for nominal in NOMINALS]

    # Confirm the existing 5 anchor levels before creating any figures.
    anchor_diffs = {}
    for row in rows:
        nominal = row["nominal"]
        if nominal in ANCHOR_GRID:
            for pool in ("ID", "OOD"):
                key = (round(nominal, 2), pool)
                anchor_diffs[f"{nominal:.2f}_{pool}"] = abs(row[f"{pool}_empirical"] - EXPECTED[key])
    if max(anchor_diffs.values()) > 0.00006:
        raise ValueError(f"existing anchor coverage does not match requested baseline: {anchor_diffs}")

    x = [r["nominal"] for r in rows]
    ce = {
        "ID_plot_grid": calibration_mae(x, [r["ID_empirical"] for r in rows]),
        "OOD_plot_grid": calibration_mae(x, [r["OOD_empirical"] for r in rows]),
        "OOD_Near_plot_grid": calibration_mae(x, [r["OOD_Near_empirical"] for r in rows]),
        "OOD_Far_plot_grid": calibration_mae(x, [r["OOD_Far_empirical"] for r in rows]),
        "ID_legacy_q10_q90": calibration_mae(x[:9], [r["ID_empirical"] for r in rows[:9]]),
        "OOD_legacy_q10_q90": calibration_mae(x[:9], [r["OOD_empirical"] for r in rows[:9]]),
        "OOD_Near_legacy_q10_q90": calibration_mae(x[:9], [r["OOD_Near_empirical"] for r in rows[:9]]),
        "OOD_Far_legacy_q10_q90": calibration_mae(x[:9], [r["OOD_Far_empirical"] for r in rows[:9]]),
        "ID_anchor_5_level": calibration_mae(ANCHOR_GRID, [next(r["ID_empirical"] for r in rows if np.isclose(r["nominal"], n)) for n in ANCHOR_GRID]),
        "OOD_anchor_5_level": calibration_mae(ANCHOR_GRID, [next(r["OOD_empirical"] for r in rows if np.isclose(r["nominal"], n)) for n in ANCHOR_GRID]),
    }
    stored_ce = {r["pool"]: r for r in source_errors}
    if abs(ce["ID_legacy_q10_q90"] - float(stored_ce["ID"]["ce_mae_0.1_0.9"])) > 1e-12:
        raise ValueError("legacy ID CE does not match audit")
    if abs(ce["OOD_legacy_q10_q90"] - float(stored_ce["OOD"]["ce_mae_0.1_0.9"])) > 1e-12:
        raise ValueError("legacy OOD CE does not match audit")

    write_csv(out / "calibration_curve_full.csv", rows)
    render_main(rows, out / "fig9_paper_style", ce)
    render_diagnostic(rows, out / "fig9_near_far_diagnostic.png", ce)
    summary = {
        "purpose": "Paper-axis/style visualization of current 100k×1 Fig.9 calibration; no training or reevaluation",
        "checkpoint": audit["checkpoint"], "evaluation_set": audit["common_eval_dir"],
        "samples_per_regime": audit["samples_per_regime"], "omitted_component_count_per_regime": 76800000,
        "formula": "df=nu-2K+1; scale^2=((kappa+1)/(kappa*df))*Psi; scale=sqrt(scale^2); two-sided Student-t interval around gamma",
        "aggregation": "existing audit coverage counts pooled omitted Real/Imag components; ID=(20ns+80ns) count pooling; OOD=(120ns+1ms) count pooling",
        "nominal_grid": NOMINALS, "nominal_grid_status": "IMPLEMENTATION-ASSUMPTION: nominal coverage grid; actual audit empirical values, no interpolation",
        "ce_definition": "mean_c |empirical(c)-nominal(c)|; exact paper CE reduction undisclosed",
        "ce": ce,
        "audit_ce_values": {r["pool"]: {"q10_q90": r["ce_mae_0.1_0.9"], "all_grid": r["ce_mae_all_grid"]} for r in source_errors},
        "anchor_validation_absolute_differences": anchor_diffs,
        "plot": {"xlim": [0, 1], "ylim": [0, 1], "ticks": [0, .2, .4, .6, .8, 1],
                 "ideal": "dark gray dashed", "ID": "#1f77b4 solid triangle", "OOD pooled": "#ff7f0e dash-dot circle", "legend": "upper left"},
        "diagnostic_plot": "OOD-Near/OOD-Far split; not a reproduction of the paper's Fig.9",
        "no_training": True, "source_audit": str(AUDIT.relative_to(ROOT)),
        "files": ["fig9_paper_style.png", "fig9_paper_style.pdf", "fig9_paper_style.svg", "fig9_near_far_diagnostic.png", "calibration_curve_full.csv"],
    }
    (out / "calibration_plot_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "ce": ce, "anchor_differences": anchor_diffs,
                      "samples_per_regime": audit["samples_per_regime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
