#!/usr/bin/env python3
"""Error-matched Ng16/Ng32 κ comparison using existing epoch3 samples."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/current_valid_baseline/epoch3_ng_delay_factorial_diagnosis_20260920/per_sample.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_ng_error_matched_kappa_20260920"
DELAYS = [20, 40, 60, 80, 100]
TOLERANCE_DB = 0.25
BOOTSTRAP = 2000
SEED = 20260920
METRICS = ["nmse", "kappa", "nu_margin", "psi", "aleatoric", "epistemic"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nearest_pairs(left, right, tolerance):
    """Greedy sorted nearest matching without replacement within common support."""
    left = sorted(left, key=lambda x: x["nmse"])
    right = sorted(right, key=lambda x: x["nmse"])
    i = j = 0
    pairs = []
    while i < len(left) and j < len(right):
        delta = left[i]["nmse"] - right[j]["nmse"]
        if abs(delta) <= tolerance:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return pairs


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def pair_summary(pairs, group):
    if not pairs:
        return {"group": group, "matched_n": 0}
    errors = np.array([a["nmse"] - b["nmse"] for a, b in pairs])
    delta = {metric: np.array([b[metric] - a[metric] for a, b in pairs]) for metric in METRICS if metric != "nmse"}
    ratio = {metric: np.array([b[metric] / a[metric] for a, b in pairs]) for metric in ["kappa", "aleatoric", "epistemic"]}
    return {"group": group, "matched_n": len(pairs), "error_difference_mean_db": float(errors.mean()), "error_difference_median_db": float(np.median(errors)), "error_difference_max_abs_db": float(np.abs(errors).max()), "kappa16_median": float(np.median([a["kappa"] for a, _ in pairs])), "kappa32_median": float(np.median([b["kappa"] for _, b in pairs])), "kappa32_minus_16_median": float(np.median(delta["kappa"])), "kappa32_over_16_median": float(np.median(ratio["kappa"])), "kappa32_below_16_fraction": float(np.mean(delta["kappa"] < 0)), "epistemic16_median": float(np.median([a["epistemic"] for a, _ in pairs])), "epistemic32_median": float(np.median([b["epistemic"] for _, b in pairs])), "epistemic32_minus_16_median": float(np.median(delta["epistemic"])), "epistemic32_over_16_median": float(np.median(ratio["epistemic"])), "epistemic32_above_16_fraction": float(np.mean(delta["epistemic"] > 0)), "aleatoric16_median": float(np.median([a["aleatoric"] for a, _ in pairs])), "aleatoric32_median": float(np.median([b["aleatoric"] for _, b in pairs])), "nu_margin16_median": float(np.median([a["nu_margin"] for a, _ in pairs])), "nu_margin32_median": float(np.median([b["nu_margin"] for _, b in pairs])), "psi16_median": float(np.median([a["psi"] for a, _ in pairs])), "psi32_median": float(np.median([b["psi"] for _, b in pairs])), "kappa_delta_bootstrap_ci": bootstrap_ci(delta["kappa"], SEED + len(pairs)), "epistemic_delta_bootstrap_ci": bootstrap_ci(delta["epistemic"], SEED + len(pairs) + 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with INPUT.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = [{key: (int(value) if key in ("ng", "delay_ns", "sample") else float(value)) for key, value in row.items()} for row in raw]
    by_condition = {(ng, delay): [row for row in rows if row["ng"] == ng and row["delay_ns"] == delay] for ng in [16, 32] for delay in [20, 40, 60, 80, 100, 120]}

    all_pairs = []
    summaries = []
    for delay in DELAYS + [120]:
        pairs = nearest_pairs(by_condition[(16, delay)], by_condition[(32, delay)], TOLERANCE_DB)
        group = f"delay_{delay}ns"
        summaries.append(pair_summary(pairs, group))
        for pair_index, (left, right) in enumerate(pairs):
            all_pairs.append({"group": group, "pair_index": pair_index, "delay_ns": delay, "sample16": left["sample"], "sample32": right["sample"], "error16": left["nmse"], "error32": right["nmse"], "error_difference": right["nmse"] - left["nmse"], **{f"{metric}16": left[metric] for metric in METRICS if metric != "nmse"}, **{f"{metric}32": right[metric] for metric in METRICS if metric != "nmse"}})

    pooled_left = [row for delay in DELAYS for row in by_condition[(16, delay)]]
    pooled_right = [row for delay in DELAYS for row in by_condition[(32, delay)]]
    pooled_pairs = nearest_pairs(pooled_left, pooled_right, TOLERANCE_DB)
    summaries.append(pair_summary(pooled_pairs, "ID_pooled_20_100ns"))
    for pair_index, (left, right) in enumerate(pooled_pairs):
        all_pairs.append({"group": "ID_pooled_20_100ns", "pair_index": pair_index, "delay_ns": "pooled", "sample16": left["sample"], "sample32": right["sample"], "error16": left["nmse"], "error32": right["nmse"], "error_difference": right["nmse"] - left["nmse"], "delay16": left["delay_ns"], "delay32": right["delay_ns"], **{f"{metric}16": left[metric] for metric in METRICS if metric != "nmse"}, **{f"{metric}32": right[metric] for metric in METRICS if metric != "nmse"}})
    write_csv(OUT / "matched_pairs.csv", all_pairs)
    write_csv(OUT / "matched_summary.csv", summaries)

    # Difficulty bins are defined within each delay using combined Ng16/Ng32 NMSE quantiles.
    bin_rows = []
    for delay in DELAYS + [120]:
        combined = np.array([row["nmse"] for row in by_condition[(16, delay)] + by_condition[(32, delay)]])
        edges = np.quantile(combined, [0, .25, .5, .75, 1])
        for bin_index in range(4):
            for ng in [16, 32]:
                subset = [row for row in by_condition[(ng, delay)] if (row["nmse"] >= edges[bin_index] and (row["nmse"] <= edges[bin_index + 1] if bin_index == 3 else row["nmse"] < edges[bin_index + 1]))]
                if subset:
                    bin_rows.append({"delay_ns": delay, "difficulty_bin": bin_index + 1, "bin_low_nmse": edges[bin_index], "bin_high_nmse": edges[bin_index + 1], "ng": ng, "n": len(subset), "nmse_median": float(np.median([x["nmse"] for x in subset])), "kappa_median": float(np.median([x["kappa"] for x in subset])), "epistemic_median": float(np.median([x["epistemic"] for x in subset])), "aleatoric_median": float(np.median([x["aleatoric"] for x in subset])), "nu_margin_median": float(np.median([x["nu_margin"] for x in subset])), "psi_median": float(np.median([x["psi"] for x in subset]))})
    write_csv(OUT / "difficulty_bins.csv", bin_rows)

    # Conditional association: log(kappa) ~ NMSE + delay + Ng32 indicator.
    id_rows = [row for row in rows if row["delay_ns"] in DELAYS and row["ng"] in [16, 32]]
    X = np.array([[1.0, row["nmse"], row["delay_ns"], 1.0 if row["ng"] == 32 else 0.0] for row in id_rows])
    y = np.log(np.array([row["kappa"] for row in id_rows]))
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coefficients
    regression = {"formula": "log(kappa) ~ NMSE + delay_ns + I(Ng=32)", "coefficients": {"intercept": float(coefficients[0]), "nmse": float(coefficients[1]), "delay_ns": float(coefficients[2]), "ng32_indicator": float(coefficients[3])}, "r2": float(1 - (residual @ residual) / ((y - y.mean()) @ (y - y.mean()))), "interpretation": "conditional association only; not a causal proof"}
    (OUT / "conditional_regression.json").write_text(json.dumps(regression, indent=2) + "\n")

    manifest = {"input": str(INPUT.relative_to(ROOT)), "checkpoint_only": True, "ngs": [16, 32], "id_delays_ns": DELAYS, "ood_reference_delay_ns": 120, "matching": "greedy sorted nearest-neighbor without replacement", "matching_tolerance_db": TOLERANCE_DB, "fallback": "pooled ID 20–100 ns matching when delay-specific support is sparse", "bootstrap_replicates": BOOTSTRAP, "no_training": True, "no_model_or_loss_changes": True}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
