#!/usr/bin/env python3
"""Audit Ng16/Ng32 ID-vs-120 ns evidential separation without retraining."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/current_valid_baseline/epoch3_id_vs_120_epistemic_20260919/per_sample.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_ng16_ng32_decomposition_20260919"
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
METRICS = ["nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic"]
REGIMES = [20, 80, 120]


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else float("nan"),
        "median": quantile(values, 0.50),
        "q05": quantile(values, 0.05),
        "q95": quantile(values, 0.95),
        "q99": quantile(values, 0.99),
        "max": max(values) if values else float("nan"),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(lambda: defaultdict(list))
    with INPUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ng = int(row["ng"])
            delay = int(float(row["delay_ns"]))
            if ng in (16, 32) and delay in REGIMES:
                for metric in METRICS:
                    grouped[(ng, delay)][metric].append(float(row[metric]))

    summaries = []
    for ng in (16, 32):
        for delay in REGIMES:
            item = {"ng": ng, "delay_ns": delay}
            for metric in METRICS:
                item[metric] = summary(grouped[(ng, delay)][metric])
            summaries.append(item)

    with (OUT / "parameter_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)

    with (OUT / "parameter_summary.csv").open("w", newline="") as handle:
        columns = ["ng", "delay_ns"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "median", "q05", "q95", "q99", "max")]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in summaries:
            row = {"ng": item["ng"], "delay_ns": item["delay_ns"]}
            for metric in METRICS:
                for stat in ("mean", "median", "q05", "q95", "q99", "max"):
                    row[f"{metric}_{stat}"] = item[metric][stat]
            writer.writerow(row)

    comparison = []
    for metric in METRICS:
        row = {"metric": metric}
        for ng in (16, 32):
            for delay in REGIMES:
                row[f"ng{ng}_delay{delay}_median"] = summary(grouped[(ng, delay)][metric])["median"]
        for ng in (16, 32):
            id80 = summary(grouped[(ng, 80)][metric])["median"]
            ood120 = summary(grouped[(ng, 120)][metric])["median"]
            row[f"ng{ng}_80_to_120_ratio"] = ood120 / id80 if id80 else float("inf")
            row[f"ng{ng}_80_to_120_delta"] = ood120 - id80
        comparison.append(row)
    with (OUT / "comparison_80_to_120.csv").open("w", newline="") as handle:
        columns = list(comparison[0])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparison)

    coverage = {
        "configured_ng": [4, 8, 16, 32],
        "first_three_epoch_sample_draws": 300000,
        "actual_per_ng_counts": "UNAVAILABLE: existing training logs do not record per-sample Ng draws",
        "actual_per_ng_ratios": "UNAVAILABLE",
        "offset_coverage": "UNAVAILABLE: existing training logs do not record random offsets",
        "imbalance": "UNDETERMINED",
        "theoretical_sampling_expectation": "Each configured Ng has probability 0.25 per sample under torch.randint; this is not an empirical audit result.",
        "reconstruction_performed": False,
        "reason": "Exact replay would require full training RNG replay, including all model/data/random operations; no fabricated counts were used.",
    }
    with (OUT / "training_mask_coverage_audit.json").open("w") as handle:
        json.dump(coverage, handle, indent=2)

    manifest = {
        "input_artifact": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "protocol": "Reused prior GPU-generated per-sample artifact; no new training or inference; Ng={16,32}, delay={20,80,120} ns, 200 samples per regime in source artifact.",
    }
    with (OUT / "analysis_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
