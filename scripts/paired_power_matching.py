#!/usr/bin/env python3
"""20 ns와 80 ns sample을 CFR power 기준으로 1:1 비교한다.

실험 목적:
    sample power를 비슷하게 맞춘 뒤에도 80 ns의 error와 uncertainty 차이가
    남는지 확인하여 power confounding을 분리한다.

주의:
    matching과 평가는 읽기 전용으로 수행하며 원 결과를 덮어쓰지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict], regime: str) -> list[dict]:
    selected = [r for r in rows if r["regime"] == regime]
    grouped = {}
    for row in selected:
        grouped.setdefault(int(row["sample_index"]), []).append(row)
    out = []
    keys = ("cfr_power", "normalized_error_mse", "normalized_psi_scale", "normalized_aleatoric")
    for sample, values in sorted(grouped.items()):
        out.append({"regime": regime, "sample_index": sample, **{key: float(np.mean([float(v[key]) for v in values])) for key in keys}, "pair_count": len(values)})
    if any(r["pair_count"] != 4 for r in out):
        raise AssertionError("Expected exactly four antenna-pair rows per sample")
    return out


def match(easy: list[dict], hard: list[dict]) -> list[dict]:
    try:
        from scipy.optimize import linear_sum_assignment
        cost = np.abs(np.asarray([r["cfr_power"] for r in easy])[:, None] - np.asarray([r["cfr_power"] for r in hard])[None, :])
        left, right = linear_sum_assignment(cost)
        matches = [(int(i), int(j)) for i, j in zip(left, right)]
        method = "minimum_total_absolute_power_distance (Hungarian assignment)"
    except ImportError:
        remaining = set(range(len(hard))); matches = []
        for i, row in enumerate(sorted(enumerate(easy), key=lambda x: x[1]["cfr_power"])):
            idx = min(remaining, key=lambda j: abs(row[1]["cfr_power"] - hard[j]["cfr_power"]))
            remaining.remove(idx); matches.append((row[0], idx))
        method = "sequential nearest without replacement"
    output = []
    for match_index, (i, j) in enumerate(sorted(matches)):
        a, b = easy[i], hard[j]
        power_diff = b["cfr_power"] - a["cfr_power"]
        power_abs = abs(power_diff)
        output.append({
            "match_index": match_index, "easy_sample_index": a["sample_index"], "hard_sample_index": b["sample_index"],
            "easy_power": a["cfr_power"], "hard_power": b["cfr_power"], "power_difference_80_minus_20": power_diff,
            "power_tolerance_absolute": power_abs, "power_tolerance_relative_to_easy": power_abs / max(abs(a["cfr_power"]), 1e-12),
            "delta_normalized_error_80_minus_20": b["normalized_error_mse"] - a["normalized_error_mse"],
            "delta_normalized_psi_80_minus_20": b["normalized_psi_scale"] - a["normalized_psi_scale"],
            "delta_normalized_aleatoric_80_minus_20": b["normalized_aleatoric"] - a["normalized_aleatoric"],
            "matching_method": method,
        })
    return output


def paired_stats(values: np.ndarray) -> dict[str, float]:
    from scipy.stats import binomtest, wilcoxon
    result = {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "positive_ratio": float(np.mean(values > 0)), "zero_ratio": float(np.mean(values == 0))}
    nonzero = values[values != 0]
    result["sign_test_p_two_sided"] = float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5, alternative="two-sided").pvalue) if len(nonzero) else 1.0
    result["wilcoxon_p_two_sided"] = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue) if len(nonzero) else 1.0
    return result


def plot(matches: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Norm. error", "Norm. Psi", "Norm. Aleatoric"]
    keys = ["delta_normalized_error_80_minus_20", "delta_normalized_psi_80_minus_20", "delta_normalized_aleatoric_80_minus_20"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, name in zip(axes, keys, names):
        values = np.asarray([r[key] for r in matches]); ax.hist(values, bins=24, alpha=.75); ax.axvline(0, color="black", linestyle="--"); ax.set_title(name); ax.set_xlabel("80 ns - 20 ns"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "paired_differences.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter([r["easy_power"] for r in matches], [r["hard_power"] for r in matches], s=10, alpha=.6); lim=[0.2,2.7]; ax.plot(lim,lim,"k--"); ax.set_xlabel("20 ns CFR power"); ax.set_ylabel("Matched 80 ns CFR power"); ax.set_title("1:1 power matching"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / "power_matching.png", dpi=160); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--input", default="runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/sample_pair_metrics.csv"); p.add_argument("--output-dir", required=True); p.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); args = p.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    with (ROOT / args.input).open() as handle: rows = list(csv.DictReader(handle))
    easy = aggregate(rows, "ID-Easy 20 ns"); hard = aggregate(rows, "ID-Hard 80 ns")
    if len(easy) != len(hard): raise AssertionError("20 ns and 80 ns sample counts differ")
    matches = match(easy, hard); stats = []
    for label, key in (("normalized_error", "delta_normalized_error_80_minus_20"), ("normalized_psi", "delta_normalized_psi_80_minus_20"), ("normalized_aleatoric", "delta_normalized_aleatoric_80_minus_20")):
        stats.append({"quantity": label, **paired_stats(np.asarray([r[key] for r in matches], dtype=float))})
    tolerance = {"mean_absolute": float(np.mean([r["power_tolerance_absolute"] for r in matches])), "median_absolute": float(np.median([r["power_tolerance_absolute"] for r in matches])), "p95_absolute": float(np.quantile([r["power_tolerance_absolute"] for r in matches], .95)), "max_absolute": float(max(r["power_tolerance_absolute"] for r in matches)), "mean_relative": float(np.mean([r["power_tolerance_relative_to_easy"] for r in matches])), "p95_relative": float(np.quantile([r["power_tolerance_relative_to_easy"] for r in matches], .95))}
    out.mkdir(parents=True, exist_ok=True); write_csv(out / "matched_pairs.csv", matches); write_csv(out / "paired_statistics.csv", stats); (out / "power_tolerance.json").write_text(json.dumps(tolerance, indent=2)); plot(matches, out)
    ck = ROOT / args.checkpoint; prov = {"input": args.input, "checkpoint": args.checkpoint, "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(), "training_performed": False, "sample_count_each": len(easy), "pair_rows_each": len(rows) // 2, "matching": "1:1 without replacement, minimum total absolute CFR-power distance", "aggregation": "mean across four antenna pairs per sample", "runtime_note": "read-only reuse of existing sample/pair inference output"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True)); (out / "config.json").write_text(json.dumps({"regimes": ["ID-Easy 20 ns", "ID-Hard 80 ns"], "matching": prov["matching"], "power_tolerance": tolerance}, indent=2)); (out / "results.json").write_text(json.dumps({"statistics": stats, "power_tolerance": tolerance, "provenance": prov}, indent=2, sort_keys=True)); print(json.dumps({"output_dir": str(out), "statistics": stats, "power_tolerance": tolerance}, indent=2))


if __name__ == "__main__": main()
