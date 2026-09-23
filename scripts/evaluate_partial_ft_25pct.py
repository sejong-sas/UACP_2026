#!/usr/bin/env python3
"""Evaluate only the new 25% checkpoint with the established evaluator protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_partial_ft_comparison import evaluate, load_model


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/current_valid_baseline_100k1_seed_20260819.json")
    parser.add_argument("--id-easy", required=True)
    parser.add_argument("--id-hard", required=True)
    parser.add_argument("--near-ood", required=True)
    parser.add_argument("--far-ood", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="partial25")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10/cuda:0 required")
    from scripts.train_predictor import load_config
    config = load_config(ROOT / args.config)
    checkpoint = ROOT / args.checkpoint
    regimes = {"20ns": args.id_easy, "80ns": args.id_hard,
               "120ns": args.near_ood, "1ms": args.far_ood}
    model = load_model(config, checkpoint, device)
    summary_rows, sample_rows = [], []
    started = time.perf_counter()
    for regime, dataset in regimes.items():
        for ng in (16, 32):
            metrics = evaluate(model, ROOT / dataset, config, device, ng,
                               args.seed + ng * 1000, args.batch_size)
            finite_count = sum(row["finite"] for row in metrics)
            for row in metrics:
                sample_rows.append({"model": args.model_label, "epoch": 3,
                                    "regime": regime, "ng": ng, **row})
            def mean(key):
                return float(np.mean([row[key] for row in metrics]))
            summary_rows.append({"model": args.model_label, "epoch": 3,
                                 "regime": regime, "ng": ng,
                                 "samples": len(metrics),
                                 "finite_samples": finite_count,
                                 "nmse_all_db": mean("nmse_all_db"),
                                 "nmse_omitted_db": mean("nmse_omitted_db"),
                                 "aleatoric": mean("aleatoric"),
                                 "epistemic": mean("epistemic"),
                                 "kappa": mean("kappa"),
                                 "nu_margin": mean("nu_margin"),
                                 "all_finite": finite_count == len(metrics)})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "sample_metrics.csv", sample_rows)
    manifest = {"checkpoint": args.checkpoint, "model_label": args.model_label, "regimes": regimes,
                "ngs": [16, 32], "seed": args.seed,
                "batch_size": args.batch_size, "device": str(device),
                "gpu": torch.cuda.get_device_name(0), "same_eval_order": True,
                "same_eval_mask": "fixed canonical offset 0 periodic mask[:,::Ng]",
                "same_eval_noise": "deterministic 15 dB complex AWGN schedule keyed by seed+Ng*1000",
                "input_is_sparse": True, "target_is_clean_full_cfr": True,
                "no_training": True,
                "evaluator_source": "scripts/evaluate_partial_ft_comparison.py",
                "elapsed_seconds": time.perf_counter() - started}
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "rows": len(summary_rows),
                      "samples": len(sample_rows),
                      "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
