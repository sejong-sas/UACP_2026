#!/usr/bin/env python3
"""Deterministic sparse pre/post evaluator for the 3.16% vs 100% experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_model(config, checkpoint: Path, device: torch.device):
    from scripts.diagnose_predictor import _make_model
    model = _make_model(config, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    return model


@torch.inference_mode()
def evaluate(model, dataset_path: Path, config: dict, device: torch.device, ng: int,
             seed: int, batch_size: int) -> list[dict]:
    from scripts.partial_ft_adapt import make_adaptation_observation
    from src.training.data import CFRNPZDataset

    dataset = CFRNPZDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    sample_index = 0
    K = int(config["paper_specified"]["num_subcarriers"])
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        mask = torch.zeros((cfr.shape[0], K), device=device)
        mask[:, ::ng] = 1.0
        x, target, _ = make_adaptation_observation(cfr, mask, seed, device, epoch=0, batch_index=batch_index)
        output = model(x)
        omitted = (1.0 - mask)[:, None, :]
        error = (output.predicted - target).square()
        nmse_all = 10.0 * torch.log10((error.sum((1, 2)) / target.square().sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        nmse_omitted = 10.0 * torch.log10(((error * omitted).sum((1, 2)) / (target.square() * omitted).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        aleatoric = (output.aleatoric * omitted).sum((1, 2)) / omitted.sum((1, 2)).clamp_min(1.0)
        epistemic = (output.epistemic * omitted).sum((1, 2)) / omitted.sum((1, 2)).clamp_min(1.0)
        kappa = output.kappa.mean((1, 2))
        nu_margin = (output.nu - (2 * K + 1.0)).mean((1, 2))
        finite = torch.isfinite(output.predicted).all((1, 2)) & torch.isfinite(aleatoric) & torch.isfinite(epistemic) & torch.isfinite(kappa) & torch.isfinite(nu_margin)
        for i in range(cfr.shape[0]):
            rows.append({"sample": sample_index + i, "nmse_all_db": float(nmse_all[i]),
                         "nmse_omitted_db": float(nmse_omitted[i]), "aleatoric": float(aleatoric[i]),
                         "epistemic": float(epistemic[i]), "kappa": float(kappa[i]),
                         "nu_margin": float(nu_margin[i]), "finite": bool(finite[i])})
        sample_index += cfr.shape[0]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", required=True)
    parser.add_argument("--partial-dir", required=True)
    parser.add_argument("--full-dir", required=True)
    parser.add_argument("--config", default="configs/current_valid_baseline_100k1_seed_20260819.json")
    parser.add_argument("--id-easy", required=True)
    parser.add_argument("--id-hard", required=True)
    parser.add_argument("--near-ood", required=True)
    parser.add_argument("--far-ood", required=True)
    parser.add_argument("--output-dir", required=True)
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
    checkpoints = [("pre", 0, ROOT / args.pre_checkpoint)]
    checkpoints += [("partial", epoch, ROOT / args.partial_dir / f"adapted_epoch_{epoch}.pt") for epoch in (1, 2, 3)]
    checkpoints += [("full", epoch, ROOT / args.full_dir / f"adapted_epoch_{epoch}.pt") for epoch in (1, 2, 3)]
    regimes = {"20ns": args.id_easy, "80ns": args.id_hard, "120ns": args.near_ood, "1ms": args.far_ood}
    summary_rows, sample_rows = [], []
    started = time.perf_counter()
    for model_name, epoch, checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = load_model(config, checkpoint, device)
        for regime, dataset in regimes.items():
            for ng in (16, 32):
                metrics = evaluate(model, ROOT / dataset, config, device, ng, args.seed + ng * 1000, args.batch_size)
                finite_count = sum(row["finite"] for row in metrics)
                for row in metrics:
                    sample_rows.append({"model": model_name, "epoch": epoch, "regime": regime, "ng": ng, **row})
                def mean(key): return float(np.mean([row[key] for row in metrics]))
                summary_rows.append({"model": model_name, "epoch": epoch, "regime": regime, "ng": ng,
                                     "samples": len(metrics), "finite_samples": finite_count,
                                     "nmse_all_db": mean("nmse_all_db"), "nmse_omitted_db": mean("nmse_omitted_db"),
                                     "aleatoric": mean("aleatoric"), "epistemic": mean("epistemic"),
                                     "kappa": mean("kappa"), "nu_margin": mean("nu_margin"),
                                     "all_finite": finite_count == len(metrics)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "sample_metrics.csv", sample_rows)
    manifest = {"pre_checkpoint": args.pre_checkpoint, "partial_dir": args.partial_dir, "full_dir": args.full_dir,
                "regimes": regimes, "ngs": [16, 32], "seed": args.seed, "batch_size": args.batch_size,
                "device": str(device), "gpu": torch.cuda.get_device_name(0), "same_eval_order": True,
                "same_eval_mask": "fixed canonical offset 0 periodic mask[:,::Ng]",
                "same_eval_noise": "deterministic 15 dB complex AWGN schedule keyed by seed+Ng*1000",
                "input_is_sparse": True, "target_is_clean_full_cfr": True,
                "no_training": True, "elapsed_seconds": time.perf_counter() - started}
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "comparison_results.json").write_text(json.dumps({"manifest": manifest, "summary_path": "summary.csv"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "rows": len(summary_rows), "samples": len(sample_rows), "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
