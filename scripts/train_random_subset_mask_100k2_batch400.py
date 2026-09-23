#!/usr/bin/env python3
"""Scratch 100k BF16 run with random-subset masks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915"
BASE_CONFIG = BASE_DIR / "config.json"
DEFAULT_TRAIN_PATH = BASE_DIR / "generated_data/train_5k_pilot.npz"
FACTORS = [4, 8, 16, 32, 64, 128]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def random_subset_mask(batch_size: int, num_subcarriers: int, factors: list[int], device: torch.device):
    """Choose Ng uniformly, then choose K/Ng positions without replacement per sample."""
    factor_tensor = torch.tensor(factors, device=device, dtype=torch.long)
    factor_indices = torch.randint(0, len(factors), (batch_size,), device=device)
    picks = factor_tensor[factor_indices]
    mask = torch.zeros((batch_size, num_subcarriers), device=device, dtype=torch.float32)
    # topk over independent random scores gives a uniform subset without replacement.
    for factor in factors:
        rows = torch.nonzero(picks == factor, as_tuple=False).flatten()
        if rows.numel():
            count = num_subcarriers // factor
            chosen = torch.rand((rows.numel(), num_subcarriers), device=device).topk(count, dim=1).indices
            selected = torch.zeros((rows.numel(), num_subcarriers), device=device, dtype=torch.float32)
            selected.scatter_(1, chosen, 1.0)
            mask.index_copy_(0, rows, selected)
    return mask, picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH.relative_to(ROOT)))
    ap.add_argument("--epochs", type=int, default=2, choices=[1, 2])
    ap.add_argument("--batch-size", type=int, default=400, choices=[8, 400])
    ap.add_argument("--expected-n", type=int, default=100000)
    args = ap.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import evaluate, load_config, write_training_csvs
    from src.models.evidential import EvidentialOutput, evidential_loss
    from src.training.data import CFRNPZDataset, build_noisy_sparse_input
    from src.training.metrics import nmse_all_db, nmse_omitted_db

    cfg = load_config(BASE_CONFIG)
    cfg = copy.deepcopy(cfg)
    assump = cfg["implementation_assumption"]
    train_path = ROOT / args.train_path
    train_data = CFRNPZDataset(train_path)
    n_train = len(train_data)
    if n_train != args.expected_n:
        raise RuntimeError(f"Expected {args.expected_n} training samples, found {n_train}")
    batch = args.batch_size
    steps = math.ceil(n_train / batch)
    total_steps = steps * args.epochs
    exposure = n_train * args.epochs
    print("=== PRE-TRAINING GATE ===", flush=True)
    print(json.dumps({
        "unique_training_cfr_count": n_train, "epochs": args.epochs,
        "batch_size": batch, "steps_per_epoch": steps,
        "total_optimizer_steps": total_steps, "total_sample_exposure": exposure,
        "cuda_available": torch.cuda.is_available(), "device": "cuda:0",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mask_factors": FACTORS, "mask_placement": "random subset without replacement",
    }, indent=2), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    expected_steps = math.ceil(100000 / batch)
    if (n_train, steps, total_steps, exposure) != (100000, expected_steps, expected_steps * args.epochs, 100000 * args.epochs):
        raise RuntimeError("Pre-training count gate failed")

    assump.update({
        "epochs": args.epochs, "pilot_epochs": args.epochs, "batch_size": batch,
        "effective_batch_size": batch, "update_mode": f"direct batch {batch}",
        "experiment_name": f"RANDOM-SUBSET-MASK-100k-{args.epochs}ep-batch{batch}-bf16-seed-20260819",
        "mask_grouping_factors": FACTORS,
        "mask_placement": "sample-wise random subset without replacement conditioned on Ng",
        "mask_sampling_probability": {str(f): 1.0 / len(FACTORS) for f in FACTORS},
        "mask_recovery": "scratch training; no checkpoint continuation",
    })
    out.mkdir(parents=True)
    seed = int(assump["seed"])
    set_seeds(seed)
    device = torch.device("cuda:0")
    loader = DataLoader(train_data, batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=True)
    val_data = CFRNPZDataset(cfg["data"]["validation_path"])
    val_loader = DataLoader(val_data, batch_size=int(assump["eval_batch_size"]), num_workers=0)
    model = _make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))
    write_json(out / "config_used.json", cfg)
    write_json(out / "run_start.json", {
        "training_archive": str(train_path.relative_to(ROOT)),
        "training_archive_sha256": sha256_file(train_path), "n_train": n_train,
        "epochs": args.epochs, "batch_size": batch, "steps_per_epoch": steps,
        "total_optimizer_steps": total_steps, "total_sample_exposure": exposure,
        "seed": seed, "device": str(device), "gpu": torch.cuda.get_device_name(0),
        "mask_factors": FACTORS, "mask_probability_each": {str(f): 1.0 / 6 for f in FACTORS},
        "mask_placement": "random subset without replacement", "ng1_training": False,
        "implementation_assumptions": [
            "IMPLEMENTATION-ASSUMPTION: random subset sampling conditioned on Ng in {4,8,16,32,64,128}, uniformly sampled",
            "BF16 backbone with FP32 evidential transform/loss",
            "Adam LR=1e-4, no scheduler, matching canonical script",
        ],
    })
    mask_counts = Counter()
    history = []
    started = time.perf_counter()
    checkpoint_tag = f"random_subset_mask_100k_{args.epochs}ep_batch{batch}_bf16"
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter(); count = 0
        epoch_start = time.perf_counter()
        for batch_data in loader:
            cfr = batch_data["cfr"].to(device, non_blocking=True)
            mask, picks = random_subset_mask(cfr.shape[0], 1024, FACTORS, device)
            for value in picks.detach().cpu().tolist():
                mask_counts[int(value)] += 1
            x, target, snr = build_noisy_sparse_input(cfr, mask, float(assump["observation_noise_snr_db"]))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = model(x)
            output = EvidentialOutput(gamma=raw.gamma.float(), kappa=raw.kappa.float(),
                                      psi=raw.psi.float(), nu=raw.nu.float(),
                                      num_subcarriers=raw.num_subcarriers)
            losses = evidential_loss(output, target.float(), float(cfg["paper_specified"]["lambda_reg"]),
                                     nll_mode=assump["nll_mode"], reg_mode=assump["reg_mode"])
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n = cfr.shape[0]; count += n
            for key in ("total", "nll", "reg", "lambda_reg_x_reg"):
                totals[key] += float(losses[key].detach().cpu()) * n
            totals["nmse_all_db"] += float(nmse_all_db(output.predicted.detach(), target).cpu()) * n
            totals["nmse_omitted_db"] += float(nmse_omitted_db(output.predicted.detach(), target, mask).cpu()) * n
            totals["measured_snr_db"] += float(snr["measured_snr_db_mean"]) * n
        train_metrics = {k: v / count for k, v in totals.items() if k != "measured_snr_db"}
        train_metrics["measured_snr_db_mean"] = totals["measured_snr_db"] / count
        train_metrics["reg_to_nll_ratio"] = abs(train_metrics["lambda_reg_x_reg"]) / (abs(train_metrics["nll"]) + 1e-8)
        validation = evaluate(model, val_loader, cfg, device)
        checkpoint = out / f"{checkpoint_tag}_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation,
               "elapsed_epoch_s": time.perf_counter() - epoch_start,
               "optimizer_steps": epoch * steps, "sample_exposure": epoch * n_train,
               "mask_draw_counts_cumulative": dict(sorted(mask_counts.items())),
               "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        history.append(row)
        write_json(out / "training_history.json", {"history": history, "mask_draw_counts": dict(sorted(mask_counts.items()))})
        with (out / "epoch_progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "checkpoint": str(checkpoint.relative_to(ROOT))}) + "\n")
        print(json.dumps({"epoch_complete": row}, sort_keys=True), flush=True)
    final = out / f"{checkpoint_tag}.pt"
    torch.save(model.state_dict(), final)
    write_training_csvs(out, [{"epoch": r["epoch"], "train": r["train"], "validation": r["validation"], "elapsed_epoch_s": r["elapsed_epoch_s"]} for r in history], {})
    result = {"experiment": f"scratch 100k unique CFR x {args.epochs} epochs x direct batch {batch} random-subset mask",
              "checkpoint": str(final.relative_to(ROOT)), "training_seconds": time.perf_counter() - started,
              "n_train": n_train, "epochs": args.epochs, "steps_per_epoch": steps,
              "total_optimizer_steps": total_steps, "total_sample_exposure": exposure,
              "device": str(device), "gpu": torch.cuda.get_device_name(0),
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
              "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
              "mask_factors": FACTORS, "mask_draw_counts": dict(sorted(mask_counts.items())),
              "history": history,
              "assumptions": ["IMPLEMENTATION-ASSUMPTION: random subset sampling conditioned on Ng in {4,8,16,32,64,128}, uniformly sampled", "BF16 backbone with FP32 evidential transform/loss", "Ng=1 excluded from training"]}
    write_json(out / "training_results.json", result)
    print(json.dumps({"complete": True, **{k: result[k] for k in ("checkpoint", "training_seconds", "mask_draw_counts")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
