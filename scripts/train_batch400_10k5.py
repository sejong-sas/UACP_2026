#!/usr/bin/env python3
"""Train the requested 10k unique CFR x 5 epoch direct-batch-400 baseline."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/current_valid_baseline_10k5_seed_20260819.json"
TRAIN = ROOT / "runs/current_valid_baseline/diversity_ablation/tenk_5ep_seed_20260819/generated_data/train_5k_pilot.npz"
COMMON = ROOT / "runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval"
REGIMES = [("ID-Easy 20 ns", "ID-Easy_20_ns.npz", 20.0), ("ID-Hard 80 ns", "ID-Hard_80_ns.npz", 80.0),
           ("OOD-Near 120 ns", "OOD-Near_120_ns.npz", 120.0), ("OOD-Far 1 ms", "OOD-Far_1_ms.npz", 1_000_000.0)]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/current_valid_baseline/batch400_10k5ep_20260917_retry")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import evaluate, load_config, set_seeds, train_one_epoch, write_training_csvs
    from src.training.data import CFRNPZDataset

    out = ROOT / args.output_dir
    if any(out.glob("uacp_predictor_batch400_10k5ep.pt")): raise FileExistsError(out)
    cfg = load_config(CONFIG)
    a = cfg["implementation_assumption"]
    a.update({"epochs": 5, "pilot_epochs": 5, "batch_size": 400,
              "experiment_name": "CURRENT-VALID-BATCH400-10k-5ep-seed-20260819",
              "update_mode": "direct batch 400", "effective_batch_size": 400,
              "implementation_assumption_batch400": "Direct batch 400; no gradient accumulation."})
    cfg["data"]["train_path"] = str(TRAIN.relative_to(ROOT))
    if len(CFRNPZDataset(TRAIN)) != 10000: raise RuntimeError("training archive is not 10,000 unique CFRs")
    out.mkdir(parents=True, exist_ok=True)
    seed = int(a["seed"]); set_seeds(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0); torch.cuda.reset_peak_memory_stats(device)
    train_data = CFRNPZDataset(TRAIN); val_data = CFRNPZDataset(cfg["data"]["validation_path"])
    train_loader = DataLoader(train_data, batch_size=400, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=int(a["eval_batch_size"]), num_workers=0)
    model = _make_model(cfg, device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history, probes = [], {}; start = time.perf_counter()
    for epoch in range(1, 6):
        ep = time.perf_counter(); train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        result = {"validation": val_metrics}
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                       "elapsed_epoch_s": time.perf_counter() - ep})
        probes[str(epoch)] = result
        torch.save(model.state_dict(), out / f"uacp_predictor_batch400_10k5ep_epoch_{epoch}.pt")
        (out / "epoch_progress.jsonl").open("a", encoding="utf-8").write(json.dumps(history[-1]) + "\n")
        (out / "training_history.json").write_text(json.dumps({"history": history, "epoch_probe_results": probes}, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                          "elapsed_epoch_s": history[-1]["elapsed_epoch_s"],
                          "gpu_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                          "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}, sort_keys=True), flush=True)
    final = out / "uacp_predictor_batch400_10k5ep.pt"; torch.save(model.state_dict(), final)
    test_results = {}
    write_training_csvs(out, history, test_results)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2) + "\n")
    summary = {"experiment": "10k unique CFR x 5 epochs x direct batch 400", "config": cfg,
       "checkpoint": str(final.relative_to(ROOT)), "device": str(device), "gpu": gpu,
       "training_seconds": time.perf_counter() - start, "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
       "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20, "peak_rss_gib": rss_gib(),
       "training_archive_sha256": sha256(TRAIN), "common_eval_dir": str(COMMON.relative_to(ROOT)),
       "history": history, "epoch_probe_results": probes,
       "assumptions": ["Diagonal-Psi approximation", "15 dB sample-wise complex AWGN on reported CFR; clean target",
                       "Uniform Ng=16 evaluation mask", "IMPLEMENTATION-ASSUMPTION: 10k sample-count scaling",
                       "IMPLEMENTATION-ASSUMPTION: direct batch 400; no gradient accumulation"]}
    (out / "training_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "checkpoint": str(final), "training_seconds": summary["training_seconds"],
                      "gpu": gpu, "peak_gpu_allocated_mib": summary["peak_gpu_allocated_mib"]}, indent=2), flush=True)

if __name__ == "__main__": main()
