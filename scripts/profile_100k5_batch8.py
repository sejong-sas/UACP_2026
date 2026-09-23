#!/usr/bin/env python3
"""Short, non-training profiling of the existing 100k x 5 batch8 FP32 path.

This script intentionally does not update any existing run. It uses the same
dataset/config/model/loss path as train_100k5_convergence.py, executes a small
representative number of optimizer steps in memory, and writes all artifacts to
the caller-provided new output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/config_used.json"
TRAIN_PATH = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/generated_data/train_5k_pilot.npz"
VAL_PATH = ROOT / "data/prototype/validation.npz"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_utilization() -> float | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        return float(value)
    except Exception:
        return None


class GPUSampler:
    def __enter__(self):
        self.values: list[float] = []
        self.stop = threading.Event()

        def worker() -> None:
            while not self.stop.is_set():
                value = gpu_utilization()
                if value is not None:
                    self.values.append(value)
                self.stop.wait(0.1)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--profile-steps", type=int, default=64)
    args = parser.parse_args()

    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))

    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    from src.models.evidential import evidential_loss
    from src.training.data import CFRNPZDataset, build_noisy_sparse_input, random_grouping_mask
    from src.training.metrics import nmse_all_db, nmse_omitted_db

    cfg = load_config(CONFIG)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("profiling requires cuda:0")
    gpu = torch.cuda.get_device_name(0)
    if "GB10" not in gpu:
        raise RuntimeError(f"expected NVIDIA GB10, found {gpu}")

    model = _make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))
    train_data = CFRNPZDataset(TRAIN_PATH)
    loader = DataLoader(
        train_data,
        batch_size=int(cfg["implementation_assumption"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["implementation_assumption"]["seed"])),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    val_data = CFRNPZDataset(VAL_PATH)
    val_loader = DataLoader(val_data, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]), num_workers=0)
    iterator = iter(loader)
    timings: list[dict[str, float | int]] = []
    gpu_samples: list[float] = []

    def run_step(step: int) -> dict[str, float | int]:
        row: dict[str, float | int] = {"step": step}
        total_start = time.perf_counter()
        stage_start = total_start
        batch = next(iterator)
        row["data_loading_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        cfr = batch["cfr"].to(device)
        sync(device)
        row["cpu_to_gpu_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        mask = random_grouping_mask(
            cfr.shape[0],
            int(cfg["paper_specified"]["num_subcarriers"]),
            list(cfg["implementation_assumption"]["mask_grouping_factors"]),
            device,
        )
        x, target, snr_stats = build_noisy_sparse_input(cfr, mask, 15.0)
        sync(device)
        row["observation_mask_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        output = model(x)
        sync(device)
        row["forward_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        losses = evidential_loss(
            output,
            target,
            float(cfg["paper_specified"]["lambda_reg"]),
            nll_mode="diagonal_multivariate",
            reg_mode="pair",
        )
        sync(device)
        row["evidential_loss_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        sync(device)
        row["backward_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        optimizer.step()
        sync(device)
        row["optimizer_step_s"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        _ = float(losses["total"].detach().cpu())
        _ = float(nmse_all_db(output.predicted.detach(), target).cpu())
        _ = float(nmse_omitted_db(output.predicted.detach(), target, mask).cpu())
        _ = float(snr_stats["measured_snr_db_mean"])
        row["post_step_metrics_logging_s"] = time.perf_counter() - stage_start
        row["total_wall_s"] = time.perf_counter() - total_start
        row["residual_other_sync_s"] = max(
            0.0,
            row["total_wall_s"]
            - sum(float(row[k]) for k in row if k.endswith("_s") and k not in {"total_wall_s", "residual_other_sync_s"}),
        )
        return row

    model.train()
    with GPUSampler() as sampler:
        for _ in range(args.warmup_steps):
            run_step(-1)
        sync(device)
        for step in range(args.profile_steps):
            timings.append(run_step(step))
        gpu_samples.extend(sampler.values)

    sync(device)
    eval_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        from scripts.train_predictor import evaluate
        validation_metrics = evaluate(model, val_loader, cfg, device)
    sync(device)
    validation_s = time.perf_counter() - eval_start

    checkpoint_start = time.perf_counter()
    torch.save(model.state_dict(), out / "profiling_checkpoint.pt")
    checkpoint_save_s = time.perf_counter() - checkpoint_start

    fields = list(timings[0])
    with (out / "step_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timings)

    phase_names = [key for key in fields if key.endswith("_s")]
    phase_summary = {name: stats([float(row[name]) for row in timings]) for name in phase_names}
    total_samples = int(args.profile_steps * int(cfg["implementation_assumption"]["batch_size"]))
    total_s = float(sum(float(row["total_wall_s"]) for row in timings))
    result = {
        "experiment": "short 100k x 5 batch8 FP32 training-path profiling",
        "training_performed": "representative in-memory optimizer steps only; no existing run modified",
        "config": {
            "config_source": str(CONFIG.relative_to(ROOT)),
            "train_path": str(TRAIN_PATH.relative_to(ROOT)),
            "batch_size": int(cfg["implementation_assumption"]["batch_size"]),
            "precision": "FP32",
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
        },
        "device": str(device),
        "gpu": gpu,
        "torch_cuda": torch.version.cuda,
        "samples_profiled": total_samples,
        "steps_profiled": args.profile_steps,
        "samples_per_second_wall": total_samples / total_s,
        "seconds_per_step_wall": total_s / args.profile_steps,
        "phase_summary": phase_summary,
        "validation": {"elapsed_s": validation_s, "metrics": validation_metrics},
        "checkpoint_save_s": checkpoint_save_s,
        "gpu_utilization_percent": stats(gpu_samples),
        "peak_gpu_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "peak_gpu_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2,
        "implementation_assumptions": [
            "Short representative profiling is not a full training run.",
            "Timing boundaries synchronize cuda:0 to expose stage wall time.",
            "The profiling checkpoint is new and is not a research candidate.",
            "FP32 and the existing model/loss/observation path are unchanged.",
        ],
    }
    (out / "profiling_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "profiling_config.json").write_text(json.dumps(result["config"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out.relative_to(ROOT)),
        "gpu": gpu,
        "seconds_per_step_wall": result["seconds_per_step_wall"],
        "samples_per_second_wall": result["samples_per_second_wall"],
        "peak_gpu_allocated_mib": result["peak_gpu_allocated_mib"],
        "validation_s": validation_s,
        "checkpoint_save_s": checkpoint_save_s,
        "gpu_util_mean": result["gpu_utilization_percent"]["mean"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
