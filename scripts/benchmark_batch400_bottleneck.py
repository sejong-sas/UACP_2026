#!/usr/bin/env python3
"""Profile 20 direct-batch-400 steps without launching full training.

Conditions:
  A: current deterministic FP32 loop, including per-step metric CPU sync.
  B: FP32 runtime optimizations (pinned transfer, non-blocking copy, no
     per-step CPU metric sync, optional torch.compile).
  C: B plus bfloat16 autocast for the backbone; evidential loss remains FP32.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json"
TRAIN_PATH = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/generated_data/train_5k_pilot.npz"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nvidia_snapshot() -> dict[str, float | None]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        fields = [x.strip() for x in raw.split(",")[:2]]
        util = float(fields[0])
        try:
            memory = float(fields[1])
        except ValueError:
            memory = None
        return {"gpu_utilization_percent": util, "memory_used_mib": memory}
    except Exception:
        return {"gpu_utilization_percent": None, "memory_used_mib": None}


class UtilizationSampler:
    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.samples: list[dict[str, float | None]] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        def run():
            while not self.stop.is_set():
                self.samples.append(nvidia_snapshot())
                self.stop.wait(self.interval_s)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def make_model(cfg: dict, device: torch.device):
    from scripts.diagnose_predictor import _make_model
    return _make_model(cfg, device)


def fp32_output(output):
    """Keep sensitive evidential transforms/loss calculations in FP32 for C."""
    from src.models.evidential import EvidentialOutput
    return EvidentialOutput(
        gamma=output.gamma.float(),
        kappa=output.kappa.float(),
        psi=output.psi.float(),
        nu=output.nu.float(),
        num_subcarriers=output.num_subcarriers,
    )


def run_condition(name: str, cfg: dict, output: Path, use_optimizations: bool,
                  use_compile: bool, use_amp: bool, steps: int, batch_size: int,
                  seed: int) -> dict:
    from scripts.train_predictor import set_seeds
    from src.models.evidential import evidential_loss
    from src.training.data import CFRNPZDataset, build_noisy_sparse_input, random_grouping_mask
    from src.training.metrics import nmse_all_db, nmse_omitted_db

    set_seeds(seed)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dataset = CFRNPZDataset(TRAIN_PATH)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=use_optimizations,
        persistent_workers=False,
    )
    model = make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))
    compile_status = "disabled"
    compile_seconds = 0.0
    if use_compile:
        compile_started = time.perf_counter()
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            compile_status = "enabled"
        except Exception as exc:
            compile_status = f"failed:{type(exc).__name__}:{exc}"
        compile_seconds = time.perf_counter() - compile_started

    model.train()
    rows = []
    sums = {k: 0.0 for k in ("data_loading", "cpu_to_gpu_transfer", "forward", "evidential_loss", "backward", "optimizer_step", "metric_sync_logging")}
    loss_values: list[float] = []
    nmse_values: list[float] = []
    deferred_loss_values: list[torch.Tensor] = []
    deferred_nmse_values: list[torch.Tensor] = []
    iterator = iter(loader)
    stream_start = time.perf_counter()
    with UtilizationSampler() as utilization:
        for step in range(steps):
            # Isolate DataLoader time from preceding GPU work.
            cuda_sync(device)
            t0 = time.perf_counter()
            batch = next(iterator)
            t1 = time.perf_counter()
            cfr_cpu = batch["cfr"]

            cuda_sync(device)
            t2 = time.perf_counter()
            cfr = cfr_cpu.to(device, non_blocking=use_optimizations)
            if use_optimizations:
                cuda_sync(device)
            t3 = time.perf_counter()

            mask = random_grouping_mask(batch_size, int(cfg["paper_specified"]["num_subcarriers"]),
                                        list(cfg["implementation_assumption"]["mask_grouping_factors"]), device)
            x, target, _ = build_noisy_sparse_input(cfr, mask, float(cfg["implementation_assumption"]["observation_noise_snr_db"]))
            cuda_sync(device)
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record();
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_output = model(x)
                output_model = fp32_output(raw_output)
            else:
                output_model = model(x)
            e1.record()
            forward_s = cuda_elapsed(e0, e1)

            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            losses = evidential_loss(output_model, target.float(), float(cfg["paper_specified"]["lambda_reg"]),
                                     nll_mode=cfg["implementation_assumption"]["nll_mode"],
                                     reg_mode=cfg["implementation_assumption"]["reg_mode"])
            e1.record()
            loss_s = cuda_elapsed(e0, e1)

            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); optimizer.zero_grad(set_to_none=True); losses["total"].backward(); e1.record()
            backward_s = cuda_elapsed(e0, e1)

            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step(); e1.record()
            optimizer_s = cuda_elapsed(e0, e1)

            # A deliberately preserves the current per-step .cpu() metric behavior.
            # B/C retain metrics on-device and transfer only once per benchmark.
            metric_start = time.perf_counter()
            nmse_gpu = nmse_all_db(output_model.predicted.detach(), target)
            nmse_omitted_gpu = nmse_omitted_db(output_model.predicted.detach(), target, mask)
            if not use_optimizations:
                loss_value = float(losses["total"].detach().cpu())
                nmse_value = float(nmse_omitted_gpu.detach().cpu())
            else:
                # Defer the only host transfer for B/C until after all steps.
                deferred_loss_values.append(losses["total"].detach())
                deferred_nmse_values.append(nmse_omitted_gpu.detach())
                loss_value = float("nan")
                nmse_value = float("nan")
            metric_end = time.perf_counter()
            metric_sync_s = metric_end - metric_start
            # Ensure stage times from CUDA events and end-to-end wall time are complete.
            cuda_sync(device)
            wall_s = time.perf_counter() - t0
            row = {"condition": name, "step": step + 1, "data_loading_s": t1 - t0,
                   "cpu_to_gpu_transfer_s": t3 - t2, "forward_s": forward_s,
                   "evidential_loss_s": loss_s, "backward_s": backward_s,
                   "optimizer_step_s": optimizer_s, "metric_sync_logging_s": metric_sync_s,
                   "wall_step_s": wall_s, "loss": loss_value, "nmse_omitted_db": nmse_value}
            rows.append(row)
            for key in sums:
                sums[key] += row[f"{key}_s"]
            if not use_optimizations:
                loss_values.append(loss_value); nmse_values.append(nmse_value)

    if use_optimizations:
        loss_values = [float(x) for x in torch.stack(deferred_loss_values).cpu()]
        nmse_values = [float(x) for x in torch.stack(deferred_nmse_values).cpu()]
        for row, loss_value, nmse_value in zip(rows, loss_values, nmse_values):
            row["loss"] = loss_value
            row["nmse_omitted_db"] = nmse_value

    # Convert on-device values for optimized paths only after all steps.
    samples = [x for x in utilization.samples if x["gpu_utilization_percent"] is not None]
    gpu_utils = [x["gpu_utilization_percent"] for x in samples]
    result = {
        "condition": name,
        "use_optimizations": use_optimizations,
        "use_torch_compile": use_compile,
        "compile_status": compile_status,
        "compile_seconds": compile_seconds,
        "use_bf16_amp": use_amp,
        "steps": steps,
        "batch_size": batch_size,
        "n_train": len(dataset),
        "steps_per_epoch": math.ceil(len(dataset) / batch_size),
        "mean_sec_per_step": float(np.mean([r["wall_step_s"] for r in rows[1:]])),
        "median_sec_per_step": float(np.median([r["wall_step_s"] for r in rows[1:]])),
        "stage_mean_sec": {key: value / steps for key, value in sums.items()},
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "gpu_utilization_samples": len(gpu_utils),
        "gpu_utilization_mean_percent": float(np.mean(gpu_utils)) if gpu_utils else None,
        "gpu_utilization_median_percent": float(np.median(gpu_utils)) if gpu_utils else None,
        "gpu_utilization_p95_percent": float(np.quantile(gpu_utils, 0.95)) if gpu_utils else None,
        "loss_first": loss_values[0],
        "loss_last": loss_values[-1],
        "loss_mean_last_5": float(np.mean(loss_values[-5:])),
        "nmse_omitted_db_first": nmse_values[0],
        "nmse_omitted_db_last": nmse_values[-1],
        "nmse_omitted_db_mean_last_5": float(np.mean(nmse_values[-5:])),
        "rows": rows,
        "assumptions": [
            "A preserves current per-step .cpu() metric synchronization.",
            "B removes per-step CPU metric conversion and uses pinned/non-blocking transfer.",
            "C uses IMPLEMENTATION-ASSUMPTION: bfloat16 autocast for backbone; evidential transforms/loss FP32.",
            "No gradient checkpointing was enabled in the model; no removal was necessary.",
            "No dense covariance is materialized by the diagonal multivariate loss.",
        ],
    }
    condition_dir = output / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    with (condition_dir / "step_timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(condition_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=400)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.train_predictor import load_config
    cfg = load_config(BASE_CONFIG)
    with np.load(TRAIN_PATH, allow_pickle=False) as archive:
        n_train = int(archive["cfr"].shape[0])
    if n_train != 100000 or args.batch_size != 400:
        raise RuntimeError(f"benchmark gate failed: n_train={n_train}, batch={args.batch_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preflight = {"n_train": n_train, "epochs_target": 5, "batch_size": args.batch_size,
                 "steps_per_epoch": math.ceil(n_train / args.batch_size),
                 "total_optimizer_steps_target": math.ceil(n_train / args.batch_size) * 5,
                 "total_sample_exposure_target": n_train * 5,
                 "device": "cuda:0", "gpu": torch.cuda.get_device_name(0),
                 "full_training_started": False}
    write_json(output / "preflight.json", preflight)
    write_json(output / "benchmark_config.json", {"base_config": str(BASE_CONFIG.relative_to(ROOT)),
                                                    "train_path": str(TRAIN_PATH.relative_to(ROOT)),
                                                    "steps": args.steps, "batch_size": args.batch_size,
                                                    "conditions": ["A_fp32_current", "B_fp32_runtime_optimized", "C_bf16_amp"]})
    results = []
    # torch.compile was explicitly probed before this steady-state comparison. On this
    # 32-block model, Inductor remained in compilation for several minutes and spawned
    # many workers, so B/C intentionally measure the actionable runtime changes without
    # compile. The review is recorded in summary.json rather than conflated with step time.
    results.append(run_condition("A_fp32_current", cfg, output, False, False, False, args.steps, args.batch_size, 20260918))
    results.append(run_condition("B_fp32_runtime_optimized", cfg, output, True, False, False, args.steps, args.batch_size, 20260918))
    results.append(run_condition("C_bf16_amp", cfg, output, True, False, True, args.steps, args.batch_size, 20260918))
    write_json(output / "summary.json", {"preflight": preflight, "results": results,
                                          "full_training_started": False,
                                          "torch_compile_review": {
                                              "status": "not_suitable_for_this_benchmark",
                                              "observed": "Inductor compilation remained active for several minutes and spawned 20 workers before steady-state steps.",
                                              "decision": "excluded from B/C steady-state timing; no evidence yet of a useful speedup",
                                          },
                                          "decision_gate": {"target_mean_sec_per_step": 8.0,
                                                            "C_near_target_required_before_full_training": True}})
    print(json.dumps({"complete": True, "output": str(output),
                      "mean_sec_per_step": {r["condition"]: r["mean_sec_per_step"] for r in results},
                      "full_training_started": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
