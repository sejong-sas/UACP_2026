#!/usr/bin/env python3
"""Train the canonical 100k CFR archive for 5 epochs with direct batch 400."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915"
BASE_CONFIG = BASE_DIR / "config.json"
BASE_CHECKPOINT = BASE_DIR / "uacp_predictor_step4a.pt"
DEFAULT_TRAIN_PATH = BASE_DIR / "generated_data/train_5k_pilot.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2


def system_memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) / 1024**2
    return {"total_gib": values.get("MemTotal"), "available_gib": values.get("MemAvailable")}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assert_conditions(base: dict, candidate: dict) -> None:
    """Require canonical data/model/loss/evaluation settings; permit epoch/batch/run metadata."""
    left = copy.deepcopy(base)
    right = copy.deepcopy(candidate)
    for cfg in (left, right):
        assumption = cfg["implementation_assumption"]
        for key in ("epochs", "pilot_epochs", "batch_size", "experiment_name", "update_mode",
                    "effective_batch_size", "implementation_assumption_batch400"):
            assumption.pop(key, None)
    if left != right:
        raise ValueError("candidate changed a canonical condition beyond epochs/batch/run metadata")


class ProgressLoader:
    def __init__(self, loader, epoch: int, output: Path, device: torch.device):
        self.loader, self.epoch, self.output, self.device = loader, epoch, output, device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        started = time.perf_counter()
        for index, batch in enumerate(self.loader, 1):
            yield batch
            if index % 25 == 0 or index == len(self.loader):
                row = {
                    "epoch": self.epoch,
                    "batches_done": index,
                    "batches_total": len(self.loader),
                    "optimizer_steps_done": (self.epoch - 1) * len(self.loader) + index,
                    "sample_exposure_done": ((self.epoch - 1) * len(self.loader) + index) * self.loader.batch_size,
                    "elapsed_s": time.perf_counter() - started,
                    "gpu_allocated_mib": torch.cuda.memory_allocated(self.device) / 2**20,
                    "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
                    "rss_peak_gib": rss_gib(),
                }
                with self.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({"batch_progress": row}), flush=True)


def train_one_epoch_bf16(model, loader, optimizer, cfg, device: torch.device) -> dict[str, float]:
    """C condition: BF16 backbone, FP32 evidential loss and metrics."""
    from src.models.evidential import evidential_loss
    from src.training.data import build_noisy_sparse_input, random_grouping_mask
    from src.training.metrics import nmse_all_db, nmse_omitted_db

    model.train()
    totals = {"total": 0.0, "nll": 0.0, "reg": 0.0, "lambda_reg_x_reg": 0.0,
              "nmse_all_db": 0.0, "nmse_omitted_db": 0.0, "measured_snr_db": 0.0}
    count = 0
    snr_count = 0
    for batch in loader:
        cfr = batch["cfr"].to(device, non_blocking=True)
        mask = random_grouping_mask(cfr.shape[0], int(cfg["paper_specified"]["num_subcarriers"]),
                                    list(cfg["implementation_assumption"]["mask_grouping_factors"]), device)
        x, target, snr_stats = build_noisy_sparse_input(
            cfr, mask, float(cfg["implementation_assumption"]["observation_noise_snr_db"])
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw_output = model(x)
        from src.models.evidential import EvidentialOutput
        output = EvidentialOutput(gamma=raw_output.gamma.float(), kappa=raw_output.kappa.float(),
                                  psi=raw_output.psi.float(), nu=raw_output.nu.float(),
                                  num_subcarriers=raw_output.num_subcarriers)
        losses = evidential_loss(output, target.float(), float(cfg["paper_specified"]["lambda_reg"]),
                                 nll_mode=cfg["implementation_assumption"]["nll_mode"],
                                 reg_mode=cfg["implementation_assumption"]["reg_mode"])
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = int(cfr.shape[0]); count += n
        for key in ("total", "nll", "reg", "lambda_reg_x_reg"):
            totals[key] += float(losses[key].detach().cpu()) * n
        totals["nmse_all_db"] += float(nmse_all_db(output.predicted.detach(), target).cpu()) * n
        totals["nmse_omitted_db"] += float(nmse_omitted_db(output.predicted.detach(), target, mask).cpu()) * n
        if snr_stats["measured_snr_db_mean"] is not None:
            totals["measured_snr_db"] += float(snr_stats["measured_snr_db_mean"]) * n
            snr_count += n
    metrics = {key: value / count for key, value in totals.items() if key != "measured_snr_db"}
    metrics["measured_snr_db_mean"] = totals["measured_snr_db"] / snr_count if snr_count else None
    metrics["reg_to_nll_ratio"] = abs(metrics["lambda_reg_x_reg"]) / (abs(metrics["nll"]) + 1e-8)
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH.relative_to(ROOT)))
    parser.add_argument("--expected-n", type=int, default=100000)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import evaluate, load_config, set_seeds, train_one_epoch, write_training_csvs
    from src.training.data import CFRNPZDataset

    train_path = ROOT / args.train_path

    base = load_config(BASE_CONFIG)
    cfg = copy.deepcopy(base)
    assumption = cfg["implementation_assumption"]
    assumption.update({
        "epochs": 5,
        "pilot_epochs": 5,
        "batch_size": 400,
        "experiment_name": "PAPER-ALIGNED-SAMPLE-COUNT-100k-5ep-batch400-seed-20260819",
        "update_mode": "direct batch 400",
        "effective_batch_size": 400,
        "implementation_assumption_batch400": "Direct batch 400; no gradient accumulation.",
    })
    assert_conditions(base, cfg)
    if not train_path.is_file() or not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError("training archive or canonical checkpoint missing")
    train_data = CFRNPZDataset(train_path)
    n_train = len(train_data)
    cfg["data"]["train_path"] = str(train_path.relative_to(ROOT))
    assumption["experiment_name"] = f"PAPER-ALIGNED-SAMPLE-COUNT-{n_train // 1000}k-5ep-batch400-bf16-seed-20260819"
    steps_per_epoch = math.ceil(n_train / 400)
    total_steps = steps_per_epoch * 5
    total_exposure = n_train * 5

    # Mandatory pre-training gate: print and validate the actual cardinality first.
    print("=== PRE-TRAINING GATE ===", flush=True)
    print(json.dumps({
        "unique_training_cfr_count": n_train,
        "expected_training_cfr_count": args.expected_n,
        "epochs": 5,
        "batch_size": 400,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "total_sample_exposure": total_exposure,
        "cuda_available": torch.cuda.is_available(),
        "device_requested": "cuda:0",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2), flush=True)
    if (n_train, steps_per_epoch, total_steps, total_exposure) != (args.expected_n, math.ceil(args.expected_n / 400), math.ceil(args.expected_n / 400) * 5, args.expected_n * 5):
        raise RuntimeError("pre-training gate failed: dataset/step/exposure counts do not match")
    if not torch.cuda.is_available():
        raise RuntimeError("pre-training gate failed: CUDA is unavailable")

    output.mkdir(parents=True, exist_ok=True)
    seed = int(assumption["seed"])
    set_seeds(seed)
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = system_memory()
    train_loader = DataLoader(train_data, batch_size=400, shuffle=True,
                               generator=torch.Generator().manual_seed(seed), num_workers=0,
                               pin_memory=True)
    val_data = CFRNPZDataset(cfg["data"]["validation_path"])
    val_loader = DataLoader(val_data, batch_size=int(assumption["eval_batch_size"]), num_workers=0)
    if len(train_loader) != steps_per_epoch:
        raise RuntimeError(f"DataLoader length mismatch: {len(train_loader)}")
    model = _make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))
    write_json(output / "config_used.json", cfg)
    write_json(output / "run_start.json", {
        "training_archive": str(train_path.relative_to(ROOT)),
        "training_archive_sha256": sha256_file(train_path),
        "canonical_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "canonical_checkpoint_sha256_before": sha256_file(BASE_CHECKPOINT),
        "n_train": n_train,
        "epochs": 5,
        "batch_size": 400,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "total_sample_exposure": total_exposure,
        "device": str(device),
        "gpu": gpu,
        "system_memory": memory_before,
    })

    history = []
    started = time.perf_counter()
    for epoch in range(1, 6):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch_bf16(model, ProgressLoader(train_loader, epoch, output / "batch_progress.jsonl", device), optimizer, cfg, device)
        validation_metrics = evaluate(model, val_loader, cfg, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics,
               "elapsed_epoch_s": time.perf_counter() - epoch_started}
        history.append(row)
        checkpoint = output / f"uacp_predictor_{n_train // 1000}k_5ep_batch400_bf16_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint)
        write_json(output / "training_history.json", {"history": history, "n_train": n_train,
                                                         "steps_per_epoch": steps_per_epoch,
                                                         "total_optimizer_steps": total_steps,
                                                         "total_sample_exposure": total_exposure})
        with (output / "epoch_progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**row, "checkpoint": str(checkpoint.relative_to(ROOT))}) + "\n")
        print(json.dumps({"epoch_complete": row, "checkpoint": str(checkpoint.relative_to(ROOT)),
                          "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}, sort_keys=True), flush=True)

    final = output / f"uacp_predictor_{n_train // 1000}k_5ep_batch400_bf16.pt"
    torch.save(model.state_dict(), final)
    write_training_csvs(output, history, {})
    result = {
        "experiment": f"{n_train} unique CFR x 5 epochs x direct batch 400",
        "config": cfg,
        "checkpoint": str(final.relative_to(ROOT)),
        "training_archive": str(train_path.relative_to(ROOT)),
        "training_archive_sha256": sha256_file(train_path),
        "device": str(device),
        "gpu": gpu,
        "training_seconds": time.perf_counter() - started,
        "n_train": n_train,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "total_sample_exposure": total_exposure,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "peak_rss_gib": rss_gib(),
        "history": history,
        "canonical_checkpoint_sha256_unchanged": sha256_file(BASE_CHECKPOINT) == json.loads((output / "run_start.json").read_text())["canonical_checkpoint_sha256_before"],
        "conditions_unchanged_except_epoch_and_batch": True,
        "assumptions": [
            "IMPLEMENTATION-ASSUMPTION: diagonal-Psi approximation",
            "IMPLEMENTATION-ASSUMPTION: reported-CFR sample-wise complex AWGN at 15 dB; clean target",
            "IMPLEMENTATION-ASSUMPTION: pair-scalar kappa/nu and diagonal multivariate Student-t NLL",
            "IMPLEMENTATION-ASSUMPTION: direct batch 400 with no gradient accumulation",
            "IMPLEMENTATION-ASSUMPTION: BF16 autocast for convolutional backbone; evidential transforms and loss in FP32",
        ],
    }
    write_json(output / "training_results.json", result)
    print(json.dumps({"complete": True, "checkpoint": result["checkpoint"],
                      "training_seconds": result["training_seconds"], "gpu": gpu,
                      "peak_gpu_allocated_mib": result["peak_gpu_allocated_mib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
