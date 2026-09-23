#!/usr/bin/env python3
"""Controlled 100k×3 retraining with explicitly balanced Ng/offset exposure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915"
BASE_CONFIG = BASE_DIR / "config.json"
BASE_CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
REGIMES = [
    ("ID-Easy 20 ns", "data/prototype/test_id_easy.npz", 20.0),
    ("ID-Hard 80 ns", "data/prototype/test_id_hard.npz", 80.0),
    ("OOD-Near 120 ns", "data/prototype/test_ood_near.npz", 120.0),
    ("OOD-Far 1 ms", "runs/baseline_reproduction/step5_snr_ablation/noisy_train_noisy_eval/generated_data/test_ood_far_1ms.npz", 1_000_000.0),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2


def system_memory() -> dict:
    values = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) / 1024**2
    return {"total_gib": values.get("MemTotal"), "available_gib": values.get("MemAvailable")}


def build_balanced_schedule(samples: int, factors: list[int], seed: int) -> list[tuple[int, int]]:
    if samples % len(factors) != 0:
        raise ValueError("sample count must divide evenly across Ng factors")
    per_factor = samples // len(factors)
    schedule = []
    for factor in factors:
        for index in range(per_factor):
            schedule.append((factor, index % factor))
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def schedule_counts(schedule: list[tuple[int, int]], factors: list[int]) -> dict:
    ng = {str(factor): 0 for factor in factors}
    offsets = {str(factor): {str(offset): 0 for offset in range(factor)} for factor in factors}
    for factor, offset in schedule:
        ng[str(factor)] += 1
        offsets[str(factor)][str(offset)] += 1
    return {"ng_counts": ng, "ng_ratios": {key: value / len(schedule) for key, value in ng.items()}, "offset_counts": offsets}


def make_balanced_mask(assignments: list[tuple[int, int]], num_subcarriers: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(assignments), num_subcarriers), dtype=torch.float32, device=device)
    for row, (factor, offset) in enumerate(assignments):
        mask[row, offset::factor] = 1.0
    return mask


def consume_legacy_mask_rng(batch_size: int, num_subcarriers: int, factors: list[int], device: torch.device) -> None:
    """Consume the same CUDA randint calls as random_grouping_mask, then discard them."""
    from src.training.data import random_grouping_mask
    random_grouping_mask(batch_size, num_subcarriers, factors, device)


def validate_config(config: dict, train_data_len: int) -> None:
    assumptions = config["implementation_assumption"]
    paper = config["paper_specified"]
    if train_data_len != 100000:
        raise RuntimeError(f"expected 100,000 training samples, found {train_data_len}")
    if int(assumptions["batch_size"]) != 8 or float(paper["learning_rate"]) != 1e-4 or float(paper["lambda_reg"]) != 1e-3:
        raise RuntimeError("canonical batch size, learning rate, or lambda_reg changed")
    if assumptions["mask_grouping_factors"] != [4, 8, 16, 32] or assumptions["device"] != "cuda:0":
        raise RuntimeError("canonical Ng factors or device changed")


def set_seeds(seed: int) -> None:
    import numpy as np
    torch.manual_seed(seed)
    import random as global_random
    global_random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.manual_seed_all(seed)


def train_balanced_epoch(model, loader, optimizer, config, device, schedule, epoch, exposure_path):
    from scripts.train_predictor import _build_observation, _loss_modes
    from src.models.evidential import evidential_loss
    from src.training.metrics import nmse_all_db, nmse_omitted_db

    model.train()
    totals = {"total": 0.0, "nll": 0.0, "reg": 0.0, "lambda_reg_x_reg": 0.0, "nmse_all_db": 0.0, "nmse_omitted_db": 0.0, "measured_snr_db": 0.0}
    count = 0
    schedule_index = 0
    nll_mode, reg_mode = _loss_modes(config)
    with exposure_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if exposure_path.stat().st_size == 0:
            writer.writerow(["epoch", "sample_index", "ng", "offset"])
        for batch in loader:
            cfr = batch["cfr"].to(device)
            assignments = schedule[schedule_index:schedule_index + cfr.shape[0]]
            schedule_index += cfr.shape[0]
            factors = list(config["implementation_assumption"]["mask_grouping_factors"])
            consume_legacy_mask_rng(cfr.shape[0], int(config["paper_specified"]["num_subcarriers"]), factors, device)
            mask = make_balanced_mask(assignments, int(config["paper_specified"]["num_subcarriers"]), device)
            writer.writerows([[epoch, index, factor, offset] for index, (factor, offset) in enumerate(assignments, schedule_index - len(assignments))])
            x, target, snr_stats = _build_observation(cfr, mask, config)
            output = model(x)
            losses = evidential_loss(output, target, float(config["paper_specified"]["lambda_reg"]), nll_mode=nll_mode, reg_mode=reg_mode)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = cfr.shape[0]
            count += batch_size
            for key in ("total", "nll", "reg", "lambda_reg_x_reg"):
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            totals["nmse_all_db"] += float(nmse_all_db(output.predicted.detach(), target).cpu()) * batch_size
            totals["nmse_omitted_db"] += float(nmse_omitted_db(output.predicted.detach(), target, mask).cpu()) * batch_size
            if snr_stats["measured_snr_db_mean"] is not None:
                totals["measured_snr_db"] += float(snr_stats["measured_snr_db_mean"]) * batch_size
    if schedule_index != len(schedule):
        raise RuntimeError(f"schedule consumption mismatch: {schedule_index} != {len(schedule)}")
    metrics = {key: value / count for key, value in totals.items() if key != "measured_snr_db"}
    metrics["measured_snr_db_mean"] = totals["measured_snr_db"] / count
    metrics["reg_to_nll_ratio"] = abs(metrics["lambda_reg_x_reg"]) / (abs(metrics["nll"]) + 1e-8)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--schedule-seed", type=int, default=20260919)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import evaluate, load_config, set_seeds as canonical_set_seeds
    from src.training.data import CFRNPZDataset

    config = load_config(BASE_CONFIG)
    config["implementation_assumption"]["epochs"] = 3
    config["implementation_assumption"]["experiment_name"] = "BALANCED-MASK-EXPOSURE-100k-3ep-seed-20260819"
    config["implementation_assumption"]["balanced_schedule_seed"] = args.schedule_seed
    config["implementation_assumption"]["rng_alignment"] = "Consume and discard canonical random_grouping_mask CUDA draws before using deterministic balanced assignments. Exact full stochastic equivalence is not guaranteed beyond those draw calls."
    train_data = CFRNPZDataset(ROOT / config["data"]["train_path"])
    validate_config(config, len(train_data))
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10 / cuda:0 is required")
    factors = list(config["implementation_assumption"]["mask_grouping_factors"])
    sample_count = args.smoke_samples if args.smoke_samples else len(train_data)
    if sample_count > len(train_data):
        raise ValueError("smoke sample count exceeds training dataset")
    if sample_count % int(config["implementation_assumption"]["batch_size"]) != 0:
        raise ValueError("sample count must be divisible by batch size")
    set_seeds(int(config["implementation_assumption"]["seed"]))
    train_loader = DataLoader(torch.utils.data.Subset(train_data, range(sample_count)), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(int(config["implementation_assumption"]["seed"])), num_workers=0)
    schedule = build_balanced_schedule(sample_count, factors, args.schedule_seed)
    counts = schedule_counts(schedule, factors)
    write_json(output / "config_used.json", config)
    write_json(output / "rng_handling.json", {"legacy_mask_rng": "random_grouping_mask called and discarded per batch", "balanced_schedule_seed": args.schedule_seed, "exact_alignment": False, "limitation": config["implementation_assumption"]["rng_alignment"]})
    write_json(output / "exposure_expected.json", counts)
    canonical_set_seeds(int(config["implementation_assumption"]["seed"]))
    from scripts.train_100k5_convergence import replay_last_probe_rng
    from scripts.generate_dataset import load_dataset_config
    replay_last_probe_rng(200, 1_000_000.0, int(config["implementation_assumption"]["seed"]), load_dataset_config(ROOT / "configs/dataset_prototype.json"))
    model = _make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    val_data = CFRNPZDataset(ROOT / config["data"]["validation_path"])
    val_loader = DataLoader(val_data, batch_size=int(config["implementation_assumption"]["eval_batch_size"]), num_workers=0)
    if args.smoke_samples:
        epoch_schedule = schedule
        metrics = train_balanced_epoch(model, train_loader, optimizer, config, device, epoch_schedule, 1, output / "exposure_epoch1.csv")
        finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters()) and all(torch.isfinite(torch.tensor(value)).item() for value in metrics.values() if value is not None)
        write_json(output / "smoke_results.json", {"samples": sample_count, "counts": counts, "train_metrics": metrics, "finite": finite, "gpu": torch.cuda.get_device_name(0), "device": str(device), "output_dir": str(output.relative_to(ROOT)), "checkpoint_written": False})
        if not finite:
            raise RuntimeError("smoke training produced non-finite values")
        print(json.dumps({"smoke_complete": True, "counts": counts, "gpu": torch.cuda.get_device_name(0), "finite": finite}, sort_keys=True), flush=True)
        return
    started = time.perf_counter()
    history = []
    for epoch in range(1, 4):
        epoch_start = time.perf_counter()
        schedule = build_balanced_schedule(len(train_data), factors, args.schedule_seed + epoch - 1)
        epoch_counts = schedule_counts(schedule, factors)
        metrics = train_balanced_epoch(model, train_loader, optimizer, config, device, schedule, epoch, output / f"exposure_epoch{epoch}.csv")
        validation = evaluate(model, val_loader, config, device)
        checkpoint = output / f"uacp_predictor_balanced_100k_3ep_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint)
        row = {"epoch": epoch, "train": metrics, "validation": validation, "exposure": epoch_counts, "elapsed_epoch_s": time.perf_counter() - epoch_start, "checkpoint": str(checkpoint.relative_to(ROOT)), "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20, "rss_peak_gib": rss_gib(), "system_memory": system_memory()}
        history.append(row)
        write_json(output / "training_history.json", {"history": history, "elapsed_s": time.perf_counter() - started})
        print(json.dumps(row, sort_keys=True), flush=True)
    result = {"experiment": "balanced Ng/offset exposure 100k×3", "history": history, "elapsed_s": time.perf_counter() - started, "device": str(device), "gpu": torch.cuda.get_device_name(0), "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20, "canonical_epoch3_checkpoint_sha256": sha256_file(BASE_CHECKPOINT), "config": config}
    write_json(output / "training_results.json", result)
    print(json.dumps({"complete": True, "elapsed_s": result["elapsed_s"], "gpu": result["gpu"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
