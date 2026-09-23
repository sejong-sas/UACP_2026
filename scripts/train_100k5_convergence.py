#!/usr/bin/env python3
"""Train a fresh 100k×5 convergence ablation using the canonical CFR archive."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
BASE_HISTORY = BASE_DIR / "training_history.json"
FAR_PROBE = BASE_DIR / "generated_data/test_ood_far_1ms.npz"
REGIMES = [("ID-Easy 20 ns", "data/prototype/test_id_easy.npz", 20.0),
           ("ID-Hard 80 ns", "data/prototype/test_id_hard.npz", 80.0),
           ("OOD-Near 120 ns", "data/prototype/test_ood_near.npz", 120.0),
           ("OOD-Far 1 ms", str(FAR_PROBE.relative_to(ROOT)), 1_000_000.0)]


def assert_epoch_only_change(base: dict, candidate: dict) -> None:
    """Ensure model/training/evaluation conditions match aside from epoch metadata."""
    reference = copy.deepcopy(base)
    current = copy.deepcopy(candidate)
    for cfg in (reference, current):
        assumption = cfg["implementation_assumption"]
        assumption.pop("epochs", None)
        assumption.pop("pilot_epochs", None)
        assumption.pop("experiment_name", None)
    if reference != current:
        raise ValueError("100k×5 config changes a condition other than epoch count/run label")
    if int(base["implementation_assumption"]["epochs"]) != 1 or int(candidate["implementation_assumption"]["epochs"]) != 5:
        raise ValueError("expected canonical 1 epoch and candidate 5 epochs")


def replay_last_probe_rng(samples: int, delay_spread_ns: float, baseline_seed: int,
                          dataset_config: dict, generate=None) -> None:
    """Restore the RNG state left by Step-4A's final generated OOD probe CFR.

    In the original run, `_step4_probe_paths` generated 200 1-ms CFR samples
    immediately before model initialization. The channel generator resets and
    advances backend RNG per sample, so replaying only the final sample has the
    same post-call RNG state while leaving all existing data artifacts intact.
    """
    if samples <= 0:
        raise ValueError("probe sample count must be positive")
    if generate is None:
        from src.channel.sionna_channel import generate_cfr_for_delay_spreads
        generate = generate_cfr_for_delay_spreads
    generation_seed = int(baseline_seed) + 94000 + int(samples) - 1
    generate(dataset_config, [float(delay_spread_ns)], generation_seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2


def system_memory() -> dict:
    values = {}
    with Path("/proc/meminfo").open() as f:
        for line in f:
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) / 1024**2
    return {"total_gib": values.get("MemTotal"), "available_gib": values.get("MemAvailable")}


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ProgressIterable:
    """Transparent DataLoader wrapper that logs progress without changing RNG use."""

    def __init__(self, loader, epoch: int, every: int, progress_path: Path, device: torch.device):
        self.loader, self.epoch, self.every = loader, epoch, every
        self.progress_path, self.device = progress_path, device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        started = time.perf_counter()
        for index, batch in enumerate(self.loader, 1):
            yield batch
            if index % self.every == 0 or index == len(self.loader):
                row = {"epoch": self.epoch, "batches_done": index, "batches_total": len(self.loader),
                       "elapsed_s": time.perf_counter() - started, "rss_peak_gib": rss_gib(),
                       "gpu_allocated_mib": torch.cuda.memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None,
                       "system_memory": system_memory()}
                with self.progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({"progress": row}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every-batches", type=int, default=250)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.baseline_reproduction import epoch_probe_row
    from scripts.diagnose_predictor import _make_model, diagnose_regime
    from scripts.train_predictor import evaluate, load_config, set_seeds, train_one_epoch, write_training_csvs
    from src.training.data import CFRNPZDataset

    if not all(p.is_file() for p in (BASE_CONFIG, BASE_CHECKPOINT, BASE_HISTORY, FAR_PROBE)):
        raise FileNotFoundError("canonical 100k×1 training inputs/checkpoint/history missing")
    config = load_config(BASE_CONFIG)
    config["implementation_assumption"]["epochs"] = 5
    config["implementation_assumption"]["pilot_epochs"] = 5
    config["implementation_assumption"]["experiment_name"] = "PAPER-ALIGNED-SAMPLE-COUNT-100k-5ep-seed-20260819"
    base_config = load_config(BASE_CONFIG)
    assert_epoch_only_change(base_config, config)
    train_path = ROOT / config["data"]["train_path"]
    if not train_path.is_file():
        raise FileNotFoundError(f"existing training CFR archive missing: {train_path}")
    if int(config["implementation_assumption"]["batch_size"]) != 8:
        raise ValueError("expected canonical batch size 8")
    if float(config["paper_specified"]["learning_rate"]) != 1e-4 or float(config["paper_specified"]["lambda_reg"]) != 1e-3:
        raise ValueError("canonical learning rate/lambda changed")
    memory_before = system_memory()
    if memory_before.get("available_gib", 0) < 16:
        raise RuntimeError(f"insufficient available system memory before training: {memory_before}")

    seed = int(config["implementation_assumption"]["seed"])
    set_seeds(seed)
    device = torch.device(config["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("conditional 100k×5 run requires cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "GB10" not in gpu_name:
        raise RuntimeError(f"expected NVIDIA GB10, found {gpu_name}")
    torch.cuda.reset_peak_memory_stats(device)

    train_data = CFRNPZDataset(train_path)
    if len(train_data) != 100000:
        raise RuntimeError(f"expected 100,000 training CFRs, found {len(train_data)}")
    val_data = CFRNPZDataset(config["data"]["validation_path"])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=int(config["implementation_assumption"]["eval_batch_size"]), num_workers=0)
    from scripts.generate_dataset import load_dataset_config
    dataset_config = load_dataset_config(ROOT / "configs/dataset_prototype.json")
    replay_last_probe_rng(samples=int(config["implementation_assumption"]["probe_samples"]),
                          delay_spread_ns=1_000_000.0, baseline_seed=seed,
                          dataset_config=dataset_config)
    model = _make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["paper_specified"]["learning_rate"]))
    baseline_history = json.loads(BASE_HISTORY.read_text(encoding="utf-8"))["history"][0]["train"]
    before = {"checkpoint_sha256": sha256_file(BASE_CHECKPOINT), "checkpoint_size_bytes": BASE_CHECKPOINT.stat().st_size,
              "train_archive": str(train_path.relative_to(ROOT)), "train_archive_size_bytes": train_path.stat().st_size,
              "n_train": len(train_data), "device": str(device), "gpu": gpu_name,
              "system_memory": memory_before, "rss_peak_gib": rss_gib(),
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
    write_json(out / "config_used.json", config)
    write_json(out / "run_start.json", before)
    history, probes = [], []
    probe_results = {}
    started = time.perf_counter()
    for epoch in range(1, 6):
        epoch_started = time.perf_counter()
        instrumented = ProgressIterable(train_loader, epoch, args.progress_every_batches, out / "batch_progress.jsonl", device)
        train_metrics = train_one_epoch(model, instrumented, optimizer, config, device)
        if epoch == 1:
            mismatches = {key: {"expected": float(baseline_history[key]), "actual": float(train_metrics[key])}
                          for key in ("total", "nll", "reg", "lambda_reg_x_reg", "nmse_all_db", "nmse_omitted_db")
                          if not np.isclose(float(baseline_history[key]), float(train_metrics[key]), rtol=1e-5, atol=1e-3)}
            if mismatches:
                torch.save(model.state_dict(), out / "checkpoint_epoch_1_alignment_failed.pt")
                write_json(out / "epoch1_alignment_failure.json", {"mismatches": mismatches, "training_metrics": train_metrics})
                raise RuntimeError(f"100k×5 epoch-1 trajectory failed to match 100k×1 baseline: {mismatches}")
        validation_metrics = evaluate(model, val_loader, config, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        epoch_result = {}
        for regime, rel_path, delay in REGIMES:
            metrics = diagnose_regime(model, ROOT / rel_path, config, device, observation_noise_snr_db=15.0)
            epoch_result[regime] = metrics
            probes.append(epoch_probe_row(epoch, regime, delay, metrics))
        probe_results[str(epoch)] = epoch_result
        checkpoint_path = out / f"uacp_predictor_100k_5ep_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        progress = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics,
                    "elapsed_epoch_s": time.perf_counter() - epoch_started,
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)), "rss_peak_gib": rss_gib(),
                    "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                    "system_memory": system_memory()}
        with (out / "epoch_progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(progress) + "\n")
        print(json.dumps(progress, sort_keys=True), flush=True)
        write_json(out / "training_history.json", {"history": history, "epoch_probe_results": probe_results,
                                                     "elapsed_s": time.perf_counter() - started})
    final_checkpoint = out / "uacp_predictor_100k_5ep.pt"
    torch.save(model.state_dict(), final_checkpoint)
    write_training_csvs(out, history, {regime: {"nmse_all_db": metrics["nmse_all_db"],
                                              "nmse_omitted_db": metrics["nmse_omitted_db"],
                                              "aleatoric": metrics["aleatoric_omitted"]["mean"],
                                              "epistemic": metrics["epistemic_omitted"]["mean"],
                                              "nll": metrics["nll"], "reg": metrics["raw_l_reg"],
                                              "lambda_reg_x_reg": metrics["lambda_reg_x_l_reg"]}
                                    for regime, metrics in probe_results["5"].items()})
    result = {"experiment": "100k×5 training convergence ablation", "no_full_psi": True,
              "config": config, "training_history": history, "epoch_probe_results": probe_results,
              "checkpoint": str(final_checkpoint.relative_to(ROOT)), "device": str(device), "gpu": gpu_name,
              "elapsed_s": time.perf_counter() - started, "peak_rss_gib": rss_gib(),
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
              "start_system_memory": memory_before, "end_system_memory": system_memory(),
              "canonical_checkpoint_sha256_unchanged": sha256_file(BASE_CHECKPOINT) == before["checkpoint_sha256"],
              "conditions_unchanged_except_epoch_count": True,
              "epoch1_metrics_match_100k1": True,
              "training_data_reused": str(train_path.relative_to(ROOT)), "new_data_generation": False}
    write_json(out / "training_results.json", result)
    print(json.dumps({"complete": True, "checkpoint": result["checkpoint"], "elapsed_s": result["elapsed_s"],
                      "peak_rss_gib": result["peak_rss_gib"], "gpu": gpu_name}, indent=2), flush=True)


if __name__ == "__main__":
    main()
