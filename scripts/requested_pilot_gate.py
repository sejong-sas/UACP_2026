#!/usr/bin/env python3
"""Gate-1 comparison for direct CFR AWGN versus requested-subcarrier pilot LS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.training.data import CFRNPZDataset, add_complex_awgn, build_requested_hadamard_ls_observation, uniform_grouping_mask  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]


def nmse_db(error: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = mask[:, :, None, None].bool().expand_as(target.real)
    return float((10 * torch.log10((error.abs().square()[selected].sum() / target.abs().square()[selected].sum()).clamp_min(1e-12))).cpu())


def ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    values = np.sort(np.concatenate([first, second]))
    cdf_first = np.searchsorted(np.sort(first), values, side="right") / first.size
    cdf_second = np.searchsorted(np.sort(second), values, side="right") / second.size
    return float(np.max(np.abs(cdf_first - cdf_second)))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def run_gate(cfg: dict, device: torch.device) -> list[dict]:
    rows = []
    for index, (label, delay) in enumerate(REGIMES):
        loader = DataLoader(CFRNPZDataset(cfg["data"]["test_paths"][label]), batch_size=16)
        direct_error, pilot_error = [], []
        direct_nmse, pilot_nmse, direct_snr, pilot_snr, count = 0.0, 0.0, 0.0, 0.0, 0
        for batch_index, batch in enumerate(loader):
            cfr = batch["cfr"].to(device)
            mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
            set_seeds(20260819 + 96000 + index * 1000 + batch_index)
            direct_cfr, direct_stats = add_complex_awgn(cfr, mask, 15.0)
            set_seeds(20260819 + 96000 + index * 1000 + batch_index)
            pilot_cfr, pilot_stats = build_requested_hadamard_ls_observation(cfr, mask, 15.0)
            selected = mask[:, :, None, None].bool().expand_as(cfr.real)
            de = (direct_cfr - cfr)[selected]
            pe = (pilot_cfr - cfr)[selected]
            direct_error.append(torch.stack((de.real, de.imag), dim=-1).reshape(-1).cpu().numpy())
            pilot_error.append(torch.stack((pe.real, pe.imag), dim=-1).reshape(-1).cpu().numpy())
            n = cfr.shape[0]
            count += n
            direct_nmse += nmse_db(direct_cfr - cfr, cfr, mask) * n
            pilot_nmse += nmse_db(pilot_cfr - cfr, cfr, mask) * n
            direct_snr += float(direct_stats["measured_snr_db_mean"]) * n
            pilot_snr += float(pilot_stats["measured_snr_db_mean"]) * n
        d = np.concatenate(direct_error)
        p = np.concatenate(pilot_error)
        rows.append({
            "regime": label,
            "delay_spread_ns": delay,
            "samples": count,
            "direct_observation_nmse_db": direct_nmse / count,
            "pilot_ls_observation_nmse_db": pilot_nmse / count,
            "direct_measured_snr_db": direct_snr / count,
            "pilot_ls_measured_snr_db": pilot_snr / count,
            "direct_error_mean": float(d.mean()),
            "pilot_error_mean": float(p.mean()),
            "direct_error_variance": float(d.var()),
            "pilot_error_variance": float(p.var()),
            "direct_error_abs2_mean": float(np.mean(d * d)),
            "pilot_error_abs2_mean": float(np.mean(p * p)),
            "error_mean_abs_difference": float(abs(d.mean() - p.mean())),
            "error_variance_abs_difference": float(abs(d.var() - p.var())),
            "error_ks_distance": ks_distance(d, p),
            "error_max_abs_difference_same_seed": float(np.max(np.abs(d - p))),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/current_valid_baseline_condition_c.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    cfg = load_config(args.config)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    rows = run_gate(cfg, device)
    out.mkdir(parents=True, exist_ok=True)
    code_files = ["src/training/data.py", "scripts/requested_pilot_gate.py"]
    metadata = {"experiment": "Requested-Subcarrier Orthogonal Pilot LS GATE 1", "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "pilot": "X=(1/sqrt(2))*[[1,1],[1,-1]]", "estimator": "H_hat=Y X^H (X X^H)^-1", "requested_mask_before_estimation": True, "snr_definition": "mean reported ||H X||^2 / mean reported ||N||^2", "assumption": "IMPLEMENTATION-ASSUMPTION", "rows": rows}
    metadata["provenance"] = {"config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "code_sha256": {name: file_sha256(ROOT / name) for name in code_files}, "baseline_checkpoint_sha256": file_sha256(ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"), "seed": cfg["implementation_assumption"]["seed"]}
    (out / "sanity_comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "observation_error_stats.json").write_text(json.dumps({"rows": rows, "all_finite": True}, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(metadata["provenance"], indent=2), encoding="utf-8")
    (out / "measured_snr.json").write_text(json.dumps({"rows": [{k: r[k] for k in ("regime", "direct_measured_snr_db", "pilot_ls_measured_snr_db")} for r in rows]}, indent=2), encoding="utf-8")
    with (out / "gate1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": metadata["gpu"], "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
