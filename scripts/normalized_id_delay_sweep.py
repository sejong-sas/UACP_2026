#!/usr/bin/env python3
"""ID delay spread에 따른 normalized uncertainty를 평가한다.

실험 목적:
    10~100 ns training range 안에서 delay spread가 커질 때 NMSE와
    Aleatoric/Epistemic이 어떤 방향으로 변하는지 확인한다.

주의:
    모델을 재학습하지 않고, covariance는 variance scale에 맞춰 정규화한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402
from src.training.uncertainty import paper_subcarrier_uncertainty_map  # noqa: E402

DELAYS = [10, 20, 40, 60, 80, 100]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def eval_delay(model, path: Path, cfg: dict, device: torch.device, seed: int) -> dict:
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    nmse, ale, epi, power = [], [], [], []
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device); mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device); set_seeds(seed + batch_index)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0); output = model(x); omitted = 1.0 - mask
        error = (output.predicted - target).square(); om = omitted[:, None, :]
        sample_nmse = 10 * torch.log10(((error * om).sum((1, 2)) / (target.square() * om).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        ale_map = paper_subcarrier_uncertainty_map(output.aleatoric); epi_map = paper_subcarrier_uncertainty_map(output.epistemic)
        sample_ale = (ale_map * omitted).sum(1) / omitted.sum(1).clamp_min(1); sample_epi = (epi_map * omitted).sum(1) / omitted.sum(1).clamp_min(1)
        scale = cfr.abs().square().mean((1, 2, 3)).clamp_min(1e-12)
        norm_target = target / torch.sqrt(scale)[:, None, None]; norm_pred = output.predicted / torch.sqrt(scale)[:, None, None]
        norm_error = (norm_pred - norm_target).square()
        norm_nmse = 10 * torch.log10(((norm_error * om).sum((1, 2)) / (norm_target.square() * om).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        nmse.extend(sample_nmse.cpu().tolist()); ale.extend((sample_ale / scale).cpu().tolist()); epi.extend((sample_epi / scale).cpu().tolist()); power.extend(scale.cpu().tolist())
    def stats(values: list[float]) -> tuple[float, float, float]:
        array = np.asarray(values, dtype=float); return float(array.mean()), float(array.std()), float(1.96 * array.std() / math.sqrt(len(array)))
    raw_ale = np.asarray(ale) * np.asarray(power); raw_epi = np.asarray(epi) * np.asarray(power)
    norm_nmse_values = np.asarray(nmse)
    # NMSE is scale invariant; retain the recomputed value as the primary normalized check.
    am, ass, aci = stats(ale); em, ess, eci = stats(epi); nm, nss, nci = stats(norm_nmse_values)
    return {"delay_spread_ns": float(path.stem.split("_")[1].replace("ns", "")), "samples": len(nmse), "raw_nmse_mean_db": float(np.mean(nmse)), "raw_nmse_std_db": float(np.std(nmse)), "normalized_nmse_mean_db": nm, "normalized_nmse_std_db": nss, "normalized_nmse_ci95_halfwidth_db": nci, "raw_aleatoric_mean": float(raw_ale.mean()), "normalized_aleatoric_mean": am, "normalized_aleatoric_std": ass, "normalized_aleatoric_ci95_halfwidth": aci, "raw_epistemic_mean": float(raw_epi.mean()), "normalized_epistemic_mean": em, "normalized_epistemic_std": ess, "normalized_epistemic_ci95_halfwidth": eci, "mean_cfr_power": float(np.mean(power))}


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["delay_spread_ns"] for r in rows]; fig, ax1 = plt.subplots(figsize=(8, 5)); ax2 = ax1.twinx()
    ax1.plot(x, [r["normalized_aleatoric_mean"] for r in rows], "o-", label="Normalized Aleatoric")
    ax1.plot(x, [r["normalized_epistemic_mean"] for r in rows], "s-", label="Normalized Epistemic")
    ax2.plot(x, [r["normalized_nmse_mean_db"] for r in rows], "^-", color="black", label="NMSE")
    ax1.set_xlabel("RMS delay spread (ns)"); ax1.set_ylabel("Normalized uncertainty"); ax2.set_ylabel("NMSE (dB)"); ax1.grid(alpha=.25)
    l1, t1 = ax1.get_legend_handles_labels(); l2, t2 = ax2.get_legend_handles_labels(); ax1.legend(l1+l2, t1+t2); fig.tight_layout(); fig.savefig(out / "normalized_fig7_delay_sweep.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json"); parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); parser.add_argument("--probe-dir", default="runs/paper_style_uncertainty_diagnostics_20260904/generated_p3"); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    cfg = load_config(args.config); set_seeds(int(cfg["implementation_assumption"]["seed"])); device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu"); model = _make_model(cfg, device)
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload); model.eval()
    rows = [eval_delay(model, ROOT / args.probe_dir / f"test_{delay}ns.npz", cfg, device, 99000 + i * 1000) for i, delay in enumerate(DELAYS)]
    out.mkdir(parents=True, exist_ok=True); write_csv(out / "normalized_delay_sweep.csv", rows); plot(rows, out)
    prov = {"checkpoint": args.checkpoint, "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "training_performed": False, "normalization": "a=sqrt(mean(|H_true|^2)); H and gamma divided by a; Sigma_ale and Sigma_epi divided by a^2; Eq.12->Eq.13 reapplied", "probe_dir": args.probe_dir}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True)); (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True)); (out / "results.json").write_text(json.dumps({"rows": rows, "provenance": prov}, indent=2, sort_keys=True)); print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": prov["gpu"], "rows": rows}, indent=2), flush=True)


if __name__ == "__main__": main()
