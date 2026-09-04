#!/usr/bin/env python3
"""Paired control/treatment diagnostics for direct versus pilot-LS observation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model, diagnose_regime  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.training.data import (  # noqa: E402
    CFRNPZDataset,
    build_noisy_sparse_input,
    build_pilot_ls_observation,
    build_pilot_sparse_input,
    build_sparse_input,
    cfr_to_real_imag,
    linear_interpolate_real_imag,
    uniform_grouping_mask,
)
from src.training.metrics import nmse_all_db, nmse_omitted_db  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def complex_nmse_db(error: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        mask = mask[:, :, None, None].bool().expand_as(error.real)
        error = error[mask]
        target = target[mask]
    return float((10.0 * torch.log10((error.abs().square().sum() / target.abs().square().sum().clamp_min(1e-12)).clamp_min(1e-12))).cpu())


def channels_to_complex_cfr(channels: torch.Tensor) -> torch.Tensor:
    real = channels[:, :4].permute(0, 2, 1)
    imag = channels[:, 4:].permute(0, 2, 1)
    return torch.complex(real, imag).reshape(channels.shape[0], channels.shape[2], 2, 2)


@torch.no_grad()
def observation_quality(cfg: dict, device: torch.device) -> tuple[list[dict], list[dict]]:
    quality, interpolation = [], []
    for index, (label, delay) in enumerate(REGIMES):
        dataset = CFRNPZDataset(cfg["data"]["test_paths"][label])
        loader = DataLoader(dataset, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
        sums = {"samples": 0, "direct_nmse_db": 0.0, "pilot_nmse_db": 0.0, "direct_snr_db": 0.0, "pilot_snr_db": 0.0, "interp_direct_all_db": 0.0, "interp_direct_omitted_db": 0.0, "interp_pilot_all_db": 0.0, "interp_pilot_omitted_db": 0.0}
        for batch in loader:
            cfr = batch["cfr"].to(device)
            mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
            set_seeds(20260819 + 94000 + index + sums["samples"])
            direct, _, direct_stats = build_noisy_sparse_input(cfr, mask, 15.0)
            set_seeds(20260819 + 94000 + index + sums["samples"])
            pilot_cfr, pilot_stats = build_pilot_ls_observation(cfr, mask, 15.0)
            direct_interp = linear_interpolate_real_imag(direct[:, :8], mask)
            pilot_x, _ = build_sparse_input(pilot_cfr, mask)
            pilot_interp = linear_interpolate_real_imag(pilot_x[:, :8], mask)
            n = cfr.shape[0]
            sums["samples"] += n
            direct_cfr = channels_to_complex_cfr(direct[:, :8])
            sums["direct_nmse_db"] += complex_nmse_db(direct_cfr - cfr, cfr, mask) * n
            sums["pilot_nmse_db"] += complex_nmse_db(pilot_cfr - cfr, cfr, mask) * n
            sums["direct_snr_db"] += float(direct_stats["measured_snr_db_mean"]) * n
            sums["pilot_snr_db"] += float(pilot_stats["measured_snr_db_mean"]) * n
            target = cfr_to_real_imag(cfr)
            sums["interp_direct_all_db"] += float(nmse_all_db(direct_interp, target).cpu()) * n
            sums["interp_direct_omitted_db"] += float(nmse_omitted_db(direct_interp, target, mask).cpu()) * n
            sums["interp_pilot_all_db"] += float(nmse_all_db(pilot_interp, target).cpu()) * n
            sums["interp_pilot_omitted_db"] += float(nmse_omitted_db(pilot_interp, target, mask).cpu()) * n
        n = sums["samples"]
        quality.append({"regime": label, "delay_spread_ns": delay, "samples": n, "direct_observation_nmse_omitted_db": sums["direct_nmse_db"] / n, "pilot_ls_observation_nmse_omitted_db": sums["pilot_nmse_db"] / n, "direct_measured_snr_db": sums["direct_snr_db"] / n, "pilot_ls_measured_snr_db": sums["pilot_snr_db"] / n})
        interpolation.append({"regime": label, "delay_spread_ns": delay, "direct_linear_nmse_all_db": sums["interp_direct_all_db"] / n, "direct_linear_nmse_omitted_db": sums["interp_direct_omitted_db"] / n, "pilot_ls_linear_nmse_all_db": sums["interp_pilot_all_db"] / n, "pilot_ls_linear_nmse_omitted_db": sums["interp_pilot_omitted_db"] / n})
    return quality, interpolation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default="configs/current_valid_baseline_condition_c.json")
    parser.add_argument("--treatment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    baseline_cfg = load_config(args.baseline_config)
    treatment_dir = ROOT / args.treatment_dir
    treatment_cfg = load_config(treatment_dir / "config.json")
    set_seeds(20260819)
    device = torch.device(baseline_cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    quality, interpolation = observation_quality(baseline_cfg, device)
    models = []
    for name, cfg, checkpoint, mode in [
        ("Direct-CFR-AWGN", baseline_cfg, ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt", "direct_cfr_awgn"),
        ("Pilot-LS", treatment_cfg, treatment_dir / "checkpoint_with_provenance.pt", "pilot_ls"),
    ]:
        model = _make_model(cfg, device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state)
        model.eval()
        for index, (label, delay) in enumerate(REGIMES):
            set_seeds(20260819 + 95000 + index)
            metrics = diagnose_regime(model, cfg["data"]["test_paths"][label], cfg, device, observation_noise_snr_db=15.0, observation_mode=mode)
            models.append({"model": name, "regime": label, "delay_spread_ns": delay, "nmse_all_db": metrics["nmse_all_db"], "nmse_omitted_db": metrics["nmse_omitted_db"], "aleatoric": metrics["aleatoric_paper_eq12_eq13"], "epistemic": metrics["epistemic_paper_eq12_eq13"], "psi_mean": metrics["psi_omitted"]["mean"], "kappa_mean": metrics["kappa_omitted"]["mean"], "nu_mean": metrics["nu_omitted"]["mean"], "denom_mean": metrics["df_cov_omitted"]["mean"], "error_aleatoric_pearson": metrics["error_aleatoric_pearson"], "error_epistemic_pearson": metrics["error_epistemic_pearson"], "measured_snr_db": metrics["measured_snr_db_mean"]})
    by = {(row["model"], row["regime"]): row for row in models}
    def row(model: str, label: str) -> dict: return by[(model, label)]
    id_labels = [REGIMES[0][0], REGIMES[1][0]]
    gaps = []
    for model in ("Direct-CFR-AWGN", "Pilot-LS"):
        a20, a80 = row(model, id_labels[0])["aleatoric"], row(model, id_labels[1])["aleatoric"]
        epi_id = max(row(model, x)["epistemic"] for x in id_labels)
        gaps.append({"model": model, "ale_gap_80_20": a80 - a20, "epi_gap_120_id": row(model, "OOD-Near 120 ns")["epistemic"] - epi_id, "epi_gap_1ms_id": row(model, "OOD-Far 1 ms")["epistemic"] - epi_id, "nmse_ordering": row(model, id_labels[0])["nmse_omitted_db"] < row(model, id_labels[1])["nmse_omitted_db"] < row(model, "OOD-Near 120 ns")["nmse_omitted_db"]})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "observation_quality.csv", quality)
    write_csv(out / "linear_reconstruction.csv", interpolation)
    write_csv(out / "model_comparison.csv", models)
    write_csv(out / "gap_summary.csv", gaps)
    (out / "results.json").write_text(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "observation_quality": quality, "linear_reconstruction": interpolation, "model_comparison": models, "gaps": gaps}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "gaps": gaps}, indent=2), flush=True)


if __name__ == "__main__":
    main()
