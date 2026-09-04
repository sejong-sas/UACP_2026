#!/usr/bin/env python3
"""Diagnostic evaluation for separating dataset difficulty from evidence behavior."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_predictor import load_config, set_seeds
from src.models.evidential import evidential_loss, pair_vectors_to_channels
from src.models.uacp_predictor import UACPEvidentialPredictor
from src.training.data import (
    CFRNPZDataset,
    build_noisy_sparse_input,
    build_pilot_sparse_input,
    build_sparse_input,
    linear_interpolate_real_imag,
    uniform_grouping_mask,
)
from src.training.metrics import nmse_all_db, nmse_omitted_db
from src.training.uncertainty import paper_omitted_uncertainty_score


def _mean_std(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is not None:
        values = tensor[mask.expand_as(tensor) > 0.5]
    else:
        values = tensor.reshape(-1)
    return {
        "mean": float(values.mean().detach().cpu()),
        "std": float(values.std(unbiased=False).detach().cpu()),
    }


def _make_model(cfg: dict[str, Any], device: torch.device) -> UACPEvidentialPredictor:
    model = UACPEvidentialPredictor(
        input_channels=int(cfg["paper_specified"]["input_channels"]),
        output_channels=int(cfg["implementation_assumption"]["target_channels"]),
        hidden_channels=int(cfg["paper_specified"]["hidden_channels"]),
        residual_blocks=int(cfg["paper_specified"]["residual_blocks"]),
        kernel_size=int(cfg["paper_specified"]["kernel_size"]),
        dropout=float(cfg["paper_specified"]["dropout"]),
        num_subcarriers=int(cfg["paper_specified"]["num_subcarriers"]),
        evidential_mode=cfg["implementation_assumption"].get("evidential_mode", "elementwise"),
        covariance_rank=int(cfg["implementation_assumption"].get("covariance_rank", 0)),
        covariance_bandwidth=int(cfg["implementation_assumption"].get("covariance_bandwidth", 32)),
        covariance_factor_scale=float(cfg["implementation_assumption"].get("covariance_factor_scale", 0.05)),
    ).to(device)
    return model


def _loss_modes(cfg: dict[str, Any]) -> tuple[str, str]:
    assumptions = cfg["implementation_assumption"]
    return assumptions.get("nll_mode", "elementwise"), assumptions.get("reg_mode", "elementwise")


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x.square()) * torch.sum(y.square())).clamp_min(1e-12)
    return float((torch.sum(x * y) / denom).cpu())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x_rank = torch.argsort(torch.argsort(x.reshape(-1).float())).float()
    y_rank = torch.argsort(torch.argsort(y.reshape(-1).float())).float()
    return _pearson(x_rank, y_rank)


def _expand_to_omitted_shape(values: torch.Tensor, omitted_mask: torch.Tensor) -> torch.Tensor:
    if values.shape == omitted_mask.shape:
        return values
    if values.ndim == 3 and values.shape[1] == 8 and values.shape[2] == omitted_mask.shape[2]:
        return values
    if values.ndim == 3 and values.shape[1] == 4 and values.shape[2] == 1:
        expanded = values.expand(-1, -1, omitted_mask.shape[2])
        return torch.cat((expanded, expanded), dim=1)
    raise ValueError(f"Cannot compare values {tuple(values.shape)} with mask {tuple(omitted_mask.shape)}")


def _error_bin_calibration(error: torch.Tensor, aleatoric: torch.Tensor, epistemic: torch.Tensor) -> list[dict[str, float | int]]:
    """Summarize uncertainty by tertiles of omitted-subcarrier squared error."""
    error = error.reshape(-1).float()
    aleatoric = aleatoric.reshape(-1).float()
    epistemic = epistemic.reshape(-1).float()
    boundaries = torch.quantile(error, torch.tensor([1 / 3, 2 / 3], device=error.device))
    labels = (error > boundaries[0]).long() + (error > boundaries[1]).long()
    result = []
    for index, label in enumerate((0, 1, 2)):
        selected = labels == label
        result.append({
            "bin": index,
            "count": int(selected.sum().item()),
            "error_mean": float(error[selected].mean().cpu()),
            "aleatoric_mean": float(aleatoric[selected].mean().cpu()),
            "epistemic_mean": float(epistemic[selected].mean().cpu()),
        })
    return result


@torch.no_grad()
def diagnose_regime(model, path: str, cfg: dict[str, Any], device: torch.device, observation_noise_snr_db: float | None = None, observation_mode: str | None = None) -> dict[str, Any]:
    loader = DataLoader(
        CFRNPZDataset(path),
        batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]),
    )
    totals: dict[str, float] = {
        "samples": 0.0,
        "nmse_all_weighted": 0.0,
        "nmse_omitted_weighted": 0.0,
        "nll_weighted": 0.0,
        "raw_reg_weighted": 0.0,
        "lambda_reg_x_reg_weighted": 0.0,
        "measured_snr_weighted": 0.0,
        "measured_snr_sq_weighted": 0.0,
        "paper_aleatoric_weighted": 0.0,
        "paper_epistemic_weighted": 0.0,
    }
    stacked: dict[str, list[torch.Tensor]] = {
        "psi": [],
        "kappa": [],
        "nu": [],
        "df_cov": [],
        "aleatoric": [],
        "epistemic": [],
        "omitted": [],
    }
    lowrank_outputs: list[torch.Tensor] = []
    error_values = []
    aleatoric_values = []
    epistemic_values = []
    nll_mode, reg_mode = _loss_modes(cfg)

    for batch in loader:
        cfr = batch["cfr"].to(device)
        batch_size = cfr.shape[0]
        mask = uniform_grouping_mask(
            batch_size,
            int(cfg["paper_specified"]["num_subcarriers"]),
            int(cfg["implementation_assumption"]["eval_grouping_factor"]),
            device,
        )
        mode = observation_mode or cfg.get("implementation_assumption", {}).get("observation_mode", "direct_cfr_awgn")
        if mode == "pilot_ls":
            x, target, snr_stats = build_pilot_sparse_input(cfr, mask, observation_noise_snr_db)
        elif mode == "direct_cfr_awgn":
            x, target, snr_stats = build_noisy_sparse_input(cfr, mask, observation_noise_snr_db)
        else:
            raise ValueError(f"Unknown observation_mode: {mode}")
        output = model(x)
        losses = evidential_loss(output, target, float(cfg["paper_specified"]["lambda_reg"]), nll_mode=nll_mode, reg_mode=reg_mode)
        omitted = (1.0 - mask)[:, None, :]
        df_cov = output.nu_expanded - 2 * output.num_subcarriers - 1
        omitted_bool = omitted.expand_as(output.aleatoric) > 0.5
        error_values.append((output.predicted - target).square()[omitted_bool].detach().cpu())
        aleatoric_values.append(output.aleatoric[omitted_bool].detach().cpu())
        epistemic_values.append(output.epistemic[omitted_bool].detach().cpu())
        totals["paper_aleatoric_weighted"] += float(paper_omitted_uncertainty_score(output.aleatoric, mask).cpu()) * batch_size
        totals["paper_epistemic_weighted"] += float(paper_omitted_uncertainty_score(output.epistemic, mask).cpu()) * batch_size

        totals["samples"] += batch_size
        totals["nmse_all_weighted"] += float(nmse_all_db(output.predicted, target).cpu()) * batch_size
        totals["nmse_omitted_weighted"] += float(nmse_omitted_db(output.predicted, target, mask).cpu()) * batch_size
        totals["nll_weighted"] += float(losses["nll"].cpu()) * batch_size
        totals["raw_reg_weighted"] += float(losses["reg"].cpu()) * batch_size
        totals["lambda_reg_x_reg_weighted"] += float(losses["lambda_reg_x_reg"].cpu()) * batch_size
        if snr_stats["measured_snr_db_mean"] is not None:
            totals["measured_snr_weighted"] += float(snr_stats["measured_snr_db_mean"]) * batch_size
            mean = float(snr_stats["measured_snr_db_mean"])
            std = float(snr_stats["measured_snr_db_std"])
            totals["measured_snr_sq_weighted"] += (std * std + mean * mean) * batch_size

        stacked["psi"].append(output.psi.detach().cpu())
        stacked["kappa"].append(output.kappa.detach().cpu())
        stacked["nu"].append(output.nu.detach().cpu())
        stacked["df_cov"].append(df_cov.detach().cpu())
        stacked["aleatoric"].append(output.aleatoric.detach().cpu())
        stacked["epistemic"].append(output.epistemic.detach().cpu())
        stacked["omitted"].append(omitted.detach().cpu())
        if output.covariance_factor is not None:
            lowrank_outputs.append(output.covariance_factor.detach().cpu())

    samples = totals["samples"]
    metrics: dict[str, Any] = {
        "samples": int(samples),
        "nmse_all_db": totals["nmse_all_weighted"] / samples,
        "nmse_omitted_db": totals["nmse_omitted_weighted"] / samples,
        "nll": totals["nll_weighted"] / samples,
        "raw_l_reg": totals["raw_reg_weighted"] / samples,
        "lambda_reg_x_l_reg": totals["lambda_reg_x_reg_weighted"] / samples,
        "reg_to_nll_ratio": abs(totals["lambda_reg_x_reg_weighted"] / samples) / (abs(totals["nll_weighted"] / samples) + 1e-8),
        "aleatoric_paper_eq12_eq13": totals["paper_aleatoric_weighted"] / samples,
        "epistemic_paper_eq12_eq13": totals["paper_epistemic_weighted"] / samples,
        "requested_snr_db": observation_noise_snr_db,
        "measured_snr_db_mean": totals["measured_snr_weighted"] / samples if observation_noise_snr_db is not None else None,
        "measured_snr_db_std": (
            max(
                totals["measured_snr_sq_weighted"] / samples
                - (totals["measured_snr_weighted"] / samples) ** 2,
                0.0,
            ) ** 0.5
            if observation_noise_snr_db is not None
            else None
        ),
    }

    omitted_mask = torch.cat(stacked["omitted"], dim=0)
    for key in ("psi", "kappa", "nu", "df_cov", "aleatoric", "epistemic"):
        values = _expand_to_omitted_shape(torch.cat(stacked[key], dim=0), omitted_mask)
        metrics[f"{key}_omitted"] = _mean_std(values, omitted_mask)
        metrics[f"{key}_all"] = _mean_std(values)

    errors = torch.cat(error_values)
    aleatoric = torch.cat(aleatoric_values)
    epistemic = torch.cat(epistemic_values)
    metrics["error_aleatoric_pearson"] = _pearson(errors, aleatoric)
    metrics["error_aleatoric_spearman"] = _spearman(errors, aleatoric)
    metrics["error_epistemic_pearson"] = _pearson(errors, epistemic)
    metrics["error_epistemic_spearman"] = _spearman(errors, epistemic)
    metrics["error_bin_calibration"] = _error_bin_calibration(errors, aleatoric, epistemic)

    if lowrank_outputs:
        factor = torch.cat(lowrank_outputs, dim=0)
        diagonal = torch.cat(stacked["psi"], dim=0)
        batch_size, pairs, dimension, _ = factor.shape
        diagonal = diagonal.reshape(batch_size, 2, 4, -1).permute(0, 2, 1, 3).reshape(batch_size, 4, dimension)
        lowrank_diag = factor.square().sum(dim=-1)
        gram = torch.matmul(factor.transpose(-2, -1), factor)
        lowrank_fro_sq = gram.square().sum(dim=(-2, -1))
        offdiag_sq = lowrank_fro_sq - lowrank_diag.square().sum(dim=-1)
        full_fro_sq = diagonal.square().sum(dim=-1) + 2.0 * (diagonal * lowrank_diag).sum(dim=-1) + lowrank_fro_sq
        offdiag_ratio = offdiag_sq / full_fro_sq.clamp_min(1e-12)
        metrics["covariance_diag_omitted"] = _mean_std(
            output_values := torch.cat([pair_vectors_to_channels(diagonal[i:i+1] + lowrank_diag[i:i+1], output.num_subcarriers) for i in range(batch_size)], dim=0),
            omitted_mask,
        )
        metrics["lowrank_diag_contribution_omitted"] = _mean_std(
            torch.cat([pair_vectors_to_channels(lowrank_diag[i:i+1], output.num_subcarriers) for i in range(batch_size)], dim=0),
            omitted_mask,
        )
        metrics["off_diagonal_energy_ratio_mean"] = float(offdiag_ratio.mean())
        metrics["off_diagonal_energy_ratio_std"] = float(offdiag_ratio.std(unbiased=False))
        for lag in (1, 2, 4, 8):
            if lag < output.num_subcarriers:
                real = factor[:, :, :output.num_subcarriers, :]
                imag = factor[:, :, output.num_subcarriers:, :]
                lag_values = torch.cat(((real[:, :, :-lag] * real[:, :, lag:]).sum(-1).abs().reshape(-1), (imag[:, :, :-lag] * imag[:, :, lag:]).sum(-1).abs().reshape(-1)))
                metrics[f"frequency_covariance_lag_{lag}_mean"] = float(lag_values.mean())
                metrics[f"frequency_covariance_lag_{lag}_std"] = float(lag_values.std(unbiased=False))
    else:
        metrics["covariance_diag_omitted"] = None
        metrics["lowrank_diag_contribution_omitted"] = None
        metrics["off_diagonal_energy_ratio_mean"] = None
        metrics["off_diagonal_energy_ratio_std"] = None

    return metrics


@torch.no_grad()
def interpolation_baseline(path: str, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    loader = DataLoader(
        CFRNPZDataset(path),
        batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]),
    )
    total = 0.0
    weighted_nmse = 0.0
    for batch in loader:
        cfr = batch["cfr"].to(device)
        batch_size = cfr.shape[0]
        mask = uniform_grouping_mask(
            batch_size,
            int(cfg["paper_specified"]["num_subcarriers"]),
            int(cfg["implementation_assumption"]["eval_grouping_factor"]),
            device,
        )
        x, target = build_sparse_input(cfr, mask)
        interpolated = linear_interpolate_real_imag(x[:, :8], mask)
        total += batch_size
        weighted_nmse += float(nmse_omitted_db(interpolated, target, mask).cpu()) * batch_size
    return {"samples": int(total), "nmse_omitted_db": weighted_nmse / total}


def training_loss_scale(training_results_path: str | Path) -> list[dict[str, Any]]:
    results = json.loads(Path(training_results_path).read_text(encoding="utf-8"))
    lambda_reg = float(results["config"]["paper_specified"]["lambda_reg"])
    rows = []
    for epoch in results["history"]:
        row = {"epoch": epoch["epoch"]}
        for split in ("train", "validation"):
            row[f"{split}_nll"] = epoch[split]["nll"]
            row[f"{split}_raw_l_reg"] = epoch[split]["reg"]
            row[f"{split}_lambda_reg_x_l_reg"] = epoch[split].get("lambda_reg_x_reg", lambda_reg * epoch[split]["reg"])
            row[f"{split}_total"] = epoch[split]["total"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_diagnostic_row(regime: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "regime": regime,
        "samples": metrics["samples"],
        "nmse_all_db": metrics["nmse_all_db"],
        "nmse_omitted_db": metrics["nmse_omitted_db"],
        "nll": metrics["nll"],
        "raw_l_reg": metrics["raw_l_reg"],
        "lambda_reg_x_l_reg": metrics["lambda_reg_x_l_reg"],
        "error_aleatoric_pearson": metrics["error_aleatoric_pearson"],
        "error_aleatoric_spearman": metrics["error_aleatoric_spearman"],
        "error_epistemic_pearson": metrics["error_epistemic_pearson"],
        "error_epistemic_spearman": metrics["error_epistemic_spearman"],
    }
    for key in ("psi", "kappa", "nu", "df_cov", "aleatoric", "epistemic"):
        for scope in ("omitted", "all"):
            stats = metrics[f"{key}_{scope}"]
            row[f"{key}_{scope}_mean"] = stats["mean"]
            row[f"{key}_{scope}_std"] = stats["std"]
    return row


def run(config_path: str | Path, checkpoint_path: str | Path, training_results_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    diagnostics = {}
    interpolation = {}
    for regime, path in cfg["data"]["test_paths"].items():
        diagnostics[regime] = diagnose_regime(model, path, cfg, device)
        interpolation[regime] = interpolation_baseline(path, cfg, device)

    loss_scale = training_loss_scale(training_results_path)
    output = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "diagnostics": diagnostics,
        "linear_interpolation_baseline": interpolation,
        "training_loss_scale": loss_scale,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "parameter_diagnostics.csv", [flatten_diagnostic_row(k, v) for k, v in diagnostics.items()])
    write_csv(
        out_dir / "linear_interpolation_baseline.csv",
        [{"regime": k, **v} for k, v in interpolation.items()],
    )
    write_csv(out_dir / "training_loss_scale.csv", loss_scale)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_prototype.json")
    parser.add_argument("--checkpoint", default="runs/prototype_predictor/uacp_predictor_prototype.pt")
    parser.add_argument("--training-results", default="runs/prototype_predictor/training_results.json")
    parser.add_argument("--output-dir", default="runs/prototype_diagnostics")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.checkpoint, args.training_results, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
