#!/usr/bin/env python3
"""Train the prototype UACP evidential predictor on the small CFR dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evidential import evidential_loss
from src.models.uacp_predictor import UACPEvidentialPredictor
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, build_pilot_sparse_input, random_grouping_mask, uniform_grouping_mask
from src.training.metrics import nmse_all_db, nmse_omitted_db


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_modes(cfg: dict[str, Any]) -> tuple[str, str]:
    assumptions = cfg["implementation_assumption"]
    return assumptions.get("nll_mode", "elementwise"), assumptions.get("reg_mode", "elementwise")


def _observation_noise_snr_db(cfg: dict[str, Any]) -> float | None:
    value = cfg.get("implementation_assumption", {}).get("observation_noise_snr_db")
    return None if value is None else float(value)


def _build_observation(cfr: torch.Tensor, mask: torch.Tensor, cfg: dict[str, Any]):
    snr_db = _observation_noise_snr_db(cfg)
    mode = cfg.get("implementation_assumption", {}).get("observation_mode", "direct_cfr_awgn")
    if mode == "pilot_ls":
        return build_pilot_sparse_input(cfr, mask, snr_db)
    if mode == "direct_cfr_awgn":
        return build_noisy_sparse_input(cfr, mask, snr_db)
    raise ValueError(f"Unknown observation_mode: {mode}")


def train_one_epoch(model, loader, optimizer, cfg, device: torch.device) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "nll": 0.0, "reg": 0.0, "lambda_reg_x_reg": 0.0, "nmse_all_db": 0.0, "nmse_omitted_db": 0.0, "measured_snr_db": 0.0}
    snr_batches = 0
    count = 0
    nll_mode, reg_mode = _loss_modes(cfg)
    for batch in loader:
        cfr = batch["cfr"].to(device)
        mask = random_grouping_mask(
            cfr.shape[0],
            int(cfg["paper_specified"]["num_subcarriers"]),
            list(cfg["implementation_assumption"]["mask_grouping_factors"]),
            device,
        )
        x, target, snr_stats = _build_observation(cfr, mask, cfg)
        output = model(x)
        losses = evidential_loss(output, target, float(cfg["paper_specified"]["lambda_reg"]), nll_mode=nll_mode, reg_mode=reg_mode)
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
            snr_batches += batch_size
    metrics = {key: value / count for key, value in totals.items() if key != "measured_snr_db"}
    metrics["measured_snr_db_mean"] = totals["measured_snr_db"] / snr_batches if snr_batches else None
    metrics["reg_to_nll_ratio"] = abs(metrics["lambda_reg_x_reg"]) / (abs(metrics["nll"]) + 1e-8)
    return metrics


@torch.no_grad()
def evaluate(model, loader, cfg, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "total": 0.0,
        "nll": 0.0,
        "reg": 0.0,
        "lambda_reg_x_reg": 0.0,
        "nmse_all_db": 0.0,
        "nmse_omitted_db": 0.0,
        "aleatoric": 0.0,
        "epistemic": 0.0,
        "aleatoric_all": 0.0,
        "epistemic_all": 0.0,
        "measured_snr_db": 0.0,
    }
    snr_samples = 0
    count = 0
    nll_mode, reg_mode = _loss_modes(cfg)
    for batch in loader:
        cfr = batch["cfr"].to(device)
        mask = uniform_grouping_mask(
            cfr.shape[0],
            int(cfg["paper_specified"]["num_subcarriers"]),
            int(cfg["implementation_assumption"]["eval_grouping_factor"]),
            device,
        )
        x, target, snr_stats = _build_observation(cfr, mask, cfg)
        output = model(x)
        losses = evidential_loss(output, target, float(cfg["paper_specified"]["lambda_reg"]), nll_mode=nll_mode, reg_mode=reg_mode)
        omitted = (1.0 - mask)[:, None, :]
        omitted_count = omitted.sum().clamp_min(1.0) * output.aleatoric.shape[1]
        aleatoric_omitted = (output.aleatoric * omitted).sum() / omitted_count
        epistemic_omitted = (output.epistemic * omitted).sum() / omitted_count

        batch_size = cfr.shape[0]
        count += batch_size
        for key in ("total", "nll", "reg", "lambda_reg_x_reg"):
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        totals["nmse_all_db"] += float(nmse_all_db(output.predicted, target).cpu()) * batch_size
        totals["nmse_omitted_db"] += float(nmse_omitted_db(output.predicted, target, mask).cpu()) * batch_size
        totals["aleatoric"] += float(aleatoric_omitted.cpu()) * batch_size
        totals["epistemic"] += float(epistemic_omitted.cpu()) * batch_size
        totals["aleatoric_all"] += float(output.aleatoric.mean().cpu()) * batch_size
        totals["epistemic_all"] += float(output.epistemic.mean().cpu()) * batch_size
        if snr_stats["measured_snr_db_mean"] is not None:
            totals["measured_snr_db"] += float(snr_stats["measured_snr_db_mean"]) * batch_size
            snr_samples += batch_size
    metrics = {key: value / count for key, value in totals.items() if key != "measured_snr_db"}
    metrics["measured_snr_db_mean"] = totals["measured_snr_db"] / snr_samples if snr_samples else None
    metrics["reg_to_nll_ratio"] = abs(metrics["lambda_reg_x_reg"]) / (abs(metrics["nll"]) + 1e-8)
    return metrics


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["implementation_assumption"]["seed"])
    set_seeds(seed)
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")

    train_data = CFRNPZDataset(cfg["data"]["train_path"])
    val_data = CFRNPZDataset(cfg["data"]["validation_path"])
    train_loader = DataLoader(
        train_data,
        batch_size=int(cfg["implementation_assumption"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_data, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))

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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))

    history = []
    start = time.perf_counter()
    for epoch in range(1, int(cfg["implementation_assumption"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    test_results = {}
    for label, path in cfg["data"]["test_paths"].items():
        loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
        test_results[label] = evaluate(model, loader, cfg, device)

    elapsed = time.perf_counter() - start
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "uacp_predictor_prototype.pt")
    results = {
        "config": cfg,
        "device": str(device),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "test_results": test_results,
        "elapsed_seconds": elapsed,
        "checkpoint": str(output_path / "uacp_predictor_prototype.pt"),
    }
    (output_path / "training_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_training_csvs(output_path, history, test_results)
    return results


def write_training_csvs(output_path: Path, history: list[dict[str, Any]], test_results: dict[str, Any]) -> None:
    with (output_path / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_total",
            "train_nll",
            "train_reg",
            "train_lambda_reg_x_reg",
            "train_reg_to_nll_ratio",
            "train_nmse_all_db",
            "train_nmse_omitted_db",
            "validation_total",
            "validation_nll",
            "validation_reg",
            "validation_lambda_reg_x_reg",
            "validation_reg_to_nll_ratio",
            "validation_nmse_all_db",
            "validation_nmse_omitted_db",
            "validation_aleatoric",
            "validation_epistemic",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_total": row["train"]["total"],
                    "train_nll": row["train"]["nll"],
                    "train_reg": row["train"]["reg"],
                    "train_lambda_reg_x_reg": row["train"]["lambda_reg_x_reg"],
                    "train_reg_to_nll_ratio": row["train"].get("reg_to_nll_ratio", abs(row["train"]["lambda_reg_x_reg"]) / (abs(row["train"]["nll"]) + 1e-8)),
                    "train_nmse_all_db": row["train"]["nmse_all_db"],
                    "train_nmse_omitted_db": row["train"]["nmse_omitted_db"],
                    "validation_total": row["validation"]["total"],
                    "validation_nll": row["validation"]["nll"],
                    "validation_reg": row["validation"]["reg"],
                    "validation_lambda_reg_x_reg": row["validation"]["lambda_reg_x_reg"],
                    "validation_reg_to_nll_ratio": row["validation"].get("reg_to_nll_ratio", abs(row["validation"]["lambda_reg_x_reg"]) / (abs(row["validation"]["nll"]) + 1e-8)),
                    "validation_nmse_all_db": row["validation"]["nmse_all_db"],
                    "validation_nmse_omitted_db": row["validation"]["nmse_omitted_db"],
                    "validation_aleatoric": row["validation"]["aleatoric"],
                    "validation_epistemic": row["validation"]["epistemic"],
                }
            )

    with (output_path / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["regime", "nmse_all_db", "nmse_omitted_db", "aleatoric", "epistemic", "nll", "raw_l_reg", "lambda_reg_x_l_reg"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime, metrics in test_results.items():
            writer.writerow(
                {
                    "regime": regime,
                    "nmse_all_db": metrics["nmse_all_db"],
                    "nmse_omitted_db": metrics["nmse_omitted_db"],
                    "aleatoric": metrics["aleatoric"],
                    "epistemic": metrics["epistemic"],
                    "nll": metrics["nll"],
                    "raw_l_reg": metrics["reg"],
                    "lambda_reg_x_l_reg": metrics["lambda_reg_x_reg"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_prototype.json")
    parser.add_argument("--output-dir", default="runs/prototype_predictor")
    args = parser.parse_args()
    results = run(args.config, args.output_dir)
    print(json.dumps({k: results[k] for k in ("device", "trainable_parameters", "total_parameters", "elapsed_seconds", "test_results")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
