#!/usr/bin/env python3
"""Psi-head-only calibration diagnostic for the current-valid baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import _build_observation, _loss_modes, load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors, evidential_loss  # noqa: E402
from src.training.data import CFRNPZDataset, random_grouping_mask, uniform_grouping_mask  # noqa: E402
from src.training.metrics import nmse_omitted_db  # noqa: E402
from src.training.uncertainty import paper_subcarrier_uncertainty_map  # noqa: E402


K = 1024
REGIMES = ("ID-Easy 20 ns", "ID-Hard 80 ns")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(model, path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {"state_dict_only": True}


def freeze_to_psi_head(model) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.psi_head.parameters():
        parameter.requires_grad = True

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [name for name in trainable_names if not name.startswith("psi_head.")]
    if invalid:
        raise AssertionError(f"Non-Psi trainable parameters found: {invalid}")
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen_parameters": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        "trainable_parameter_names": trainable_names,
        "invalid_trainable_parameter_names": invalid,
    }


def make_fixed_batches(path: str, cfg: dict[str, Any], device: torch.device, seed: int, uniform: bool = True) -> list[dict[str, torch.Tensor]]:
    dataset = CFRNPZDataset(path)
    loader = DataLoader(dataset, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    batches = []
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        if uniform:
            mask = uniform_grouping_mask(cfr.shape[0], K, 16, device)
        else:
            mask = uniform_grouping_mask(cfr.shape[0], K, 16, device)
        set_seeds(seed + batch_index)
        x, target, _ = _build_observation(cfr, mask, cfg)
        batches.append({"cfr": cfr, "x": x.detach(), "target": target.detach(), "mask": mask.detach()})
    return batches


def normalized_metrics(output, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    cfr = batch["cfr"]
    mask = batch["mask"]
    scale = cfr.abs().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-12)
    psi_norm = output.psi / scale.square()[:, None, None]
    aleatoric_norm = output.aleatoric / scale.square()[:, None, None]
    epistemic_norm = output.epistemic / scale.square()[:, None, None]
    omitted = 1.0 - mask

    def omitted_score(values: torch.Tensor) -> torch.Tensor:
        score_map = paper_subcarrier_uncertainty_map(values)
        return (score_map * omitted).sum(dim=-1) / omitted.sum(dim=-1).clamp_min(1.0)

    pair_psi = channels_to_pair_vectors(psi_norm)
    return {
        "nmse_omitted_db": float(nmse_omitted_db(output.predicted, batch["target"], mask).detach().cpu()),
        "normalized_psi_mean": float(psi_norm.mean().detach().cpu()),
        "normalized_psi_median": float(psi_norm.median().detach().cpu()),
        "normalized_psi_pair_mean": float(pair_psi.mean().detach().cpu()),
        "nu_mean": float(output.nu.mean().detach().cpu()),
        "denominator_mean": float((output.nu - 2 * K - 1).mean().detach().cpu()),
        "normalized_aleatoric": float(omitted_score(aleatoric_norm).mean().detach().cpu()),
        "normalized_epistemic": float(omitted_score(epistemic_norm).mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_fixed(model, batches: list[dict[str, torch.Tensor]], cfg: dict[str, Any], include_loss: bool = True) -> dict[str, float]:
    model.eval()
    nll_mode, reg_mode = _loss_modes(cfg)
    totals: dict[str, float] = {"nmse_omitted_db": 0.0, "normalized_psi_mean": 0.0, "normalized_psi_median": 0.0, "nu_mean": 0.0, "denominator_mean": 0.0, "normalized_aleatoric": 0.0, "normalized_epistemic": 0.0, "nll": 0.0}
    count = 0
    for batch in batches:
        output = model(batch["x"])
        metrics = normalized_metrics(output, batch)
        batch_size = int(batch["x"].shape[0])
        count += batch_size
        for key in totals:
            if key in metrics:
                totals[key] += metrics[key] * batch_size
        if include_loss:
            losses = evidential_loss(output, batch["target"], float(cfg["paper_specified"]["lambda_reg"]), nll_mode=nll_mode, reg_mode=reg_mode)
            totals["nll"] += float(losses["nll"].detach().cpu()) * batch_size
    return {key: value / count for key, value in totals.items()}


def snapshot_outputs(model, batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    model.eval()
    rows = {key: [] for key in ("gamma", "kappa", "nu", "psi")}
    for batch in batches:
        output = model(batch["x"])
        for key in rows:
            rows[key].append(getattr(output, key).detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}


def train_psi_epoch(model, loader, optimizer, cfg: dict[str, Any], device: torch.device) -> dict[str, float]:
    # Keep the entire model in eval mode: frozen residual-block dropout is disabled.
    model.eval()
    nll_mode, reg_mode = _loss_modes(cfg)
    totals = {"nll": 0.0, "reg": 0.0, "total": 0.0, "psi_gradient_norm": 0.0}
    count = 0
    for batch in loader:
        cfr = batch["cfr"].to(device)
        mask = random_grouping_mask(cfr.shape[0], K, list(cfg["implementation_assumption"]["mask_grouping_factors"]), device)
        x, target, _ = _build_observation(cfr, mask, cfg)
        output = model(x)
        losses = evidential_loss(output, target, float(cfg["paper_specified"]["lambda_reg"]), nll_mode=nll_mode, reg_mode=reg_mode)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_sq = 0.0
        for parameter in model.psi_head.parameters():
            if parameter.grad is not None:
                grad_sq += float(parameter.grad.detach().square().sum().cpu())
        grad_norm = grad_sq ** 0.5
        torch.nn.utils.clip_grad_norm_(model.psi_head.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = int(cfr.shape[0])
        count += batch_size
        for key in ("nll", "reg", "total"):
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        totals["psi_gradient_norm"] += grad_norm * batch_size
    return {key: value / count for key, value in totals.items()}


def freeze_sanity(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "gamma_max_abs_diff": float((before["gamma"] - after["gamma"]).abs().max()),
        "kappa_max_abs_diff": float((before["kappa"] - after["kappa"]).abs().max()),
        "nu_max_abs_diff": float((before["nu"] - after["nu"]).abs().max()),
        "psi_max_abs_diff": float((before["psi"] - after["psi"]).abs().max()),
    }


def counterfactual_summary(path: Path) -> dict[str, float]:
    values = {}
    with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["quantity"]] = float(row["mean"])
    return {
        "C_Psi": values["C_Psi_symmetric"],
        "C_Nu": values["C_Nu_symmetric"],
        "A20_20": values["A20_20"],
        "A80_80": values["A80_80"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json")
    parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-seed-easy", type=int, default=97000)
    parser.add_argument("--eval-seed-hard", type=int, default=98000)
    args = parser.parse_args()

    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / args.config)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    checkpoint_path = ROOT / args.checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    before_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    freeze_info = freeze_to_psi_head(model)
    print(json.dumps({"device": str(device), "cuda_available": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, **freeze_info}, indent=2), flush=True)

    train_data = CFRNPZDataset(cfg["data"]["train_path"])
    train_loader = DataLoader(
        train_data,
        batch_size=int(cfg["implementation_assumption"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["implementation_assumption"]["seed"])),
    )
    eval_batches = {
        REGIMES[0]: make_fixed_batches(cfg["data"]["test_paths"][REGIMES[0]], cfg, device, args.eval_seed_easy),
        REGIMES[1]: make_fixed_batches(cfg["data"]["test_paths"][REGIMES[1]], cfg, device, args.eval_seed_hard),
    }
    val_batches = make_fixed_batches(cfg["data"]["validation_path"], cfg, device, 66000)
    before_outputs = {regime: snapshot_outputs(model, batches) for regime, batches in eval_batches.items()}
    before_metrics = {regime: evaluate_fixed(model, batches, cfg) for regime, batches in eval_batches.items()}
    before_val = evaluate_fixed(model, val_batches, cfg)

    optimizer = torch.optim.Adam(model.psi_head.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_psi_epoch(model, train_loader, optimizer, cfg, device)
        epoch_metrics = {regime: evaluate_fixed(model, batches, cfg) for regime, batches in eval_batches.items()}
        val_metrics = evaluate_fixed(model, val_batches, cfg)
        row = {
            "epoch": epoch,
            "train_nll": train_metrics["nll"],
            "train_reg": train_metrics["reg"],
            "train_total": train_metrics["total"],
            "psi_gradient_norm": train_metrics["psi_gradient_norm"],
            "validation_nll": val_metrics["nll"],
        }
        for regime, prefix in ((REGIMES[0], "20ns"), (REGIMES[1], "80ns")):
            row[f"{prefix}_nmse_omitted_db"] = epoch_metrics[regime]["nmse_omitted_db"]
            row[f"{prefix}_psi_norm"] = epoch_metrics[regime]["normalized_psi_mean"]
            row[f"{prefix}_aleatoric_norm"] = epoch_metrics[regime]["normalized_aleatoric"]
            row[f"{prefix}_epistemic_norm"] = epoch_metrics[regime]["normalized_epistemic"]
            row[f"{prefix}_nu"] = epoch_metrics[regime]["nu_mean"]
            row[f"{prefix}_denominator"] = epoch_metrics[regime]["denominator_mean"]
        row["delta_psi"] = row["80ns_psi_norm"] - row["20ns_psi_norm"]
        row["delta_aleatoric"] = row["80ns_aleatoric_norm"] - row["20ns_aleatoric_norm"]
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    after_outputs = {regime: snapshot_outputs(model, batches) for regime, batches in eval_batches.items()}
    after_metrics = {regime: evaluate_fixed(model, batches, cfg) for regime, batches in eval_batches.items()}
    after_val = evaluate_fixed(model, val_batches, cfg)
    sanity = {regime: freeze_sanity(before_outputs[regime], after_outputs[regime]) for regime in REGIMES}

    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "source_checkpoint": args.checkpoint, "experiment": "Psi-Head-Only Calibration Diagnostic"}, out / "calibrated_checkpoint.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out / "training_history.csv", history)

    before_rows = []
    after_rows = []
    for regime in REGIMES:
        for label, metrics in (("before", before_metrics[regime]), ("after", after_metrics[regime])):
            row = {"regime": regime, "state": label, **metrics}
            (before_rows if label == "before" else after_rows).append(row)
    write_csv(out / "evaluation_before.csv", before_rows)
    write_csv(out / "evaluation_after.csv", after_rows)

    parameter_before = []
    parameter_after = []
    for regime in REGIMES:
        for label, metrics in (("before", before_metrics[regime]), ("after", after_metrics[regime])):
            row = {"regime": regime, "state": label, "normalized_psi_mean": metrics["normalized_psi_mean"], "normalized_psi_median": metrics["normalized_psi_median"], "nu_mean": metrics["nu_mean"], "denominator_mean": metrics["denominator_mean"], "normalized_aleatoric": metrics["normalized_aleatoric"], "normalized_epistemic": metrics["normalized_epistemic"]}
            (parameter_before if label == "before" else parameter_after).append(row)
    write_csv(out / "parameter_stats_before.csv", parameter_before)
    write_csv(out / "parameter_stats_after.csv", parameter_after)

    before_cf_dir = ROOT / "runs/current_valid_baseline/diagnostics/psi_nu_counterfactual_20260911_final"
    cf_after_dir = out / "counterfactual_after"
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/psi_nu_counterfactual_diagnostic.py"),
        "--config", args.config,
        "--checkpoint", str(out.relative_to(ROOT) / "calibrated_checkpoint.pt"),
        "--output-dir", str(cf_after_dir.relative_to(ROOT)),
        "--permutations", "100",
        "--permutation-seed", "20260911",
    ], check=True)
    before_cf = counterfactual_summary(before_cf_dir)
    after_cf = counterfactual_summary(cf_after_dir)

    analysis = {
        "experiment": "Psi-Head-Only Calibration Diagnostic",
        "hypothesis": "A fixed shared representation may contain difficulty information while joint training leaves the Psi head insufficiently calibrated.",
        "classification_labels": {
            "ten_epoch_head_only_training": "IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC",
            "dropout_disabled_by_model_eval": "IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC",
            "diagonal_psi_instead_of_full_LL_transpose": "APPROXIMATION",
            "direct_cfr_awgn_at_15_db": "IMPLEMENTATION-ASSUMPTION",
        },
        "freeze_info": freeze_info,
        "freeze_sanity": sanity,
        "validation_before": before_val,
        "validation_after": after_val,
        "before": {"metrics": before_metrics, "counterfactual": before_cf},
        "after": {"metrics": after_metrics, "counterfactual": after_cf},
        "deltas": {
            "before_delta_psi": before_metrics[REGIMES[1]]["normalized_psi_mean"] - before_metrics[REGIMES[0]]["normalized_psi_mean"],
            "after_delta_psi": after_metrics[REGIMES[1]]["normalized_psi_mean"] - after_metrics[REGIMES[0]]["normalized_psi_mean"],
            "before_delta_aleatoric": before_metrics[REGIMES[1]]["normalized_aleatoric"] - before_metrics[REGIMES[0]]["normalized_aleatoric"],
            "after_delta_aleatoric": after_metrics[REGIMES[1]]["normalized_aleatoric"] - after_metrics[REGIMES[0]]["normalized_aleatoric"],
        },
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "calibrated_checkpoint_sha256": hashlib.sha256((out / "calibrated_checkpoint.pt").read_bytes()).hexdigest(),
    }
    (out / "provenance.json").write_text(json.dumps({
        "config": args.config,
        "source_checkpoint": args.checkpoint,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "learning_rate": float(cfg["paper_specified"]["learning_rate"]),
        "batch_size": int(cfg["implementation_assumption"]["batch_size"]),
        "train_path": cfg["data"]["train_path"],
        "observation_mode": cfg["implementation_assumption"].get("observation_mode", "direct_cfr_awgn"),
        "observation_snr_db": cfg["implementation_assumption"].get("observation_noise_snr_db"),
        "eval_grouping_factor": 16,
        "eval_seeds": {"ID-Easy 20 ns": args.eval_seed_easy, "ID-Hard 80 ns": args.eval_seed_hard},
        "dropout_mode": "model.eval() throughout calibration; diagnostic implementation assumption",
        "trainable_parameter_names": freeze_info["trainable_parameter_names"],
        "optimizer": "Adam on model.psi_head.parameters() only",
        "loss_reused": "src.models.evidential.evidential_loss with config nll_mode/reg_mode",
        "training_performed": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "analysis": analysis}, indent=2), flush=True)


if __name__ == "__main__":
    main()
