#!/usr/bin/env python3
"""Run staged diagnostics for UACP baseline reproduction experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_predictor import _make_model, diagnose_regime, flatten_diagnostic_row
from scripts.generate_dataset import load_dataset_config, save_split_npz
from scripts.train_predictor import evaluate, load_config, set_seeds, train_one_epoch, write_training_csvs
from src.channel.sionna_channel import assumption_settings, generate_cfr_for_delay_spreads, paper_settings, unknown_settings
from src.models.evidential import evidential_loss
from src.training.data import CFRNPZDataset, build_sparse_input, uniform_grouping_mask
from src.training.uncertainty import paper_omitted_uncertainty_score


def delay_label(delay_spread_ns: float) -> str:
    if abs(delay_spread_ns - 1_000_000.0) < 1e-6:
        return "1 ms"
    if float(delay_spread_ns).is_integer():
        return f"{int(delay_spread_ns)} ns"
    return f"{delay_spread_ns:g} ns"


def step_row(experiment: str, regime: str, delay_spread_ns: float, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "experiment": experiment,
        "regime": regime,
        "delay_spread_ns": float(delay_spread_ns),
        "samples": metrics["samples"],
        "nmse_all_db": metrics["nmse_all_db"],
        "nmse_omitted_db": metrics["nmse_omitted_db"],
        "psi_mean": metrics["psi_omitted"]["mean"],
        "psi_std": metrics["psi_omitted"]["std"],
        "kappa_mean": metrics["kappa_omitted"]["mean"],
        "kappa_std": metrics["kappa_omitted"]["std"],
        "nu_mean": metrics["nu_omitted"]["mean"],
        "nu_std": metrics["nu_omitted"]["std"],
        "df_cov_mean": metrics["df_cov_omitted"]["mean"],
        "df_cov_std": metrics["df_cov_omitted"]["std"],
        "aleatoric_mean": metrics["aleatoric_omitted"]["mean"],
        "aleatoric_std": metrics["aleatoric_omitted"]["std"],
        "epistemic_mean": metrics["epistemic_omitted"]["mean"],
        "epistemic_std": metrics["epistemic_omitted"]["std"],
        "error_aleatoric_pearson": metrics["error_aleatoric_pearson"],
        "error_aleatoric_spearman": metrics["error_aleatoric_spearman"],
        "error_epistemic_pearson": metrics["error_epistemic_pearson"],
        "error_epistemic_spearman": metrics["error_epistemic_spearman"],
        "nll": metrics.get("nll", ""),
        "raw_l_reg": metrics.get("raw_l_reg", ""),
        "lambda_reg_x_l_reg": metrics.get("lambda_reg_x_l_reg", ""),
        "reg_to_nll_ratio": metrics.get(
            "reg_to_nll_ratio",
            abs(float(metrics.get("lambda_reg_x_l_reg", 0.0))) / (abs(float(metrics.get("nll", 0.0))) + 1e-8),
        ),
        "requested_snr_db": metrics.get("requested_snr_db"),
        "measured_snr_db_mean": metrics.get("measured_snr_db_mean"),
        "measured_snr_db_std": metrics.get("measured_snr_db_std"),
        "covariance_diag_mean": (metrics.get("covariance_diag_omitted") or {}).get("mean", ""),
        "covariance_diag_std": (metrics.get("covariance_diag_omitted") or {}).get("std", ""),
        "lowrank_diag_contribution_mean": (metrics.get("lowrank_diag_contribution_omitted") or {}).get("mean", ""),
        "lowrank_diag_contribution_std": (metrics.get("lowrank_diag_contribution_omitted") or {}).get("std", ""),
        "off_diagonal_energy_ratio_mean": metrics.get("off_diagonal_energy_ratio_mean", ""),
        "off_diagonal_energy_ratio_std": metrics.get("off_diagonal_energy_ratio_std", ""),
    }
    for lag in (1, 2, 4, 8):
        row[f"frequency_covariance_lag_{lag}_mean"] = metrics.get(f"frequency_covariance_lag_{lag}_mean", "")
        row[f"frequency_covariance_lag_{lag}_std"] = metrics.get(f"frequency_covariance_lag_{lag}_std", "")
    for item in metrics.get("error_bin_calibration", []):
        index = item["bin"]
        row[f"error_bin_{index}_error_mean"] = item["error_mean"]
        row[f"error_bin_{index}_aleatoric_mean"] = item["aleatoric_mean"]
        row[f"error_bin_{index}_epistemic_mean"] = item["epistemic_mean"]
    return row


def epoch_probe_row(epoch: int, regime: str, delay_spread_ns: float, metrics: dict[str, Any]) -> dict[str, Any]:
    row = step_row("epoch_probe", regime, delay_spread_ns, metrics)
    row.pop("experiment")
    return {"epoch": int(epoch), **row}


def probe_metrics_as_evaluation(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "nmse_all_db": metrics["nmse_all_db"],
        "nmse_omitted_db": metrics["nmse_omitted_db"],
        "aleatoric": metrics["aleatoric_omitted"]["mean"],
        "epistemic": metrics["epistemic_omitted"]["mean"],
        "nll": metrics["nll"],
        "reg": metrics["raw_l_reg"],
        "lambda_reg_x_reg": metrics["lambda_reg_x_l_reg"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_curve_pngs(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {
        "delay_vs_nmse_omitted.png": ("nmse_omitted_db", "NMSE omitted (dB)"),
        "delay_vs_aleatoric.png": ("aleatoric_mean", "Aleatoric uncertainty"),
        "delay_vs_epistemic.png": ("epistemic_mean", "Epistemic uncertainty"),
        "delay_vs_psi.png": ("psi_mean", "psi"),
        "delay_vs_kappa.png": ("kappa_mean", "kappa"),
    }
    sorted_rows = sorted(rows, key=lambda row: float(row["delay_spread_ns"]))
    x = [float(row["delay_spread_ns"]) for row in sorted_rows]
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (metric, ylabel) in curves.items():
        y = [float(row[metric]) for row in sorted_rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("RMS delay spread (ns)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def write_epoch_curve_pngs(training_history: list[dict[str, Any]], probe_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(path: Path, ylabel: str, series: dict[str, tuple[list[int], list[float]]]) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, (x, y) in series.items():
            ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    epochs = [int(row["epoch"]) for row in training_history]
    save(
        output_dir / "epoch_vs_train_validation_nmse.png",
        "NMSE omitted (dB)",
        {
            "train": (epochs, [float(row["train"]["nmse_omitted_db"]) for row in training_history]),
            "validation": (epochs, [float(row["validation"]["nmse_omitted_db"]) for row in training_history]),
        },
    )
    save(
        output_dir / "epoch_vs_nll.png",
        "NLL",
        {
            "train": (epochs, [float(row["train"]["nll"]) for row in training_history]),
            "validation": (epochs, [float(row["validation"]["nll"]) for row in training_history]),
        },
    )
    save(
        output_dir / "epoch_vs_lambda_reg_x_l_reg.png",
        "lambda_reg x L_reg",
        {
            "train": (epochs, [float(row["train"]["lambda_reg_x_reg"]) for row in training_history]),
            "validation": (epochs, [float(row["validation"]["lambda_reg_x_reg"]) for row in training_history]),
        },
    )
    save(
        output_dir / "epoch_vs_reg_to_nll_ratio.png",
        "abs(lambda_reg x L_reg) / abs(NLL)",
        {
            "train": (epochs, [float(row["train"].get("reg_to_nll_ratio", 0.0)) for row in training_history]),
            "validation": (epochs, [float(row["validation"].get("reg_to_nll_ratio", 0.0)) for row in training_history]),
        },
    )

    for metric, ylabel, filename in [
        ("aleatoric_mean", "Aleatoric uncertainty", "epoch_vs_aleatoric.png"),
        ("epistemic_mean", "Epistemic uncertainty", "epoch_vs_epistemic.png"),
        ("psi_mean", "psi", "epoch_vs_psi.png"),
        ("kappa_mean", "kappa", "epoch_vs_kappa.png"),
        ("df_cov_mean", "nu - 2K - 1", "epoch_vs_df_cov.png"),
    ]:
        regimes = sorted({row["regime"] for row in probe_rows})
        series = {}
        for regime in regimes:
            subset = sorted([row for row in probe_rows if row["regime"] == regime], key=lambda row: int(row["epoch"]))
            series[regime] = ([int(row["epoch"]) for row in subset], [float(row[metric]) for row in subset])
        save(output_dir / filename, ylabel, series)

    return written


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in sorted(vars(args).items()) if not callable(value)}


def _metadata(dataset_cfg: dict[str, Any], split: str, delay_spread_ns: float) -> dict[str, Any]:
    return {
        "split": split,
        "delay_spread_ns": float(delay_spread_ns),
        "paper_specified": paper_settings(dataset_cfg),
        "implementation_assumption": assumption_settings(dataset_cfg),
        "unknown": unknown_settings(dataset_cfg),
        "layout": "[sample, subcarrier, rx_ant, tx_ant]",
        "snr_policy": "clean CFR; SNR is recorded but not applied",
    }


def generate_fixed_delay_split(
    dataset_config_path: str | Path,
    output_dir: Path,
    split: str,
    regime: str,
    delay_spread_ns: float,
    samples: int,
    seed: int,
) -> Path:
    dataset_cfg = load_dataset_config(dataset_config_path)
    delays = np.full(samples, float(delay_spread_ns), dtype=np.float32)
    labels = np.array([regime] * samples)
    start = time.perf_counter()
    cfr = generate_cfr_for_delay_spreads(dataset_cfg, delays, seed=seed)
    metadata = _metadata(dataset_cfg, split, delay_spread_ns)
    metadata["generation_seconds"] = time.perf_counter() - start
    path = output_dir / f"{split}.npz"
    save_split_npz(path, cfr, delays, labels, metadata)
    return path


def generate_uniform_train_split(
    dataset_config_path: str | Path,
    output_dir: Path,
    samples: int,
    seed: int,
) -> Path:
    dataset_cfg = load_dataset_config(dataset_config_path)
    low, high = dataset_cfg["training_delay_spread_ns"]
    rng = np.random.default_rng(seed)
    delays = rng.uniform(float(low), float(high), size=int(samples)).astype(np.float32)
    labels = np.array(["train"] * int(samples))
    start = time.perf_counter()
    cfr = generate_cfr_for_delay_spreads(dataset_cfg, delays, seed=seed + 1000)
    metadata = {
        "split": "train_5k_pilot",
        "paper_specified": paper_settings(dataset_cfg),
        "implementation_assumption": assumption_settings(dataset_cfg),
        "pilot": {
            "train_samples": int(samples),
            "delay_spread_distribution": f"Uniform[{float(low)}, {float(high)}] ns",
        },
        "unknown": unknown_settings(dataset_cfg),
        "layout": "[sample, subcarrier, rx_ant, tx_ant]",
        "snr_policy": "clean CFR; SNR is recorded but not applied",
        "generation_seconds": time.perf_counter() - start,
    }
    path = output_dir / "train_5k_pilot.npz"
    save_split_npz(path, cfr, delays, labels, metadata)
    return path


def _load_model(model_cfg: dict[str, Any], checkpoint_path: str | Path, device: torch.device):
    model = _make_model(model_cfg, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _new_model(model_cfg: dict[str, Any], device: torch.device):
    model = _make_model(model_cfg, device)
    model.train()
    return model


def _write_summary_md(path: Path, title: str, rows: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {regime} | {nmse_all_db:.4f} | {nmse_omitted_db:.4f} | {aleatoric_mean:.6f} | "
            "{epistemic_mean:.6f} | {psi_mean:.6f} | {kappa_mean:.6f} | {df_cov_mean:.6f} | "
            "{error_aleatoric_pearson:.4f}/{error_aleatoric_spearman:.4f} | "
            "{error_epistemic_pearson:.4f}/{error_epistemic_spearman:.4f} |".format(**row)
        )
    lines.extend(["", "## Interpretation", ""])
    for key, value in interpretation.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_regimes(
    experiment_name: str,
    model_cfg: dict[str, Any],
    checkpoint_path: str | Path,
    regimes: list[tuple[str, float, Path]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seeds(int(model_cfg["implementation_assumption"]["seed"]))
    device = torch.device(model_cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _load_model(model_cfg, checkpoint_path, device)
    diagnostics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for regime, delay_spread_ns, path in regimes:
        metrics = diagnose_regime(model, str(path), model_cfg, device)
        diagnostics[regime] = metrics
        rows.append(step_row(experiment_name, regime, delay_spread_ns, metrics))
    return rows, diagnostics


def _step1_interpretation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime = {row["regime"]: row for row in rows}
    epi_1ms = by_regime["OOD-Far 1 ms"]["epistemic_mean"]
    id_epi = max(by_regime["ID-Easy 20 ns"]["epistemic_mean"], by_regime["ID-Hard 80 ns"]["epistemic_mean"])
    ale_hard_gt_easy = by_regime["ID-Hard 80 ns"]["aleatoric_mean"] > by_regime["ID-Easy 20 ns"]["aleatoric_mean"]
    epi_near_gt_id = by_regime["OOD-Near 120 ns"]["epistemic_mean"] > id_epi
    epi_far_gt_id = epi_1ms > id_epi
    case = "Case A: OOD-Far epistemic increases" if epi_far_gt_id else "Case B: OOD-Far epistemic does not increase"
    return {
        "Aleatoric(80) > Aleatoric(20)": ale_hard_gt_easy,
        "Epistemic(120) > Epistemic(ID)": epi_near_gt_id,
        "Epistemic(1 ms) > Epistemic(ID)": epi_far_gt_id,
        "Step 1 case": case,
        "Next step": "Proceed to delay sweep to locate trend shape before changing training objective.",
    }


def run_step1(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "generated_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = load_config(args.model_config)
    samples = int(args.samples)
    seed = int(model_cfg["implementation_assumption"]["seed"]) + 91000
    ood_far_path = generate_fixed_delay_split(
        args.dataset_config,
        data_dir,
        "test_ood_far_1ms",
        "OOD-Far 1 ms",
        1_000_000.0,
        samples,
        seed,
    )
    regimes = [
        ("ID-Easy 20 ns", 20.0, Path(model_cfg["data"]["test_paths"]["ID-Easy 20 ns"])),
        ("ID-Hard 80 ns", 80.0, Path(model_cfg["data"]["test_paths"]["ID-Hard 80 ns"])),
        ("OOD-Near 120 ns", 120.0, Path(model_cfg["data"]["test_paths"]["OOD-Near 120 ns"])),
        ("OOD-Far 1 ms", 1_000_000.0, ood_far_path),
    ]
    rows, diagnostics = _evaluate_regimes(args.experiment_name, model_cfg, args.checkpoint, regimes)
    interpretation = _step1_interpretation(rows)
    output = {
        "step": "STEP 1 OOD-Far 1 ms",
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "dataset_config": str(args.dataset_config),
        "generated_ood_far_path": str(ood_far_path),
        "rows": rows,
        "diagnostics": diagnostics,
        "interpretation": interpretation,
    }
    (out_dir / "config.json").write_text(json.dumps(serializable_args(args), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    _write_summary_md(out_dir / "summary.md", "STEP 1 OOD-Far 1 ms", rows, interpretation)
    return output


def _step2_interpretation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: row["delay_spread_ns"])
    id_rows = [row for row in sorted_rows if row["delay_spread_ns"] <= 100.0]
    ood_rows = [row for row in sorted_rows if row["delay_spread_ns"] > 100.0]
    nmse_worse = sorted_rows[-1]["nmse_omitted_db"] > sorted_rows[0]["nmse_omitted_db"]
    ale_id_increases = id_rows[-1]["aleatoric_mean"] > id_rows[0]["aleatoric_mean"]
    epi_ood_increases = bool(ood_rows) and max(row["epistemic_mean"] for row in ood_rows) > max(row["epistemic_mean"] for row in id_rows)
    return {
        "NMSE worsens from smallest to largest delay": nmse_worse,
        "ID-range aleatoric increases": ale_id_increases,
        "OOD epistemic exceeds ID epistemic": epi_ood_increases,
        "Next step": "If uncertainty trends still fail, verify paper-style aggregation before retraining.",
    }


def interpret_step4(probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_epoch = max(int(row["epoch"]) for row in probe_rows)
    final = {row["regime"]: row for row in probe_rows if int(row["epoch"]) == latest_epoch}
    id_epi = max(final["ID-Easy 20 ns"]["epistemic_mean"], final["ID-Hard 80 ns"]["epistemic_mean"])
    ale_ok = final["ID-Hard 80 ns"]["aleatoric_mean"] > final["ID-Easy 20 ns"]["aleatoric_mean"]
    epi_near_ok = final["OOD-Near 120 ns"]["epistemic_mean"] > id_epi
    epi_far_ok = final["OOD-Far 1 ms"]["epistemic_mean"] > id_epi
    nmse_ood_worse = final["OOD-Far 1 ms"]["nmse_omitted_db"] > final["ID-Easy 20 ns"]["nmse_omitted_db"]
    correlations_positive = all(
        row["error_aleatoric_pearson"] > 0.0 and row["error_epistemic_pearson"] > 0.0
        for row in final.values()
    )

    early_epochs = sorted({int(row["epoch"]) for row in probe_rows})
    early = {row["regime"]: row for row in probe_rows if int(row["epoch"]) == early_epochs[0]}
    early_id_epi = max(early["ID-Easy 20 ns"]["epistemic_mean"], early["ID-Hard 80 ns"]["epistemic_mean"])
    early_epi_ood_ok = early["OOD-Far 1 ms"]["epistemic_mean"] > early_id_epi
    if ale_ok or epi_near_ok or epi_far_ok:
        case = "Case A"
        proceed = True
    elif early_epi_ood_ok and not epi_far_ok:
        case = "Case C"
        proceed = False
    else:
        case = "Case B"
        proceed = False

    return {
        "case": case,
        "latest_epoch": latest_epoch,
        "Aleatoric(80) > Aleatoric(20)": ale_ok,
        "Epistemic(120) > Epistemic(ID)": epi_near_ok,
        "Epistemic(1 ms) > Epistemic(ID)": epi_far_ok,
        "Early epoch OOD-Far epistemic > ID": early_epi_ood_ok,
        "OOD-Far NMSE worse than ID-Easy": nmse_ood_worse,
        "Final error-uncertainty Pearson correlations positive": correlations_positive,
        "proceed_to_step4b": proceed,
    }


def _aggregation_row(regime: str, delay_spread_ns: float, current_metrics: dict[str, Any], paper_metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "regime": regime,
        "delay_spread_ns": delay_spread_ns,
        "aleatoric_current": current_metrics["aleatoric_omitted"]["mean"],
        "aleatoric_paper_aggregation": paper_metrics["aleatoric_paper_aggregation"],
        "epistemic_current": current_metrics["epistemic_omitted"]["mean"],
        "epistemic_paper_aggregation": paper_metrics["epistemic_paper_aggregation"],
        "total_current": current_metrics["aleatoric_omitted"]["mean"] + current_metrics["epistemic_omitted"]["mean"],
        "total_paper_aggregation": paper_metrics["aleatoric_paper_aggregation"] + paper_metrics["epistemic_paper_aggregation"],
        "samples": current_metrics["samples"],
    }


@torch.no_grad()
def diagnose_paper_aggregation(model, path: str | Path, cfg: dict[str, Any], device: torch.device) -> dict[str, float]:
    loader = DataLoader(
        CFRNPZDataset(path),
        batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]),
    )
    total = 0
    aleatoric_weighted = 0.0
    epistemic_weighted = 0.0
    for batch in loader:
        cfr = batch["cfr"].to(device)
        batch_size = cfr.shape[0]
        mask = uniform_grouping_mask(
            batch_size,
            int(cfg["paper_specified"]["num_subcarriers"]),
            int(cfg["implementation_assumption"]["eval_grouping_factor"]),
            device,
        )
        x, _ = build_sparse_input(cfr, mask)
        output = model(x)
        aleatoric_weighted += float(paper_omitted_uncertainty_score(output.aleatoric, mask).cpu()) * batch_size
        epistemic_weighted += float(paper_omitted_uncertainty_score(output.epistemic, mask).cpu()) * batch_size
        total += batch_size
    return {
        "aleatoric_paper_aggregation": aleatoric_weighted / total,
        "epistemic_paper_aggregation": epistemic_weighted / total,
    }


def run_step2(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "generated_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = load_config(args.model_config)
    samples = int(args.samples)
    regimes: list[tuple[str, float, Path]] = []
    for index, delay in enumerate(args.delay_spreads_ns):
        label = delay_label(float(delay))
        split = "test_delay_" + label.replace(" ", "_").replace(".", "p")
        path = generate_fixed_delay_split(
            args.dataset_config,
            data_dir,
            split,
            label,
            float(delay),
            samples,
            int(model_cfg["implementation_assumption"]["seed"]) + 92000 + index * 1000,
        )
        regimes.append((label, float(delay), path))
    rows, diagnostics = _evaluate_regimes(args.experiment_name, model_cfg, args.checkpoint, regimes)
    curves = write_curve_pngs(rows, out_dir)
    interpretation = _step2_interpretation(rows)
    output = {
        "step": "STEP 2 Delay Spread Sweep",
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "dataset_config": str(args.dataset_config),
        "curve_files": [str(path) for path in curves],
        "rows": rows,
        "diagnostics": diagnostics,
        "interpretation": interpretation,
    }
    (out_dir / "config.json").write_text(json.dumps(serializable_args(args), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    _write_summary_md(out_dir / "summary.md", "STEP 2 Delay Spread Sweep", rows, interpretation)
    return output


def run_step3(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "generated_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = load_config(args.model_config)
    seed = int(model_cfg["implementation_assumption"]["seed"]) + 93000
    ood_far_path = generate_fixed_delay_split(
        args.dataset_config,
        data_dir,
        "test_ood_far_1ms",
        "OOD-Far 1 ms",
        1_000_000.0,
        int(args.samples),
        seed,
    )
    regimes = [
        ("ID-Easy 20 ns", 20.0, Path(model_cfg["data"]["test_paths"]["ID-Easy 20 ns"])),
        ("ID-Hard 80 ns", 80.0, Path(model_cfg["data"]["test_paths"]["ID-Hard 80 ns"])),
        ("OOD-Near 120 ns", 120.0, Path(model_cfg["data"]["test_paths"]["OOD-Near 120 ns"])),
        ("OOD-Far 1 ms", 1_000_000.0, ood_far_path),
    ]
    set_seeds(int(model_cfg["implementation_assumption"]["seed"]))
    device = torch.device(model_cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _load_model(model_cfg, args.checkpoint, device)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for regime, delay_spread_ns, path in regimes:
        current = diagnose_regime(model, str(path), model_cfg, device)
        paper = diagnose_paper_aggregation(model, path, model_cfg, device)
        diagnostics[regime] = {"current": current, "paper_aggregation": paper}
        rows.append(_aggregation_row(regime, delay_spread_ns, current, paper))
    by_regime = {row["regime"]: row for row in rows}
    id_epi = max(by_regime["ID-Easy 20 ns"]["epistemic_paper_aggregation"], by_regime["ID-Hard 80 ns"]["epistemic_paper_aggregation"])
    interpretation = {
        "Aggregation scale": "For diagonal covariance, paper-style trace aggregation is 2x the current 8-channel mean.",
        "Aleatoric(80) > Aleatoric(20) under paper aggregation": by_regime["ID-Hard 80 ns"]["aleatoric_paper_aggregation"] > by_regime["ID-Easy 20 ns"]["aleatoric_paper_aggregation"],
        "Epistemic(120) > Epistemic(ID) under paper aggregation": by_regime["OOD-Near 120 ns"]["epistemic_paper_aggregation"] > id_epi,
        "Epistemic(1 ms) > Epistemic(ID) under paper aggregation": by_regime["OOD-Far 1 ms"]["epistemic_paper_aggregation"] > id_epi,
        "Next step": "Aggregation does not change ordering if it is only a constant scale; proceed to scaled training if trends still fail.",
    }
    output = {
        "step": "STEP 3 Uncertainty Score Aggregation",
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "dataset_config": str(args.dataset_config),
        "rows": rows,
        "diagnostics": diagnostics,
        "interpretation": interpretation,
    }
    (out_dir / "config.json").write_text(json.dumps(serializable_args(args), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    lines = ["# STEP 3 Uncertainty Score Aggregation", ""]
    lines.append("| Regime | Ale current | Ale paper | Epi current | Epi paper | Total current | Total paper |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {regime} | {aleatoric_current:.6f} | {aleatoric_paper_aggregation:.6f} | "
            "{epistemic_current:.6f} | {epistemic_paper_aggregation:.6f} | "
            "{total_current:.6f} | {total_paper_aggregation:.6f} |".format(**row)
        )
    lines.extend(["", "## Interpretation", ""])
    for key, value in interpretation.items():
        lines.append(f"- {key}: {value}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _step4_probe_paths(args: argparse.Namespace, model_cfg: dict[str, Any], output_dir: Path) -> list[tuple[str, float, Path]]:
    data_dir = output_dir / "generated_data"
    ood_far_path = generate_fixed_delay_split(
        args.dataset_config,
        data_dir,
        "test_ood_far_1ms",
        "OOD-Far 1 ms",
        1_000_000.0,
        int(args.probe_samples),
        int(model_cfg["implementation_assumption"]["seed"]) + 94000,
    )
    return [
        ("ID-Easy 20 ns", 20.0, Path(model_cfg["data"]["test_paths"]["ID-Easy 20 ns"])),
        ("ID-Hard 80 ns", 80.0, Path(model_cfg["data"]["test_paths"]["ID-Hard 80 ns"])),
        ("OOD-Near 120 ns", 120.0, Path(model_cfg["data"]["test_paths"]["OOD-Near 120 ns"])),
        ("OOD-Far 1 ms", 1_000_000.0, ood_far_path),
    ]


def _write_step4_summary(path: Path, history: list[dict[str, Any]], probe_rows: list[dict[str, Any]], interpretation: dict[str, Any], runtime: dict[str, float], title: str = "STEP 4A Scaled-Training Pilot") -> None:
    latest = int(interpretation["latest_epoch"])
    final_rows = [row for row in probe_rows if int(row["epoch"]) == latest]
    lines = [f"# {title}", ""]
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Estimated training seconds: `{runtime['estimated_training_seconds']:.1f}`")
    lines.append(f"- Dataset generation seconds: `{runtime['dataset_generation_seconds']:.1f}`")
    lines.append(f"- Actual training/probe seconds: `{runtime['training_probe_seconds']:.1f}`")
    lines.append("")
    lines.append("## Final Epoch Probe")
    lines.append("")
    lines.append("| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in final_rows:
        lines.append(
            "| {regime} | {nmse_all_db:.4f} | {nmse_omitted_db:.4f} | {aleatoric_mean:.6f} | "
            "{epistemic_mean:.6f} | {psi_mean:.6f} | {kappa_mean:.6f} | {df_cov_mean:.6f} | "
            "{error_aleatoric_pearson:.4f}/{error_aleatoric_spearman:.4f} | "
            "{error_epistemic_pearson:.4f}/{error_epistemic_spearman:.4f} |".format(**row)
        )
    lines.extend(["", "## Interpretation", ""])
    for key, value in interpretation.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step4a(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "generated_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = load_config(args.model_config)
    model_cfg["implementation_assumption"]["epochs"] = int(args.epochs)
    lambda_override = getattr(args, "lambda_reg_override", None)
    if lambda_override is not None:
        model_cfg["paper_specified"]["lambda_reg"] = float(lambda_override)
    seed = int(model_cfg["implementation_assumption"]["seed"])
    set_seeds(seed)
    device = torch.device(model_cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")

    estimated_training_seconds = float(args.estimated_seconds_per_sample_epoch) * float(args.train_samples) * float(args.epochs)
    data_start = time.perf_counter()
    train_path = generate_uniform_train_split(args.dataset_config, data_dir, int(args.train_samples), seed + 94500)
    dataset_generation_seconds = time.perf_counter() - data_start
    model_cfg["data"]["train_path"] = str(train_path)
    model_cfg["implementation_assumption"]["experiment_name"] = getattr(args, "experiment_name", "STEP-4A-D-5k-10ep-pilot")
    model_cfg["implementation_assumption"]["pilot_train_samples"] = int(args.train_samples)
    model_cfg["implementation_assumption"]["pilot_epochs"] = int(args.epochs)
    model_cfg["implementation_assumption"]["pilot_note"] = "IMPLEMENTATION-ASSUMPTION / PILOT, not paper-specified"
    observation_noise_snr_db = getattr(args, "observation_noise_snr_db", None)
    model_cfg["implementation_assumption"]["observation_noise_snr_db"] = observation_noise_snr_db
    mode_override = getattr(args, "evidential_mode_override", None)
    nll_override = getattr(args, "nll_mode_override", None)
    if mode_override is not None:
        model_cfg["implementation_assumption"]["evidential_mode"] = mode_override
    if nll_override is not None:
        model_cfg["implementation_assumption"]["nll_mode"] = nll_override
    covariance_rank = getattr(args, "covariance_rank", None)
    if covariance_rank is not None:
        model_cfg["implementation_assumption"]["covariance_rank"] = int(covariance_rank)
    model_cfg["implementation_assumption"]["observation_noise_note"] = (
        "IMPLEMENTATION-ASSUMPTION: observed complex CFR AWGN; target remains clean CFR"
        if observation_noise_snr_db is not None else "clean observation"
    )

    train_loader = DataLoader(
        CFRNPZDataset(train_path),
        batch_size=int(model_cfg["implementation_assumption"]["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        CFRNPZDataset(model_cfg["data"]["validation_path"]),
        batch_size=int(model_cfg["implementation_assumption"]["eval_batch_size"]),
    )
    probe_paths = _step4_probe_paths(args, model_cfg, output_dir)
    model = _new_model(model_cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["paper_specified"]["learning_rate"]))

    history: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    probe_results_by_epoch: dict[str, Any] = {}
    train_start = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, model_cfg, device)
        validation_metrics = evaluate(model, val_loader, model_cfg, device)
        history_row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(history_row)
        epoch_results = {}
        for regime, delay_spread_ns, path in probe_paths:
            metrics = diagnose_regime(model, str(path), model_cfg, device, observation_noise_snr_db=observation_noise_snr_db)
            epoch_results[regime] = metrics
            probe_rows.append(epoch_probe_row(epoch, regime, delay_spread_ns, metrics))
        probe_results_by_epoch[str(epoch)] = epoch_results
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "probe": epoch_results}, sort_keys=True), flush=True)
    training_probe_seconds = time.perf_counter() - train_start

    interpretation = interpret_step4(probe_rows)
    runtime = {
        "estimated_training_seconds": estimated_training_seconds,
        "dataset_generation_seconds": dataset_generation_seconds,
        "training_probe_seconds": training_probe_seconds,
    }
    final_results = {
        "step": getattr(args, "step_label", "STEP 4A Scaled-Training Pilot"),
        "model_config": str(args.model_config),
        "dataset_config": str(args.dataset_config),
        "train_path": str(train_path),
        "validation_path": model_cfg["data"]["validation_path"],
        "config": model_cfg,
        "history": history,
        "epoch_probe_results": probe_results_by_epoch,
        "interpretation": interpretation,
        "runtime": runtime,
        "device": str(device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    torch.save(model.state_dict(), output_dir / "uacp_predictor_step4a.pt")
    (output_dir / "config.json").write_text(json.dumps(model_cfg, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps({"history": history, "runtime": runtime}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "final_results.json").write_text(json.dumps(final_results, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "epoch_probe_results.csv", probe_rows)
    write_csv(output_dir / "summary.csv", [row for row in probe_rows if int(row["epoch"]) == int(args.epochs)])
    write_training_csvs(output_dir, history, {regime: probe_metrics_as_evaluation(metrics) for regime, metrics in probe_results_by_epoch[str(args.epochs)].items()})
    curve_files = write_epoch_curve_pngs(history, probe_rows, output_dir)
    final_results["curve_files"] = [str(path) for path in curve_files]
    (output_dir / "final_results.json").write_text(json.dumps(final_results, indent=2, sort_keys=True), encoding="utf-8")
    _write_step4_summary(output_dir / "summary.md", history, probe_rows, interpretation, runtime, title=final_results["step"])
    return final_results


def run_step4a_finalize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    final_results_path = output_dir / "final_results.json"
    final_results = json.loads(final_results_path.read_text(encoding="utf-8"))
    history = final_results["history"]
    probe_rows: list[dict[str, Any]] = []
    for epoch, regimes in sorted(final_results["epoch_probe_results"].items(), key=lambda item: int(item[0])):
        for regime, metrics in regimes.items():
            delay_spread_ns = 1_000_000.0 if regime == "OOD-Far 1 ms" else float(regime.split()[1])
            probe_rows.append(epoch_probe_row(int(epoch), regime, delay_spread_ns, metrics))
    interpretation = interpret_step4(probe_rows)
    final_results["interpretation"] = interpretation
    write_csv(output_dir / "epoch_probe_results.csv", probe_rows)
    write_csv(output_dir / "summary.csv", [row for row in probe_rows if int(row["epoch"]) == int(interpretation["latest_epoch"])])
    write_training_csvs(
        output_dir,
        history,
        {regime: probe_metrics_as_evaluation(metrics) for regime, metrics in final_results["epoch_probe_results"][str(interpretation["latest_epoch"])].items()},
    )
    curve_files = write_epoch_curve_pngs(history, probe_rows, output_dir)
    final_results["curve_files"] = [str(path) for path in curve_files]
    (output_dir / "final_results.json").write_text(json.dumps(final_results, indent=2, sort_keys=True), encoding="utf-8")
    _write_step4_summary(output_dir / "summary.md", history, probe_rows, interpretation, final_results["runtime"])
    return final_results


def run_step5_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.model_config)
    noise_snr_db = None if getattr(args, "clean_eval", False) else float(args.observation_noise_snr_db)
    cfg["implementation_assumption"]["observation_noise_snr_db"] = noise_snr_db
    cfg["implementation_assumption"]["observation_noise_note"] = "IMPLEMENTATION-ASSUMPTION: observed complex CFR AWGN; clean target"
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    probe_args = argparse.Namespace(dataset_config=args.dataset_config, probe_samples=int(args.samples))
    probe_paths = _step4_probe_paths(probe_args, cfg, output_dir)
    rows = []
    diagnostics = {}
    for regime, delay_spread_ns, path in probe_paths:
        metrics = diagnose_regime(model, str(path), cfg, device, observation_noise_snr_db=noise_snr_db)
        diagnostics[regime] = metrics
        rows.append(step_row(args.condition, regime, delay_spread_ns, metrics))
    result = {
        "step": "STEP 5 Clean vs 15 dB Observation Noise",
        "condition": args.condition,
        "config": cfg,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "diagnostics": diagnostics,
        "rows": rows,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "final_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "summary.csv", rows)
    write_csv(output_dir / "parameter_diagnostics.csv", rows)
    (output_dir / "summary.md").write_text(
        "# STEP 5 Evaluation\n\n"
        f"- Condition: `{args.condition}`\n- Observation noise: `{noise_snr_db}` dB\n"
        "- Target: clean full CFR\n",
        encoding="utf-8",
    )
    return result


def _load_step5_rows(path: Path) -> list[dict[str, Any]]:
    return list(csv.DictReader((path / "summary.csv").open(encoding="utf-8")))


def run_step5_compare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, path in [("A Clean->Clean", Path(args.clean_dir)), ("B Clean->Noisy", Path(args.clean_noisy_dir)), ("C Noisy->Noisy", Path(args.noisy_dir))]:
        source = _load_step5_rows(path)
        for row in source:
            rows.append({"condition": condition, **row})
    if args.noisy_clean_dir:
        for row in _load_step5_rows(Path(args.noisy_clean_dir)):
            rows.append({"condition": "D Noisy->Clean", **row})
    write_csv(output_dir / "summary.csv", rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel, filename in [
        ("nmse_omitted_db", "NMSE omitted (dB)", "condition_vs_nmse_omitted.png"),
        ("aleatoric_mean", "Aleatoric uncertainty", "condition_vs_aleatoric.png"),
        ("epistemic_mean", "Epistemic uncertainty", "condition_vs_epistemic.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for condition in sorted({row["condition"] for row in rows}):
            subset = [row for row in rows if row["condition"] == condition]
            subset.sort(key=lambda row: float(row["delay_spread_ns"]))
            ax.plot([float(row["delay_spread_ns"]) for row in subset], [float(row[metric]) for row in subset], marker="o", label=condition)
        ax.set_xscale("log")
        ax.set_xlabel("RMS delay spread (ns)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    (output_dir / "final_results.json").write_text(json.dumps({"step": "STEP 5 comparison", "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text("# STEP 5 Comparison\n\n" + "\n".join(
        f"- {r['condition']} / {r['regime']}: NMSE omitted {float(r['nmse_omitted_db']):.4f}, Aleatoric {float(r['aleatoric_mean']):.6f}, Epistemic {float(r['epistemic_mean']):.6f}"
        for r in rows
    ) + "\n", encoding="utf-8")
    return {"rows": rows}


def run_step6_compare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [(0.0, Path(args.lambda_0)), (1e-4, Path(args.lambda_1e4)), (1e-3, Path(args.lambda_1e3)), (1e-2, Path(args.lambda_1e2))]
    rows: list[dict[str, Any]] = []
    error_bins: list[dict[str, Any]] = []
    for lambda_value, source in sources:
        final = json.loads((source / "final_results.json").read_text(encoding="utf-8"))
        if "epoch_probe_results" in final:
            metrics_by_regime = final["epoch_probe_results"][str(max(map(int, final["epoch_probe_results"]))) ]
        else:
            metrics_by_regime = final["diagnostics"]
        for regime, metrics in metrics_by_regime.items():
            row = step_row(f"lambda={lambda_value:g}", regime, 1_000_000.0 if regime == "OOD-Far 1 ms" else float(regime.split()[1]), metrics)
            row["lambda_reg"] = lambda_value
            rows.append(row)
            for item in metrics.get("error_bin_calibration", []):
                error_bins.append({"lambda_reg": lambda_value, "regime": regime, **item})
    if args.lambda_1e3_error_bins:
        reference = json.loads((Path(args.lambda_1e3_error_bins) / "final_results.json").read_text(encoding="utf-8"))
        existing = {(float(row["lambda_reg"]), row["regime"]) for row in error_bins}
        for regime, metrics in reference.get("diagnostics", {}).items():
            if (1e-3, regime) in existing:
                continue
            for item in metrics.get("error_bin_calibration", []):
                error_bins.append({"lambda_reg": 1e-3, "regime": regime, **item})

    by_lambda = {}
    for row in rows:
        by_lambda.setdefault(float(row["lambda_reg"]), {})[row["regime"]] = row
    gap_rows = []
    for lambda_value, regimes in sorted(by_lambda.items()):
        id20 = regimes["ID-Easy 20 ns"]
        id80 = regimes["ID-Hard 80 ns"]
        near = regimes["OOD-Near 120 ns"]
        far = regimes["OOD-Far 1 ms"]
        ale_gap = float(id80["aleatoric_mean"]) - float(id20["aleatoric_mean"])
        epi_gap_near = float(near["epistemic_mean"]) - max(float(id20["epistemic_mean"]), float(id80["epistemic_mean"]))
        epi_gap_far = float(far["epistemic_mean"]) - max(float(id20["epistemic_mean"]), float(id80["epistemic_mean"]))
        gap_rows.append({
            "lambda_reg": lambda_value,
            "ale_gap_80_20": ale_gap,
            "epi_gap_120_id": epi_gap_near,
            "epi_gap_1ms_id": epi_gap_far,
            "mean_id_nmse_omitted_db": (float(id20["nmse_omitted_db"]) + float(id80["nmse_omitted_db"])) / 2.0,
            "decision": "candidate" if ale_gap > 0 and epi_gap_near > 0 and epi_gap_far > 0 else "not_full_behavior",
        })
    write_csv(output_dir / "lambda_comparison.csv", rows)
    write_csv(output_dir / "lambda_gap_summary.csv", gap_rows)
    write_csv(output_dir / "error_bin_calibration.csv", error_bins)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_gap(filename: str, field: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["lambda_reg"] for r in gap_rows], [r[field] for r in gap_rows], marker="o")
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.set_xlabel("lambda_reg")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    plot_gap("lambda_vs_ale_gap_80_20.png", "ale_gap_80_20", "Aleatoric(80)-Aleatoric(20)")
    plot_gap("lambda_vs_epi_gap_120_id.png", "epi_gap_120_id", "Epistemic(120)-max(ID)")
    plot_gap("lambda_vs_epi_gap_1ms_id.png", "epi_gap_1ms_id", "Epistemic(1 ms)-max(ID)")
    for metric, ylabel, filename in [("nmse_omitted_db", "NMSE omitted (dB)", "lambda_vs_final_nmse_omitted.png"), ("aleatoric_mean", "Aleatoric", "lambda_vs_final_aleatoric.png"), ("epistemic_mean", "Epistemic", "lambda_vs_final_epistemic.png")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        for regime in ("ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"):
            selected = [r for r in rows if r["regime"] == regime]
            selected.sort(key=lambda r: float(r["lambda_reg"]))
            ax.plot([float(r["lambda_reg"]) for r in selected], [float(r[metric]) for r in selected], marker="o", label=regime)
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.set_xlabel("lambda_reg")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    result = {"step": "STEP 6 Lambda / Evidence Calibration", "rows": rows, "gap_rows": gap_rows, "error_bins": error_bins}
    (output_dir / "final_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# STEP 6 Lambda / Evidence Calibration", "", "| Lambda | Ale gap 80-20 | Epi gap 120-ID | Epi gap 1ms-ID | Mean ID NMSE | Decision |", "| ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.extend(f"| {r['lambda_reg']:g} | {r['ale_gap_80_20']:.8f} | {r['epi_gap_120_id']:.8f} | {r['epi_gap_1ms_id']:.8f} | {r['mean_id_nmse_omitted_db']:.4f} | {r['decision']} |" for r in gap_rows)
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="step", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--model-config", default="configs/train_formulation_c_diag_mvnll.json")
        subparser.add_argument("--checkpoint", default="runs/formulation_c_diag_mvnll/uacp_predictor_prototype.pt")
        subparser.add_argument("--dataset-config", default="configs/dataset_prototype.json")
        subparser.add_argument("--samples", type=int, default=200)
        subparser.add_argument("--experiment-name", default="Experiment C + diag MV NLL")

    step1 = subparsers.add_parser("step1-ood-far")
    add_common(step1)
    step1.add_argument("--output-dir", default="runs/baseline_reproduction/step1_ood_far")
    step1.set_defaults(func=run_step1)

    step2 = subparsers.add_parser("step2-delay-sweep")
    add_common(step2)
    step2.add_argument("--delay-spreads-ns", type=float, nargs="+", default=[10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 200.0, 500.0, 1_000_000.0])
    step2.add_argument("--output-dir", default="runs/baseline_reproduction/step2_delay_sweep")
    step2.set_defaults(func=run_step2)

    step3 = subparsers.add_parser("step3-uncertainty-aggregation")
    add_common(step3)
    step3.add_argument("--output-dir", default="runs/baseline_reproduction/step3_uncertainty_aggregation")
    step3.set_defaults(func=run_step3)

    step4a = subparsers.add_parser("step4a-scaled-training-pilot")
    step4a.add_argument("--model-config", default="configs/train_formulation_d_pair_reg.json")
    step4a.add_argument("--dataset-config", default="configs/dataset_prototype.json")
    step4a.add_argument("--train-samples", type=int, default=5000)
    step4a.add_argument("--probe-samples", type=int, default=200)
    step4a.add_argument("--epochs", type=int, default=10)
    step4a.add_argument("--observation-noise-snr-db", type=float, default=None)
    step4a.add_argument("--step-label", default="STEP 4A Scaled-Training Pilot")
    step4a.add_argument("--lambda-reg-override", type=float, default=None)
    step4a.add_argument("--experiment-name", default="STEP-4A-D-5k-10ep-pilot")
    step4a.add_argument("--evidential-mode-override")
    step4a.add_argument("--nll-mode-override")
    step4a.add_argument("--covariance-rank", type=int)
    step4a.add_argument("--estimated-seconds-per-sample-epoch", type=float, default=0.0266)
    step4a.add_argument("--output-dir", default="runs/baseline_reproduction/step4_scaled_training/pilot_5k_10ep")
    step4a.set_defaults(func=run_step4a)

    step4a_finalize = subparsers.add_parser("step4a-finalize")
    step4a_finalize.add_argument("--output-dir", default="runs/baseline_reproduction/step4_scaled_training/pilot_5k_10ep")
    step4a_finalize.set_defaults(func=run_step4a_finalize)

    step5_eval = subparsers.add_parser("step5-evaluate-noisy")
    step5_eval.add_argument("--model-config", default="configs/train_formulation_d_pair_reg.json")
    step5_eval.add_argument("--dataset-config", default="configs/dataset_prototype.json")
    step5_eval.add_argument("--checkpoint", required=True)
    step5_eval.add_argument("--condition", required=True)
    step5_eval.add_argument("--observation-noise-snr-db", type=float, default=15.0)
    step5_eval.add_argument("--clean-eval", action="store_true")
    step5_eval.add_argument("--samples", type=int, default=200)
    step5_eval.add_argument("--output-dir", required=True)
    step5_eval.set_defaults(func=run_step5_evaluation)

    step5_compare = subparsers.add_parser("step5-compare")
    step5_compare.add_argument("--clean-dir", required=True)
    step5_compare.add_argument("--clean-noisy-dir", required=True)
    step5_compare.add_argument("--noisy-dir", required=True)
    step5_compare.add_argument("--noisy-clean-dir")
    step5_compare.add_argument("--output-dir", required=True)
    step5_compare.set_defaults(func=run_step5_compare)

    step6_compare = subparsers.add_parser("step6-compare")
    step6_compare.add_argument("--lambda-0", required=True)
    step6_compare.add_argument("--lambda-1e4", required=True)
    step6_compare.add_argument("--lambda-1e3", required=True)
    step6_compare.add_argument("--lambda-1e2", required=True)
    step6_compare.add_argument("--lambda-1e3-error-bins")
    step6_compare.add_argument("--output-dir", required=True)
    step6_compare.set_defaults(func=run_step6_compare)

    args = parser.parse_args()
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
