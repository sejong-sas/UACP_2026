#!/usr/bin/env python3
"""Train and evaluate the bounded frequency-local covariance treatment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model, diagnose_regime, flatten_diagnostic_row  # noqa: E402
from scripts.train_predictor import load_config, set_seeds, train_one_epoch, evaluate  # noqa: E402
from src.training.data import CFRNPZDataset  # noqa: E402


REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or "NOT RECORDED"
    except OSError:
        return "NOT RECORDED"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(key for key in row if key not in fields)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def covariance_structure_summary(model, path: str, cfg: dict, device: torch.device) -> dict:
    """Summarize learned block-banded Psi without a 2048x2048 matrix."""
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    bandwidth = int(cfg["implementation_assumption"]["covariance_bandwidth"])
    block_size = bandwidth + 1
    lag_values = {lag: [] for lag in (1, 8, 16, 32)}
    energy_values = []
    for batch in loader:
        cfr = batch["cfr"].to(device)
        from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
        mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
        x, _, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)
        factors = output.banded_factor
        block_cov = factors @ factors.transpose(-2, -1)
        diagonal = torch.diagonal(block_cov, dim1=-2, dim2=-1)
        diag_energy = diagonal.square().sum(dim=(-1, -2))
        total_energy = block_cov.square().sum(dim=(-1, -2, -3))
        off_energy = total_energy - diag_energy
        energy_values.extend(torch.stack((diag_energy, off_energy, off_energy / total_energy.clamp_min(1e-12)), dim=-1).reshape(-1, 3).cpu().tolist())
        for lag in lag_values:
            per_lag = []
            for block in range(factors.shape[2]):
                rows = torch.arange(0, block_size - lag, device=device)
                first = factors[:, :, block, 2 * rows, :]
                second = factors[:, :, block, 2 * (rows + lag), :]
                first_im = factors[:, :, block, 2 * rows + 1, :]
                second_im = factors[:, :, block, 2 * (rows + lag) + 1, :]
                # Mean absolute entries of the 2x2 covariance between frequency blocks.
                cov00 = (first * second).sum(-1)
                cov01 = (first * second_im).sum(-1)
                cov10 = (first_im * second).sum(-1)
                cov11 = (first_im * second_im).sum(-1)
                per_lag.append(torch.stack((cov00, cov01, cov10, cov11), dim=-1).abs().mean(dim=(-1, -2)))
            lag_values[lag].extend(torch.cat(per_lag, dim=-1).reshape(-1).cpu().tolist())
    energy = torch.tensor(energy_values, dtype=torch.float64)
    return {
        "diag_energy_mean": float(energy[:, 0].mean()),
        "offdiag_energy_mean": float(energy[:, 1].mean()),
        "offdiag_total_ratio_mean": float(energy[:, 2].mean()),
        "lag_covariance": {str(lag): {"mean": float(torch.tensor(values).mean()), "std": float(torch.tensor(values).std(unbiased=False))} for lag, values in lag_values.items()},
    }


def feasibility(config_path: Path) -> None:
    cfg = load_config(config_path)
    cfg["paper_specified"]["num_subcarriers"] = 1024
    cfg["implementation_assumption"]["covariance_bandwidth"] = 32
    set_seeds(20260819)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    x = torch.randn(1, 9, 1024, device=device)
    target = torch.randn(1, 8, 1024, device=device)
    from src.models.evidential import evidential_loss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(x)
    losses = evidential_loss(output, target, 1e-3, nll_mode="banded_multivariate", reg_mode="pair")
    losses["total"].backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    print(json.dumps({"mode": "feasibility", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "seconds": time.perf_counter() - start, "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None, "finite": bool(torch.isfinite(losses["total"]).item()), "factor_shape": list(output.banded_factor.shape)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/covariance_banded_lag32_5k_10ep.json")
    parser.add_argument("--output-dir", default="runs/current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep")
    parser.add_argument("--feasibility-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    if args.feasibility_only:
        feasibility(config_path); return
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    cfg = load_config(config_path)
    seed = int(cfg["implementation_assumption"]["seed"])
    set_seeds(seed)
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    train_loader = DataLoader(CFRNPZDataset(cfg["data"]["train_path"]), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(CFRNPZDataset(cfg["data"]["validation_path"]), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    code_files = ["src/models/evidential.py", "src/models/uacp_predictor.py", "src/training/data.py", "src/training/uncertainty.py", "scripts/train_predictor.py", "scripts/diagnose_predictor.py", "scripts/train_banded_covariance.py"]
    provenance = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_output(["git", "rev-parse", "HEAD"]), "git_status": git_output(["git", "status", "--short"]), "code_sha256": {name: sha256(ROOT / name) for name in code_files}, "config_sha256": sha256(config_path), "seed": seed, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    history, probe_rows, probe_results = [], [], {}
    start = time.perf_counter()
    for epoch in range(1, 11):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics, "epoch_seconds": time.perf_counter() - epoch_start})
        epoch_probes = {}
        for index, regime in enumerate(REGIMES):
            set_seeds(seed + 92000 + index)
            metrics = diagnose_regime(model, cfg["data"]["test_paths"][regime], cfg, device, observation_noise_snr_db=15.0)
            epoch_probes[regime] = metrics
            row = {"epoch": epoch, **flatten_diagnostic_row(regime, metrics)}
            probe_rows.append(row)
        probe_results[str(epoch)] = epoch_probes
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": val_metrics, "epoch_seconds": history[-1]["epoch_seconds"]}, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - start
    model.eval()
    final_rows = []
    structure_rows = []
    for regime in REGIMES:
        metrics = probe_results["10"][regime]
        final_rows.append({"regime": regime, "nmse_all_db": metrics["nmse_all_db"], "nmse_omitted_db": metrics["nmse_omitted_db"], "aleatoric": metrics["aleatoric_paper_eq12_eq13"], "epistemic": metrics["epistemic_paper_eq12_eq13"], "error_aleatoric_pearson": metrics["error_aleatoric_pearson"], "error_epistemic_pearson": metrics["error_epistemic_pearson"]})
        structure = covariance_structure_summary(model, cfg["data"]["test_paths"][regime], cfg, device)
        structure_rows.append({"regime": regime, **structure, **{f"lag_{lag}_mean": structure["lag_covariance"][str(lag)]["mean"] for lag in (1, 8, 16, 32)}})
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    (output / "evaluation_setup.json").write_text(json.dumps({"test_paths": cfg["data"]["test_paths"], "Ng": 16, "noise_snr_db": 15.0, "probe_seed_offset": 92000}, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "provenance": provenance, "epoch": 10, "experiment_name": cfg["implementation_assumption"]["experiment_name"]}, output / "checkpoint_with_provenance.pt")
    torch.save(model.state_dict(), output / "model_state_dict.pt")
    (output / "training_history.json").write_text(json.dumps({"history": history, "runtime_seconds": elapsed}, indent=2), encoding="utf-8")
    (output / "final_results.json").write_text(json.dumps({"final_results": final_rows, "structure": structure_rows, "epoch_probe_results": probe_results, "provenance": provenance, "runtime_seconds": elapsed, "device": str(device), "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, indent=2), encoding="utf-8")
    write_csv(output / "epoch_probe_results.csv", probe_rows)
    write_csv(output / "final_results.csv", final_rows)
    write_csv(output / "learned_covariance_summary.csv", structure_rows)
    (output / "summary.md").write_text("# Frequency-Local Banded Psi Pilot\n\n" + f"Bandwidth: `{cfg['implementation_assumption']['covariance_bandwidth']}`; device: `{device}`; runtime: `{elapsed:.3f}` s.\n\n" + "| Regime | NMSE omitted | Aleatoric | Epistemic |\n| --- | ---: | ---: | ---: |\n" + "\n".join(f"| {r['regime']} | {r['nmse_omitted_db']:.6f} | {r['aleatoric']:.9f} | {r['epistemic']:.9f} |" for r in final_rows) + "\n\n## Learned covariance\n\n" + "\n".join(f"- {r['regime']}: off/total={r['offdiag_total_ratio_mean']:.6f}, lag32={r['lag_32_mean']:.9f}" for r in structure_rows) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "device": str(device), "gpu": provenance["gpu_name"], "seconds": elapsed, "checkpoint": str(output / "checkpoint_with_provenance.pt")}, indent=2))


if __name__ == "__main__":
    main()
