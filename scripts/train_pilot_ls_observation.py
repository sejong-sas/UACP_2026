#!/usr/bin/env python3
"""Train the fixed UACP model with the pilot-based LS observation pipeline."""

from __future__ import annotations

import argparse
import copy
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

from scripts.baseline_reproduction import epoch_probe_row, step_row, write_csv, write_epoch_curve_pngs  # noqa: E402
from scripts.diagnose_predictor import _make_model, diagnose_regime  # noqa: E402
from scripts.train_predictor import evaluate, load_config, set_seeds, train_one_epoch  # noqa: E402
from src.models.evidential import channels_to_pair_vectors  # noqa: E402
from src.training.data import CFRNPZDataset  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(args: list[str]) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip() or "NOT RECORDED"
    except OSError:
        return "NOT RECORDED"


def mapping_assert() -> dict:
    values = torch.arange(1.0, 9.0).reshape(1, 8, 1).expand(1, 8, 3)
    pairs = channels_to_pair_vectors(values)
    expected = torch.tensor([[[1., 1., 1., 5., 5., 5.], [2., 2., 2., 6., 6., 6.], [3., 3., 3., 7., 7., 7.], [4., 4., 4., 8., 8., 8.]]])
    if not torch.equal(pairs, expected):
        raise AssertionError("Corrected real/imag pair mapping failed")
    return {"layout": "[Re(pair0..3), Im(pair0..3)]", "pair_channels": {f"pair{i}": [i, i + 4] for i in range(4)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/current_valid_baseline_condition_c.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    cfg["implementation_assumption"]["observation_mode"] = "pilot_ls"
    cfg["implementation_assumption"]["observation_noise_note"] = "IMPLEMENTATION-ASSUMPTION: X=I2 orthogonal pilot, received-pilot complex AWGN, LS CFR estimate; clean target"
    cfg["implementation_assumption"]["pilot_sequence"] = "X=I2, two orthogonal pilot slots"
    cfg["implementation_assumption"]["pilot_amplitude"] = 1.0
    cfg["implementation_assumption"]["estimator"] = "least_squares"
    cfg["implementation_assumption"]["experiment_name"] = "OBSERVATION-PILOT-LS-15dB-5k-10ep"
    seed = int(cfg["implementation_assumption"]["seed"])
    set_seeds(seed)
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    code_files = ["src/models/evidential.py", "src/models/uacp_predictor.py", "src/training/data.py", "src/training/uncertainty.py", "scripts/train_predictor.py", "scripts/diagnose_predictor.py", "scripts/train_pilot_ls_observation.py"]
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status": git_output(["git", "status", "--short"]),
        "config_sha256": config_hash,
        "code_sha256": {name: sha256(ROOT / name) for name in code_files},
        "seed": seed,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    train_loader = DataLoader(CFRNPZDataset(cfg["data"]["train_path"]), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(CFRNPZDataset(cfg["data"]["validation_path"]), batch_size=16)
    model = _make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["paper_specified"]["learning_rate"]))
    history, probe_rows, probe_results = [], [], {}
    start = time.perf_counter()
    for epoch in range(1, int(cfg["implementation_assumption"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        epoch_metrics = {}
        for label, delay in REGIMES:
            metrics = diagnose_regime(model, cfg["data"]["test_paths"][label], cfg, device, observation_noise_snr_db=15.0, observation_mode="pilot_ls")
            epoch_metrics[label] = metrics
            row = epoch_probe_row(epoch, label, delay, metrics)
            row["aleatoric_eq12_eq13"] = metrics["aleatoric_paper_eq12_eq13"]
            row["epistemic_eq12_eq13"] = metrics["epistemic_paper_eq12_eq13"]
            probe_rows.append(row)
        probe_results[str(epoch)] = epoch_metrics
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": val_metrics}, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - start
    final_metrics = probe_results[str(cfg["implementation_assumption"]["epochs"])]
    final_rows = []
    for label, delay in REGIMES:
        row = step_row("PILOT-LS-TREATMENT", label, delay, final_metrics[label])
        row["aleatoric_eq12_eq13"] = final_metrics[label]["aleatoric_paper_eq12_eq13"]
        row["epistemic_eq12_eq13"] = final_metrics[label]["epistemic_paper_eq12_eq13"]
        final_rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    cfg["pair_mapping"] = mapping_assert()
    cfg["provenance"] = prov
    cfg["evaluation_protocol"] = {"mask": "uniform_grouping_mask", "Ng": 16, "omitted_polarity": "1-mask", "score": "paper_subcarrier_uncertainty_map then paper_omitted_uncertainty_score"}
    (output / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    (output / "observation_config.json").write_text(json.dumps({"mode": "pilot_ls", "noise_stage": "received pilot Y", "snr_db": 15.0, "target": "clean full CFR", "assumption": "IMPLEMENTATION-ASSUMPTION"}, indent=2), encoding="utf-8")
    (output / "pilot_config.json").write_text(json.dumps({"X": [[1.0, 0.0], [0.0, 1.0]], "slots": 2, "amplitude": 1.0, "orthogonal": True, "assumption": "IMPLEMENTATION-ASSUMPTION"}, indent=2), encoding="utf-8")
    (output / "estimator_config.json").write_text(json.dumps({"estimator": "LS", "formula": "H_hat=Y X^H (X X^H)^-1", "for_X": "I2, H_hat=Y", "assumption": "IMPLEMENTATION-ASSUMPTION"}, indent=2), encoding="utf-8")
    (output / "evaluation_setup.json").write_text(json.dumps({"regimes": REGIMES, "paths": cfg["data"]["test_paths"], "ng": 16, "noise_snr_db": 15.0, "observation_mode": "pilot_ls"}, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "git_commit": prov["git_commit"], "code_hashes": prov["code_sha256"], "seed": seed, "epoch": 10, "experiment_name": cfg["implementation_assumption"]["experiment_name"]}, output / "checkpoint_with_provenance.pt")
    torch.save(model.state_dict(), output / "model_state_dict.pt")
    result = {"experiment": "Observation Pilot LS", "config": cfg, "provenance": prov, "history": history, "epoch_probe_results": probe_results, "final_results": final_rows, "runtime": {"training_probe_seconds": elapsed}, "device": str(device), "gpu": prov["gpu_name"], "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (output / "training_history.json").write_text(json.dumps({"history": history, "runtime": {"training_probe_seconds": elapsed}}, indent=2, sort_keys=True), encoding="utf-8")
    (output / "final_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output / "epoch_probe_results.csv", probe_rows)
    write_csv(output / "summary.csv", final_rows)
    curves = write_epoch_curve_pngs(history, probe_rows, output)
    result["curve_files"] = [str(path) for path in curves]
    (output / "final_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "device": str(device), "gpu": prov["gpu_name"], "seconds": elapsed, "checkpoint": str(output / "checkpoint_with_provenance.pt")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
