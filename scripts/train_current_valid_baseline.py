#!/usr/bin/env python3
"""수정된 Condition C CURRENT VALID BASELINE을 학습한다.

실험 목적:
    pair mapping을 바로잡은 현재 기준 모델을 한 번 학습하고 이후 진단의
    공통 checkpoint로 저장한다.

입력과 출력:
    CFR 데이터와 sparse 관측을 입력으로 받아 checkpoint, 학습 기록,
    20/80/120 ns 및 1 ms 평가 결과를 저장한다.

주의:
    이 파일은 baseline을 새로 만들 때 사용하는 학습 파일이다.
"""

from __future__ import annotations

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

from scripts.baseline_reproduction import (  # noqa: E402
    _make_model,
    epoch_probe_row,
    step_row,
    write_csv,
    write_epoch_curve_pngs,
)
from scripts.diagnose_predictor import diagnose_regime  # noqa: E402
from scripts.train_predictor import evaluate, load_config, set_seeds, train_one_epoch, write_training_csvs  # noqa: E402
from src.models.evidential import channels_to_pair_vectors  # noqa: E402
from src.training.data import CFRNPZDataset, uniform_grouping_mask  # noqa: E402


CONFIG_PATH = ROOT / "configs/current_valid_baseline_condition_c.json"
OUTPUT_DIR = ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected"
REGIMES = [
    ("ID-Easy 20 ns", 20.0),
    ("ID-Hard 80 ns", 80.0),
    ("OOD-Near 120 ns", 120.0),
    ("OOD-Far 1 ms", 1_000_000.0),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "NOT RECORDED"


def provenance(config: dict) -> dict:
    code_files = [
        "src/models/evidential.py",
        "src/models/uacp_predictor.py",
        "src/training/data.py",
        "src/training/uncertainty.py",
        "scripts/train_predictor.py",
        "scripts/diagnose_predictor.py",
        "scripts/train_current_valid_baseline.py",
    ]
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]) or "NOT RECORDED",
        "git_status": command_output(["git", "status", "--short"]),
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "code_sha256": {path: sha256(ROOT / path) for path in code_files},
        "seed": config["implementation_assumption"]["seed"],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def pair_mapping_assert() -> dict:
    values = torch.arange(1.0, 9.0).reshape(1, 8, 1).expand(1, 8, 3)
    pairs = channels_to_pair_vectors(values)
    expected = torch.tensor([
        [[1., 1., 1., 5., 5., 5.], [2., 2., 2., 6., 6., 6.],
         [3., 3., 3., 7., 7., 7.], [4., 4., 4., 8., 8., 8.]],
    ])
    if not torch.equal(pairs, expected):
        raise AssertionError("Corrected real/imag antenna-pair mapping failed.")
    return {"layout": "[Re(pair0..3), Im(pair0..3)]", "pair_channels": {f"pair{i}": [i, i + 4] for i in range(4)}}


def summary_markdown(config: dict, prov: dict, rows: list[dict], runtime: dict) -> str:
    lines = [
        "# CURRENT VALID BASELINE - Condition C",
        "",
        "This run was trained from scratch with the corrected real/imag antenna-pair mapping. It is the current control for later diagnostics.",
        "",
        "## Conditions",
        "",
        "- 5,000 train samples, 10 epochs, batch size 8, Adam, learning rate 1e-4",
        "- Uniform `[10,100] ns` training delay spread, TDL-A channel (`IMPLEMENTATION-ASSUMPTION`)",
        "- 15 dB reported-CFR complex AWGN for both train and evaluation (`IMPLEMENTATION-ASSUMPTION`); clean target",
        "- Experiment D: pair-scalar kappa/nu, diagonal Psi, diagonal multivariate NLL, pair regularizer",
        "- Fixed evaluation grouping `Ng=16`, seed `20260819`, device `cuda:0`",
        "",
        "`SNR=15 dB` is PAPER-SPECIFIED; its exact paper application stage is UNKNOWN. The 5k/10 epoch pilot and TDL-A are not paper-specified.",
        "",
        "## Final Probe (Eq.12 -> Eq.13 scores)",
        "",
        "| Regime | NMSE all | NMSE omitted | Aleatoric | Epistemic | Err-Ale P | Err-Epi P |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['regime']} | {row['nmse_all_db']:.4f} | {row['nmse_omitted_db']:.4f} | {row['aleatoric_eq12_eq13']:.6f} | {row['epistemic_eq12_eq13']:.6f} | {row['error_aleatoric_pearson']:.4f} | {row['error_epistemic_pearson']:.4f} |")
    lines += [
        "",
        "## Provenance",
        "",
        f"- Git commit: `{prov['git_commit']}`",
        f"- Config SHA-256: `{prov['config_sha256']}`",
        f"- GPU: `{prov['gpu_name']}`; CUDA available: `{prov['cuda_available']}`",
        f"- Training/probe seconds: `{runtime['training_probe_seconds']:.3f}`",
        "- Wrapped checkpoint: `checkpoint_with_provenance.pt`",
        "- Compatibility state dict: `model_state_dict.pt`",
        "",
        "## Known Approximation",
        "",
        "The paper specifies full pair covariance `Psi=L L^T`; this baseline uses diagonal `Psi` and must be treated as an APPROXIMATION. Historical STEP 6 Condition C is pre-fix and remains invalid for uncertainty comparison.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    config = load_config(CONFIG_PATH)
    mapping = pair_mapping_assert()
    set_seeds(int(config["implementation_assumption"]["seed"]))
    device = torch.device(config["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    prov = provenance(config)

    train_data = CFRNPZDataset(config["data"]["train_path"])
    val_data = CFRNPZDataset(config["data"]["validation_path"])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(20260819))
    val_loader = DataLoader(val_data, batch_size=16)
    model = _make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    probe_paths = {label: config["data"]["test_paths"][label] for label, _ in REGIMES}

    history = []
    probe_rows = []
    probe_results = {}
    start = time.perf_counter()
    for epoch in range(1, 11):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device)
        val_metrics = evaluate(model, val_loader, config, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        epoch_metrics = {}
        for label, delay in REGIMES:
            metrics = diagnose_regime(model, probe_paths[label], config, device, observation_noise_snr_db=15.0)
            epoch_metrics[label] = metrics
            row = epoch_probe_row(epoch, label, delay, metrics)
            row["aleatoric_eq12_eq13"] = metrics["aleatoric_paper_eq12_eq13"]
            row["epistemic_eq12_eq13"] = metrics["epistemic_paper_eq12_eq13"]
            probe_rows.append(row)
        probe_results[str(epoch)] = epoch_metrics
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": val_metrics}, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - start

    final_metrics = probe_results["10"]
    final_rows = []
    for label, delay in REGIMES:
        metrics = final_metrics[label]
        row = step_row("CURRENT-VALID-BASELINE", label, delay, metrics)
        row["aleatoric_eq12_eq13"] = metrics["aleatoric_paper_eq12_eq13"]
        row["epistemic_eq12_eq13"] = metrics["epistemic_paper_eq12_eq13"]
        final_rows.append(row)
    runtime = {"training_probe_seconds": elapsed}
    config["provenance"] = prov
    config["pair_mapping"] = mapping
    config["evaluation_protocol"] = {"mask": "uniform_grouping_mask", "Ng": 16, "omitted_polarity": "1-mask", "score": "paper_subcarrier_uncertainty_map then paper_omitted_uncertainty_score"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "evaluation_setup.json").write_text(json.dumps({"regimes": REGIMES, "paths": probe_paths, "ng": 16, "noise_snr_db": 15.0, "mask_indices": list(range(0, 1024, 16)), "mask_seed": 20260819}, indent=2, sort_keys=True), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "git_commit": prov["git_commit"], "code_hashes": prov["code_sha256"], "seed": 20260819, "epoch": 10, "experiment_name": config["implementation_assumption"]["experiment_name"]}, OUTPUT_DIR / "checkpoint_with_provenance.pt")
    torch.save(model.state_dict(), OUTPUT_DIR / "model_state_dict.pt")
    final = {"step": "CURRENT VALID BASELINE Condition C", "config": config, "provenance": prov, "history": history, "epoch_probe_results": probe_results, "final_results": final_rows, "runtime": runtime, "device": str(device), "total_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps({"history": history, "runtime": runtime}, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "final_results.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(OUTPUT_DIR / "epoch_probe_results.csv", probe_rows)
    write_csv(OUTPUT_DIR / "summary.csv", final_rows)
    write_training_csvs(OUTPUT_DIR, history, {label: {"nmse_all_db": final_metrics[label]["nmse_all_db"], "nmse_omitted_db": final_metrics[label]["nmse_omitted_db"], "aleatoric": final_metrics[label]["aleatoric_paper_eq12_eq13"], "epistemic": final_metrics[label]["epistemic_paper_eq12_eq13"], "nll": final_metrics[label]["nll"], "reg": final_metrics[label]["raw_l_reg"], "lambda_reg_x_reg": final_metrics[label]["lambda_reg_x_l_reg"]} for label, _ in REGIMES})
    curves = write_epoch_curve_pngs(history, probe_rows, OUTPUT_DIR)
    final["curve_files"] = [str(p) for p in curves]
    (OUTPUT_DIR / "final_results.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "summary.md").write_text(summary_markdown(config, prov, final_rows, runtime), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "device": str(device), "gpu": prov["gpu_name"], "seconds": elapsed, "checkpoint": str(OUTPUT_DIR / "checkpoint_with_provenance.pt")}, indent=2))


if __name__ == "__main__":
    main()
