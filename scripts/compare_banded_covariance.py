#!/usr/bin/env python3
"""Read-only comparison artifacts for the diagonal control and banded pilot."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_predictor import _make_model, diagnose_regime  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402

BASE = ROOT / "runs/current_valid_baseline"
CONTROL = BASE / "condition_c_5k_10ep_lambda1e3_corrected"
TREATMENT = BASE / "covariance_experiments/banded_lag32_5k_10ep"
OUT = TREATMENT / "comparison"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    control_cfg = load_config(CONTROL / "config.json")
    treatment_cfg = load_config(TREATMENT / "config.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    control = _make_model(control_cfg, device)
    treatment = _make_model(treatment_cfg, device)
    control.load_state_dict(torch.load(CONTROL / "model_state_dict.pt", map_location=device, weights_only=False))
    treatment.load_state_dict(torch.load(TREATMENT / "model_state_dict.pt", map_location=device, weights_only=False))
    control.eval(); treatment.eval()
    rows = []
    for name, model, cfg in (("Diagonal current-valid control", control, control_cfg), ("Banded lag32", treatment, treatment_cfg)):
        for index, regime in enumerate(REGIMES):
            # Same seed protocol for both models makes this a paired read-only comparison.
            set_seeds(20260819 + 92000 + index)
            metrics = diagnose_regime(model, cfg["data"]["test_paths"][regime], cfg, device, observation_noise_snr_db=15.0)
            rows.append({"model": name, "regime": regime, "nmse_all_db": metrics["nmse_all_db"], "nmse_omitted_db": metrics["nmse_omitted_db"], "aleatoric": metrics["aleatoric_paper_eq12_eq13"], "epistemic": metrics["epistemic_paper_eq12_eq13"], "error_aleatoric_pearson": metrics["error_aleatoric_pearson"], "error_epistemic_pearson": metrics["error_epistemic_pearson"]})
    by_model = {name: {row["regime"]: row for row in rows if row["model"] == name} for name in ("Diagonal current-valid control", "Banded lag32")}
    gaps = []
    for name, values in by_model.items():
        id_max = max(values[REGIMES[0]]["epistemic"], values[REGIMES[1]]["epistemic"])
        gaps.append({"model": name, "ale_gap_80_20": values[REGIMES[1]]["aleatoric"] - values[REGIMES[0]]["aleatoric"], "epi_gap_120_id": values[REGIMES[2]]["epistemic"] - id_max, "epi_gap_1ms_id": values[REGIMES[3]]["epistemic"] - id_max, "mean_id_nmse_omitted_db": (values[REGIMES[0]]["nmse_omitted_db"] + values[REGIMES[1]]["nmse_omitted_db"]) / 2})
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "condition_comparison.csv", rows)
    write_csv(OUT / "gap_summary.csv", gaps)
    structure = json.loads((TREATMENT / "final_results.json").read_text())
    learned_rows = [{"model": "Banded lag32", "regime": row["regime"], "diag_energy": row["diag_energy_mean"], "offdiag_energy": row["offdiag_energy_mean"], "offdiag_total_ratio": row["offdiag_total_ratio_mean"], "lag_1": row["lag_1_mean"], "lag_8": row["lag_8_mean"], "lag_16": row["lag_16_mean"], "lag_32": row["lag_32_mean"]} for row in structure["structure"]]
    write_csv(OUT / "learned_covariance.csv", learned_rows)
    control_params = sum(p.numel() for p in control.parameters()); treatment_params = sum(p.numel() for p in treatment.parameters())
    cost_rows = []
    for name, model, params in (("Diagonal current-valid control", control, control_params), ("Banded lag32", treatment, treatment_params)):
        x = torch.zeros(1, 9, 1024, device=device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        for _ in range(2):
            with torch.no_grad(): model(x)
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(5):
            with torch.no_grad(): model(x)
        if device.type == "cuda": torch.cuda.synchronize(device)
        cost_rows.append({"model": name, "total_parameters": params, "parameter_increase_vs_diagonal": params - control_params, "inference_ms": (time.perf_counter() - start) * 1000 / 5, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None})
    write_csv(OUT / "cost_comparison.csv", cost_rows)
    historical_root = ROOT / "runs/baseline_reproduction/step7_covariance_structure/comparison"
    historical_rows = []
    if (historical_root / "covariance_comparison.csv").exists():
        with (historical_root / "covariance_comparison.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                historical_rows.append({"model": "Historical STEP7 rank4", "regime": row["regime"], "nmse_omitted_db": row["nmse_omitted_db"], "aleatoric": row["aleatoric_mean"], "epistemic": row["epistemic_mean"], "off_diagonal_energy_ratio": row["off_diagonal_energy_ratio_mean"]})
    write_csv(OUT / "historical_rank4_comparison.csv", historical_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, filename, ylabel in (("nmse_omitted_db", "nmse_comparison.png", "NMSE omitted (dB)"), ("aleatoric", "aleatoric_comparison.png", "Aleatoric"), ("epistemic", "epistemic_comparison.png", "Epistemic")):
        fig, ax = plt.subplots(figsize=(9, 5)); positions = range(len(REGIMES)); width = .35
        for offset, name in enumerate(("Diagonal current-valid control", "Banded lag32")):
            vals = [by_model[name][regime][metric] for regime in REGIMES]
            ax.bar([p + (offset - .5) * width for p in positions], vals, width, label=name)
        ax.set_xticks(list(positions), REGIMES, rotation=20); ax.set_ylabel(ylabel); ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(OUT / filename, dpi=140); plt.close(fig)
    with (TREATMENT / "epoch_probe_results.csv").open(newline="", encoding="utf-8") as handle:
        history_rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for regime in REGIMES:
        values = [row for row in history_rows if row["regime"] == regime]
        epochs = [int(row["epoch"]) for row in values]
        axes[0].plot(epochs, [float(row["nmse_omitted_db"]) for row in values], marker="o", label=regime)
        axes[1].plot(epochs, [float(row["aleatoric_omitted_mean"]) for row in values], marker="o", label=regime)
        axes[2].plot(epochs, [float(row["epistemic_omitted_mean"]) for row in values], marker="o", label=regime)
    for axis, title in zip(axes, ("NMSE omitted (dB)", "Aleatoric", "Epistemic")):
        axis.set_title(title); axis.set_xlabel("Epoch"); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "epoch_probe_curves.png", dpi=140); plt.close(fig)
    treatment_final = json.loads((TREATMENT / "final_results.json").read_text())
    (OUT / "resource_summary.json").write_text(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "training_seconds": treatment_final["runtime_seconds"], "training_peak_memory_recorded": False, "feasibility_peak_memory_mb": 200.3330078125, "paired_inference_cost": cost_rows, "note": "Training peak memory was not captured by the original training process; feasibility and paired inference measurements are recorded."}, indent=2), encoding="utf-8")
    summary = ["# Diagonal vs Banded lag32", "", "Paired read-only evaluation uses the same deterministic probe seeds.", "", "| Model | Regime | NMSE omitted | Aleatoric | Epistemic |", "| --- | --- | ---: | ---: | ---: |"]
    summary += [f"| {r['model']} | {r['regime']} | {r['nmse_omitted_db']:.6f} | {r['aleatoric']:.9f} | {r['epistemic']:.9f} |" for r in rows]
    summary += ["", "| Model | Ale 80-20 | Epi 120-ID | Epi 1ms-ID | Mean ID NMSE |", "| --- | ---: | ---: | ---: | ---: |"]
    summary += [f"| {r['model']} | {r['ale_gap_80_20']:.9g} | {r['epi_gap_120_id']:.9g} | {r['epi_gap_1ms_id']:.9g} | {r['mean_id_nmse_omitted_db']:.6f} |" for r in gaps]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({"control": str(CONTROL), "treatment": str(TREATMENT), "paired_probe_seed_base": 20260819 + 92000, "device": str(device)}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
