#!/usr/bin/env python3
"""Per-sample κ-gradient diagnosis using the repository's actual evidential loss."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
DATA = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
OUT = ROOT / "runs/current_valid_baseline/epoch3_kappa_loss_gradient_diagnosis_20260920"
REGIMES = [80, 120]
NGS = [16, 32]
SAMPLES = 200
BATCH_SIZE = 8
SEED = 20262000
K = 1024
METRICS = ["nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic"]


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)), "q05": float(np.quantile(values, .05)), "q50": float(np.quantile(values, .50)), "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99)), "max": float(values.max())}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import _build_observation, evidential_loss, load_config, set_seeds
    from src.models.evidential import EvidentialOutput
    from src.training.data import cfr_to_real_imag
    from src.training.metrics import nmse_omitted_db

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10 / cuda:0 is required")
    config = load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json")
    model = _make_model(config, device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state)
    model.eval()
    config["implementation_assumption"]["nll_mode"] = "diagonal_multivariate"
    config["implementation_assumption"]["reg_mode"] = "pair"
    lambda_reg = float(config["paper_specified"]["lambda_reg"])

    raw_capture = {}
    def capture(_module, _inputs, output):
        raw_capture["raw_kappa"] = output
    hook = model.kappa_head.register_forward_hook(capture)
    rows = []
    for regime_index, delay in enumerate(REGIMES):
        with np.load(DATA / f"test_delay_{delay}_ns.npz") as archive:
            cfr_all = np.asarray(archive["cfr"][:SAMPLES], dtype=np.complex64)
        for ng in NGS:
            for start in range(0, SAMPLES, BATCH_SIZE):
                end = min(start + BATCH_SIZE, SAMPLES)
                set_seeds(SEED + regime_index * 100000 + ng + start)
                cfr = torch.from_numpy(cfr_all[start:end]).to(device)
                mask = torch.zeros((len(cfr), K), dtype=torch.float32, device=device)
                mask[:, ::ng] = 1.0
                x, target, _ = _build_observation(cfr, mask, config)
                model.zero_grad(set_to_none=True)
                raw_capture.clear()
                output = model(x)
                raw_kappa = raw_capture["raw_kappa"]
                for local in range(end - start):
                    index = start + local
                    out_one = EvidentialOutput(gamma=output.gamma[local:local + 1], kappa=output.kappa[local:local + 1], psi=output.psi[local:local + 1], nu=output.nu[local:local + 1], num_subcarriers=output.num_subcarriers)
                    target_one = target[local:local + 1]
                    losses = evidential_loss(out_one, target_one, lambda_reg, nll_mode="diagonal_multivariate", reg_mode="pair")
                    gradients = {}
                    for loss_name, loss_value in (("nll", losses["nll"]), ("reg", losses["lambda_reg_x_reg"]), ("total", losses["total"])):
                        grad_kappa, grad_raw_full = torch.autograd.grad(loss_value, (out_one.kappa, raw_kappa), retain_graph=True, allow_unused=False)
                        grad_raw = grad_raw_full[local:local + 1]
                        kappa_values = grad_kappa.detach().reshape(-1).cpu().numpy()
                        raw_values = grad_raw.detach().reshape(-1).cpu().numpy()
                        gradients[f"{loss_name}_kappa_grad_mean"] = float(kappa_values.mean())
                        gradients[f"{loss_name}_kappa_grad_abs_mean"] = float(np.abs(kappa_values).mean())
                        gradients[f"{loss_name}_kappa_grad_q05"] = float(np.quantile(kappa_values, .05))
                        gradients[f"{loss_name}_kappa_grad_q95"] = float(np.quantile(kappa_values, .95))
                        gradients[f"{loss_name}_raw_kappa_grad_mean"] = float(raw_values.mean())
                        gradients[f"{loss_name}_raw_kappa_grad_abs_mean"] = float(np.abs(raw_values).mean())
                        gradients[f"{loss_name}_lower_kappa_fraction"] = float(np.mean(kappa_values > 0.0))
                        gradients[f"{loss_name}_higher_kappa_fraction"] = float(np.mean(kappa_values < 0.0))
                    omitted = (1.0 - mask[local:local + 1])
                    ale_map = out_one.aleatoric
                    epi_map = out_one.epistemic
                    omitted_expanded = omitted[:, None, :].expand_as(ale_map)
                    ale = (ale_map * omitted_expanded).sum() / omitted_expanded.sum().clamp_min(1.0)
                    epi = (epi_map * omitted_expanded).sum() / omitted_expanded.sum().clamp_min(1.0)
                    params = {
                        "nmse": float(nmse_omitted_db(out_one.predicted, target_one, mask[local:local + 1]).detach().cpu()),
                        "psi": float((out_one.psi * omitted_expanded).sum().detach().cpu() / omitted_expanded.sum().clamp_min(1.0).detach().cpu()),
                        "kappa": float((out_one.kappa_expanded * omitted_expanded).sum().detach().cpu() / omitted_expanded.sum().clamp_min(1.0).detach().cpu()),
                        "nu_margin": float(((out_one.nu_expanded - (2 * K + 1)) * omitted_expanded).sum().detach().cpu() / omitted_expanded.sum().clamp_min(1.0).detach().cpu()),
                        "aleatoric": float(ale.detach().cpu()), "epistemic": float(epi.detach().cpu()),
                    }
                    row = {"delay_ns": delay, "ng": ng, "sample": index, **params, **gradients}
                    rows.append(row)
                del cfr, mask, x, target, output
                torch.cuda.empty_cache()
    hook.remove()
    write_csv(OUT / "per_sample_gradients.csv", rows)

    summary_rows = []
    for ng in NGS:
        for delay in REGIMES:
            subset = [row for row in rows if row["ng"] == ng and row["delay_ns"] == delay]
            row = {"ng": ng, "delay_ns": delay}
            for metric in METRICS:
                for key, value in summary([r[metric] for r in subset]).items():
                    row[f"{metric}_{key}"] = value
            for loss_name in ["nll", "reg", "total"]:
                for grad_name in ["kappa_grad_mean", "kappa_grad_abs_mean", "kappa_grad_q05", "kappa_grad_q95", "raw_kappa_grad_mean", "raw_kappa_grad_abs_mean", "lower_kappa_fraction", "higher_kappa_fraction"]:
                    values = [r[f"{loss_name}_{grad_name}"] for r in subset]
                    row[f"{loss_name}_{grad_name}_mean"] = float(np.mean(values))
                    row[f"{loss_name}_{grad_name}_median"] = float(np.median(values))
            summary_rows.append(row)
    write_csv(OUT / "gradient_summary.csv", summary_rows)
    ratio_rows = []
    for ng in NGS:
        id_row = next(r for r in summary_rows if r["ng"] == ng and r["delay_ns"] == 80)
        ood_row = next(r for r in summary_rows if r["ng"] == ng and r["delay_ns"] == 120)
        row = {"ng": ng}
        for loss_name in ["nll", "reg", "total"]:
            for grad_name in ["kappa_grad_abs_mean", "raw_kappa_grad_abs_mean"]:
                row[f"{loss_name}_{grad_name}_120_over_80"] = ood_row[f"{loss_name}_{grad_name}_mean"] / id_row[f"{loss_name}_{grad_name}_mean"]
        ratio_rows.append(row)
    write_csv(OUT / "gradient_ratio_120_over_80.csv", ratio_rows)
    manifest = {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "device": str(device), "gpu": torch.cuda.get_device_name(0), "regimes_ns": REGIMES, "ngs": NGS, "samples_per_condition": SAMPLES, "batch_size": BATCH_SIZE, "lambda_reg": lambda_reg, "nll_mode": "diagonal_multivariate", "reg_mode": "pair", "loss_source": "src/models/evidential.py:evidential_loss", "optimizer_step": False, "gradient_interpretation": "positive gradient means gradient descent lowers kappa; negative gradient means gradient descent raises kappa", "per_sample_loss_scaling": "Each sample loss was evaluated separately; training batch averaging only changes magnitude by a positive batch factor, not sign."}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
