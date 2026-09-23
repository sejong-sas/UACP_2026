#!/usr/bin/env python3
"""Read-only Psi/Nu counterfactual diagnostic for the current-valid baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors, expand_pair_scalar, pair_vectors_to_channels  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402
from src.training.uncertainty import paper_omitted_uncertainty_score, paper_subcarrier_uncertainty_map  # noqa: E402


REGIMES = ("ID-Easy 20 ns", "ID-Hard 80 ns")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counterfactual_score(psi_norm: torch.Tensor, nu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the existing pair mapping and Eq.12 -> Eq.13 reduction."""
    nu_pair = nu.expand(-1, -1, psi_norm.shape[-1])
    aleatoric_pair = psi_norm / (nu_pair - 2 * 1024 - 1)
    aleatoric_channels = pair_vectors_to_channels(aleatoric_pair, 1024)
    score_map = paper_subcarrier_uncertainty_map(aleatoric_channels)
    omitted = 1.0 - mask
    return (score_map * omitted).sum(dim=-1) / omitted.sum(dim=-1).clamp_min(1.0)


def evaluate_state(model, path: str, cfg: dict, device: torch.device, seed: int) -> dict[str, torch.Tensor | float | int]:
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    psi_rows, nu_rows, scale_rows, mask_rows, actual_rows, pair_rows, check_rows = [], [], [], [], [], [], []
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
        set_seeds(seed + batch_index)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)
        scale = cfr.abs().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-12)
        psi = channels_to_pair_vectors(output.psi)
        psi_norm = psi / scale.square()[:, None, None]
        actual_norm = output.aleatoric / scale.square()[:, None, None]
        actual_score = paper_omitted_uncertainty_score(actual_norm, mask)
        actual_score = (paper_subcarrier_uncertainty_map(actual_norm) * (1.0 - mask)).sum(dim=-1) / (1.0 - mask).sum(dim=-1).clamp_min(1.0)
        custom_score = counterfactual_score(psi_norm, output.nu, mask)
        denom = output.nu.expand(-1, -1, psi_norm.shape[-1]) - 2 * 1024 - 1
        pair_ale = psi_norm / denom
        omitted = (1.0 - mask).unsqueeze(1)
        pair_actual = pair_ale[:, :, :1024] + pair_ale[:, :, 1024:]
        pair_actual = (pair_actual * omitted).sum(dim=-1) / omitted.sum(dim=-1).clamp_min(1.0)
        psi_rows.append(psi_norm.detach().cpu())
        nu_rows.append(output.nu.detach().cpu())
        scale_rows.append(scale.detach().cpu())
        mask_rows.append(mask.detach().cpu())
        actual_rows.append(actual_score.detach().cpu())
        pair_rows.append(pair_actual.detach().cpu())
        check_rows.append((custom_score - actual_score).detach().cpu())
    return {
        "psi_norm": torch.cat(psi_rows),
        "nu": torch.cat(nu_rows),
        "scale": torch.cat(scale_rows),
        "mask": torch.cat(mask_rows),
        "actual": torch.cat(actual_rows),
        "pair_actual": torch.cat(pair_rows),
        "custom_check": torch.cat(check_rows),
    }


def mixed_score(
    psi_state: dict[str, torch.Tensor],
    psi_indices: np.ndarray,
    nu_state: dict[str, torch.Tensor],
    nu_indices: np.ndarray,
    mask_state: dict[str, torch.Tensor],
    mask_indices: np.ndarray,
) -> np.ndarray:
    psi = psi_state["psi_norm"][torch.as_tensor(psi_indices)]
    nu = nu_state["nu"][torch.as_tensor(nu_indices)]
    mask = mask_state["mask"][torch.as_tensor(mask_indices)]
    return counterfactual_score(psi, nu, mask).numpy()


def parameter_stats(label: str, state: dict[str, torch.Tensor]) -> list[dict]:
    psi = state["psi_norm"]
    nu = state["nu"]
    denom = nu - 2 * 1024 - 1
    pair_psi = psi.reshape(psi.shape[0], 4, 2, 1024).mean(dim=(2, 3))
    pair_ale = state["pair_actual"]
    rows = [{
        "regime": label,
        "scope": "overall",
        "pair": "all",
        "samples": int(psi.shape[0]),
        "normalized_psi_mean": float(psi.mean()),
        "normalized_psi_median": float(psi.median()),
        "nu_mean": float(nu.mean()),
        "denominator_mean": float(denom.mean()),
        "actual_normalized_aleatoric_mean": float(state["actual"].mean()),
        "actual_normalized_aleatoric_median": float(state["actual"].median()),
    }]
    for pair in range(4):
        rows.append({
            "regime": label,
            "scope": "pair",
            "pair": pair,
            "samples": int(psi.shape[0]),
            "normalized_psi_mean": float(pair_psi[:, pair].mean()),
            "normalized_psi_median": float(pair_psi[:, pair].median()),
            "nu_mean": float(nu[:, pair, 0].mean()),
            "denominator_mean": float(denom[:, pair, 0].mean()),
            "actual_normalized_aleatoric_mean": float(pair_ale[:, pair].mean()),
            "actual_normalized_aleatoric_median": float(pair_ale[:, pair].median()),
        })
    return rows


def make_plots(summary: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["A20,20", "A80,20", "A20,80", "A80,80"]
    values = [summary[0]["mean"], summary[1]["mean"], summary[2]["mean"], summary[3]["mean"]]
    errors = [summary[0]["std"], summary[1]["std"], summary[2]["std"], summary[3]["std"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.set_ylabel("Normalized Aleatoric")
    ax.set_title("Psi/Nu counterfactual states")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "counterfactual_aleatoric_bar.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["C_Psi", "C_Nu"], [summary[4]["mean"], summary[5]["mean"]], yerr=[summary[4]["std"], summary[5]["std"]], capsize=4)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("Symmetric contribution")
    ax.set_title("Psi vs Nu contribution")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "symmetric_contribution_bar.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json")
    parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--permutation-seed", type=int, default=20260911)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    cfg = load_config(ROOT / args.config)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint)
    model.eval()

    paths = {
        "ID-Easy 20 ns": cfg["data"]["test_paths"]["ID-Easy 20 ns"],
        "ID-Hard 80 ns": cfg["data"]["test_paths"]["ID-Hard 80 ns"],
    }
    states = {
        "ID-Easy 20 ns": evaluate_state(model, paths["ID-Easy 20 ns"], cfg, device, 97000),
        "ID-Hard 80 ns": evaluate_state(model, paths["ID-Hard 80 ns"], cfg, device, 98000),
    }
    n = int(states[REGIMES[0]]["psi_norm"].shape[0])
    if n != int(states[REGIMES[1]]["psi_norm"].shape[0]):
        raise ValueError("Expected equal 20 ns and 80 ns sample counts.")

    rng = np.random.default_rng(args.permutation_seed)
    permutations = [rng.permutation(n) for _ in range(args.permutations)]
    records = []
    for perm_id, perm in enumerate(permutations):
        i = np.arange(n)
        actual20 = mixed_score(states[REGIMES[0]], i, states[REGIMES[0]], i, states[REGIMES[0]], i)
        psi80_nu20 = mixed_score(states[REGIMES[1]], perm, states[REGIMES[0]], i, states[REGIMES[0]], i)
        psi20_nu80 = mixed_score(states[REGIMES[0]], i, states[REGIMES[1]], perm, states[REGIMES[0]], i)
        actual80 = mixed_score(states[REGIMES[1]], perm, states[REGIMES[1]], perm, states[REGIMES[1]], perm)
        for sample in range(n):
            records.append({
                "permutation": perm_id,
                "easy_sample": sample,
                "hard_sample": int(perm[sample]),
                "A20_20": float(actual20[sample]),
                "A80_20": float(psi80_nu20[sample]),
                "A20_80": float(psi20_nu80[sample]),
                "A80_80": float(actual80[sample]),
                "delta_psi_given_20": float(psi80_nu20[sample] - actual20[sample]),
                "delta_nu_given_20": float(psi20_nu80[sample] - actual20[sample]),
                "delta_actual_80_minus_20": float(actual80[sample] - actual20[sample]),
            })

    fields = ("A20_20", "A80_20", "A20_80", "A80_80", "delta_psi_given_20", "delta_nu_given_20", "delta_actual_80_minus_20")
    summary = []
    for key in fields:
        values = np.asarray([row[key] for row in records], dtype=float)
        summary.append({"quantity": key, "mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values)), "positive_ratio": float(np.mean(values > 0)), "records": int(values.size)})
    a20 = summary[0]["mean"]; a80 = summary[3]["mean"]
    c_psi = .5 * (summary[1]["mean"] - a20 + a80 - summary[2]["mean"])
    c_nu = .5 * (summary[2]["mean"] - a20 + a80 - summary[1]["mean"])
    summary.extend([
        {"quantity": "C_Psi_symmetric", "mean": c_psi, "std": float(np.std([.5 * (r["A80_20"]-r["A20_20"] + r["A80_80"]-r["A20_80"]) for r in records])), "median": 0.0, "positive_ratio": 0.0, "records": len(records)},
        {"quantity": "C_Nu_symmetric", "mean": c_nu, "std": float(np.std([.5 * (r["A20_80"]-r["A20_20"] + r["A80_80"]-r["A80_20"]) for r in records])), "median": 0.0, "positive_ratio": 0.0, "records": len(records)},
    ])

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "counterfactual_results.csv", records)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "parameter_stats.csv", parameter_stats(REGIMES[0], states[REGIMES[0]]) + parameter_stats(REGIMES[1], states[REGIMES[1]]))
    state_rows = []
    for regime, state in states.items():
        for idx in range(n):
            state_rows.append({"regime": regime, "sample": idx, "scale_a": float(state["scale"][idx]), "actual_normalized_aleatoric": float(state["actual"][idx]), "custom_vs_actual_max_check": float(abs(state["custom_check"][idx]))})
    write_csv(out / "state_sample_summary.csv", state_rows)
    make_plots(summary, out)

    pairing = {
        "status": "independent_datasets_not_indexwise_paired",
        "reason": "generate_dataset.py uses regime-dependent seeds seed+3001+index*1000; no shared base realization metadata exists",
        "method": "deterministic permutation pairing without replacement",
        "permutations": args.permutations,
        "permutation_seed": args.permutation_seed,
        "assumption_label": "IMPLEMENTATION-ASSUMPTION",
    }
    provenance = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_performed": False,
        "optimizer_step_called": False,
        "observation_snr_db": 15.0,
        "eval_grouping_factor": 16,
        "normalization": "a=sqrt(mean(|H_true|^2)); Psi_norm=Psi/a^2; A_norm=Psi_norm/(nu-2K-1)",
        "pair_mapping": "[Re(pair0..3), Im(pair0..3)] -> pairs (0,4),(1,5),(2,6),(3,7)",
        "aggregation": "existing paper_subcarrier_uncertainty_map -> omitted Eq.13 score",
        "pairing": pairing,
    }
    analysis = {
        "hypothesis": "separate the contribution of normalized Psi state change and nu state change to the 20-vs-80 normalized Aleatoric gap",
        "definitions": {
            "A20_20": "F(Psi20_norm, nu20)",
            "A80_20": "F(Psi80_norm, nu20)",
            "A20_80": "F(Psi20_norm, nu80)",
            "A80_80": "F(Psi80_norm, nu80)",
            "C_Psi": "0.5*((A80_20-A20_20)+(A80_80-A20_80))",
            "C_Nu": "0.5*((A20_80-A20_20)+(A80_80-A80_20))",
        },
        "identity_mean_residual": float((a80 - a20) - c_psi - c_nu),
        "state_check_max_abs": float(max(abs(float(x)) for state in states.values() for x in state["custom_check"])),
        "summary": summary,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": provenance["gpu"], "pairing": pairing, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
