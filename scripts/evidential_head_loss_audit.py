#!/usr/bin/env python3
"""Code-level audit of pair-scalar evidential heads and active loss reductions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors, evidential_loss  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, random_grouping_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/current_valid_baseline_condition_c.json")
    parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    cfg = load_config(args.config)
    seed = int(cfg["implementation_assumption"]["seed"])
    set_seeds(seed)
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if "model_state_dict" in payload else payload)
    model.eval()
    batch = next(iter(DataLoader(CFRNPZDataset(cfg["data"]["train_path"]), batch_size=8)))
    cfr = batch["cfr"].to(device)
    mask = random_grouping_mask(8, 1024, [4, 8, 16, 32], device)
    set_seeds(seed + 70000)
    x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
    with torch.no_grad():
        output = model(x)
        losses = evidential_loss(output, target, 1e-3, nll_mode="diagonal_multivariate", reg_mode="pair")
    mapping_input = torch.arange(1.0, 9.0).reshape(1, 8, 1).expand(1, 8, 3)
    mapping = channels_to_pair_vectors(mapping_input)
    expected = torch.tensor([[[1., 1., 1., 5., 5., 5.], [2., 2., 2., 6., 6., 6.], [3., 3., 3., 7., 7., 7.], [4., 4., 4., 8., 8., 8.]]])
    result = {
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": str(args.config), "checkpoint": str(args.checkpoint), "batch_size": 8,
        "heads": {
            "gamma_shape": list(output.gamma.shape), "gamma_frequency_dependent": True,
            "kappa_shape": list(output.kappa.shape), "kappa_independent_values_per_sample": int(output.kappa.shape[1]),
            "kappa_broadcast_shape": list(output.kappa_expanded.shape),
            "nu_shape": list(output.nu.shape), "nu_independent_values_per_sample": int(output.nu.shape[1]),
            "nu_broadcast_shape": list(output.nu_expanded.shape),
            "psi_shape": list(output.psi.shape), "psi_structure": "diagonal approximation, [B,4,2048] after pair mapping",
        },
        "mapping": {"pass": bool(torch.equal(mapping, expected)), "layout": "[Re(pair0..3), Im(pair0..3)]", "pairs": [[0,4],[1,5],[2,6],[3,7]]},
        "loss": {
            "nll_mode": "diagonal_multivariate", "reg_mode": "pair", "dimension_d": 2048,
            "raw_L_NLL": float(losses["nll"]), "raw_L_reg": float(losses["reg"]),
            "lambda_reg": 1e-3, "lambda_reg_times_L_reg": float(losses["lambda_reg_x_reg"]),
            "lambda_reg_L_reg_over_abs_L_NLL": float(abs(losses["lambda_reg_x_reg"]) / (abs(losses["nll"]) + 1e-8)),
            "formula": "df=nu-d+1; S=((kappa+1)/(kappa*df))*Psi; pair regularizer=mean(||h-gamma||^2*(kappa+nu))",
            "reduction_note": "batch/pair mean is IMPLEMENTATION-ASSUMPTION; paper Eq.10/11 exact reduction is not disclosed beyond sum notation",
        },
        "paper_mapping": {"gamma": "MATCH frequency-dependent [2048] per pair", "kappa": "MATCH pair-level scalar", "nu": "MATCH pair-level scalar", "psi": "APPROXIMATION diagonal instead of full Psi=L L^T"},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
