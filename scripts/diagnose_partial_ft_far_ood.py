#!/usr/bin/env python3
"""Root-cause diagnosis for the 25% Partial FT Far-OOD epistemic collapse.

No training is performed. The established sparse evaluation protocol is reused,
and state-dict component swaps are evaluated in memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K = 1024
REGIMES = ("20ns", "80ns", "120ns", "1ms")
NGS = (16, 32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_state(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state


def load_model(config, state, device):
    from scripts.diagnose_predictor import _make_model
    model = _make_model(config, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def hybrid_state(pre, post, kind: str):
    state = {key: value.clone() for key, value in pre.items()}
    if kind in ("H25", "H-BLOCK"):
        prefixes = tuple(f"residual_blocks.{i}." for i in range(24, 32))
        for key in state:
            if key.startswith(prefixes):
                state[key] = post[key].clone()
    if kind in ("H25", "H-HEAD", "H-GAMMA", "H-UNC"):
        heads = {"H-HEAD": ("gamma_head.", "psi_head.", "kappa_head.", "nu_head."),
                 "H-GAMMA": ("gamma_head." ,),
                 "H-UNC": ("psi_head.", "kappa_head.", "nu_head.")}
        selected = ("gamma_head.", "psi_head.", "kappa_head.", "nu_head.") if kind == "H25" else heads[kind]
        for key in state:
            if key.startswith(selected):
                state[key] = post[key].clone()
    return state


def make_input(cfr, mask, seed, device, epoch=0, batch_index=0):
    from scripts.partial_ft_adapt import make_adaptation_observation
    return make_adaptation_observation(cfr, mask, seed, device, epoch=epoch, batch_index=batch_index)


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def forward_raw(model, x):
    h = model.input_projection(x)
    block23 = None
    for index, block in enumerate(model.residual_blocks):
        h = block(h)
        if index == 23:
            block23 = h
    block31 = h
    raw_gamma = model.gamma_head(h)
    raw_psi = model.psi_head(h)
    pooled = model.pool(h)
    raw_kappa = model.kappa_head(pooled)
    raw_nu = model.nu_head(pooled)
    from src.models.evidential import constrain_pair_scalar_evidential
    output = constrain_pair_scalar_evidential(raw_gamma, raw_psi, raw_kappa, raw_nu, K)
    return output, {"raw_gamma": raw_gamma, "raw_psi": raw_psi, "raw_kappa": raw_kappa,
                    "raw_nu": raw_nu, "block23": block23, "block31": block31,
                    "head_input": block31}


def per_sample_metrics(output, raw, target, mask):
    omitted = 1.0 - mask
    omitted8 = omitted[:, None, :]
    error = (output.gamma - target).square()
    nmse = 10.0 * torch.log10(((error * omitted8).sum((1, 2)) /
        (target.square() * omitted8).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
    psi = output.psi
    nu_margin = output.nu_expanded - 2 * K - 1.0
    kappa = output.kappa_expanded
    ale = psi / nu_margin
    epi = ale / kappa
    def omitted_mean(value):
        if value.shape[1] == 4:
            return value.mean(dim=(1, 2))
        return (value * omitted8).sum((1, 2)) / omitted8.sum((1, 2)).clamp_min(1.0)
    values = {"nmse_omitted_db": nmse, "psi": omitted_mean(psi),
              "nu_margin": omitted_mean(nu_margin), "kappa": omitted_mean(kappa),
              "aleatoric": omitted_mean(ale), "epistemic": omitted_mean(epi),
              "total_uncertainty": omitted_mean(ale + epi),
              "raw_psi": omitted_mean(raw["raw_psi"]),
              "raw_nu": raw["raw_nu"].mean((1, 2)),
              "raw_kappa": raw["raw_kappa"].mean((1, 2))}
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def activation_summary(pre_model, post_model, dataset_path, config, device, ng, seed, batch_size):
    from src.training.data import CFRNPZDataset
    dataset = CFRNPZDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            cfr = batch["cfr"].to(device)
            mask = torch.zeros((cfr.shape[0], K), device=device)
            mask[:, ::ng] = 1.0
            x, _, _ = make_input(cfr, mask, seed, device, epoch=0, batch_index=batch_index)
            _, a = forward_raw(pre_model, x)
            _, b = forward_raw(post_model, x)
            for name in ("block23", "block31", "head_input"):
                av = a[name].flatten(1); bv = b[name].flatten(1)
                diff = (bv - av).norm(dim=1) / av.norm(dim=1).clamp_min(1e-12)
                cosine = torch.nn.functional.cosine_similarity(av, bv, dim=1)
                for i in range(cfr.shape[0]):
                    rows.append({"ng": ng, "feature": name, "sample": len(rows),
                                 "relative_l2": float(diff[i]), "cosine": float(cosine[i])})
    return rows


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "median": float(np.median(values)), "p10": float(np.quantile(values, .10)),
            "p90": float(np.quantile(values, .90))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/current_valid_baseline_100k1_seed_20260819.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10/cuda:0 required")
    from scripts.train_predictor import load_config
    config = load_config(ROOT / args.config)
    base_path = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
    p3_path = ROOT / "runs/current_valid_baseline/partial_ft_20260921_sparse_3ep/adapted_epoch_3.pt"
    p25_path = ROOT / "runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_3ep/adapted_epoch_3.pt"
    full_path = ROOT / "runs/current_valid_baseline/full_ft_20260921_sparse_3ep/adapted_epoch_3.pt"
    data_dir = ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data"
    regimes = {"20ns": data_dir / "test_id_easy.npz", "80ns": data_dir / "test_id_hard.npz",
               "120ns": data_dir / "test_ood_near.npz", "1ms": data_dir / "test_ood_far.npz"}
    states = {"Pre": load_state(base_path), "3.16%": load_state(p3_path),
              "25%": load_state(p25_path), "Full": load_state(full_path)}
    states.update({kind: hybrid_state(states["Pre"], states["25%"], kind)
                   for kind in ("H-BLOCK", "H-HEAD", "H-GAMMA", "H-UNC", "H25")})
    model_names = ("Pre", "3.16%", "25%", "Full", "H-BLOCK", "H-HEAD", "H-GAMMA", "H-UNC", "H25")
    models = {name: load_model(config, states[name], device) for name in model_names}
    started = time.perf_counter()
    sample_rows, raw_rows = [], []
    for model_name in model_names:
        model = models[model_name]
        for regime in REGIMES:
            from src.training.data import CFRNPZDataset
            loader = DataLoader(CFRNPZDataset(regimes[regime]), batch_size=args.batch_size, shuffle=False, num_workers=0)
            for ng in NGS:
                group = []
                with torch.inference_mode():
                    for batch_index, batch in enumerate(loader):
                        cfr = batch["cfr"].to(device)
                        mask = torch.zeros((cfr.shape[0], K), device=device); mask[:, ::ng] = 1.0
                        x, target, _ = make_input(cfr, mask, 20260921 + ng * 1000, device, epoch=0, batch_index=batch_index)
                        output, raw = forward_raw(model, x)
                        vals = per_sample_metrics(output, raw, target, mask)
                        for i in range(cfr.shape[0]):
                            group.append({"model": model_name, "regime": regime, "ng": ng,
                                          "sample": len(group), **{k: float(v[i]) for k, v in vals.items()}})
                sample_rows.extend(group)
                for quantity in ("psi", "nu_margin", "kappa", "aleatoric", "epistemic", "total_uncertainty", "raw_psi", "raw_nu", "raw_kappa"):
                    row = {"model": model_name, "regime": regime, "ng": ng, "quantity": quantity,
                           **stats([r[quantity] for r in group])}
                    raw_rows.append(row)
    write_csv(out / "sample_metrics.csv", sample_rows)
    write_csv(out / "raw_evidential_stats.csv", raw_rows)
    # AUROC: ID is pooled 20/80 ns; compare against 120 ns and 1 ms separately.
    auc_rows = []
    for model in model_names:
        for ng in NGS:
            id_values = np.asarray([r["epistemic"] for r in sample_rows if r["model"] == model and r["ng"] == ng and r["regime"] in ("20ns", "80ns")])
            for ood in ("120ns", "1ms"):
                ood_values = np.asarray([r["epistemic"] for r in sample_rows if r["model"] == model and r["ng"] == ng and r["regime"] == ood])
                auc_rows.append({"model": model, "ng": ng, "comparison": f"ID(20+80) vs {ood}",
                                 "id_mean": float(id_values.mean()), "ood_mean": float(ood_values.mean()),
                                 "id_median": float(np.median(id_values)), "ood_median": float(np.median(ood_values)),
                                 "auroc": auc_rank(ood_values, id_values)})
    write_csv(out / "auroc.csv", auc_rows)
    # 1 ms distribution plots for direct visual confirmation of distribution shift.
    for ng in NGS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in ("Pre", "3.16%", "25%", "Full", "H-BLOCK", "H-HEAD", "H-GAMMA", "H-UNC"):
            vals = np.asarray([r["epistemic"] for r in sample_rows if r["model"] == model and r["ng"] == ng and r["regime"] == "1ms"])
            ax.hist(np.log10(np.maximum(vals, 1e-30)), bins=40, alpha=.35, label=model)
        ax.set_xlabel("log10(Epistemic Eq.(8))"); ax.set_ylabel("Count"); ax.set_title(f"1 ms Epistemic distribution, Ng={ng}")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / f"epistemic_1ms_hist_ng{ng}.png", dpi=180); plt.close(fig)
    # Representation drift and parameter drift.
    pre_model, post_model = models["Pre"], models["25%"]
    activation_rows = []
    for regime in REGIMES:
        for ng in NGS:
            rows = activation_summary(pre_model, post_model, regimes[regime], config, device, ng, 20260921 + ng * 1000, args.batch_size)
            for row in rows:
                row["regime"] = regime
            activation_rows.extend(rows)
    write_csv(out / "activation_drift.csv", activation_rows)
    parameter_rows = []
    for prefix in ["residual_blocks.%d" % i for i in range(32)] + ["gamma_head", "psi_head", "kappa_head", "nu_head"]:
        keys = [k for k in states["Pre"] if k.startswith(prefix + ".")]
        pre_norm = float(torch.sqrt(sum((states["Pre"][k].float() ** 2).sum() for k in keys)))
        delta_norm = float(torch.sqrt(sum(((states["25%"][k] - states["Pre"][k]).float() ** 2).sum() for k in keys)))
        parameter_rows.append({"component": prefix, "parameter_count": int(sum(states["Pre"][k].numel() for k in keys)),
                               "pre_l2": pre_norm, "delta_l2": delta_norm,
                               "relative_drift": delta_norm / max(pre_norm, 1e-12),
                               "trainable_in_25pct": prefix.startswith("residual_blocks.") and int(prefix.split(".")[1]) >= 24 or prefix in ("gamma_head", "psi_head", "kappa_head", "nu_head")})
    write_csv(out / "parameter_drift.csv", parameter_rows)
    manifest = {"no_training": True, "device": str(device), "gpu": torch.cuda.get_device_name(0),
                "checkpoint_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
                "fixed_protocol": {"seed": 20260921, "ngs": list(NGS), "regimes": list(REGIMES), "batch_size": args.batch_size,
                                    "mask": "fixed canonical offset 0 periodic mask[:,::Ng]",
                                    "noise": "deterministic 15 dB complex AWGN keyed by seed+Ng*1000",
                                    "input_is_sparse": True, "target_is_clean_full_cfr": True},
                "states": list(model_names), "swap_definitions": {
                    "H-BLOCK": "25% ResidualBlock 24-31 only; Pre heads and remaining backbone",
                    "H-HEAD": "25% all four evidential heads only; Pre backbone",
                    "H-GAMMA": "25% gamma head only; Pre uncertainty heads and backbone",
                    "H-UNC": "25% psi/kappa/nu heads only; Pre gamma head and backbone",
                    "H25": "25% ResidualBlock 24-31 plus all four heads"},
                "equations_audited": {"aleatoric": "Psi/(nu-2K-1)", "epistemic": "aleatoric/kappa",
                                      "implementation": "diagonal Psi; pooled kappa/nu; Eq.(12)/(13) omitted aggregation"},
                "elapsed_seconds": time.perf_counter() - started}
    (out / "diagnosis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "sample_rows": len(sample_rows), "raw_rows": len(raw_rows),
                      "auc_rows": len(auc_rows), "device": str(device), "gpu": manifest["gpu"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
