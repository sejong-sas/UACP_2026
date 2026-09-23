#!/usr/bin/env python3
"""Fixed-grid, multi-seed and multi-duration validation for the final Partial scope."""
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
SEEDS = (20260921, 20260922, 20260923)
SCOPES = {"partial_3pct": "last_block_plus_head", "partial_last4": "last_4_blocks_plus_head",
          "partial_25pct": "last_8_blocks_plus_head", "full": "full"}
EPOCHS = (1, 3, 5, 10)
REGIMES = ("20ns", "80ns", "120ns", "1ms")
NGS = (4, 8, 16, 32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, "") for key in fields})


def state(path: Path):
    x = torch.load(path, map_location="cpu", weights_only=False)
    return x["model_state_dict"] if isinstance(x, dict) and "model_state_dict" in x else x


def model_from_state(config, value, device):
    from scripts.diagnose_predictor import _make_model
    m = _make_model(config, device); m.load_state_dict(value); m.eval(); return m


def make_input(cfr, mask, seed, device, batch_index):
    from scripts.partial_ft_adapt import make_adaptation_observation
    return make_adaptation_observation(cfr, mask, seed, device, epoch=0, batch_index=batch_index)


def auc(pos, neg):
    scores = np.r_[neg, pos]; labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), float)
    ordered = scores[order]; start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2; start = end
    p = ranks[labels == 1]
    return float((p.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def describe(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x)),
            "p10": float(np.quantile(x, .10)), "p90": float(np.quantile(x, .90))}


def bootstrap_ci_difference(a, b, seed, repetitions=1000):
    rng = np.random.default_rng(seed); n = len(a); values = np.empty(repetitions)
    delta = np.asarray(a) - np.asarray(b)
    for i in range(repetitions): values[i] = np.mean(delta[rng.integers(0, n, n)])
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def forward_stats(model, x, target, mask):
    out = model(x)
    omitted = (1.0 - mask)[:, None, :]
    error = (out.gamma - target).square()
    nmse = 10 * torch.log10(((error * omitted).sum((1, 2)) /
        (target.square() * omitted).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
    nu_margin = out.nu_expanded - 2 * K - 1
    ale = out.psi / nu_margin
    epi = ale / out.kappa_expanded
    def mean_omitted(x):
        return (x * omitted).sum((1, 2)) / omitted.sum((1, 2)).clamp_min(1)
    return {"nmse": nmse.detach().cpu().numpy(), "psi": mean_omitted(out.psi).detach().cpu().numpy(),
            "kappa": mean_omitted(out.kappa_expanded).detach().cpu().numpy(),
            "nu_margin": mean_omitted(nu_margin).detach().cpu().numpy(),
            "aleatoric": mean_omitted(ale).detach().cpu().numpy(),
            "epistemic": mean_omitted(epi).detach().cpu().numpy(),
            "total_uncertainty": mean_omitted(ale + epi).detach().cpu().numpy()}


def forward_with_activation(model, x):
    h = model.input_projection(x); boundary = {}
    for i, block in enumerate(model.residual_blocks):
        h = block(h)
        if i in (22, 26, 30, 31): boundary[f"block{i+1}"] = h
    return boundary, h


def activation_drift(pre_model, model, x, boundary):
    with torch.inference_mode():
        pre, pre_final = forward_with_activation(pre_model, x)
        cur, cur_final = forward_with_activation(model, x)
    names = ["block23", "block27", "block31"]
    rows = {}
    for name in names:
        a = pre[name].flatten(1); b = cur[name].flatten(1)
        rows[name] = ((b-a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).detach().cpu().numpy()
        rows[name+"_cosine"] = torch.nn.functional.cosine_similarity(a, b, dim=1).detach().cpu().numpy()
    a = pre_final.flatten(1); b = cur_final.flatten(1)
    rows["head_input"] = ((b-a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).detach().cpu().numpy()
    rows["head_input_cosine"] = torch.nn.functional.cosine_similarity(a, b, dim=1).detach().cpu().numpy()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    root = ROOT / args.output_dir
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True); (root / "plots").mkdir()
    from scripts.train_predictor import load_config
    config = load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0): raise RuntimeError("GB10/cuda:0 required")
    data_dir = ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data"
    paths = {"20ns": data_dir/"test_id_easy.npz", "80ns": data_dir/"test_id_hard.npz",
             "120ns": data_dir/"test_ood_near.npz", "1ms": data_dir/"test_ood_far.npz"}
    base = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
    pre_state = state(base); pre_model = model_from_state(config, pre_state, device)
    # Cache fixed evaluation arrays once; all models and seeds use these exact arrays.
    cached = {}
    for regime in REGIMES:
        from src.training.data import CFRNPZDataset
        loader = DataLoader(CFRNPZDataset(paths[regime]), batch_size=args.batch_size, shuffle=False, num_workers=0)
        for ng in NGS:
            vals = {k: [] for k in ("nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic", "total_uncertainty")}
            acts = {k: [] for k in ("block23", "block27", "block31", "head_input", "block23_cosine", "block27_cosine", "block31_cosine", "head_input_cosine")}
            for bi, batch in enumerate(loader):
                cfr = batch["cfr"].to(device); mask = torch.zeros((cfr.shape[0], K), device=device); mask[:, ::ng] = 1
                x, target, _ = make_input(cfr, mask, 20260921 + ng*1000, device, bi)
                with torch.inference_mode():
                    pv = forward_stats(pre_model, x, target, mask)
                for k in vals: vals[k].append(pv[k])
                del cfr, mask, x, target
            cached[(regime,ng)] = {k: np.concatenate(v) for k,v in vals.items()}
    trajectory, drift_rows, auc_rows = [], [], []
    # Include Pre as epoch 0 in trajectory tables.
    for regime in REGIMES:
        for ng in NGS:
            v = cached[(regime,ng)]
            for q in ("psi","kappa","nu_margin","aleatoric","epistemic","total_uncertainty"):
                trajectory.append({"seed":"all","scope":"Pre","epoch":0,"regime":regime,"ng":ng,"quantity":q,**describe(v[q])})
    for seed in SEEDS:
        for scope, scope_name in SCOPES.items():
            for epoch in EPOCHS:
                cp = ROOT / args.training_root / f"seed_{seed}" / scope / f"adapted_epoch_{epoch}.pt"
                if not cp.is_file(): raise FileNotFoundError(cp)
                model = model_from_state(config, state(cp), device)
                for regime in REGIMES:
                    from src.training.data import CFRNPZDataset
                    for ng in NGS:
                        loader = DataLoader(CFRNPZDataset(paths[regime]), batch_size=args.batch_size, shuffle=False, num_workers=0)
                        got = {k: [] for k in ("nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic", "total_uncertainty")}
                        drift = {k: [] for k in ("block23", "block27", "block31", "head_input", "block23_cosine", "block27_cosine", "block31_cosine")}
                        for bi, batch in enumerate(loader):
                            cfr = batch["cfr"].to(device); mask = torch.zeros((cfr.shape[0], K), device=device); mask[:, ::ng] = 1
                            x, target, _ = make_input(cfr, mask, 20260921 + ng*1000, device, bi)
                            with torch.inference_mode(): cv = forward_stats(model, x, target, mask)
                            for k in got: got[k].append(cv[k])
                            if epoch in EPOCHS:
                                dv = activation_drift(pre_model, model, x, None)
                                for k in drift: drift[k].append(dv[k])
                            del cfr, mask, x, target
                        got = {k: np.concatenate(v) for k,v in got.items()}
                        for q in ("psi","kappa","nu_margin","aleatoric","epistemic","total_uncertainty"):
                            trajectory.append({"seed":seed,"scope":scope_name,"epoch":epoch,"regime":regime,"ng":ng,"quantity":q,**describe(got[q])})
                        for k in drift:
                            drift_rows.append({"seed":seed,"scope":scope_name,"epoch":epoch,"regime":regime,"ng":ng,"feature":k,**describe(np.concatenate(drift[k]))})
                        pre = cached[(regime,ng)]
                        if epoch == 10:
                            for comparison in ("120ns","1ms"):
                                if regime != comparison: continue
                            id_epi = np.concatenate([cached[("20ns",ng)]["epistemic"], cached[("80ns",ng)]["epistemic"]])
                            auc_rows.append({"seed":seed,"scope":scope_name,"ng":ng,"comparison":"ID_vs_120ns","auroc":auc(got["epistemic"],id_epi) if regime=="120ns" else ""})
                            auc_rows.append({"seed":seed,"scope":scope_name,"ng":ng,"comparison":"ID_vs_1ms","auroc":auc(got["epistemic"],id_epi) if regime=="1ms" else ""})
                        if epoch == 10:
                            # One row per seed/scope/ng/regime, including paired bootstrap for NMSE and Epi.
                            imp = pre["nmse"] - got["nmse"]
                            trajectory.append({"seed":seed,"scope":scope_name,"epoch":epoch,"regime":regime,"ng":ng,"quantity":"nmse_improvement_pre_minus_post",**describe(imp),"ci95_low":bootstrap_ci_difference(pre["nmse"],got["nmse"],seed+ng),"ci95_high":bootstrap_ci_difference(pre["nmse"],got["nmse"],seed+ng)[1]})
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
    # The trajectory table includes all raw uncertainty descriptors; derive final summary from it.
    write_csv(root/"epoch_trajectory.csv", trajectory)
    write_csv(root/"representation_drift.csv", drift_rows)
    # Build epoch-10 performance summaries from trajectory rows and use fixed Pre/Full values.
    rec_rows=[]; unc_rows=[]
    for seed in SEEDS:
        for scope, scope_name in SCOPES.items():
            for ng in NGS:
                pre120 = cached[("120ns",ng)]["nmse"]
                # Re-read exact epoch-10 data by compactly using trajectory values for uncertainty; NMSE values are stored in diagnostic rows.
                for regime in REGIMES:
                    for q in ("nmse_improvement_pre_minus_post",):
                        rr=[r for r in trajectory if str(r["seed"])==str(seed) and r["scope"]==scope_name and int(r["epoch"])==10 and r["regime"]==regime and int(r["ng"])==ng and r["quantity"]==q]
                        if rr: rec_rows.append(rr[0])
                    for q in ("aleatoric","epistemic","total_uncertainty"):
                        rr=[r for r in trajectory if str(r["seed"])==str(seed) and r["scope"]==scope_name and int(r["epoch"])==10 and r["regime"]==regime and int(r["ng"])==ng and r["quantity"]==q]
                        if rr: unc_rows.append(rr[0])
    write_csv(root/"reconstruction_summary.csv", rec_rows); write_csv(root/"uncertainty_summary.csv", unc_rows); write_csv(root/"auroc_summary.csv", auc_rows)
    # Efficiency from manifests.
    eff=[]
    for seed in SEEDS:
        for scope, scope_name in SCOPES.items():
            m=json.loads((ROOT/args.training_root/f"seed_{seed}"/scope/"readiness_manifest.json").read_text())
            r=m["dry_run_result"]
            eff.append({"seed":seed,"scope":scope_name,"trainable_params":r["optimizer_trainable_params"],"trainable_percent":100*r["optimizer_trainable_params"]/11815320,"training_seconds":r["elapsed_seconds"],"seconds_per_step":r["step_seconds_mean"],"peak_vram_mib":r["peak_allocated_vram_mib"]})
    write_csv(root/"efficiency_summary.csv", eff)
    (root/"validation_manifest.json").write_text(json.dumps({"training_root":args.training_root,"seeds":SEEDS,"scopes":SCOPES,"epochs":EPOCHS,"ngs":NGS,"regimes":REGIMES,"device":str(device),"gpu":torch.cuda.get_device_name(0),"checkpoint_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),"same_eval_data_mask_noise":True,"no_new_scope_sweep":True,"primary_endpoint":"epoch10"},indent=2)+"\n")
    print(json.dumps({"output":str(root.relative_to(ROOT)),"device":str(device),"gpu":torch.cuda.get_device_name(0),"trajectory_rows":len(trajectory),"drift_rows":len(drift_rows),"elapsed_seconds":0},indent=2))


if __name__ == "__main__": main()
