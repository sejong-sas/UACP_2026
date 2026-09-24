#!/usr/bin/env python3
"""Experiment A: epistemic severity versus minimum Partial-FT scope.

This runner is deliberately self-contained at the experiment level, while
reusing the repository's model, loss, channel generator, sparse observation,
and deterministic adaptation schedule. It never writes to prior artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
CONFIG = ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json"
THRESHOLD_FILE = ROOT / "runs/current_valid_baseline/epoch3_ng_conditioned_threshold_fig11_20260919_final/id_calibration_thresholds.csv"
SCOPES = [
    ("partial_small", "last_block_plus_head", 1),
    ("partial_medium", "last_4_blocks_plus_head", 2),
    ("partial_large", "last_8_blocks_plus_head", 3),
    ("full", "full", 4),
]
DELAYS = [110, 120, 140, 160, 200, 250]
NGS = [4, 8, 16, 32]
PRIMARY_NGS = [16, 32]
SEED = 20260921
EPOCHS = 3
BATCH = 8
LR = 1e-4
SAMPLES_TRAIN = 1000
SAMPLES_VAL = 200
SAMPLES_TEST = 1000


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_hashes(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as d:
        return {hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in d["cfr"]}


def model_and_config(device):
    from scripts.train_predictor import load_config
    from scripts.diagnose_predictor import _make_model
    cfg = load_config(CONFIG)
    return _make_model(cfg, device), cfg


def load_state(model, path: Path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)


def write_npz(path: Path, cfr: np.ndarray, delays: np.ndarray, label: str, metadata: dict) -> None:
    np.savez(path, cfr=cfr.astype(np.complex64), delay_spread_ns=delays.astype(np.float32),
             regime_label=np.array([label] * len(cfr), dtype="U32"),
             metadata_json=np.array(json.dumps(metadata, sort_keys=True)))


def generate_dataset_bundle(out: Path) -> dict:
    from src.channel.sionna_channel import generate_cfr_for_delay_spreads, load_sectioned_config, environment_report, paper_settings, assumption_settings, unknown_settings
    base_cfg = load_sectioned_config(ROOT / "configs/adaptation_120ns_protocol_20260921.json")
    data_dir = out / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    specs = {}
    generation_rows = []
    all_paths = []
    for delay in DELAYS:
        root = data_dir / f"delay_{delay}ns"
        root.mkdir(parents=True, exist_ok=True)
        for split, count, offset in [("adapt_train", SAMPLES_TRAIN, 11), ("adapt_val", SAMPLES_VAL, 22), ("target_test", SAMPLES_TEST, 33)]:
            path = root / f"{split}.npz"
            rng = np.random.default_rng(SEED + delay * 100 + offset)
            delays = np.full(count, float(delay), dtype=np.float32)
            started = time.perf_counter()
            cfr = generate_cfr_for_delay_spreads(base_cfg, delays, seed=SEED + delay * 100 + offset)
            elapsed = time.perf_counter() - started
            meta = {"split": split, "delay_ns": delay, "seed": SEED + delay * 100 + offset,
                    "role": "RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION",
                    "paper_specified": paper_settings(base_cfg),
                    "implementation_assumption": assumption_settings(base_cfg), "unknown": unknown_settings(base_cfg),
                    "environment": environment_report(), "generation_seconds": elapsed}
            write_npz(path, cfr, delays, f"{delay} ns", meta)
            generation_rows.append({"delay_ns": delay, "split": split, "samples": count, "path": str(path.relative_to(ROOT)), "generation_seconds": elapsed})
            specs[f"{delay}ns_{split}"] = path
            all_paths.append(path)
    common = {"id20": 20, "id80": 80, "far1ms": 1_000_000}
    for name, delay in common.items():
        path = data_dir / f"common_{name}.npz"
        delays = np.full(SAMPLES_TEST, float(delay), dtype=np.float32)
        started = time.perf_counter()
        cfr = generate_cfr_for_delay_spreads(base_cfg, delays, seed=SEED + 90000 + delay)
        elapsed = time.perf_counter() - started
        meta = {"split": name, "delay_ns": delay, "seed": SEED + 90000 + delay,
                "role": "common quick evaluation only; 1 ms blind to training/model selection",
                "paper_specified": paper_settings(base_cfg), "implementation_assumption": assumption_settings(base_cfg),
                "unknown": unknown_settings(base_cfg), "environment": environment_report(), "generation_seconds": elapsed}
        write_npz(path, cfr, delays, name, meta)
        generation_rows.append({"delay_ns": delay, "split": name, "samples": SAMPLES_TEST, "path": str(path.relative_to(ROOT)), "generation_seconds": elapsed})
        specs[name] = path; all_paths.append(path)
    sets = {str(p): sample_hashes(p) for p in all_paths}
    overlaps = []
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n = len(sets[a] & sets[b])
            if n: overlaps.append({"a": a, "b": b, "duplicates": n})
    write_csv(out / "generation_summary.csv", generation_rows)
    (out / "leakage_summary.json").write_text(json.dumps({"overlap_pairs": overlaps, "all_zero": not overlaps}, indent=2), encoding="utf-8")
    return specs


def configure(model, scope: str) -> None:
    from scripts.partial_ft_adapt import configure_trainable_scope
    configure_trainable_scope(model, scope)


def mask_and_observe(cfr, ng, seed, batch_index, device):
    from scripts.partial_ft_adapt import make_adaptation_observation, scheduled_mask
    mask = scheduled_mask(cfr.shape[0], 1024, 0, batch_index, seed + ng * 10000, device)
    # For evaluation, use fixed periodic Ng masks, matching the validated static evaluator.
    mask.zero_(); mask[:, ::ng] = 1.0
    return make_adaptation_observation(cfr, mask, seed + ng * 10000, device, epoch=0, batch_index=batch_index), mask


@torch.inference_mode()
def evaluate_model(model, path: Path, cfg: dict, device, seed: int, include_samples=False) -> tuple[list[dict], dict]:
    from src.training.data import CFRNPZDataset
    ds = CFRNPZDataset(path)
    rows = []
    # Held-out validation can use a larger batch without changing the
    # observation schedule or metric definition.
    eval_batch_size = int(os.environ.get("UACP_EVAL_BATCH_SIZE", "32"))
    loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    for bi, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        xs, targets, masks = [], [], []
        for ng in NGS:
            (x, target, _), mask = mask_and_observe(cfr, ng, seed, bi, device)
            xs.append(x); targets.append(target); masks.append(mask)
        out = model(torch.cat(xs, dim=0))
        for ng_index, ng in enumerate(NGS):
            start = ng_index * cfr.shape[0]
            stop = start + cfr.shape[0]
            target = targets[ng_index]
            mask = masks[ng_index]
            gamma = out.gamma[start:stop]
            epistemic = out.epistemic[start:stop]
            aleatoric_tensor = out.aleatoric[start:stop]
            omitted = (1.0 - mask)[:, None, :]
            err = (gamma - target).square()
            nmse = 10 * torch.log10((err.mul(omitted).sum((1,2)) / (target.square().mul(omitted).sum((1,2)).clamp_min(1e-12))).clamp_min(1e-12))
            epi = (epistemic * omitted).sum((1,2)) / omitted.sum((1,2)).clamp_min(1.0)
            ale = (aleatoric_tensor * omitted).sum((1,2)) / omitted.sum((1,2)).clamp_min(1.0)
            nmse_np = nmse.detach().cpu().numpy(); epi_np = epi.detach().cpu().numpy(); ale_np = ale.detach().cpu().numpy()
            finite_np = np.isfinite(nmse_np) & np.isfinite(epi_np) & np.isfinite(ale_np)
            for i, (nmse_value, epi_value, ale_value) in enumerate(zip(nmse_np, epi_np, ale_np)):
                row = {"sample": int(bi * eval_batch_size + i), "ng": ng, "nmse_omitted_db": float(nmse_value),
                       "epistemic": float(epi_value), "aleatoric": float(ale_value), "finite": bool(finite_np[i])}
                rows.append(row)
    summary = []
    for ng in NGS:
        q = [r for r in rows if r["ng"] == ng]
        for metric in ["nmse_omitted_db", "epistemic", "aleatoric"]:
            vals = np.array([r[metric] for r in q], dtype=float)
            summary.append({"ng": ng, "metric": metric, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
                            "q90": float(np.quantile(vals,.90)), "q95": float(np.quantile(vals,.95)), "q99": float(np.quantile(vals,.99)),
                            "finite": bool(np.isfinite(vals).all())})
    return (rows if include_samples else []), {f"{r['ng']}_{r['metric']}": r for r in summary}


def train_one(scope_name: str, scope: str, delay: int, paths: dict, out: Path, device, cfg: dict, seed: int = SEED) -> dict:
    from scripts.train_predictor import set_seeds
    from scripts.partial_ft_adapt import make_adaptation_observation, scheduled_mask, optimizer_for_trainable
    from src.models.evidential import evidential_loss
    from src.training.data import CFRNPZDataset
    run = out / "runs" / f"delay_{delay}ns" / scope_name
    run.mkdir(parents=True, exist_ok=True)
    model, _ = model_and_config(device)
    load_state(model, CHECKPOINT, device)
    configure(model, scope)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    frozen_before = {n: p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    set_seeds(seed)
    train_loader = DataLoader(CFRNPZDataset(paths[f"{delay}ns_adapt_train"]), batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(CFRNPZDataset(paths[f"{delay}ns_adapt_val"]), batch_size=BATCH, shuffle=False, num_workers=0)
    opt = optimizer_for_trainable(model, LR)
    history=[]; steps=0; step_times=[]
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train(); losses=[]
        for bi,batch in enumerate(train_loader):
            t=time.perf_counter(); cfr=batch["cfr"].to(device)
            mask=scheduled_mask(cfr.shape[0],1024,epoch,bi,seed,device)
            x,target,_=make_adaptation_observation(cfr,mask,seed,device,epoch,bi)
            opt.zero_grad(set_to_none=True); result=model(x)
            ls=evidential_loss(result,target,float(cfg["paper_specified"]["lambda_reg"]),nll_mode=cfg["implementation_assumption"].get("nll_mode","elementwise"),reg_mode=cfg["implementation_assumption"].get("reg_mode","elementwise"))
            if not torch.isfinite(ls["total"]): raise FloatingPointError(f"nonfinite loss delay={delay} scope={scope_name} epoch={epoch}")
            ls["total"].backward(); opt.step(); torch.cuda.synchronize(device)
            losses.append(float(ls["total"].detach().cpu())); steps+=1; step_times.append(time.perf_counter()-t)
        model.eval(); v=[]
        with torch.inference_mode():
            for bi,batch in enumerate(val_loader):
                cfr=batch["cfr"].to(device); mask=scheduled_mask(cfr.shape[0],1024,epoch,bi,seed+700000,device)
                x,target,_=make_adaptation_observation(cfr,mask,seed+700000,device,epoch,bi)
                ls=evidential_loss(model(x),target,float(cfg["paper_specified"]["lambda_reg"]),nll_mode=cfg["implementation_assumption"].get("nll_mode","elementwise"),reg_mode=cfg["implementation_assumption"].get("reg_mode","elementwise")); v.append(float(ls["total"].cpu()))
        ck=run/f"adapted_epoch_{epoch}.pt"; torch.save(model.state_dict(),ck)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":float(np.mean(v)),"optimizer_steps":steps,"checkpoint":str(ck.relative_to(ROOT))})
    elapsed=time.perf_counter()-started
    frozen_diff=max((float((p.detach()-frozen_before[n]).abs().max().cpu()) for n,p in model.named_parameters() if n in frozen_before), default=0.0)
    manifest={"delay_ns":delay,"scope_name":scope_name,"scope":scope,"seed":seed,"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"checkpoint_sha256":sha256_file(CHECKPOINT),"total_params":total,"trainable_params":trainable,"trainable_percent":100*trainable/total,"trainable_names":names,"epochs":EPOCHS,"batch_size":BATCH,"lr":LR,"lambda_reg":float(cfg["paper_specified"]["lambda_reg"]),"device":"cuda:0","gpu":torch.cuda.get_device_name(0),"elapsed_seconds":elapsed,"sec_per_step_mean":float(np.mean(step_times)),"sec_per_step_total":float(np.sum(step_times)),"peak_allocated_vram_mib":float(torch.cuda.max_memory_allocated(device)/2**20),"frozen_parameter_max_diff":frozen_diff,"history":history,"implementation_assumption":True}
    (run/"training_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    del model; torch.cuda.empty_cache()
    return manifest


def severity(pre_rows, thresholds, out):
    rows=[]; summary=[]
    for r in pre_rows:
        tau=thresholds[r["ng"]]; z=math.log10(max(r["epistemic"],1e-30)/tau)
        rows.append({**r,"tau_ng":tau,"z_log10_epi_over_tau":z})
    for ng in NGS:
        vals=np.array([r["z_log10_epi_over_tau"] for r in rows if r["ng"]==ng])
        summary.append({"ng":ng,"median_z":float(np.median(vals)),"mean_z":float(np.mean(vals)),"q90_z":float(np.quantile(vals,.9)),"q95_z":float(np.quantile(vals,.95)),"q99_z":float(np.quantile(vals,.99))})
    write_csv(out/"pre_adaptation_severity_samples.csv",rows); write_csv(out/"pre_adaptation_severity_summary.csv",summary)
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--reuse-dataset-dir",default=None); args=p.parse_args()
    out=ROOT/args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True)
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0): raise RuntimeError("NVIDIA GB10/cuda:0 required")
    device=torch.device("cuda:0")
    cfg = __import__('scripts.train_predictor',fromlist=['load_config']).load_config(CONFIG)
    metadata={"experiment":"uncertainty-guided minimum adaptation scope Experiment A","checkpoint_sha256":sha256_file(CHECKPOINT),"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"device":"cuda:0","gpu":torch.cuda.get_device_name(0),"delays_ns":DELAYS,"scopes":[x[0] for x in SCOPES],"seed":SEED,"epochs":EPOCHS,"batch_size":BATCH,"lr":LR,"ngs":NGS,"primary_ngs":PRIMARY_NGS,"research_extension":True,"implementation_assumptions":["delay-specific 1000/200/1000 data counts","TDL-A current repository channel generator","15 dB complex AWGN on reported sparse CFR","clean full-CFR target","diagonal Psi and current UACP loss","Ng-conditioned ID-only q99 thresholds reused without recalibration"]}
    (out/"experiment_config.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    thresholds={int(r["ng"]):float(r["tau_ng"] if "tau_ng" in r else r["q99"]) for r in csv.DictReader(THRESHOLD_FILE.open())}
    write_csv(out/"fixed_thresholds.csv",[{"ng":k,"tau_ng":v,"source":str(THRESHOLD_FILE.relative_to(ROOT))} for k,v in thresholds.items()])
    if args.reuse_dataset_dir:
        source = ROOT / args.reuse_dataset_dir
        paths = {}
        for delay in DELAYS:
            for split in ("adapt_train", "adapt_val", "target_test"):
                paths[f"{delay}ns_{split}"] = source / "datasets" / f"delay_{delay}ns" / f"{split}.npz"
        paths.update({"id20": source / "datasets/common_id20.npz", "id80": source / "datasets/common_id80.npz", "far1ms": source / "datasets/common_far1ms.npz"})
        if not all(p.is_file() for p in paths.values()):
            raise FileNotFoundError("reuse dataset directory is incomplete")
        (out / "dataset_reuse_manifest.json").write_text(json.dumps({"source": str(source.relative_to(ROOT)), "paths": {k: str(v.relative_to(ROOT)) for k,v in paths.items()}, "reused_without_modification": True}, indent=2), encoding="utf-8")
    else:
        paths=generate_dataset_bundle(out)
    pre_model,_=model_and_config(device); load_state(pre_model,CHECKPOINT,device); pre_model.eval()
    all_sev=[]; pre_eval_rows=[]
    for delay in DELAYS:
        rows,summary=evaluate_model(pre_model,paths[f"{delay}ns_target_test"],cfg,device,SEED,include_samples=True)
        for r in rows: r.update({"delay_ns":delay,"model":"pre"})
        pre_eval_rows.extend(rows); all_sev.extend(rows)
    severity_summary=severity(all_sev,thresholds,out)
    write_csv(out/"pre_evaluation_samples.csv",pre_eval_rows)
    del pre_model; torch.cuda.empty_cache()
    manifests=[]; eval_rows=[]
    for delay in DELAYS:
        for scope_name,scope,ordinal in SCOPES:
            print(json.dumps({"status":"training","delay_ns":delay,"scope":scope_name}),flush=True)
            m=train_one(scope_name,scope,delay,paths,out,device,cfg); manifests.append(m)
            model,_=model_and_config(device); load_state(model,out/"runs"/f"delay_{delay}ns"/scope_name/"adapted_epoch_3.pt",device); model.eval()
            for regime,path in [("target",paths[f"{delay}ns_target_test"]),("id20",paths["id20"]),("id80",paths["id80"]),("far1ms",paths["far1ms"])]:
                rows,summary=evaluate_model(model,path,cfg,device,SEED,include_samples=False)
                for key,val in summary.items():
                    ng,metric=key.split("_", 1)
                    eval_rows.append({"delay_ns":delay,"scope":scope_name,"scope_ordinal":ordinal,"regime":regime,"ng":int(ng),**val})
            del model; torch.cuda.empty_cache()
    write_csv(out/"training_summary.csv",manifests); write_csv(out/"evaluation_summary.csv",eval_rows)
    # Minimum sufficient scope, requiring both primary Ng values to be within 0.5 dB of Full.
    full={(r["delay_ns"],r["ng"]):r["median"] for r in eval_rows if r["scope"]=="full" and r["regime"]=="target" and r["metric"]=="nmse_omitted_db"}
    req=[]
    for delay in DELAYS:
        chosen=None
        for name,scope,ordinal in SCOPES:
            okay=True
            for ng in PRIMARY_NGS:
                row=next(r for r in eval_rows if r["delay_ns"]==delay and r["scope"]==name and r["regime"]=="target" and r["ng"]==ng and r["metric"]=="nmse_omitted_db")
                okay &= float(row["median"]) <= full[(delay,ng)] + .5
            if okay: chosen=(name,ordinal); break
        sev=np.mean([r["median_z"] for r in severity_summary if r["ng"] in PRIMARY_NGS])
        req.append({"delay_ns":delay,"severity_median_z_primary":float(sev),"minimum_scope":chosen[0] if chosen else "none","scope_ordinal":chosen[1] if chosen else np.nan,"criterion_db":.5})
    write_csv(out/"minimum_sufficient_scope.csv",req)
    x=np.array([r["severity_median_z_primary"] for r in req if r["minimum_scope"]!="none"],float); y=np.array([r["scope_ordinal"] for r in req if r["minimum_scope"]!="none"],float)
    from scipy.stats import spearmanr
    rho=float(spearmanr(x,y).statistic) if len(x)>1 and len(set(y))>1 else float("nan")
    nondec=all(y[i]<=y[i+1] for i in range(len(y)-1)) if len(y)>1 else False
    h1={"rho":rho,"conditions_with_scope":int(len(y)),"scope_ordinals":y.tolist(),"non_decreasing_by_delay":nondec,"passes_rho":bool(np.isfinite(rho) and rho>=.7),"passes_condition_count":len(y)>=4,"nontrivial_scope_variation":len(set(y))>1,"h1_supported":bool(np.isfinite(rho) and rho>=.7 and len(y)>=4 and len(set(y))>1 and nondec)}
    (out/"h1_gate.json").write_text(json.dumps(h1,indent=2),encoding="utf-8")
    (out/"integrity_summary.json").write_text(json.dumps({"pre_checkpoint_sha256":sha256_file(CHECKPOINT),"expected_pre_checkpoint_sha256":"d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986","checkpoint_unchanged":sha256_file(CHECKPOINT)=="d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986","leakage_summary":"leakage_summary.json","all_evaluation_finite":all(bool(r["finite"]) for r in eval_rows),"gpu":torch.cuda.get_device_name(0)},indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out.relative_to(ROOT)),"h1":h1,"checkpoint_sha256":sha256_file(CHECKPOINT)},indent=2),flush=True)

if __name__ == "__main__": main()
