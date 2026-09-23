#!/usr/bin/env python3
"""STEP 4: evaluate an ID-only Ng-conditioned threshold on epoch3.

The threshold is calibrated from the saved 10--100 ns ID score table only.
This is an IMPLEMENTATION-ASSUMPTION diagnostic, not a paper-specified rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
DATA_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
K = 1024
NGS = [4, 8, 16, 32, 64, 128]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    m = _make_model(load_config("configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    p = torch.load(CHECKPOINT, map_location=device, weights_only=False); state = p["model_state_dict"] if isinstance(p, dict) and "model_state_dict" in p else p
    m.load_state_dict(state); m.eval(); return m


def cfr(delay, count):
    with np.load(DATA_DIR / f"test_delay_{delay}_ns.npz") as d: return np.asarray(d["cfr"][:count], dtype=np.complex64)


def score(model, sample, ng, seed, device):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    set_seeds(seed); b = torch.from_numpy(sample[None]).to(device); mask = torch.zeros((1, K), device=device); mask[:, ::ng] = 1
    x, target, _ = build_noisy_sparse_input(b, mask, 15.0)
    with torch.inference_mode():
        o = model(x); ale_map = o.psi / (o.nu_expanded - 2*K - 1); epi_map = ale_map / o.kappa_expanded
        pair_a = ale_map.reshape(1,2,4,K).permute(0,2,1,3).sum(2).mean(1); pair_e = epi_map.reshape(1,2,4,K).permute(0,2,1,3).sum(2).mean(1); omitted = 1-mask
        ale = float(((pair_a*omitted).sum(1)/omitted.sum(1).clamp_min(1)).cpu()); epi = float(((pair_e*omitted).sum(1)/omitted.sum(1).clamp_min(1)).cpu())
    return ale, epi, int(omitted.sum().cpu())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir", required=True); ap.add_argument("--output-dir", required=True); ap.add_argument("--samples",type=int,default=40); ap.add_argument("--seed",type=int,default=20262000)
    a=ap.parse_args(); inp=ROOT/a.input_dir; out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT)); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    model = load_model(device)
    thresholds={}; raw=list(csv.DictReader((inp/"step3_id_ng_scores.csv").open()))
    for ng in NGS: thresholds[ng]=float(np.quantile([float(r["epistemic"]) for r in raw if int(r["ng"])==ng],.99))
    global_tau=float(json.load((inp/"results.json").open())["global_threshold"])
    prior=json.loads((ROOT/"runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json").read_text()); probes=prior["epoch_probe_results"]["1"]; ale_target=(probes["ID-Easy 20 ns"]["aleatoric_paper_eq12_eq13"]+probes["ID-Hard 80 ns"]["aleatoric_paper_eq12_eq13"])/2; ale_delta=.1*ale_target
    from scripts.fig11_dynamic_runtime_validation import CANDIDATE_NGS
    rows=[]; current=128; step=0
    for delay in [20,80,10,40,60,120]:
        data=cfr(delay,a.samples)
        for i,sample in enumerate(data):
            ale,epi,omitted=score(model,sample,current,a.seed+step,device); tau=thresholds.get(current,global_tau); trigger=bool(np.isfinite(epi) and epi>tau)
            if trigger: next_ng=1; reason="epistemic_threshold"
            elif current==1 or not np.isfinite(ale): next_ng=current; reason="hold_no_omitted_set"
            else:
                idx=CANDIDATE_NGS.index(current)
                if ale>ale_target+ale_delta: next_ng=CANDIDATE_NGS[min(idx+1,len(CANDIDATE_NGS)-1)]; reason="aleatoric_denser"
                elif ale<ale_target-ale_delta: next_ng=CANDIDATE_NGS[max(idx-1,0)]; reason="aleatoric_sparser"
                else: next_ng=current; reason="hysteresis_hold"
            rows.append({"time_step":step,"delay_ns":delay,"regime":f"{delay} ns","current_ng":current,"tau_ng":tau,"global_tau":global_tau,"epistemic":epi,"ratio_to_tau":epi/tau if np.isfinite(epi) else np.nan,"aleatoric":ale,"omitted_count":omitted,"trigger":trigger,"next_ng":next_ng,"reason":reason})
            current=int(next_ng); step+=1
    write_csv(out/"ng_conditioned_trace.csv",rows); write_csv(out/"ng_conditioned_thresholds.csv",[{"ng":ng,"tau_id_q99":tau,"global_tau":global_tau} for ng,tau in thresholds.items()])
    summary=[]
    for delay in [20,80,10,40,60,120]:
        s=[r for r in rows if int(r["delay_ns"])==delay]; summary.append({"delay_ns":delay,"trigger_count":sum(bool(r["trigger"]) for r in s),"first_trigger_step":next((r["time_step"] for r in s if r["trigger"]),""),"mean_ng":float(np.mean([r["current_ng"] for r in s])),"fallback_after_trigger":any(bool(r["trigger"]) and int(r["next_ng"])==1 for r in s)})
    write_csv(out/"ng_conditioned_summary.csv",summary)
    fig,ax=plt.subplots(figsize=(10,5)); x=[r["time_step"] for r in rows]; ax.plot(x,[r["epistemic"] for r in rows],label="epistemic"); ax.plot(x,[r["tau_ng"] for r in rows],label="tau_Ng"); ax.set_yscale("symlog"); ax.set_xlabel("time step"); ax.set_ylabel("score / threshold"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"ng_conditioned_trace.png",dpi=180); plt.close(fig)
    (out/"results.json").write_text(json.dumps({"checkpoint":str(CHECKPOINT),"threshold_definition":"ID 10/20/40/60/80/100 ns pooled q99 per Ng","thresholds":thresholds,"global_threshold":global_tau,"trigger_count":sum(bool(r["trigger"]) for r in rows),"trigger_rows":[r for r in rows if r["trigger"]],"IMPLEMENTATION_ASSUMPTION":["per-Ng q99 is an unpublished controller calibration rule","OOD 120 ns/1 ms were excluded from threshold calibration","existing hysteresis target/delta retained","Ng=1 means full-feedback and omitted metrics are N/A"],"no_training":True},indent=2,allow_nan=True))


if __name__=="__main__": main()
