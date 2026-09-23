#!/usr/bin/env python3
"""Fig.11 candidate-set-only ablation for the frozen epoch-3 checkpoint.

Only adaptive sparse candidates change from the canonical set to {32,16,8,4}.
The canonical initial state (Ng=128), threshold, hysteresis, seed, sequence,
observation protocol, and Ng=1 fallback are retained.
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
K = 1024
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
SEQUENCE_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
REGIME_ORDER = [("20 ns", 20.0), ("80 ns", 80.0), ("10 ns", 10.0), ("40 ns", 40.0), ("60 ns", 60.0), ("120 ns", 120.0)]
RESTRICTED_CANDIDATES = [32, 16, 8, 4]
INITIAL_NG = 128
THRESHOLD = 52.45694942474356


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    m = _make_model(load_config("configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    p = torch.load(CHECKPOINT, map_location=device, weights_only=False); state = p["model_state_dict"] if isinstance(p, dict) and "model_state_dict" in p else p
    m.load_state_dict(state); m.eval(); return m


def make_mask(batch, ng, device):
    mask = torch.zeros((batch, K), dtype=torch.float32, device=device); mask[:, ::ng] = 1.0; return mask


def evaluate_step(model, cfr_np, ng, seed, device):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    set_seeds(seed); cfr = torch.from_numpy(cfr_np[None]).to(device); mask = make_mask(1, ng, device); x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
    with torch.inference_mode():
        o = model(x); ale_map = o.psi / (o.nu_expanded - 2*K - 1.0); epi_map = ale_map / o.kappa_expanded
        pair_a = ale_map.reshape(1,2,4,K).permute(0,2,1,3).sum(2).mean(1); pair_e = epi_map.reshape(1,2,4,K).permute(0,2,1,3).sum(2).mean(1); omitted=1-mask; count=omitted.sum(1)
        ale=float(((pair_a*omitted).sum(1)/count.clamp_min(1)).cpu()) if int(count.sum()) else float('nan'); epi=float(((pair_e*omitted).sum(1)/count.clamp_min(1)).cpu()) if int(count.sum()) else float('nan')
        nmse=float((10*torch.log10((((o.gamma-target).square()*omitted[:,None,:]).sum()/(target.square()*omitted[:,None,:]).sum().clamp_min(1e-12)).clamp_min(1e-12))).cpu()) if int(count.sum()) else float('nan')
    return ale, epi, nmse, int(mask.sum()), int(count.sum())


def decision(current, ale, epi, ale_target, ale_delta):
    if np.isfinite(epi) and epi > THRESHOLD:
        return 1, True, "epistemic_threshold"
    if current == 1 or not np.isfinite(ale): return current, False, "hold_no_omitted_set"
    if current not in RESTRICTED_CANDIDATES:
        return RESTRICTED_CANDIDATES[0], False, "restricted_candidate_projection"
    i = RESTRICTED_CANDIDATES.index(current)
    if ale > ale_target + ale_delta: return RESTRICTED_CANDIDATES[min(i+1,len(RESTRICTED_CANDIDATES)-1)], False, "aleatoric_denser"
    if ale < ale_target - ale_delta: return RESTRICTED_CANDIDATES[max(i-1,0)], False, "aleatoric_sparser"
    return current, False, "hysteresis_hold"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--segment-length',type=int,default=40); ap.add_argument('--eval-seed',type=int,default=20262000); a=ap.parse_args()
    out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT)); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    model=load_model(device)
    prior=json.loads((ROOT/'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json').read_text()); probes=prior['epoch_probe_results']['1']; ale_target=(probes['ID-Easy 20 ns']['aleatoric_paper_eq12_eq13']+probes['ID-Hard 80 ns']['aleatoric_paper_eq12_eq13'])/2; ale_delta=.1*ale_target
    rows=[]; current=INITIAL_NG; step=0
    for regime,delay in REGIME_ORDER:
        path=SEQUENCE_DIR/('test_delay_1_ms.npz' if delay==1_000_000 else f'test_delay_{int(delay)}_ns.npz')
        with np.load(path) as d: cfr=d['cfr'][:a.segment_length]
        for sample in cfr:
            ale,epi,nmse,observed,omitted=evaluate_step(model,sample,current,a.eval_seed+step,device); next_ng,trigger,reason=decision(current,ale,epi,ale_target,ale_delta)
            rows.append({'time_step':step,'regime':regime,'delay_ns':delay,'current_ng':current,'next_ng':next_ng,'aleatoric':ale,'epistemic':epi,'epistemic_threshold':THRESHOLD,'ood_trigger':trigger,'nmse_db':nmse,'observed_count':observed,'omitted_count':omitted,'decision_reason':reason,'ood':delay>100}); current=int(next_ng); step+=1
    write_csv(out/'dynamic_trace.csv',rows)
    summary=[]
    for regime,delay in REGIME_ORDER:
        s=[r for r in rows if r['regime']==regime]; valid=[r for r in s if np.isfinite(float(r['nmse_db']))]; summary.append({'regime':regime,'delay_ns':delay,'mean_current_ng':float(np.mean([r['current_ng'] for r in s])),'mean_nmse_db':float(np.mean([r['nmse_db'] for r in valid])) if valid else 'N/A (full-feedback / empty omitted set)','mean_aleatoric':float(np.nanmean([r['aleatoric'] for r in s])) if any(np.isfinite(float(r['aleatoric'])) for r in s) else 'N/A (full-feedback / empty omitted set)','mean_epistemic':float(np.nanmean([r['epistemic'] for r in s])) if any(np.isfinite(float(r['epistemic'])) for r in s) else 'N/A (full-feedback / empty omitted set)','false_ood_trigger_count':sum(bool(r['ood_trigger']) and not r['ood'] for r in s),'ood_trigger_count':sum(bool(r['ood_trigger']) for r in s),'first_trigger_step':next((r['time_step'] for r in s if r['ood_trigger']), '')})
    write_csv(out/'regime_summary.csv',summary)
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True); x=[r['time_step'] for r in rows]; ax[0].step(x,[r['current_ng'] for r in rows],where='post'); ax[0].set_ylabel('Ng'); ax[0].grid(alpha=.25); ax[1].plot(x,[float(r['nmse_db']) if np.isfinite(float(r['nmse_db'])) else np.nan for r in rows]); ax[1].set_ylabel('NMSE [dB]'); ax[1].set_xlabel('time step'); ax[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'fig11_restricted_candidates.png',dpi=180); plt.close(fig)
    result={'checkpoint':str(CHECKPOINT),'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'candidate_ngs':RESTRICTED_CANDIDATES,'initial_ng':INITIAL_NG,'initial_state_projection':'initial Ng=128 retained for first observation; first adaptive non-fallback action projects to max allowed candidate Ng=32 because 128 is outside restricted candidate set','threshold':THRESHOLD,'threshold_unchanged':True,'ale_target':ale_target,'ale_delta':ale_delta,'seed':a.eval_seed,'segment_length':a.segment_length,'sequence':[r for r,_ in REGIME_ORDER],'no_training':True,'implementation_assumptions':['only sparse candidate set changed','initial 128 is a one-step retained canonical state, then projected to allowed candidate set','Ng=1 fallback unchanged','full-feedback metrics are N/A when omitted set is empty'],'trigger_count':sum(bool(r['ood_trigger']) for r in rows),'false_trigger_count':sum(bool(r['ood_trigger']) and not r['ood'] for r in rows),'ood_trigger_count':sum(bool(r['ood_trigger']) and r['ood'] for r in rows)}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=True))


if __name__=='__main__': main()
