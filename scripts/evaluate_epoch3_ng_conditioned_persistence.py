#!/usr/bin/env python3
"""STEP 5: minimal N=2 persistence test after Ng-conditioned calibration."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'runs/baseline_reproduction/step2_delay_sweep_repro/generated_data'
CKPT=ROOT/'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt'
K=1024; NGS=[4,8,16,32,64,128]

def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--samples',type=int,default=40); ap.add_argument('--seed',type=int,default=20262000); a=ap.parse_args()
    inp=ROOT/a.input_dir; out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT))
    from scripts.evaluate_epoch3_ng_conditioned_fig11 import load_model, score
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    model=load_model(device)
    thresholds={int(r['ng']):float(r['tau_id_q99']) for r in csv.DictReader((inp/'ng_conditioned_thresholds.csv').open())}
    prior=json.loads((ROOT/'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json').read_text()); probes=prior['epoch_probe_results']['1']; ale_target=(probes['ID-Easy 20 ns']['aleatoric_paper_eq12_eq13']+probes['ID-Hard 80 ns']['aleatoric_paper_eq12_eq13'])/2; ale_delta=.1*ale_target
    from scripts.fig11_dynamic_runtime_validation import CANDIDATE_NGS
    rows=[]; current=128; consecutive=0; step=0
    for delay in [20,80,10,40,60,120]:
        with np.load(DATA/f'test_delay_{delay}_ns.npz') as d: cfr=np.asarray(d['cfr'][:a.samples],dtype=np.complex64)
        for sample in cfr:
            ale,epi,omitted=score(model,sample,current,a.seed+step,device); tau=thresholds[current]; above=bool(np.isfinite(epi) and epi>tau); consecutive=consecutive+1 if above else 0; trigger=consecutive>=2
            if trigger: next_ng=1; reason='epistemic_threshold_persistence_N2'; consecutive=0
            elif current==1 or not np.isfinite(ale): next_ng=current; reason='hold_no_omitted_set'
            else:
                idx=CANDIDATE_NGS.index(current)
                if ale>ale_target+ale_delta: next_ng=CANDIDATE_NGS[min(idx+1,len(CANDIDATE_NGS)-1)]; reason='aleatoric_denser'
                elif ale<ale_target-ale_delta: next_ng=CANDIDATE_NGS[max(idx-1,0)]; reason='aleatoric_sparser'
                else: next_ng=current; reason='hysteresis_hold'
            rows.append({'time_step':step,'delay_ns':delay,'current_ng':current,'tau_ng':tau,'epistemic':epi,'ratio_to_tau':epi/tau if np.isfinite(epi) else np.nan,'aleatoric':ale,'omitted_count':omitted,'above_threshold':above,'consecutive_above':consecutive,'trigger':trigger,'next_ng':next_ng,'reason':reason}); current=int(next_ng); step+=1
    write_csv(out/'persistence_trace.csv',rows)
    summary=[]
    for delay in [20,80,10,40,60,120]:
        s=[r for r in rows if r['delay_ns']==delay]; summary.append({'delay_ns':delay,'trigger_count':sum(bool(r['trigger']) for r in s),'first_trigger_step':next((r['time_step'] for r in s if r['trigger']),''),'mean_ng':float(np.mean([r['current_ng'] for r in s])),'fallback_after_trigger':any(bool(r['trigger']) for r in s)})
    write_csv(out/'persistence_summary.csv',summary)
    (out/'results.json').write_text(json.dumps({'checkpoint':str(CKPT),'persistence_N':2,'threshold_source':str(inp/'ng_conditioned_thresholds.csv'),'summary':summary,'trigger_rows':[r for r in rows if r['trigger']],'IMPLEMENTATION_ASSUMPTION':['N=2 consecutive threshold exceedances is a minimal controller persistence rule, not paper-specified','Ng-conditioned thresholds are ID-only q99; OOD excluded','Ng=1 is full-feedback/empty omitted set']},indent=2,allow_nan=True))

if __name__=='__main__': main()
