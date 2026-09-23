#!/usr/bin/env python3
"""ID-only Ng-conditioned q99 calibration and restricted Fig.11 replay."""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt'
DATA_DIR=ROOT/'runs/baseline_reproduction/step2_delay_sweep_repro/generated_data'
SCORE_FILE=ROOT/'runs/current_valid_baseline/epoch3_threshold_diagnosis_20260919/step1_step3_final/step3_id_ng_scores.csv'
NGS=[4,8,16,32]; INITIAL_NG=128; K=1024; GLOBAL=52.45694942474356
REGIMES=[('20 ns',20.0),('80 ns',80.0),('10 ns',10.0),('40 ns',40.0),('60 ns',60.0),('120 ns',120.0)]

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def calibrate():
    rows=list(csv.DictReader(SCORE_FILE.open())); values={ng:[] for ng in NGS}
    test_rows=[]
    for r in rows:
        ng=int(r['ng']); delay=int(r['delay_ns']); idx=int(r['sample']); v=float(r['epistemic'])
        if ng in values:
            if idx<100: values[ng].append(v)
            else: test_rows.append((delay,ng,idx,v))
    thresholds={ng:float(np.quantile(v,.99)) for ng,v in values.items()}
    cal_rows=[]
    for ng in NGS:
        v=np.asarray(values[ng]); cal_rows.append({'ng':ng,'sample_count':len(v),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'q99':float(np.quantile(v,.99)),'max':float(np.max(v)),'global_threshold':GLOBAL})
    test_summary=[]
    for delay in [10,20,40,60,80,100]:
        for ng in NGS:
            v=np.asarray([x[3] for x in test_rows if x[0]==delay and x[1]==ng]); tau=thresholds[ng]
            test_summary.append({'delay_ns':delay,'ng':ng,'sample_count':len(v),'threshold':tau,'false_trigger_count':int(np.sum(v>tau)),'false_trigger_rate':float(np.mean(v>tau)),'median':float(np.median(v)),'max':float(np.max(v))})
    return thresholds,cal_rows,test_summary

def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    m=_make_model(load_config('configs/current_valid_baseline_100k1_seed_20260819.json'),device); p=torch.load(CHECKPOINT,map_location=device,weights_only=False); s=p['model_state_dict'] if isinstance(p,dict) and 'model_state_dict' in p else p; m.load_state_dict(s); m.eval(); return m

def decision(current,ale,epi,tau,ale_target,ale_delta):
    if np.isfinite(epi) and epi>tau: return 1,True,'epistemic_threshold'
    if current==1 or not np.isfinite(ale): return current,False,'hold_no_omitted_set'
    candidates=[32,16,8,4]
    if current not in candidates: return 32,False,'restricted_candidate_projection'
    i=candidates.index(current)
    if ale>ale_target+ale_delta: return candidates[min(i+1,len(candidates)-1)],False,'aleatoric_denser'
    if ale<ale_target-ale_delta: return candidates[max(i-1,0)],False,'aleatoric_sparser'
    return current,False,'hysteresis_hold'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--segment-length',type=int,default=40); ap.add_argument('--eval-seed',type=int,default=20262000); a=ap.parse_args(); out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); thresholds,cal_rows,test_rows=calibrate(); write_csv(out/'id_calibration_thresholds.csv',cal_rows); write_csv(out/'id_holdout_false_trigger.csv',test_rows)
    sys.path.insert(0,str(ROOT)); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    from scripts.evaluate_epoch3_restricted_ng_fig11 import evaluate_step
    model=load_model(device); prior=json.loads((ROOT/'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json').read_text()); probes=prior['epoch_probe_results']['1']; ale_target=(probes['ID-Easy 20 ns']['aleatoric_paper_eq12_eq13']+probes['ID-Hard 80 ns']['aleatoric_paper_eq12_eq13'])/2; ale_delta=.1*ale_target
    rows=[]; current=INITIAL_NG; step=0
    for regime,delay in REGIMES:
        path=DATA_DIR/('test_delay_1_ms.npz' if delay==1_000_000 else f'test_delay_{int(delay)}_ns.npz')
        with np.load(path) as d: cfr=d['cfr'][:a.segment_length]
        for sample in cfr:
            tau=thresholds.get(current,thresholds[32]); ale,epi,nmse,observed,omitted=evaluate_step(model,sample,current,a.eval_seed+step,device); nxt,trig,reason=decision(current,ale,epi,tau,ale_target,ale_delta)
            rows.append({'time_step':step,'regime':regime,'delay_ns':delay,'current_ng':current,'next_ng':nxt,'aleatoric':ale,'epistemic':epi,'tau_ng':tau,'epi_tau_ratio':epi/tau if np.isfinite(epi) else np.nan,'ood_trigger':trig,'nmse_db':nmse,'observed_count':observed,'omitted_count':omitted,'decision_reason':reason,'ood':delay>100}); current=int(nxt); step+=1
    write_csv(out/'dynamic_trace.csv',rows)
    summary=[]
    for regime,delay in REGIMES:
        s=[r for r in rows if r['regime']==regime]; valid=[r for r in s if np.isfinite(float(r['nmse_db']))]; summary.append({'regime':regime,'delay_ns':delay,'mean_current_ng':float(np.mean([r['current_ng'] for r in s])),'mean_nmse_db':float(np.mean([r['nmse_db'] for r in valid])) if valid else 'N/A (full-feedback / empty omitted set)','false_trigger_count':sum(bool(r['ood_trigger']) and not r['ood'] for r in s),'ood_trigger_count':sum(bool(r['ood_trigger']) and r['ood'] for r in s),'first_trigger_step':next((r['time_step'] for r in s if r['ood_trigger']), '')})
    write_csv(out/'regime_summary.csv',summary)
    comparison=[]
    old=list(csv.DictReader((ROOT/'runs/current_valid_baseline/epoch3_baseline_validation_20260919/fig11/regime_summary.csv').open())); restricted=list(csv.DictReader((ROOT/'runs/current_valid_baseline/epoch3_restricted_ng_fig11_20260919/regime_summary.csv').open()))
    for label,data in [('A_global_full',old),('B_global_restricted',restricted),('C_conditioned_restricted',summary)]:
        for r in data: comparison.append({'condition':label,**r})
    write_csv(out/'controller_comparison.csv',comparison)
    (out/'results.json').write_text(json.dumps({'checkpoint':str(CHECKPOINT),'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'calibration_source':str(SCORE_FILE),'calibration_delays_ns':[10,20,40,60,80,100],'calibration_samples_per_delay_ng':100,'threshold_definition':'tau_Ng = pooled ID-only q99 per Ng from first 100 samples; second 100 samples holdout','thresholds':thresholds,'global_threshold':GLOBAL,'candidate_ngs':NGS,'initial_ng':INITIAL_NG,'sequence':[x[0] for x in REGIMES],'OOD_used_for_calibration':False,'no_training':True,'IMPLEMENTATION_ASSUMPTION':['per-Ng pooled ID q99 threshold is not paper-specified','first/second split of existing 200-sample ID artifacts is used for calibration/holdout','restricted candidate semantics retain initial 128 for first observation then project to 32','Ng=1 metrics are N/A after full-feedback fallback']},indent=2,allow_nan=True))

if __name__=='__main__': main()
