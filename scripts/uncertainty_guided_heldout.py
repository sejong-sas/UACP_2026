#!/usr/bin/env python3
"""Experiment B after a passed Experiment-A H1 gate."""
from pathlib import Path
import json, hashlib, time, shutil, os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("UACP_EVAL_BATCH_SIZE", "128")
from scripts.train_predictor import load_config
from scripts.uncertainty_guided_scope_experiment import (
    ROOT,CHECKPOINT,CONFIG,NGS,PRIMARY_NGS,SCOPES,SEED,EPOCHS,BATCH,LR,
    model_and_config,load_state,train_one,evaluate_model,write_csv,sha256_file
)

OUT=ROOT/'runs/uncertainty_guided_adaptation_20260923_v9/heldout'
DELAYS=[130,180,220]; SEEDS=[20260921,20260922,20260923]; N=5000

def gen_data():
    from src.channel.sionna_channel import generate_cfr_for_delay_spreads, load_sectioned_config
    cfg=load_sectioned_config(ROOT/'configs/adaptation_120ns_protocol_20260921.json'); d=OUT/'datasets'; d.mkdir(parents=True,exist_ok=True); paths={}; rows=[]
    specs=[]
    for delay in DELAYS:
        for split,count,off in [('adapt_train',1000,11),('adapt_val',200,22),('target_test',N,33)]: specs.append((f'{delay}ns_{split}',delay,split,count,off))
    for name,delay,split,count,off in specs:
        p=d/f'{delay}ns_{split}.npz';
        if not p.exists():
            cfr=generate_cfr_for_delay_spreads(cfg,np.full(count,float(delay),np.float32),seed=SEEDS[0]+delay*100+off)
            np.savez(p,cfr=cfr.astype(np.complex64),delay_spread_ns=np.full(count,float(delay),np.float32),regime_label=np.array([split]*count,dtype='U32'),metadata_json=np.array(json.dumps({'delay_ns':delay,'split':split,'blind_1ms':False})))
        paths[name]=p
    for name,delay in [('id20',20),('id80',80),('far1ms',1_000_000)]:
        p=d/f'common_{name}.npz'
        if not p.exists():
            cfr=generate_cfr_for_delay_spreads(cfg,np.full(N,float(delay),np.float32),seed=SEEDS[0]+90000+delay)
            np.savez(p,cfr=cfr.astype(np.complex64),delay_spread_ns=np.full(N,float(delay),np.float32),regime_label=np.array([name]*N,dtype='U32'),metadata_json=np.array(json.dumps({'delay_ns':delay,'split':name,'blind_1ms':delay==1_000_000})))
        paths[name]=p
    return paths

def main():
    if not json.loads((ROOT/'runs/uncertainty_guided_adaptation_20260923_v8/h1_gate_corrected.json').read_text())['h1_supported']:
        raise RuntimeError('H1 gate is not supported; Experiment B must not run')
    OUT.mkdir(parents=True,exist_ok=True); paths=gen_data(); cfg=load_config(CONFIG); device=torch.device('cuda:0')
    # Ordered policy learned from calibration only: the midpoint between the
    # highest small-scope and lowest medium-scope calibration severities.
    cal=pd.read_csv(ROOT/'runs/uncertainty_guided_adaptation_20260923_v8/h1_plot_data.csv'); t=float((cal[cal.delay_ns==160].severity_median_z_primary.iloc[0]+cal[cal.delay_ns==200].severity_median_z_primary.iloc[0])/2)
    (OUT/'policy.json').write_text(json.dumps({'policy':'z < t -> partial_small; z >= t -> partial_medium','t':t,'calibration_only':True},indent=2))
    # Pre severity and selected scope per held-out delay.
    pre,_=model_and_config(device); load_state(pre,CHECKPOINT,device); pre.eval(); thresholds=pd.read_csv(ROOT/'runs/current_valid_baseline/epoch3_ng_conditioned_threshold_fig11_20260919_final/id_calibration_thresholds.csv').set_index('ng')['q99'].to_dict()
    from scripts.uncertainty_guided_scope_experiment import evaluate_model
    selections=[]
    for delay in DELAYS:
        rows,_=evaluate_model(pre,paths[f'{delay}ns_target_test'],cfg,device,SEED,include_samples=True)
        z=[np.log10(max(r['epistemic'],1e-30)/thresholds[r['ng']]) for r in rows if r['ng'] in PRIMARY_NGS]
        med=float(np.median(z)); choice='partial_small' if med<t else 'partial_medium'; selections.append({'delay_ns':delay,'severity_median_z_primary':med,'selected_scope':choice,'policy_threshold':t})
    del pre; torch.cuda.empty_cache(); write_csv(OUT/'heldout_policy_selection.csv',selections)
    training=[]; evaluation=[]
    for delay,sel in [(r['delay_ns'],r['selected_scope']) for r in selections]:
        for seed in SEEDS:
            for scope_name,scope,ord_ in SCOPES:
                print(json.dumps({'status':'heldout_training','delay_ns':delay,'seed':seed,'scope':scope_name}),flush=True)
                run=OUT/f'delay_{delay}ns'/f'seed_{seed}'/scope_name; run.parent.mkdir(parents=True,exist_ok=True)
                run_root=OUT/'training_runs'/f'delay_{delay}ns'/f'seed_{seed}'
                m=train_one(scope_name,scope,delay,paths,run_root,device,cfg,seed=seed)
                # train_one uses its own delay/seed directory; preserve a provenance link by copying the run.
                src=run_root/'runs'/f'delay_{delay}ns'/scope_name
                dest=run
                if not dest.exists(): shutil.copytree(src,dest)
                training.append({**m,'heldout_seed':seed,'heldout_delay_ns':delay})
                model,_=model_and_config(device); load_state(model,src/'adapted_epoch_3.pt',device); model.eval()
                for regime,path in [('target',paths[f'{delay}ns_target_test']),('id20',paths['id20']),('id80',paths['id80']),('far1ms',paths['far1ms'])]:
                    _,s=evaluate_model(model,path,cfg,device,seed,include_samples=False)
                    for key,val in s.items():
                        ng,metric=key.split('_',1); evaluation.append({'delay_ns':delay,'seed':seed,'scope':scope_name,'regime':regime,'ng':int(ng),**val})
                del model; torch.cuda.empty_cache()
    write_csv(OUT/'training_summary.csv',training); write_csv(OUT/'evaluation_summary.csv',evaluation)
    (OUT/'integrity_summary.json').write_text(json.dumps({'checkpoint_sha256':sha256_file(CHECKPOINT),'expected_sha256':'d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986','cuda':torch.cuda.get_device_name(0),'one_ms_training_used':False},indent=2))

if __name__=='__main__': main()
