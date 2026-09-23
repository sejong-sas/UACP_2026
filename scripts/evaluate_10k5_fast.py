#!/usr/bin/env python3
"""Fast common-set evaluation for the three 10k x 5 checkpoints only."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.diagnose_predictor import _make_model
from scripts.train_predictor import load_config
from scripts.evaluate_diversity_reproducibility import REGIMES, evaluate_model

def main():
 p=argparse.ArgumentParser(); p.add_argument('--common-eval-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--samples-per-regime',type=int,default=10000); p.add_argument('--batch-size',type=int,default=256); a=p.parse_args()
 out=ROOT/a.output_dir; out.mkdir(parents=True,exist_ok=True)
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); common=ROOT/a.common_eval_dir
 models=[]
 for seed in (20260819,20260915,20260916):
  models.append((f'D{seed} 10k x 5',load_config(f'configs/current_valid_baseline_10k5_seed_{seed}.json'),ROOT/f'runs/current_valid_baseline/diversity_ablation/tenk_5ep_seed_{seed}/uacp_predictor_step4a.pt'))
 paths={label:str(common/(label.replace(' ','_').replace('/','_')+'.npz')) for label,_ in REGIMES}; rows=[]
 for name,cfg,ckpt in models:
  m=_make_model(cfg,device); payload=torch.load(ckpt,map_location=device,weights_only=False); m.load_state_dict(payload.get('model_state_dict',payload) if isinstance(payload,dict) else payload); m.eval()
  rows.extend(evaluate_model(m,paths,device,a.samples_per_regime,a.batch_size,20262000,name))
 summary=[]
 for name,_,_ in models:
  for label,_ in REGIMES:
   z=[r for r in rows if r['model']==name and r['regime']==label]
   summary.append({'model':name,'regime':label,'samples':len(z),**{f'{k}_{s}':float((np.median(v) if s=='median' else getattr(v,s)())) for k in ('nmse_omitted_db','aleatoric','epistemic') for s,v in ((s,np.asarray([r[k] for r in z])) for s in ('mean','median','std'))}})
 by={(r['model'],r['regime']):r for r in summary}; gaps=[]
 for name,_,_ in models:
  idmax=max(by[(name,'ID-Easy 20 ns')]['epistemic_mean'],by[(name,'ID-Hard 80 ns')]['epistemic_mean'])
  gaps.append({'model':name,'delta_ale':by[(name,'ID-Hard 80 ns')]['aleatoric_mean']-by[(name,'ID-Easy 20 ns')]['aleatoric_mean'],'near_gap':by[(name,'OOD-Near 120 ns')]['epistemic_mean']-idmax,'far_gap':by[(name,'OOD-Far 1 ms')]['epistemic_mean']-idmax})
 for fn,data in [('summary.csv',summary),('gaps.csv',gaps),('per_sample.csv',rows)]:
  with (out/fn).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
 (out/'results.json').write_text(json.dumps({'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'samples_per_regime':a.samples_per_regime,'updates':6250,'gaps':gaps},indent=2))
 print(json.dumps({'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gaps':gaps},indent=2),flush=True)
if __name__=='__main__': main()
