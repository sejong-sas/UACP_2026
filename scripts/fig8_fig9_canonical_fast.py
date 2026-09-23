#!/usr/bin/env python3
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_predictor import _make_model
from scripts.train_predictor import load_config
from scripts.final_pretrained_fig8_fig9_validation import eval_model,REGIMES,NOMINALS,pairwise_auc

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--common-eval-dir',required=True);p.add_argument('--samples-per-regime',type=int,default=10000);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--config',default='configs/current_valid_baseline_10k5_seed_20260819.json');p.add_argument('--checkpoint',default='runs/current_valid_baseline/diversity_ablation/tenk_5ep_seed_20260819/uacp_predictor_step4a.pt');p.add_argument('--with-calibration',action='store_true');a=p.parse_args();out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
 cfg=load_config(a.config);ck=ROOT/a.checkpoint;dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');m=_make_model(cfg,dev);pay=torch.load(ck,map_location=dev,weights_only=False);m.load_state_dict(pay.get('model_state_dict',pay) if isinstance(pay,dict) else pay);m.eval();common=ROOT/a.common_eval_dir
 # Reuse the validated evaluator, including exact common CFR/mask/noise realization.
 rows,comps=eval_model(m,common,dev,a.samples_per_regime,a.batch_size,20262000,do_calibration=a.with_calibration)
 sums=[]
 for label,_ in REGIMES:
  z=np.array([r['epistemic'] for r in rows if r['regime']==label]);sums.append({'regime':label,'mean':z.mean(),'median':np.median(z),'std':z.std(),'p05':np.quantile(z,.05),'p25':np.quantile(z,.25),'p75':np.quantile(z,.75),'p95':np.quantile(z,.95)})
 idv=np.array([r['epistemic'] for r in rows if r['regime'] in ('ID-Easy 20 ns','ID-Hard 80 ns')]);near=np.array([r['epistemic'] for r in rows if r['regime']=='OOD-Near 120 ns']);far=np.array([r['epistemic'] for r in rows if r['regime']=='OOD-Far 1 ms']);au={'overall_id_vs_ood':pairwise_auc(np.r_[idv,near,far],np.r_[np.zeros(len(idv)),np.ones(len(near)+len(far))]),'id_vs_near':pairwise_auc(np.r_[idv,near],np.r_[np.zeros(len(idv)),np.ones(len(near))]),'id_vs_far':pairwise_auc(np.r_[idv,far],np.r_[np.zeros(len(idv)),np.ones(len(far))])}
 cal=[]
 if comps:
  for nominal in NOMINALS:
   for pool,labels in [('ID',('ID-Easy 20 ns','ID-Hard 80 ns')),('OOD',('OOD-Near 120 ns','OOD-Far 1 ms'))]:
    z=[x for x in comps if x['nominal']==float(nominal) and x['regime'] in labels];cal.append({'pool':pool,'nominal':float(nominal),'empirical':sum(x['covered'] for x in z)/sum(x['count'] for x in z),'count':sum(x['count'] for x in z)})
 ce={pool:float(np.mean([abs(x['empirical']-x['nominal']) for x in cal if x['pool']==pool])) for pool in ('ID','OOD')} if cal else {}
 outputs=[('fig8_distribution.csv',sums)] + ([('fig9_calibration.csv',cal)] if cal else [])
 for fn,data in outputs:
  with(out/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 result={'device':str(dev),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'checkpoint':str(ck),'fig8_distribution':sums,'fig8_auroc':au,'fig9_calibration_error_mae':ce,'fig9_calibration':cal,'assumptions':['Diagonal-Psi approximation','Eq.(5) marginal Student-t intervals','ID=(20,80), OOD=(120,1ms) equal-count pooling for AUROC','Coverage aggregates omitted subcarrier real/imag components','Calibration error is MAE over nominal 0.1..0.9 grid']};(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
