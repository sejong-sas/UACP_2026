#!/usr/bin/env python3
"""Fig.8-only 100k-vs-50k evaluation; no Fig.9/calibration code."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_predictor import _make_model
from scripts.train_predictor import load_config
from scripts.final_pretrained_fig8_fig9_validation import REGIMES,eval_model

def auc(s,l):
 p=s[l==1];n=s[l==0];return float((p[:,None]>n[None,:]).mean()+.5*(p[:,None]==n[None,:]).mean())
def boot(s,l,seed=20261501,draws=500):
 rng=np.random.default_rng(seed);v=[]
 for _ in range(draws):
  i=rng.integers(0,len(s),len(s));
  if len(np.unique(l[i]))==2:v.append(auc(s[i],l[i]))
 return {'mean':float(np.mean(v)),'ci95_low':float(np.quantile(v,.025)),'ci95_high':float(np.quantile(v,.975)),'draws':len(v),'seed':seed}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--common-eval-dir',required=True);p.add_argument('--batch-size',type=int,default=256);a=p.parse_args();out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
 cfg=load_config('configs/current_valid_baseline_100k1_seed_20260819.json');ck=ROOT/'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt';dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');m=_make_model(cfg,dev);pay=torch.load(ck,map_location=dev,weights_only=False);m.load_state_dict(pay.get('model_state_dict',pay) if isinstance(pay,dict) else pay);m.eval()
 rows,_=eval_model(m,ROOT/a.common_eval_dir,dev,10000,a.batch_size,20261509)
 for r in rows:r['epistemic_db']=10*np.log10(max(r['epistemic'],1e-12))
 stats=[]
 for label,_ in REGIMES:
  x=np.array([r['epistemic_db'] for r in rows if r['regime']==label]);stats.append({'regime':label,'mean_db':x.mean(),'median_db':np.median(x),'std_db':x.std(),'q05_db':np.quantile(x,.05),'q25_db':np.quantile(x,.25),'q75_db':np.quantile(x,.75),'q95_db':np.quantile(x,.95),'linear_mean':np.mean([r['epistemic'] for r in rows if r['regime']==label])})
 by={x['regime']:x for x in rows}; id80=np.array([r['epistemic_db'] for r in rows if r['regime']=='ID-Hard 80 ns']); near=np.array([r['epistemic_db'] for r in rows if r['regime']=='OOD-Near 120 ns']); id20=np.array([r['epistemic_db'] for r in rows if r['regime']=='ID-Easy 20 ns']); far=np.array([r['epistemic_db'] for r in rows if r['regime']=='OOD-Far 1 ms']); idpool=np.r_[id20,id80]
 aucs={'id_vs_ood_pooled':auc(np.r_[idpool,near,far],np.r_[np.zeros(len(idpool)),np.ones(len(near)+len(far))]),'id_vs_near':auc(np.r_[idpool,near],np.r_[np.zeros(len(idpool)),np.ones(len(near))]),'id_vs_far':auc(np.r_[idpool,far],np.r_[np.zeros(len(idpool)),np.ones(len(far))]),'id_hard_80_vs_near_120':auc(np.r_[id80,near],np.r_[np.zeros(len(id80)),np.ones(len(near))])}
 labels=np.r_[np.zeros(len(idpool)),np.ones(len(near)+len(far))];scores=np.r_[idpool,near,far]; boots={'id_vs_ood_pooled':boot(scores,labels),'id_vs_near':boot(np.r_[idpool,near],np.r_[np.zeros(len(idpool)),np.ones(len(near))]),'id_vs_far':boot(np.r_[idpool,far],np.r_[np.zeros(len(idpool)),np.ones(len(far))]),'id_hard_80_vs_near_120':boot(np.r_[id80,near],np.r_[np.zeros(len(id80)),np.ones(len(near))])}
 # Reuse the previously saved 50k sample-level scores for an apples-to-apples dB plot.
 old=ROOT/'runs/current_valid_baseline/diversity_ablation/reproducibility_20260914_retry/per_sample.csv'; oldrows=[]
 if old.exists():
  with old.open() as f:
   for r in csv.DictReader(f):
    if r['model'].startswith('B1 '):oldrows.append({'regime':r['regime'],'epistemic_db':10*np.log10(max(float(r['epistemic']),1e-12)),'model':'50k x 1'})
 for r in rows:r['model']='100k x 1'
 with(out/'per_sample.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for fn,data in [('distribution.csv',stats),('auroc.csv',[{'metric':k,'value':v} for k,v in aucs.items()])]:
  with(out/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.figure(figsize=(9,5));
 for label,_ in REGIMES:
  x=[r['epistemic_db'] for r in rows if r['regime']==label];plt.hist(x,bins=45,density=True,histtype='step',linewidth=1.8,label=label)
 plt.xlabel('Epistemic Uncertainty (dB)');plt.ylabel('Density');plt.grid(alpha=.2);plt.legend();plt.tight_layout();plt.savefig(out/'fig8_100k_epistemic_db.png',dpi=180);plt.close()
 plt.figure(figsize=(9,5));
 for model,rr in [('50k x 1',oldrows),('100k x 1',rows)]:
  for label,_ in REGIMES:
   x=[r['epistemic_db'] for r in rr if r['regime']==label];
   if x:plt.hist(x,bins=45,density=True,histtype='step',linewidth=1.2,label=f'{model} {label}')
 plt.xlabel('Epistemic Uncertainty (dB)');plt.ylabel('Density');plt.grid(alpha=.2);plt.legend(fontsize=7,ncol=2);plt.tight_layout();plt.savefig(out/'fig8_50k_vs_100k_epistemic_db.png',dpi=180);plt.close()
 result={'device':str(dev),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'checkpoint':str(ck),'samples_per_regime':10000,'dB_conversion':'IMPLEMENTATION-ASSUMPTION: sample-level U_epi_dB=10*log10(U_epi), because Eq.(8)/(12)/(13) score is variance/covariance-trace-like','distribution':stats,'auroc':aucs,'bootstrap':boots,'comparison_50k_source':str(old),'no_fig9':True}
 (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
