#!/usr/bin/env python3
"""12-combination quick gate for frozen/recovered mask-coverage checkpoints."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
def q(v):
 a=np.asarray(v,float); return {'mean':float(a.mean()),'median':float(np.median(a)),'p05':float(np.quantile(a,.05)),'p95':float(np.quantile(a,.95))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--samples-per-combination',type=int,default=50);p.add_argument('--batch-size',type=int,default=25);a=p.parse_args();out=ROOT/a.output_dir
 if out.exists() and any(out.iterdir()): raise FileExistsError(out)
 out.mkdir(parents=True);sys.path.insert(0,str(ROOT))
 from scripts.train_predictor import load_config,set_seeds
 from scripts.diagnose_predictor import _make_model
 from scripts.fig11_dynamic_runtime_validation import aggregate_omitted_scores,make_mask
 from src.training.data import build_noisy_sparse_input
 cfg=load_config('runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json');d=torch.device('cuda:0');m=_make_model(cfg,d);pay=torch.load(ROOT/a.checkpoint,map_location=d,weights_only=False);m.load_state_dict(pay.get('model_state_dict',pay) if isinstance(pay,dict) else pay);m.eval();rows=[];delays=[20,80,120];ngs=[16,32,64,128]
 for di,delay in enumerate(delays):
  fn=f'test_delay_{delay}_ns.npz';cfr=np.load(ROOT/'runs/baseline_reproduction/step2_delay_sweep_repro/generated_data'/fn)['cfr'][:a.samples_per_combination]
  for ng in ngs:
   avs=[];evs=[];alls=[];oms=[];nf=0
   for st in range(0,len(cfr),a.batch_size):
    set_seeds(20263000+di*10000+ng*100+st);c=torch.from_numpy(cfr[st:st+a.batch_size]).to(d);mask=make_mask(c.shape[0],ng,d);x,t,_=build_noisy_sparse_input(c,mask,15.0)
    with torch.inference_mode():o=m(x);e=(o.predicted-t).square();om=(1-mask)[:,None,:].expand_as(t);alls.extend((10*torch.log10((e.sum((1,2))/t.square().sum((1,2)).clamp_min(1e-12)).clamp_min(1e-12))).cpu().numpy().tolist());den=t.square().mul(om).sum((1,2)).clamp_min(1e-12);oms.extend((10*torch.log10((e.mul(om).sum((1,2))/den).clamp_min(1e-12))).cpu().numpy().tolist());ale=o.aleatoric;epi=o.epistemic;pa=ale.reshape(ale.shape[0],2,4,1024).sum(1).mean(1);pe=epi.reshape(epi.shape[0],2,4,1024).sum(1).mean(1);keep=1-mask;cnt=keep.sum(1).clamp_min(1);avs.extend(((pa*keep).sum(1)/cnt).cpu().numpy().tolist());evs.extend(((pe*keep).sum(1)/cnt).cpu().numpy().tolist());nf+=int((~torch.isfinite(o.nu_expanded)).sum().cpu())
   rows.append({'delay_ns':delay,'ng':ng,'observed_subcarriers':1024//ng,'samples':len(alls),'all_nmse_mean_db':q(alls)['mean'],'omitted_nmse_mean_db':q(oms)['mean'],'aleatoric_mean':q(avs)['mean'],'epistemic_mean':q(evs)['mean'],'nu_nonfinite_components':nf,'all_nmse_median_db':q(alls)['median'],'omitted_nmse_median_db':q(oms)['median'],'aleatoric_median':q(avs)['median'],'epistemic_median':q(evs)['median']})
 with (out/'quick_gate.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'checkpoint':a.checkpoint,'samples_per_combination':a.samples_per_combination,'combinations':len(rows),'delays_ns':delays,'ngs':ngs,'training_performed':False,'nu_nonfinite_total':sum(int(r['nu_nonfinite_components']) for r in rows)}; (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
