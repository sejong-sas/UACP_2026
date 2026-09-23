#!/usr/bin/env python3
"""Inference-only absolute CFR amplitude scale sensitivity diagnostic."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CHECKPOINT=ROOT/"runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
COMMON=ROOT/"runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval"
def write(path,rows):
 with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True);p.add_argument("--samples",type=int,default=200);p.add_argument("--seed",type=int,default=20262021);args=p.parse_args();out=ROOT/args.output_dir
 if out.exists() and any(out.iterdir()): raise FileExistsError(out)
 out.mkdir(parents=True);device=torch.device("cuda:0")
 if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0): raise RuntimeError("NVIDIA GB10/cuda:0 required")
 sys.path.insert(0,str(ROOT));from scripts.diagnose_predictor import _make_model;from scripts.train_predictor import load_config
 model=_make_model(load_config(ROOT/"configs/current_valid_baseline_100k1_seed_20260819.json"),device);state=torch.load(CHECKPOINT,map_location=device,weights_only=False);model.load_state_dict(state["model_state_dict"] if isinstance(state,dict) and "model_state_dict" in state else state);model.eval()
 from src.training.data import build_sparse_input
 rows=[]; scales=[.75,1.0,1.25]
 for delay,name in [(20,"ID-Easy_20_ns.npz"),(60,None),(100,None)]:
  if name: pth=COMMON/name
  else: pth=ROOT/"runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"/f"test_delay_{delay}_ns.npz"
  with np.load(pth) as z:cfr_np=np.asarray(z["cfr"][:args.samples],np.complex64)
  base=cfr_np
  for ng in (16,32):
   mask=torch.zeros((len(base),1024),device=device);mask[:,::ng]=1.0
   torch.manual_seed(args.seed+delay*100+ng)
   cfr=torch.from_numpy(base).to(device);signal=(cfr.abs().square()*mask[:,:,None,None]).sum((1,2,3))/mask[:,:,None,None].expand_as(cfr.real).sum((1,2,3)).clamp_min(1)
   noise=torch.complex(torch.randn_like(cfr.real),torch.randn_like(cfr.real));noise=noise*torch.sqrt((signal/(10**1.5))[:,None,None,None]/2)*mask[:,:,None,None]
   for alpha in scales:
    noisy=alpha*cfr+alpha*noise;x,target=build_sparse_input(noisy,mask);clean_target=target
    with torch.inference_mode():
     o=model(x);den=o.nu_expanded-2049;ale=o.psi/den;epi=ale/o.kappa_expanded;om=(1-mask)[:,None,:];
     nmse=10*torch.log10((((o.gamma-clean_target).square()*om).sum((1,2))/(clean_target.square()*om).sum((1,2)).clamp_min(1e-12)).clamp_min(1e-12));a=(ale*om).sum((1,2))/om.sum((1,2)).clamp_min(1);e=(epi*om).sum((1,2))/om.sum((1,2)).clamp_min(1);om2=om.expand_as(ale);k=(o.kappa_expanded*om2).sum((1,2))/om2.sum((1,2)).clamp_min(1);nu=(den*om2).sum((1,2))/om2.sum((1,2)).clamp_min(1);ps=(o.psi*om2).sum((1,2))/om2.sum((1,2)).clamp_min(1)
    for i in range(len(base)):rows.append({"delay_ns":delay,"ng":ng,"sample":i,"alpha":alpha,"nmse":float(nmse[i]),"kappa":float(k[i]),"nu_margin":float(nu[i]),"psi":float(ps[i]),"aleatoric":float(a[i]),"epistemic":float(e[i])})
 write(out/"sample_level.csv",rows); base_rows={(r["delay_ns"],r["ng"],r["sample"]):r for r in rows if r["alpha"]==1.0}; summary=[]
 for d in (20,60,100):
  for ng in (16,32):
   for alpha in scales:
    part=[r for r in rows if r["delay_ns"]==d and r["ng"]==ng and r["alpha"]==alpha]; ratios=[(r["kappa"]/base_rows[(d,ng,r["sample"])] ["kappa"],r["epistemic"]/base_rows[(d,ng,r["sample"])] ["epistemic"]) for r in part];summary.append({"delay_ns":d,"ng":ng,"alpha":alpha,"n":len(part),"kappa_ratio_median":float(np.median([x[0] for x in ratios])),"kappa_ratio_q05":float(np.quantile([x[0] for x in ratios],.05)),"kappa_ratio_q95":float(np.quantile([x[0] for x in ratios],.95)),"epistemic_ratio_median":float(np.median([x[1] for x in ratios])),"epistemic_ratio_q05":float(np.quantile([x[1] for x in ratios],.05)),"epistemic_ratio_q95":float(np.quantile([x[1] for x in ratios],.95))})
 write(out/"scale_summary.csv",summary);(out/"analysis_manifest.json").write_text(json.dumps({"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"device":str(device),"gpu":torch.cuda.get_device_name(0),"alphas":scales,"delays_ns":[20,60,100],"ngs":[16,32],"same_shape":True,"same_target_scale":True,"same_snr_relation":True,"mask_scaled":False,"normalize_channel":False,"normalize_channel_label":"IMPLEMENTATION-ASSUMPTION / diagnostic only","no_training":True,"diagnostic_only":True},indent=2)+"\n");print(json.dumps({"output":str(out.relative_to(ROOT)),"gpu":torch.cuda.get_device_name(0),"summary":summary},indent=2))
if __name__=="__main__":main()
