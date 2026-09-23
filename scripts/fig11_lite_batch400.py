#!/usr/bin/env python3
"""Fig.11-lite: only causal Ng and omitted-NMSE trajectory."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.fig11_dynamic_runtime_validation import aggregate_omitted_scores, make_mask, nmse_omitted_db, select_next_ng
from scripts.diagnose_predictor import _make_model
from scripts.train_predictor import load_config, set_seeds
from src.training.data import build_noisy_sparse_input
from src.channel.sionna_channel import load_sectioned_config
REGIMES=[("20 ns",20), ("80 ns",80), ("10 ns",10), ("40 ns",40), ("60 ns",60), ("120 ns",120)]
NGS=[128,64,32,16,8,4,1]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--segment-length",type=int,default=40); a=p.parse_args()
 out=ROOT/a.output_dir
 if out.exists() and any(out.iterdir()): raise FileExistsError(out)
 out.mkdir(parents=True); cfg=load_config("configs/current_valid_baseline_10k5_seed_20260819.json"); dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 if dev.type!="cuda": raise RuntimeError("CUDA required")
 model=_make_model(cfg,dev); pay=torch.load(ROOT/a.checkpoint,map_location=dev,weights_only=False); model.load_state_dict(pay.get("model_state_dict",pay) if isinstance(pay,dict) else pay); model.eval()
 seqdir=ROOT/"runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"; seed=20262000; sequence=[]
 for regime,delay in REGIMES:
  fn="test_delay_1_ms.npz" if delay==1_000_000 else f"test_delay_{delay}_ns.npz"
  with np.load(seqdir/fn) as z: sequence.extend((regime,delay,x) for x in z["cfr"][:a.segment_length])
 # Fixed ID-only threshold and aleatoric target, matching the existing controller evaluator.
 threshold=0.0; ale_target=0.0; threshold_values=[]; probe_values=[]
 for ri,delay in enumerate((20,80)):
  with np.load(seqdir/f"test_delay_{delay}_ns.npz") as z: cfr=z["cfr"][:40]
  vals=[]
  for start in range(0,len(cfr),8):
   set_seeds(seed+500000+ri*10000+start); c=torch.from_numpy(cfr[start:start+8]).to(dev); mask=make_mask(c.shape[0],16,dev); x,_,_=build_noisy_sparse_input(c,mask,15.0)
   with torch.inference_mode(): o=model(x); ale=o.psi/(o.nu_expanded-2049.0); epi=ale/o.kappa_expanded; av,ev=aggregate_omitted_scores(ale,epi,mask)
   vals.append(ev); threshold_values.extend([ev]); probe_values.append(av)
 threshold=float(np.quantile(np.asarray(threshold_values),.99)); ale_target=float(np.mean(probe_values)); ale_delta=.1*ale_target
 trace=[]; current=128; started=time.perf_counter()
 for step,(regime,delay,cfr_np) in enumerate(sequence):
  set_seeds(seed+step); c=torch.from_numpy(cfr_np[None]).to(dev); mask=make_mask(1,current,dev); x,target,_=build_noisy_sparse_input(c,mask,15.0)
  with torch.inference_mode():
   o=model(x); ale=o.psi/(o.nu_expanded-2049.0); epi=ale/o.kappa_expanded; av,ev=aggregate_omitted_scores(ale,epi,mask); nm=nmse_omitted_db(o.gamma,target,mask)
   nm_all=float((10*torch.log10(((o.gamma-target).square().sum()/target.square().sum()).clamp_min(1e-12))).detach().cpu())
  d=select_next_ng(current,av,ev,threshold,ale_target,ale_delta); trace.append({"time_step":step,"regime":regime,"delay_spread_ns":delay,"current_ng":current,"predicted_next_ng":d["next_ng"],"epistemic_score":ev,"epistemic_threshold":threshold,"aleatoric_score":av,"nmse_omitted_db":nm,"nmse_all_db":nm_all,"decision_reason":d["decision_reason"]}); current=int(d["next_ng"])
 rows=[]
 for regime,delay in REGIMES:
  z=[r for r in trace if r["regime"]==regime]; rows.append({"regime":regime,"delay_spread_ns":delay,"steps":len(z),"mean_ng":float(np.mean([r["current_ng"] for r in z])),"median_ng":float(np.median([r["current_ng"] for r in z])),"mean_nmse_omitted_db":float(np.nanmean([r["nmse_omitted_db"] for r in z])),"min_nmse_omitted_db":float(np.nanmin([r["nmse_omitted_db"] for r in z])) if np.isfinite(np.asarray([r["nmse_omitted_db"] for r in z])).any() else None,"mean_nmse_all_db":float(np.mean([r["nmse_all_db"] for r in z]))})
 for fn,data in [("dynamic_trace.csv",trace),("regime_summary.csv",rows)]:
  with (out/fn).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
 x=np.arange(len(trace)); fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True); ax[0].step(x,[r["current_ng"] for r in trace],where="post",color="#1f77b4"); ax[0].set_ylabel("Ng"); ax[0].grid(alpha=.25); ax[1].plot(x,[r["nmse_omitted_db"] for r in trace],color="#d62728"); ax[1].set_ylabel("NMSE [dB]"); ax[1].set_xlabel("Time step"); ax[1].grid(alpha=.25)
 for i in range(1,len(trace)):
  if trace[i]["regime"]!=trace[i-1]["regime"]: ax[0].axvline(i,color="k",alpha=.2); ax[1].axvline(i,color="k",alpha=.2)
 fig.tight_layout(); fig.savefig(out/"fig11_lite_ng_nmse.png",dpi=240); fig.savefig(out/"fig11_lite_ng_nmse.pdf"); plt.close(fig)
 result={"checkpoint":a.checkpoint,"device":str(dev),"gpu":torch.cuda.get_device_name(0),"sequence":REGIMES,"segment_length":a.segment_length,"steps":len(trace),"threshold":threshold,"ale_target":ale_target,"runtime_seconds":time.perf_counter()-started,"training_performed":False,"adaptation_update":False,"assumptions":["IMPLEMENTATION-ASSUMPTION: existing delay-sweep samples represent ordered online sequence; no temporal-correlated generator found","IMPLEMENTATION-ASSUMPTION: ID-only 99th percentile Eq.(13) threshold and 10% aleatoric hysteresis","Ng/NMSE only; BER/EVM/precoding/adaptation omitted"],"regime_summary":rows}
 (out/"results.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
