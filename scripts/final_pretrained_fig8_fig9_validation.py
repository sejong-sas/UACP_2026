#!/usr/bin/env python3
"""Final pretrained-recipe comparison and Fig.8/Fig.9-style diagnostics."""
from __future__ import annotations

import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.diagnose_predictor import _make_model
from scripts.train_predictor import load_config, set_seeds
from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
from src.channel.sionna_channel import load_sectioned_config

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]
NOMINALS = np.arange(0.1, 1.0, 0.1)

def boot_auc(scores, labels, seed=20261101, draws=2000):
    rng=np.random.default_rng(seed); vals=[]; n=len(scores)
    for _ in range(draws):
        ix=rng.integers(0,n,n)
        if len(np.unique(labels[ix]))<2: continue
        vals.append(pairwise_auc(scores[ix], labels[ix]))
    return {"mean":float(np.mean(vals)),"ci95_low":float(np.quantile(vals,.025)),"ci95_high":float(np.quantile(vals,.975)),"draws":len(vals),"seed":seed}

def pairwise_auc(scores, labels):
    pos=scores[labels==1]; neg=scores[labels==0]
    return float((pos[:,None]>neg[None,:]).mean()+0.5*(pos[:,None]==neg[None,:]).mean())

def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

@torch.no_grad()
def eval_model(model, common_dir, device, count, batch_size, model_seed, do_calibration=False):
    sample_rows=[]; component_rows=[]
    for ri,(label,_) in enumerate(REGIMES):
        cfr=np.load(common_dir/(label.replace(" ","_").replace("/","_")+".npz"))["cfr"]
        for start in range(0,count,batch_size):
            c=torch.from_numpy(cfr[start:start+batch_size]).to(device); n=c.shape[0]
            mask=uniform_grouping_mask(n,1024,16,device); set_seeds(model_seed+ri*100000+start)
            x,target,_=build_noisy_sparse_input(c,mask,15.0); out=model(x)
            omitted=(1-mask).bool(); err=(out.predicted-target).square()
            nmse=10*torch.log10((err.mul(omitted[:,None,:]).sum((1,2))/target.square().mul(omitted[:,None,:]).sum((1,2)).clamp_min(1e-12)).clamp_min(1e-12))
            # Eq.(12)/(13): pair Re/Imag trace, four-pair mean, omitted-subcarrier mean.
            ale=out.aleatoric.reshape(n,2,4,1024).sum(1).mean(1); epi=out.epistemic.reshape(n,2,4,1024).sum(1).mean(1)
            om_count=omitted.sum(1).clamp_min(1)
            ale=((ale*omitted).sum(1)/om_count).cpu().numpy(); epi=((epi*omitted).sum(1)/om_count).cpu().numpy()
            for i in range(n): sample_rows.append({"regime":label,"sample":start+i,"nmse_db":float(nmse[i]),"aleatoric":float(ale[i]),"epistemic":float(epi[i])})
            if do_calibration:
                # Eq.(5) marginal interval: df=nu-d+1 and scale_sq=((kappa+1)/(kappa*df))*Psi.
                df=out.nu_expanded-2*1024+1; scale_sq=((out.kappa_expanded+1)/(out.kappa_expanded*df))*out.psi
                for nominal in NOMINALS:
                    q=torch.as_tensor(student_t.ppf((1+nominal)/2,df.detach().cpu().numpy()),device=device,dtype=scale_sq.dtype)
                    half=q*torch.sqrt(scale_sq.clamp_min(1e-12)); covered=((target-out.gamma).abs()<=half)
                    keep=omitted[:,None,:].expand_as(covered)
                    component_rows.append({"regime":label,"nominal":float(nominal),"covered":int(covered[keep].sum()),"count":int(keep.sum())})
    return sample_rows,component_rows

def summary_rows(rows):
    out=[]
    for label,_ in REGIMES:
        a=[r for r in rows if r["regime"]==label]
        out.append({"regime":label,"samples":len(a),"nmse_mean":np.mean([r["nmse_db"] for r in a]),"nmse_median":np.median([r["nmse_db"] for r in a]),"nmse_std":np.std([r["nmse_db"] for r in a]),"ale_mean":np.mean([r["aleatoric"] for r in a]),"ale_median":np.median([r["aleatoric"] for r in a]),"ale_std":np.std([r["aleatoric"] for r in a]),"epi_mean":np.mean([r["epistemic"] for r in a]),"epi_median":np.median([r["epistemic"] for r in a]),"epi_std":np.std([r["epistemic"] for r in a])})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--common-eval-dir",required=True); p.add_argument("--samples-per-regime",type=int,default=10000); p.add_argument("--batch-size",type=int,default=64); args=p.parse_args()
    out=ROOT/args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); common=ROOT/args.common_eval_dir; device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base=load_config("configs/current_valid_baseline_condition_c.json")
    models=[
      ("A 5k x 10",base,ROOT/"runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt",20260819),
      ("B1 50k x 1",base,ROOT/"runs/current_valid_baseline/diversity_ablation/unique_50k_1ep/uacp_predictor_step4a.pt",20260819),
      ("B2 50k x 1",load_config("configs/current_valid_baseline_seed_20260915.json"),ROOT/"runs/current_valid_baseline/diversity_ablation/seed_20260915/uacp_predictor_step4a.pt",20260915),
      ("B3 50k x 1",load_config("configs/current_valid_baseline_seed_20260916.json"),ROOT/"runs/current_valid_baseline/diversity_ablation/seed_20260916/uacp_predictor_step4a.pt",20260916),
    ]
    models += [(f"D{seed} 10k x 5",load_config(f"configs/current_valid_baseline_10k5_seed_{seed}.json"),ROOT/f"runs/current_valid_baseline/diversity_ablation/tenk_5ep_seed_{seed}/uacp_predictor_step4a.pt",seed) for seed in (20260819,20260915,20260916)]
    all_summary=[]; all_gaps=[]; all_boot={}; fig8=[]; cal=[]
    for name,cfg,ckpt,seed in models:
        m=_make_model(cfg,device); payload=torch.load(ckpt,map_location=device,weights_only=False); m.load_state_dict(payload["model_state_dict"] if isinstance(payload,dict) and "model_state_dict" in payload else payload); m.eval()
        rows, comps=eval_model(m,common,device,args.samples_per_regime,args.batch_size,20262000,do_calibration=(name == "D20260819 10k x 5"))
        for r in summary_rows(rows): r["model"]=name; all_summary.append(r)
        by={r["regime"]:r for r in summary_rows(rows)}; da=by["ID-Hard 80 ns"]["ale_mean"]-by["ID-Easy 20 ns"]["ale_mean"]; idmax=max(by["ID-Easy 20 ns"]["epi_mean"],by["ID-Hard 80 ns"]["epi_mean"])
        gaps={"model":name,"delta_ale":da,"near_gap":by["OOD-Near 120 ns"]["epi_mean"]-idmax,"far_gap":by["OOD-Far 1 ms"]["epi_mean"]-idmax}; all_gaps.append(gaps)
        idv=np.array([r["epistemic"] for r in rows if r["regime"] in ("ID-Easy 20 ns","ID-Hard 80 ns")]); near=np.array([r["epistemic"] for r in rows if r["regime"]=="OOD-Near 120 ns"]); far=np.array([r["epistemic"] for r in rows if r["regime"]=="OOD-Far 1 ms"])
        scores=np.r_[idv,near,far]; labels=np.r_[np.zeros(len(idv)),np.ones(len(near)+len(far))]
        auc={"model":name,"overall":pairwise_auc(scores,labels),"near":pairwise_auc(np.r_[idv,near],np.r_[np.zeros(len(idv)),np.ones(len(near))]),"far":pairwise_auc(np.r_[idv,far],np.r_[np.zeros(len(idv)),np.ones(len(far))])}; fig8.append(auc); all_boot[name]=boot_auc(scores,labels)
        for label,_ in REGIMES:
            v=np.array([r["epistemic"] for r in rows if r["regime"]==label]); fig8.append({"model":name,"regime":label,"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"p05":float(np.quantile(v,.05)),"p25":float(np.quantile(v,.25)),"p75":float(np.quantile(v,.75)),"p95":float(np.quantile(v,.95))})
        if comps:
          for nominal in NOMINALS:
            cc=[x for x in comps if x["nominal"]==float(nominal)]; den=sum(x["count"] for x in cc); num=sum(x["covered"] for x in cc); cal.append({"model":name,"pool":"all","nominal":float(nominal),"empirical":num/den,"count":den})
            for pool,labels_pool in (("ID",("ID-Easy 20 ns","ID-Hard 80 ns")),("OOD",("OOD-Near 120 ns","OOD-Far 1 ms"))):
                cc=[x for x in comps if x["nominal"]==float(nominal) and x["regime"] in labels_pool]; den=sum(x["count"] for x in cc); num=sum(x["covered"] for x in cc); cal.append({"model":name,"pool":pool,"nominal":float(nominal),"empirical":num/den,"count":den})
    # Calibration error is mean absolute deviation from nominal over the displayed grid.
    ces=[]
    for name in [m[0] for m in models]:
        for pool in ("ID","OOD"):
            z=[r for r in cal if r["model"]==name and r["pool"]==pool]; ces.append({"model":name,"pool":pool,"calibration_error_mae":float(np.mean([abs(r["empirical"]-r["nominal"]) for r in z]))})
    for fn,data in (("summary.csv",all_summary),("gaps.csv",all_gaps),("fig8_distributions.csv",[x for x in fig8 if "regime" in x]),("fig8_auroc.csv",[x for x in fig8 if "overall" in x]),("calibration.csv",cal),("calibration_error.csv",ces)):
        write_rows(out/fn,data)
    # Plot artifacts are deliberately simple Fig.8/Fig.9-style diagnostics.
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(9,5));
    for name in [m[0] for m in models]:
        for label,_ in REGIMES:
            z=[x for x in fig8 if x.get("model")==name and x.get("regime")==label]
            if z: plt.plot([label],[z[0]["mean"]],"o",label=f"{name}: {label}")
    plt.ylabel("Epistemic uncertainty (raw Eq.12/13 score)"); plt.xticks(rotation=25); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/"fig8_epistemic_regime_summary.png",dpi=160); plt.close()
    for pool in ("ID","OOD"):
        plt.figure(figsize=(6,5)); plt.plot([0,1],[0,1],"k--",label="ideal")
        z=[r for r in cal if r["model"]=="D20260819 10k x 5" and r["pool"]==pool]; plt.plot([r["nominal"] for r in z],[r["empirical"] for r in z],"o-",label=f"{pool} D20260819")
        plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage"); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out/f"fig9_calibration_{pool.lower()}.png",dpi=160); plt.close()
    result={"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"samples_per_regime":args.samples_per_regime,"common_eval_dir":str(common),"updates":{"10k_x_5":6250,"50k_x_1":6250},"gaps":all_gaps,"fig8_auroc":fig8,"fig8_bootstrap_overall":all_boot,"calibration_error":ces,"assumptions":["IMPLEMENTATION-ASSUMPTION: equal-count ID=(20,80) and OOD=(120,1ms) pooling for AUROC","IMPLEMENTATION-ASSUMPTION: omitted-subcarrier component pooling for calibration","IMPLEMENTATION-ASSUMPTION: MAE over nominal 0.1..0.9 grid as calibration error","Diagonal-Psi approximation; Eq.(5) marginal Student-t interval, not full-covariance reproduction"]}
    (out/"results.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
