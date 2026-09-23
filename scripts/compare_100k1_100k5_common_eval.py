#!/usr/bin/env python3
"""RAM-safe paired Fig.8/Fig.9/error-scale comparison for frozen checkpoints."""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata, t as student_t

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
REGIMES = [("ID-Easy 20 ns", "ID-Easy_20_ns.npz"), ("ID-Hard 80 ns", "ID-Hard_80_ns.npz"),
           ("OOD-Near 120 ns", "OOD-Near_120_ns.npz"), ("OOD-Far 1 ms", "OOD-Far_1_ms.npz")]
NOMINALS = [0.50, 0.80, 0.90, 0.95, 0.99]
K = 1024

def rss_gib(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
def auc(a, b):
    x = np.concatenate([a, b]); ranks = rankdata(x, method="average")
    return float((ranks[len(a):].sum() - len(b)*(len(b)+1)/2) / (len(a)*len(b)))
def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def reservoir_update(state, values, max_size=8192):
    values = np.asarray(values, dtype=np.float64).reshape(-1); state["sum"] += float(values.sum()); state["n"] += len(values)
    if state["sample"].size < max_size:
        take = min(max_size-state["sample"].size, len(values)); state["sample"] = np.concatenate((state["sample"], values[:take])); values = values[take:]
    if len(values):
        slots = state["rng"].integers(0, state["n"], size=len(values)); keep = slots < max_size; state["sample"][slots[keep]] = values[keep]
def stats(state):
    return {"n": state["n"], "mean": state["sum"]/max(1,state["n"]), "median": float(np.quantile(state["sample"],.5)), "std_reservoir": float(np.std(state["sample"])), "q95": float(np.quantile(state["sample"],.95))}

def load_model(cfg_path, ckpt, device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    model = _make_model(load_config(cfg_path), device)
    payload = torch.load(ROOT/ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload) if isinstance(payload,dict) else payload); model.eval(); return model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--common-eval-dir",required=True); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--samples-per-regime",type=int,default=10000); a=p.parse_args()
    out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
    cfg="configs/current_valid_baseline_100k1_seed_20260819.json"
    models=[("100k×1", "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"),
            ("100k×5", "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep.pt")]
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    loaded_models=[(name,load_model(cfg,ck,device)) for name,ck in models]
    scores={(m,r):[] for m,_ in models for r,_ in REGIMES}; errors={(m,r):[] for m,_ in models for r,_ in REGIMES}; scales={(m,r):[] for m,_ in models for r,_ in REGIMES}; ratios={(m,r):[] for m,_ in models for r,_ in REGIMES}
    cov={(m,r,n):[0,0] for m,_ in models for r,_ in REGIMES for n in NOMINALS}
    started=time.perf_counter(); common=ROOT/a.common_eval_dir
    for ri,(regime,fn) in enumerate(REGIMES):
        with np.load(common/fn) as z: cfr=np.asarray(z["cfr"][:a.samples_per_regime],dtype=np.complex64)
        if len(cfr)!=a.samples_per_regime: raise RuntimeError(f"{regime}: sample count mismatch")
        for start in range(0,len(cfr),a.batch_size):
            c=torch.from_numpy(cfr[start:start+a.batch_size]).to(device); mask=uniform_grouping_mask(len(c),K,16,device)
            # Fig.8 and Fig.9 use their established seeds; models share each generated realization.
            set_seeds(20261509+ri*100000+start); x8,_,_=build_noisy_sparse_input(c,mask,15.0)
            set_seeds(20262000+ri*100000+start); x9,target,_=build_noisy_sparse_input(c,mask,15.0)
            omitted=(1-mask).bool()
            with torch.inference_mode():
                for name,model in loaded_models:
                    o8=model(x8); epi=o8.epistemic.reshape(len(c),2,4,K).sum(1).mean(1); sc=epi[omitted].detach().cpu().numpy().reshape(len(c),-1).mean(1); scores[(name,regime)].extend(sc.tolist()); del o8,epi,sc
                    o=model(x9); err=(o.gamma-target).abs(); df=o.nu_expanded-2*K+1; scale_sq=((o.kappa_expanded+1)/(o.kappa_expanded*df))*o.psi; scale=torch.sqrt(scale_sq.clamp_min(1e-12)); keep=omitted[:,None,:].expand_as(err)
                    e=err[keep].detach().cpu().numpy().reshape(len(c),-1).mean(1); s=scale[keep].detach().cpu().numpy().reshape(len(c),-1).mean(1); rat=(err/(scale.clamp_min(1e-12)))[keep].detach().cpu().numpy().reshape(len(c),-1).mean(1); errors[(name,regime)].extend(e.tolist()); scales[(name,regime)].extend(s.tolist()); ratios[(name,regime)].extend(rat.tolist())
                    for nominal in NOMINALS:
                        q=student_t.ppf((1+nominal)/2,df.detach().cpu().numpy()); q=torch.as_tensor(q,dtype=scale.dtype,device=device); covered=(err<=q*scale); n=int((covered&keep).sum().item()); d=int(keep.sum().item()); cov[(name,regime,nominal)][0]+=n; cov[(name,regime,nominal)][1]+=d
                    del o,err,df,scale_sq,scale,keep
            del c,mask,x8,x9,target,omitted
        del cfr; gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()
        print(json.dumps({"regime":regime,"rss_gib":rss_gib()}),flush=True)
    fig8=[]; summary=[]; comparison=[]
    for name,_ in models:
        vals={r:np.asarray(scores[(name,r)]) for r,_ in REGIMES}; idv=np.concatenate([vals[REGIMES[0][0]],vals[REGIMES[1][0]]]); near=vals[REGIMES[2][0]]; far=vals[REGIMES[3][0]]
        au={"id_vs_near":auc(idv,near),"id_vs_far":auc(idv,far),"id_vs_pooled_ood":auc(idv,np.concatenate([near,far])),"80_vs_120":auc(vals[REGIMES[1][0]],near)}
        for r,_ in REGIMES:
            v=vals[r]; db=10*np.log10(v); finite=np.isfinite(v); row={"model":name,"regime":r,"samples":len(v),"epi_finite_fraction":float(np.mean(finite)),"epi_inf_fraction":float(np.mean(np.isinf(v))),"epi_db_mean":float(np.mean(db)),"epi_db_median":float(np.median(db)),"epi_db_std_finite":float(np.std(db[finite])) if np.any(finite) else float("nan"),"epi_linear_mean":float(np.mean(v)),"nmse_error_mean":float(np.mean(errors[(name,r)])),"scale_mean":float(np.mean(scales[(name,r)])),"error_scale_mean":float(np.mean(ratios[(name,r)]))}; fig8.append(row)
        for pool,rs in [("ID",REGIMES[:2]),("Near",[REGIMES[2]]),("Far",[REGIMES[3]]),("OOD-pooled",REGIMES[2:])]:
            for nominal in NOMINALS:
                n=sum(cov[(name,r,nominal)][0] for r,_ in rs); d=sum(cov[(name,r,nominal)][1] for r,_ in rs); comparison.append({"model":name,"pool":pool,"nominal":nominal,"empirical":n/d,"covered":n,"component_count":d})
    ce=[]
    for name,_ in models:
        for pool in ("ID","Near","Far","OOD-pooled"):
            z=[r for r in comparison if r["model"]==name and r["pool"]==pool]; ce.append({"model":name,"pool":pool,"ce_mae_5_level":float(np.mean([abs(r["empirical"]-r["nominal"]) for r in z]))})
    for name,_ in models:
        for r,_ in REGIMES:
            x=next(x for x in fig8 if x["model"]==name and x["regime"]==r); summary.append(x)
    write_csv(out/"fig8_summary.csv",fig8); write_csv(out/"calibration.csv",comparison); write_csv(out/"calibration_error.csv",ce)
    # Compact comparison table requested by the experiment.
    b1=next(x for x in fig8 if x["model"]=="100k×1" and x["regime"]=="ID-Easy 20 ns"); b5=next(x for x in fig8 if x["model"]=="100k×5" and x["regime"]=="ID-Easy 20 ns")
    au_rows=[]
    for name,_ in models:
        vals={r:np.asarray(scores[(name,r)]) for r,_ in REGIMES}; idv=np.concatenate([vals[REGIMES[0][0]],vals[REGIMES[1][0]]]); near=vals[REGIMES[2][0]]; far=vals[REGIMES[3][0]]
        au_rows.append({"model":name,"ID_vs_Near_AUROC":auc(idv,near),"ID_vs_Far_AUROC":auc(idv,far),"ID_vs_pooled_AUROC":auc(idv,np.concatenate([near,far])),"80_vs_120_AUROC":auc(vals[REGIMES[1][0]],near),"ID_CE":next(x["ce_mae_5_level"] for x in ce if x["model"]==name and x["pool"]=="ID"),"Near_CE":next(x["ce_mae_5_level"] for x in ce if x["model"]==name and x["pool"]=="Near"),"Far_CE":next(x["ce_mae_5_level"] for x in ce if x["model"]==name and x["pool"]=="Far"),"OOD_pooled_CE":next(x["ce_mae_5_level"] for x in ce if x["model"]==name and x["pool"]=="OOD-pooled"),"Near_cov_0.9":next(x["empirical"] for x in comparison if x["model"]==name and x["pool"]=="Near" and x["nominal"]==.9),"Far_cov_0.9":next(x["empirical"] for x in comparison if x["model"]==name and x["pool"]=="Far" and x["nominal"]==.9),"1ms_error_scale":next(x["error_scale_mean"] for x in fig8 if x["model"]==name and x["regime"]=="OOD-Far 1 ms")})
    write_csv(out/"comparison_100k1_vs_100k5.csv",au_rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5));
    for name,style in [("100k×1","o-"),("100k×5","s--")]: plt.plot([x["regime"] for x in fig8 if x["model"]==name],[x["nmse_error_mean"] for x in fig8 if x["model"]==name],style,label=f"{name} error"); plt.plot([x["regime"] for x in fig8 if x["model"]==name],[x["scale_mean"] for x in fig8 if x["model"]==name],style,label=f"{name} scale")
    plt.ylabel("Omitted-component magnitude"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out/"error_vs_scale.png",dpi=180); plt.close()
    plt.figure(figsize=(7,5));
    for name,style in [("100k×1","o-"),("100k×5","s--")]:
        z=[r for r in comparison if r["model"]==name and r["pool"] in ("ID","OOD-pooled")]; plt.plot([r["nominal"] for r in z if r["pool"]=="ID"],[r["empirical"] for r in z if r["pool"]=="ID"],style,label=f"{name} ID"); plt.plot([r["nominal"] for r in z if r["pool"]=="OOD-pooled"],[r["empirical"] for r in z if r["pool"]=="OOD-pooled"],style,label=f"{name} OOD")
    plt.plot([0,1],[0,1],"k--",label="Ideal"); plt.xlim(0,1); plt.ylim(0,1); plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out/"calibration_100k1_vs_100k5.png",dpi=180); plt.close()
    result={"models":models,"common_eval_dir":str(common.relative_to(ROOT)),"samples_per_regime":a.samples_per_regime,"batch_size":a.batch_size,"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"fig8":fig8,"calibration":comparison,"calibration_error":ce,"comparison":au_rows,"peak_rss_gib":rss_gib(),"peak_gpu_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,"seeds":{"fig8":20261509,"fig9_error_scale":20262000},"assumptions":["paired common CFR/mask/noise realization per metric between checkpoints","omitted-subcarrier Eq.(13) primary aggregation","Eq.(5) marginal Student-t coverage","five-level CE is MAE over 0.50,0.80,0.90,0.95,0.99","diagonal-Psi baseline"]}
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"output":str(out),"peak_rss_gib":result["peak_rss_gib"],"peak_gpu_allocated_mib":result["peak_gpu_allocated_mib"]},indent=2),flush=True)
if __name__=="__main__": main()
