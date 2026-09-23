#!/usr/bin/env python3
"""Large-sample conditional association diagnostic for the epoch-3 checkpoint."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
COMMON = ROOT / "runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval"
FIXED = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
DELAY_LIST = [20, 40, 60, 80, 100]

def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load_data(delay, limit):
    source = COMMON / ("ID-Easy_20_ns.npz" if delay == 20 else "ID-Hard_80_ns.npz") if delay in (20,80) else FIXED / f"test_delay_{delay}_ns.npz"
    with np.load(source) as z: return np.asarray(z["cfr"][:limit], np.complex64), str(source.relative_to(ROOT))

def load_model(device):
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    model = _make_model(load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval(); return model

def evaluate(model, cfr_np, ng, seed, device):
    from src.training.data import build_noisy_sparse_input
    K = 1024; cfr = torch.from_numpy(cfr_np).to(device)
    mask = torch.zeros((len(cfr), K), device=device); mask[:, ::ng] = 1.0
    torch.manual_seed(seed)
    x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
    with torch.inference_mode():
        out = model(x); denom = out.nu_expanded - 2*K - 1.0
        ale_map = out.psi / denom; epi_map = ale_map / out.kappa_expanded
        omitted = 1-mask; count = omitted.sum(1).clamp_min(1)
        pair_ale = ale_map.reshape(len(cfr),2,4,K).permute(0,2,1,3).sum(2).mean(1)
        pair_epi = epi_map.reshape(len(cfr),2,4,K).permute(0,2,1,3).sum(2).mean(1)
        ale = (pair_ale*omitted).sum(1)/count; epi=(pair_epi*omitted).sum(1)/count
        nmse = 10*torch.log10((((out.gamma-target).square()*omitted[:,None,:]).sum((1,2))/(target.square()*omitted[:,None,:]).sum((1,2)).clamp_min(1e-12)).clamp_min(1e-12))
        om=omitted[:,None,:].expand_as(ale_map)
        def masked(v): return (v*om).sum((1,2))/om.sum((1,2)).clamp_min(1)
        mag=cfr.abs(); adj=(cfr[:,1:]-cfr[:,:-1]).abs()
        vals={"nmse":nmse,"kappa":masked(out.kappa_expanded),"nu_margin":masked(denom),"psi":masked(out.psi),"aleatoric":ale,"epistemic":epi,
              "rms_magnitude":torch.sqrt((mag.square()).mean((1,2,3))),"magnitude_mean":mag.mean((1,2,3)),"magnitude_std":mag.std((1,2,3),unbiased=False),"adjacent_variation":adj.mean((1,2,3)),"peak_to_mean":mag.amax((1,2,3))/mag.mean((1,2,3)).clamp_min(1e-12)}
    return {k:v.detach().cpu().numpy() for k,v in vals.items()}

def fit_conditional(rows, out):
    id_rows=[r for r in rows if r["ng"] in (16,32) and r["delay_ns"] in DELAY_LIST]
    train=[r for r in id_rows if r["sample"]%2==0]; test=[r for r in id_rows if r["sample"]%2==1]
    names=["rms_magnitude","adjacent_variation","nmse","delay_ns","ng32_indicator"]
    means={n:float(np.mean([r[n] for r in train])) for n in names[:-1]}; scales={n:float(np.std([r[n] for r in train])) or 1.0 for n in names[:-1]}
    def design(part): return np.array([[1]+[(r[n]-means[n])/scales[n] for n in names[:-1]]+[r["ng32_indicator"] for _ in [0]] for r in part],float)
    y=np.log(np.array([r["kappa"] for r in train])); X=design(train); coef=np.linalg.lstsq(X,y,rcond=None)[0]
    def score(part):
        yp=design(part)@coef; yt=np.log(np.array([r["kappa"] for r in part])); return float(1-((yt-yp)**2).sum()/((yt-yt.mean())**2).sum()), yp, yt
    r2,yp,yt=score(test)
    for r,p,t in zip(test,yp,yt): r["conditional_log_kappa"] = float(p); r["log_kappa_residual"] = float(t-p); r["kappa_residual_ratio"] = float(np.exp(t-p))
    result={"formula":"log(kappa) ~ scaled RMS + scaled adjacent/frequency variation + scaled NMSE + scaled delay + Ng32 indicator","train_n":len(train),"heldout_n":len(test),"coefficients":{"intercept":float(coef[0]),**{n:float(v) for n,v in zip(names[0:4],coef[1:5])},"ng32_indicator":float(coef[5])},"feature_means":means,"feature_scales":scales,"heldout_r2":r2,"interpretation":"conditional association diagnostic only; not causal proof"}
    (out/"conditional_regression.json").write_text(json.dumps(result,indent=2)+"\n")
    return test,result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--samples-20-80",type=int,default=2000); p.add_argument("--samples-other",type=int,default=200); p.add_argument("--seed",type=int,default=20262020); args=p.parse_args()
    out=ROOT/args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True)
    device=torch.device("cuda:0")
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0): raise RuntimeError("NVIDIA GB10/cuda:0 required")
    model=load_model(device); rows=[]; sources={}
    for di,delay in enumerate(DELAY_LIST):
        cfr,source=load_data(delay,args.samples_20_80 if delay in (20,80) else args.samples_other); sources[str(delay)]=source
        for ng in (16,32):
            vals=evaluate(model,cfr,ng,args.seed+di*100000+ng,device)
            for i in range(len(cfr)):
                rows.append({"sample":i,"delay_ns":delay,"ng":ng,"ng32_indicator":int(ng==32),**{k:float(v[i]) for k,v in vals.items()}})
    write_csv(out/"sample_level.csv",rows)
    id32=[r for r in rows if r["ng"]==32]; q99=float(np.quantile([r["epistemic"] for r in id32],.99));
    heldout,reg=fit_conditional(rows,out)
    tail=[r for r in heldout if r["ng"]==32 and r["epistemic"]>=q99]
    ng_stats=[]
    for ng in (16,32):
        x=[r for r in heldout if r["ng"]==ng]; res=np.array([r["kappa_residual_ratio"] for r in x])
        ng_stats.append({"ng":ng,"n":len(x),"residual_median":float(np.median(res)),"residual_q05":float(np.quantile(res,.05)),"residual_q95":float(np.quantile(res,.95)),"residual_q99":float(np.quantile(res,.99))})
    tail_stats={"pooled_ng32_id_q99_threshold":q99,"heldout_q99_n":len(tail),"heldout_q99_residual_kappa_median":float(np.median([r["kappa_residual_ratio"] for r in tail])) if tail else None,"heldout_q99_residual_kappa_q05":float(np.quantile([r["kappa_residual_ratio"] for r in tail],.05)) if tail else None,"heldout_q99_residual_kappa_q95":float(np.quantile([r["kappa_residual_ratio"] for r in tail],.95)) if tail else None}
    write_csv(out/"heldout_residuals.csv",heldout); write_csv(out/"residual_ng_summary.csv",ng_stats)
    (out/"residual_tail.json").write_text(json.dumps(tail_stats,indent=2)+"\n")
    (out/"analysis_manifest.json").write_text(json.dumps({"checkpoint":str(CHECKPOINT.relative_to(ROOT)),"device":str(device),"gpu":torch.cuda.get_device_name(0),"sources":sources,"samples_per_delay":{str(d):(args.samples_20_80 if d in (20,80) else args.samples_other) for d in DELAY_LIST},"normalize_channel":False,"normalize_channel_label":"IMPLEMENTATION-ASSUMPTION: not paper-confirmed; predictor input was not normalized","no_training":True,"diagnostic_only":True,"matching": "not attempted in sparse regimes; residual model is primary"},indent=2)+"\n")
    print(json.dumps({"output":str(out.relative_to(ROOT)),"gpu":torch.cuda.get_device_name(0),"regression":reg,"tail":tail_stats},indent=2))

if __name__=="__main__": main()
