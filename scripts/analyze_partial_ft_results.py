#!/usr/bin/env python3
"""Summarize the controlled sparse Partial-vs-Full adaptation experiment."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out=ROOT/"runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation"
    summary=rows(out/"summary.csv")
    for r in summary:
        for k in ("epoch","ng","samples","finite_samples"): r[k]=int(r[k])
        for k in ("nmse_all_db","nmse_omitted_db","aleatoric","epistemic","kappa","nu_margin"): r[k]=float(r[k])
        r["all_finite"]=r["all_finite"]=="True"
    primary=[]; uncertainty=[]
    for model in ("pre","partial","full"):
        epoch=0 if model=="pre" else 3
        for ng in (16,32):
            for regime in ("20ns","80ns","120ns","1ms"):
                r=next(x for x in summary if x["model"]==model and x["epoch"]==epoch and x["regime"]==regime and x["ng"]==ng)
                primary.append({"model":model,"ng":ng,"regime":regime,"nmse_all_db":r["nmse_all_db"],"nmse_omitted_db":r["nmse_omitted_db"]})
                uncertainty.append({"model":model,"ng":ng,"regime":regime,"aleatoric":r["aleatoric"],"epistemic":r["epistemic"],"kappa":r["kappa"],"nu_margin":r["nu_margin"],"all_finite":r["all_finite"]})
    write_csv(out/"primary_performance_epoch3.csv",primary); write_csv(out/"uncertainty_epoch3.csv",uncertainty)
    def get(model,epoch,regime,ng,key): return next(x[key] for x in summary if x["model"]==model and x["epoch"]==epoch and x["regime"]==regime and x["ng"]==ng)
    comparison=[]
    for ng in (16,32):
        for model in ("partial","full"):
            npre=get("pre",0,"120ns",ng,"nmse_omitted_db"); npost=get(model,3,"120ns",ng,"nmse_omitted_db")
            epre=get("pre",0,"120ns",ng,"epistemic"); epost=get(model,3,"120ns",ng,"epistemic")
            comparison.append({"metric":"120_nmse_improvement_pre_minus_post","model":model,"ng":ng,"value":npre-npost,"pre":npre,"post":npost})
            comparison.append({"metric":"120_epistemic_change_post_minus_pre","model":model,"ng":ng,"value":epost-epre,"pre":epre,"post":epost})
            for regime in ("20ns","80ns"):
                pre=get("pre",0,regime,ng,"nmse_omitted_db"); post=get(model,3,regime,ng,"nmse_omitted_db")
                comparison.append({"metric":f"{regime}_nmse_change_post_minus_pre","model":model,"ng":ng,"value":post-pre,"pre":pre,"post":post})
            far_pre=get("pre",0,"1ms",ng,"epistemic")/get("pre",0,"120ns",ng,"epistemic")
            far_post=get(model,3,"1ms",ng,"epistemic")/get(model,3,"120ns",ng,"epistemic")
            comparison.append({"metric":"1ms_to_120ns_epistemic_ratio","model":model,"ng":ng,"value":far_post,"pre":far_pre,"post":far_post})
            full_improvement=npre-get("full",3,"120ns",ng,"nmse_omitted_db")
            partial_improvement=npre-get("partial",3,"120ns",ng,"nmse_omitted_db")
            comparison.append({"metric":"partial_nmse_retention_ratio_vs_full","model":"partial","ng":ng,"value":partial_improvement/full_improvement if full_improvement else float("nan"),"pre":partial_improvement,"post":full_improvement})
    write_csv(out/"adaptation_forgetting.csv",comparison)
    run_manifests={"partial":json.loads((ROOT/"runs/current_valid_baseline/partial_ft_20260921_sparse_3ep/readiness_manifest.json").read_text()),"full":json.loads((ROOT/"runs/current_valid_baseline/full_ft_20260921_sparse_3ep/readiness_manifest.json").read_text())}
    efficiency=[]
    for model,manifest in run_manifests.items():
        result=manifest["dry_run_result"]; efficiency.append({"model":model,"trainable_params":manifest["trainable_parameter_names"],"trainable_parameter_count":result["optimizer_trainable_params"],"trainable_percent":100*result["optimizer_trainable_params"]/11815320,"training_seconds":result["elapsed_seconds"],"seconds_per_step":result["step_seconds_mean"],"samples_per_second":3000/result["elapsed_seconds"],"peak_allocated_vram_mib":result["peak_allocated_vram_mib"],"optimizer_state_mib_fp32":result["optimizer_trainable_params"]*2*4/2**20})
    write_csv(out/"efficiency.csv",efficiency)
    pre=ROOT/"runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
    partial=ROOT/"runs/current_valid_baseline/partial_ft_20260921_sparse_3ep/adapted_epoch_3.pt"
    full=ROOT/"runs/current_valid_baseline/full_ft_20260921_sparse_3ep/adapted_epoch_3.pt"
    pre_state=torch.load(pre,map_location="cpu",weights_only=False); partial_state=torch.load(partial,map_location="cpu",weights_only=False); full_state=torch.load(full,map_location="cpu",weights_only=False)
    if "model_state_dict" in pre_state: pre_state=pre_state["model_state_dict"]
    def state_dict(x): return x["model_state_dict"] if isinstance(x,dict) and "model_state_dict" in x else x
    partial_state=state_dict(partial_state); full_state=state_dict(full_state)
    frozen=[k for k in pre_state if not (k.startswith("residual_blocks.31.") or k.startswith(("gamma_head.","psi_head.","kappa_head.","nu_head.")))]
    frozen_diffs={k:float((partial_state[k]-pre_state[k]).abs().max()) for k in frozen}
    full_changed=sum(bool((full_state[k]-pre_state[k]).abs().max()>0) for k in pre_state)
    scope_check={"pre_checkpoint_sha256":sha(pre),"partial_epoch3_sha256":sha(partial),"full_epoch3_sha256":sha(full),"partial_frozen_parameter_count":len(frozen),"partial_frozen_max_abs_diff":max(frozen_diffs.values()),"partial_frozen_all_identical":max(frozen_diffs.values())==0.0,"full_changed_parameter_tensor_count":full_changed}
    write(out/"scope_verification.json",scope_check)
    finite=all(r["all_finite"] for r in summary)
    aggregate={}
    for model in ("pre","partial","full"):
        for regime in ("20ns","80ns","120ns","1ms"):
            for ng in (16,32): aggregate[f"{model}_{regime}_ng{ng}"]={k:get(model,0 if model=="pre" else 3,regime,ng,k) for k in ("nmse_omitted_db","aleatoric","epistemic","kappa","nu_margin")}
    write(out/"analysis_summary.json",{"primary_endpoint":"epoch3","all_evaluation_samples_finite":finite,"aggregate_epoch3":aggregate,"comparison_path":"adaptation_forgetting.csv","efficiency_path":"efficiency.csv","scope_verification":scope_check,"interpretation_label":"RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION; not paper Partial FT reproduction"})

if __name__=="__main__": main()
