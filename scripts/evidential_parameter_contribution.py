#!/usr/bin/env python3
"""Psi, Nu, Kappa가 Aleatoric/Epistemic 변화에 기여하는 정도를 분리한다.

실험 목적:
    현재 uncertainty가 어떤 evidential parameter 경로를 통해 조절되는지
    local derivative와 signed contribution으로 확인한다.

주의:
    현재 valid checkpoint를 고정하며 weight와 optimizer 상태를 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors, constrain_pair_scalar_evidential  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402

REGIMES = [("ID-Easy 20 ns", "test_20ns.npz"), ("ID-Hard 80 ns", "test_80ns.npz")]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def pair_nll(output, target):
    eps = 1e-8; d = 2 * output.num_subcarriers
    residual = channels_to_pair_vectors(target - output.gamma); psi = channels_to_pair_vectors(output.psi).clamp_min(eps)
    df = (output.nu - d + 1).clamp_min(eps); scale = ((output.kappa + 1) / (output.kappa * df)) * psi
    mahal = (residual.square() / scale.clamp_min(eps)).sum(-1); logdet = scale.clamp_min(eps).log().sum(-1); df = df.squeeze(-1)
    logp = torch.lgamma((df + d) / 2) - torch.lgamma(df / 2) - .5 * (d * torch.log(df * torch.pi) + logdet) - ((df + d) / 2) * torch.log1p(mahal / df)
    return -logp


def forward_raw(model, x):
    h = model.input_projection(x)
    for block in model.residual_blocks: h = block(h)
    pooled = model.pool(h); rg = model.gamma_head(h); rp = model.psi_head(h); rk = model.kappa_head(pooled); rn = model.nu_head(pooled)
    return constrain_pair_scalar_evidential(rg, rp, rk, rn, model.num_subcarriers), rp, rk, rn


def grads(loss, tensors):
    values = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    return tuple(v if v is not None else torch.zeros_like(t) for v, t in zip(values, tensors))


@torch.enable_grad()
def evaluate(model, path, cfg, device, seed, regime):
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    rows=[]; lam=float(cfg["paper_specified"]["lambda_reg"])
    for bi,batch in enumerate(loader):
        cfr=batch["cfr"].to(device); mask=uniform_grouping_mask(cfr.shape[0],1024,16,device); set_seeds(seed+bi); x,target,_=build_noisy_sparse_input(cfr,mask,15.0)
        output,rp,rk,rn=forward_raw(model,x); tp=channels_to_pair_vectors(target); gp=channels_to_pair_vectors(output.gamma); pp=channels_to_pair_vectors(output.psi)
        ale=channels_to_pair_vectors(output.aleatoric); epi=channels_to_pair_vectors(output.epistemic); scale=cfr.abs().square().mean((1,2,3)).sqrt().clamp_min(1e-12); scale2=scale.square(); ale_n=ale/scale2[:,None,None]; epi_n=epi/scale2[:,None,None]; pp_n=pp/scale2[:,None,None]
        om=torch.cat(((1-mask).bool(),(1-mask).bool()),dim=1)[:,None,:].expand(-1,4,-1); residual=tp-gp; err_n=(residual/scale[:,None,None]).square(); err=(residual.square()); err_n=(err_n*om).sum(-1)/om.sum(-1).clamp_min(1); err=(err*om).sum(-1)/om.sum(-1).clamp_min(1)
        nll=pair_nll(output,target); reg=residual.square().sum(-1)*(output.kappa+output.nu).squeeze(-1); total=nll+lam*reg
        g_nll=grads(nll.mean(),(rp,rk,rn)); g_reg=grads((lam*reg).mean(),(rp,rk,rn)); g_tot=grads(total.mean(),(rp,rk,rn))
        # Derivatives of normalized pair Aleatoric/Epistemic scores through the current graph.
        ale_score=ale_n.mean(-1); epi_score=epi_n.mean(-1)
        da=grads(ale_score.sum(),(rp,rk,rn)); de=grads(epi_score.sum(),(rp,rk,rn))
        def pair_grad(g): return channels_to_pair_vectors(g)
        gpsi_n,gk_n,gnu_n=pair_grad(g_nll[0]),g_nll[1].squeeze(-1),g_nll[2].squeeze(-1); gpsi_r,gk_r,gnu_r=pair_grad(g_reg[0]),g_reg[1].squeeze(-1),g_reg[2].squeeze(-1); gpsi_t,gk_t,gnu_t=pair_grad(g_tot[0]),g_tot[1].squeeze(-1),g_tot[2].squeeze(-1)
        da_p,da_k,da_n=pair_grad(da[0]),da[1].squeeze(-1),da[2].squeeze(-1); de_p,de_k,de_n=pair_grad(de[0]),de[1].squeeze(-1),de[2].squeeze(-1)
        for i in range(cfr.shape[0]):
            for p in range(4):
                # Sum over the 2K Psi components to obtain one pair-level signed contribution.
                s_ap=-(da_p[i,p]*gpsi_t[i,p]).sum(); s_an=-(da_n[i,p]*gnu_t[i,p]); s_ak=-(da_k[i,p]*gk_t[i,p]); s_at=s_ap+s_an+s_ak
                s_ep=-(de_p[i,p]*gpsi_t[i,p]).sum(); s_en=-(de_n[i,p]*gnu_t[i,p]); s_ek=-(de_k[i,p]*gk_t[i,p]); s_et=s_ep+s_en+s_ek
                rows.append({"regime":regime,"sample_index":i+bi*int(cfg["implementation_assumption"]["eval_batch_size"]),"pair":p,"normalized_error":float(err_n[i,p].detach().cpu()),"absolute_error":float(err[i,p].detach().cpu()),"raw_psi":float(pp[i,p].mean().detach().cpu()),"normalized_psi":float(pp_n[i,p].mean().detach().cpu()),"normalized_aleatoric":float(ale_n[i,p].mean().detach().cpu()),"normalized_epistemic":float(epi_n[i,p].mean().detach().cpu()),"kappa":float(output.kappa[i,p,0].detach().cpu()),"nu":float(output.nu[i,p,0].detach().cpu()),"denom":float((output.nu[i,p,0]-2049).detach().cpu()),"nll":float(nll[i,p].detach().cpu()),"raw_l_reg":float(reg[i,p].detach().cpu()),"lambda_l_reg":float((lam*reg[i,p]).detach().cpu()),"grad_nll_psi_mean":float(gpsi_n[i,p].mean().detach().cpu()),"grad_reg_psi_mean":float(gpsi_r[i,p].mean().detach().cpu()),"grad_total_psi_mean":float(gpsi_t[i,p].mean().detach().cpu()),"grad_nll_nu":float(gnu_n[i,p].detach().cpu()),"grad_reg_nu":float(gnu_r[i,p].detach().cpu()),"grad_total_nu":float(gnu_t[i,p].detach().cpu()),"grad_nll_kappa":float(gk_n[i,p].detach().cpu()),"grad_reg_kappa":float(gk_r[i,p].detach().cpu()),"grad_total_kappa":float(gk_t[i,p].detach().cpu()),"s_ale_psi":float(s_ap.detach().cpu()),"s_ale_nu":float(s_an.detach().cpu()),"s_ale_kappa":float(s_ak.detach().cpu()),"s_ale_total":float(s_at.detach().cpu()),"s_ale_psi_nll":float((-(da_p[i,p]*gpsi_n[i,p]).sum()).detach().cpu()),"s_ale_nu_nll":float((-(da_n[i,p]*gnu_n[i,p])).detach().cpu()),"s_ale_psi_reg":float((-(da_p[i,p]*gpsi_r[i,p]).sum()).detach().cpu()),"s_ale_nu_reg":float((-(da_n[i,p]*gnu_r[i,p])).detach().cpu()),"s_epi_psi":float(s_ep.detach().cpu()),"s_epi_nu":float(s_en.detach().cpu()),"s_epi_kappa":float(s_ek.detach().cpu()),"s_epi_total":float(s_et.detach().cpu())})
    return rows


def summarize(rows):
    summaries=[]; correlations=[]; bins=[]
    keys=("normalized_error","normalized_psi","normalized_aleatoric","normalized_epistemic","s_ale_psi","s_ale_nu","s_ale_total","s_epi_psi","s_epi_nu","s_epi_kappa","s_epi_total","s_ale_psi_nll","s_ale_nu_nll","s_ale_psi_reg","s_ale_nu_reg","grad_total_psi_mean","grad_total_nu","grad_total_kappa","nll","raw_l_reg","lambda_l_reg")
    for regime in ("ID-Easy 20 ns","ID-Hard 80 ns"):
        rr=[r for r in rows if r["regime"]==regime]; out={"regime":regime,"samples":len({r["sample_index"] for r in rr}),"sample_pairs":len(rr)}
        for k in keys:
            v=np.array([r[k] for r in rr],float); out[k+"_mean"]=float(v.mean()); out[k+"_median"]=float(np.median(v)); out[k+"_std"]=float(v.std());
            if k.startswith("s_ale_"): out[k+"_positive_ratio"]=float(np.mean(v>0))
        denom=np.abs(np.array([r["s_ale_psi"] for r in rr]))+np.abs(np.array([r["s_ale_nu"] for r in rr]))+1e-12; out["ale_psi_dominance_mean"]=float(np.mean(np.abs([r["s_ale_psi"] for r in rr])/denom)); out["ale_nu_dominance_mean"]=float(np.mean(np.abs([r["s_ale_nu"] for r in rr])/denom))
        ed=np.abs(np.array([r["s_epi_psi"] for r in rr]))+np.abs(np.array([r["s_epi_nu"] for r in rr]))+np.abs(np.array([r["s_epi_kappa"] for r in rr]))+1e-12; out["epi_psi_dominance_mean"]=float(np.mean(np.abs([r["s_epi_psi"] for r in rr])/ed)); out["epi_nu_dominance_mean"]=float(np.mean(np.abs([r["s_epi_nu"] for r in rr])/ed)); out["epi_kappa_dominance_mean"]=float(np.mean(np.abs([r["s_epi_kappa"] for r in rr])/ed)); summaries.append(out)
        for x,y in (("normalized_error","s_ale_psi"),("normalized_error","s_ale_nu"),("normalized_error","s_ale_total"),("normalized_error","normalized_psi"),("normalized_error","denom")):
            a=np.array([r[x] for r in rr]); b=np.array([r[y] for r in rr]); p,s=(0.,0.) if np.std(a)==0 or np.std(b)==0 else (float(np.corrcoef(a,b)[0,1]),float(np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1])); correlations.append({"regime":regime,"x":x,"y":y,"pearson":p,"spearman":s,"observations":len(rr)})
    values=np.array([r["normalized_error"] for r in rows]); edges=np.quantile(values,np.linspace(0,1,6)); edges[0]-=1e-12; edges[-1]+=1e-12
    for r in rows: r["error_bin"]=int(min(max(np.searchsorted(edges,r["normalized_error"],side="right")-1,0),4))
    for regime in ("ID-Easy 20 ns","ID-Hard 80 ns"):
        for b in range(5):
            rr=[r for r in rows if r["regime"]==regime and r["error_bin"]==b]
            if rr: bins.append({"regime":regime,"error_bin":b,"count":len(rr),**{k+"_mean":float(np.mean([r[k] for r in rr])) for k in ("normalized_error","normalized_psi","normalized_aleatoric","s_ale_psi","s_ale_nu","s_ale_total")}})
    return summaries,correlations,bins


def gradient_summary(rows):
    out=[]
    for regime in ("ID-Easy 20 ns","ID-Hard 80 ns"):
        rr=[r for r in rows if r["regime"]==regime]
        for component in ("nll","reg","total"):
            for parameter,key in (("raw_Psi",f"grad_{component}_psi_mean"),("raw_nu",f"grad_{component}_nu"),("raw_kappa",f"grad_{component}_kappa")):
                v=np.asarray([r[key] for r in rr],float); out.append({"regime":regime,"loss_component":component,"parameter":parameter,"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"positive_ratio":float(np.mean(v>0)),"negative_ratio":float(np.mean(v<0))})
        for parameter,key in (("S_Ale_Psi","s_ale_psi"),("S_Ale_Nu","s_ale_nu"),("S_Ale_Total","s_ale_total"),("S_Epi_Kappa","s_epi_kappa")):
            v=np.asarray([r[key] for r in rr],float); out.append({"regime":regime,"loss_component":"uncertainty_contribution","parameter":parameter,"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"positive_ratio":float(np.mean(v>0)),"negative_ratio":float(np.mean(v<0))})
    return out


def matched_contribution(rows, match_path):
    from scipy.stats import wilcoxon
    by={}
    for r in rows: by.setdefault((r["regime"],int(r["sample_index"])),[]).append(r)
    def sample_values(reg):
        return {i:{k:float(np.mean([x[k] for x in rr])) for k in ("normalized_error","s_ale_psi","s_ale_nu","s_ale_total")} for (rg,i),rr in by.items() if rg==reg}
    easy=sample_values("ID-Easy 20 ns"); hard=sample_values("ID-Hard 80 ns"); pairs=list(csv.DictReader(Path(match_path).open()))
    differences=[]
    for m in pairs:
        a=easy[int(m["easy_sample_index"])]; b=hard[int(m["hard_sample_index"])]
        differences.append({"match_index":m["match_index"],**{f"delta_{k}_80_minus_20":b[k]-a[k] for k in ("normalized_error","s_ale_psi","s_ale_nu","s_ale_total")}})
    summary=[]
    for label,key in (("normalized_error","delta_normalized_error_80_minus_20"),("S_Ale_Psi","delta_s_ale_psi_80_minus_20"),("S_Ale_Nu","delta_s_ale_nu_80_minus_20"),("S_Ale_Total","delta_s_ale_total_80_minus_20")):
        v=np.asarray([r[key] for r in differences],float); nonzero=v[v!=0]; summary.append({"quantity":label,"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"positive_ratio":float(np.mean(v>0)),"negative_ratio":float(np.mean(v<0)),"wilcoxon_p_two_sided":float(wilcoxon(v).pvalue) if len(nonzero) else 1.0})
    return summary,differences


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json"); p.add_argument("--checkpoint",default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); p.add_argument("--probe-dir",default="runs/paper_style_uncertainty_diagnostics_20260904/generated_p3"); p.add_argument("--matching",default="runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/paired_power_matching/matched_pairs.csv"); p.add_argument("--output-dir",required=True); a=p.parse_args(); out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    cfg=load_config(a.config); set_seeds(int(cfg["implementation_assumption"]["seed"])); device=torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu"); model=_make_model(cfg,device); payload=torch.load(ROOT/a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload,dict) and "model_state_dict" in payload else payload); model.eval(); rows=[]
    for i,(regime,file) in enumerate(REGIMES): rows.extend(evaluate(model,ROOT/a.probe_dir/file,cfg,device,99000+i*1000,regime))
    summaries,corrs,bins=summarize(rows); grad_rows=gradient_summary(rows); matched_stats,matched_rows=matched_contribution(rows,ROOT/a.matching); out.mkdir(parents=True,exist_ok=True); write_csv(out/"sample_pair_contributions.csv",rows); write_csv(out/"summary.csv",summaries); write_csv(out/"gradient_summary.csv",grad_rows); write_csv(out/"correlations.csv",corrs); write_csv(out/"error_bin_summary.csv",bins); write_csv(out/"matched_contribution_statistics.csv",matched_stats); write_csv(out/"matched_contribution_differences.csv",matched_rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5)); labels=["20 ns","80 ns"]; x=np.arange(2); width=.22
    for j,key in enumerate(("s_ale_psi_mean","s_ale_nu_mean","s_ale_total_mean")): ax.bar(x+(j-1)*width,[r[key] for r in summaries],width,label=key.replace("_mean",""))
    ax.set_xticks(x,labels); ax.set_ylabel("Signed contribution"); ax.axhline(0,color="black",lw=.8); ax.legend(); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out/"aleatoric_contribution_bar.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4));
    for ax,key,title in zip(axes,("s_ale_psi","s_ale_nu","s_ale_total"),("Psi contribution","Nu contribution","Total contribution")):
        for regime,color in (("ID-Easy 20 ns","tab:blue"),("ID-Hard 80 ns","tab:orange")):
            rr=[r for r in bins if r["regime"]==regime]; ax.plot([r["error_bin"] for r in rr],[r[key+"_mean"] for r in rr],"o-",label=regime,color=color)
        ax.axhline(0,color="black",lw=.7); ax.set_title(title); ax.set_xlabel("Error quintile"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/"error_bin_contributions.png",dpi=160); plt.close(fig)
    ck=ROOT/a.checkpoint; prov={"checkpoint":a.checkpoint,"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"training_performed":False,"optimizer_step_called":False,"normalization":"sample-wise CFR power normalization; variance quantities scaled by 1/a^2","loss":"current diagonal multivariate Student-t NLL + pair-level regularizer","lambda_reg":float(cfg["paper_specified"]["lambda_reg"])}
    (out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)); (out/"config.json").write_text(json.dumps({"regimes":[r[0] for r in REGIMES],"snr_db":15.0,"ng":16,"lambda_reg":prov["lambda_reg"],"matching":a.matching},indent=2)); (out/"results.json").write_text(json.dumps({"summary":summaries,"correlations":corrs,"error_bins":bins,"gradient_summary":grad_rows,"matched_contribution_statistics":matched_stats,"provenance":prov},indent=2,sort_keys=True)); print(json.dumps({"output_dir":str(out),"device":str(device),"gpu":prov["gpu"],"summary":summaries,"correlations":corrs,"matched":matched_stats},indent=2))


if __name__=="__main__": main()
