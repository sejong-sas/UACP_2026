#!/usr/bin/env python3
"""loss가 evidential parameter를 어느 방향으로 움직이는지 검사한다.

실험 목적:
    어려운 sample에서 Psi를 키우는 gradient가 있는지 NLL과 regularizer로
    나누어 확인한다.

주의:
    autograd만 사용하고 optimizer.step()은 호출하지 않는다.
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


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return 0.0, 0.0
    p = float(np.corrcoef(x, y)[0, 1]); rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return p, float(np.corrcoef(rx, ry)[0, 1])


def pair_nll(output, target: torch.Tensor) -> torch.Tensor:
    """Return one exact diagonal multivariate Student-t NLL per sample/pair."""
    eps = 1e-8; d = 2 * output.num_subcarriers
    residual = channels_to_pair_vectors(target - output.gamma)
    psi = torch.clamp(channels_to_pair_vectors(output.psi), min=eps)
    degrees = torch.clamp(output.nu - d + 1, min=eps)
    scale = torch.clamp(((output.kappa + 1) / (output.kappa * degrees)) * psi, min=eps)
    mahal = (residual.square() / scale).sum(dim=-1); logdet = torch.log(scale).sum(dim=-1)
    df = degrees.squeeze(-1)
    log_prob = (torch.lgamma((df + d) / 2) - torch.lgamma(df / 2) - 0.5 * (d * torch.log(df * torch.pi) + logdet) - ((df + d) / 2) * torch.log1p(mahal / df))
    return -log_prob


def forward_with_raw(model, x: torch.Tensor):
    h = model.input_projection(x)
    for block in model.residual_blocks: h = block(h)
    pooled = model.pool(h)
    raw_gamma = model.gamma_head(h); raw_psi = model.psi_head(h); raw_kappa = model.kappa_head(pooled); raw_nu = model.nu_head(pooled)
    output = constrain_pair_scalar_evidential(raw_gamma, raw_psi, raw_kappa, raw_nu, model.num_subcarriers)
    return output, raw_psi, raw_kappa, raw_nu


@torch.enable_grad()
def evaluate(model, path: Path, cfg: dict, device: torch.device, seed: int, regime: str) -> list[dict]:
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    rows = []
    lambda_reg = float(cfg["paper_specified"]["lambda_reg"])
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device); mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device); set_seeds(seed + batch_index)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        output, raw_psi, raw_kappa, raw_nu = forward_with_raw(model, x)
        target_pair = channels_to_pair_vectors(target); gamma_pair = channels_to_pair_vectors(output.gamma); psi_pair = channels_to_pair_vectors(output.psi)
        scale = cfr.abs().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-12); scale2 = scale.square()
        residual = target_pair - gamma_pair; normalized_residual = residual / scale[:, None, None]
        normalized_psi = psi_pair / scale2[:, None, None]
        omitted = (1.0 - mask).bool()
        # Mask the pair vector as [Re(omitted), Im(omitted)].
        pair_mask = torch.cat((omitted, omitted), dim=1)[:, None, :].expand(-1, 4, -1)
        pair_error = (residual.square() * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(1)
        pair_error_norm = (normalized_residual.square() * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(1)
        pair_psi_norm = normalized_psi.mean(dim=-1)
        pair_ale_norm = channels_to_pair_vectors(output.aleatoric).div(scale2[:, None, None]).mean(dim=-1)
        nll_pair = pair_nll(output, target)
        reg_pair = residual.square().sum(dim=-1) * (output.kappa + output.nu).squeeze(-1)
        # Mean reductions match the production loss up to the same global B*4 factor.
        nll_loss = nll_pair.mean(); reg_loss = reg_pair.mean(); weighted_reg = lambda_reg * reg_loss; total_loss = nll_loss + weighted_reg
        grads = {}
        for name, loss in (("nll", nll_loss), ("reg", weighted_reg), ("total", total_loss)):
            values = torch.autograd.grad(loss, (raw_psi, raw_kappa, raw_nu), retain_graph=True, allow_unused=True)
            grads[name] = tuple(value if value is not None else torch.zeros_like(reference) for value, reference in zip(values, (raw_psi, raw_kappa, raw_nu)))
        gpsi_nll = channels_to_pair_vectors(grads["nll"][0]); gpsi_reg = channels_to_pair_vectors(grads["reg"][0]); gpsi_total = channels_to_pair_vectors(grads["total"][0])
        gk_nll, gn_nll = grads["nll"][1].squeeze(-1), grads["nll"][2].squeeze(-1); gk_reg, gn_reg = grads["reg"][1].squeeze(-1), grads["reg"][2].squeeze(-1); gk_total, gn_total = grads["total"][1].squeeze(-1), grads["total"][2].squeeze(-1)
        pressure = -gpsi_total
        for i in range(cfr.shape[0]):
            for pair in range(4):
                pp = pressure[i, pair];
                rows.append({
                    "regime": regime, "sample_index": i + batch_index * int(cfg["implementation_assumption"]["eval_batch_size"]), "pair": pair,
                    "normalized_error": float(pair_error_norm[i, pair].detach().cpu()), "absolute_error": float(pair_error[i, pair].detach().cpu()),
                    "normalized_psi": float(pair_psi_norm[i, pair].detach().cpu()), "normalized_aleatoric": float(pair_ale_norm[i, pair].detach().cpu()),
                    "kappa": float(output.kappa[i, pair, 0].detach().cpu()), "nu": float(output.nu[i, pair, 0].detach().cpu()),
                    "nll": float(nll_pair[i, pair].detach().cpu()), "raw_l_reg": float(reg_pair[i, pair].detach().cpu()), "lambda_l_reg": float((lambda_reg * reg_pair[i, pair]).detach().cpu()),
                    "psi_pressure_mean": float(pp.mean().detach().cpu()), "psi_pressure_median": float(pp.median().detach().cpu()), "psi_pressure_std": float(pp.std(unbiased=False).detach().cpu()), "psi_pressure_l2": float(pp.norm().detach().cpu()), "psi_pressure_positive_ratio": float((pp > 0).float().mean().detach().cpu()), "psi_pressure_negative_ratio": float((pp < 0).float().mean().detach().cpu()),
                    "grad_nll_psi_mean": float(gpsi_nll[i, pair].mean().detach().cpu()), "grad_reg_psi_mean": float(gpsi_reg[i, pair].mean().detach().cpu()), "grad_total_psi_mean": float(gpsi_total[i, pair].mean().detach().cpu()), "pressure_nll_psi_mean": float((-gpsi_nll[i, pair]).mean().detach().cpu()), "pressure_reg_psi_mean": float((-gpsi_reg[i, pair]).mean().detach().cpu()),
                    "grad_nll_psi_l2": float(gpsi_nll[i, pair].norm().detach().cpu()), "grad_reg_psi_l2": float(gpsi_reg[i, pair].norm().detach().cpu()), "grad_total_psi_l2": float(gpsi_total[i, pair].norm().detach().cpu()),
                    "grad_nll_kappa_abs": float(gk_nll[i, pair].abs().detach().cpu()), "grad_reg_kappa_abs": float(gk_reg[i, pair].abs().detach().cpu()), "grad_total_kappa_abs": float(gk_total[i, pair].abs().detach().cpu()),
                    "grad_nll_nu_abs": float(gn_nll[i, pair].abs().detach().cpu()), "grad_reg_nu_abs": float(gn_reg[i, pair].abs().detach().cpu()), "grad_total_nu_abs": float(gn_total[i, pair].abs().detach().cpu()),
                })
        del output, raw_psi, raw_kappa, raw_nu
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    summaries=[]; correlations=[]; bins=[]
    for regime in ("ID-Easy 20 ns", "ID-Hard 80 ns"):
        rr=[r for r in rows if r["regime"]==regime]
        summary={"regime":regime,"samples":len({r["sample_index"] for r in rr}),"sample_pairs":len(rr)}
        for k in ("normalized_error","normalized_psi","normalized_aleatoric","psi_pressure_mean","psi_pressure_l2","psi_pressure_positive_ratio","psi_pressure_negative_ratio","grad_nll_psi_mean","grad_reg_psi_mean","grad_total_psi_mean","pressure_nll_psi_mean","pressure_reg_psi_mean","grad_nll_psi_l2","grad_reg_psi_l2","grad_total_psi_l2","grad_total_kappa_abs","grad_total_nu_abs","nll","raw_l_reg","lambda_l_reg"):
            v=np.array([r[k] for r in rr],float); summary[k+"_mean"]=float(v.mean()); summary[k+"_median"]=float(np.median(v)); summary[k+"_std"]=float(v.std())
        summaries.append(summary)
        for x,y in (("normalized_error","psi_pressure_mean"),("normalized_error","normalized_psi"),("normalized_error","grad_total_kappa_abs"),("normalized_error","grad_total_nu_abs")):
            p,s=corr(np.array([r[x] for r in rr]),np.array([r[y] for r in rr])); correlations.append({"regime":regime,"x":x,"y":y,"pearson":p,"spearman":s,"observations":len(rr)})
    values=np.array([r["normalized_error"] for r in rows]); edges=np.quantile(values,np.linspace(0,1,6)); edges[0]-=1e-12; edges[-1]+=1e-12
    for r in rows: r["error_bin"]=int(min(max(np.searchsorted(edges,r["normalized_error"],side="right")-1,0),4))
    for regime in ("ID-Easy 20 ns","ID-Hard 80 ns"):
        for b in range(5):
            rr=[r for r in rows if r["regime"]==regime and r["error_bin"]==b]
            if rr: bins.append({"regime":regime,"error_bin":b,"count":len(rr),**{k+"_mean":float(np.mean([r[k] for r in rr])) for k in ("normalized_error","normalized_psi","normalized_aleatoric","psi_pressure_mean","grad_total_kappa_abs","grad_total_nu_abs")}})
    return summaries, correlations, bins


def plot(rows, bins, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(13,4))
    for ax,y,title in zip(axes,("psi_pressure_mean","normalized_psi","normalized_aleatoric"),("Error vs Psi pressure","Error vs normalized Psi","Error vs normalized Aleatoric")):
        for reg,color in (("ID-Easy 20 ns","tab:blue"),("ID-Hard 80 ns","tab:orange")):
            rr=[r for r in rows if r["regime"]==reg]; ax.scatter([r["normalized_error"] for r in rr],[r[y] for r in rr],s=7,alpha=.25,label=reg,color=color)
        ax.set_xlabel("Normalized omitted error"); ax.set_ylabel(y); ax.set_title(title); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/"gradient_alignment_scatter.png",dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,("psi_pressure_mean","normalized_psi","normalized_aleatoric"),("Psi increase pressure","Normalized Psi","Normalized Aleatoric")):
        for reg,color in (("ID-Easy 20 ns","tab:blue"),("ID-Hard 80 ns","tab:orange")):
            rr=[r for r in bins if r["regime"]==reg]; ax.plot([r["error_bin"] for r in rr],[r[key+"_mean"] for r in rr],"o-",label=reg,color=color)
        ax.set_xlabel("Global normalized-error quintile"); ax.set_title(title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/"error_bin_gradient_trends.png",dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json"); p.add_argument("--checkpoint",default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); p.add_argument("--probe-dir",default="runs/paper_style_uncertainty_diagnostics_20260904/generated_p3"); p.add_argument("--output-dir",required=True); a=p.parse_args(); out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    cfg=load_config(a.config); set_seeds(int(cfg["implementation_assumption"]["seed"])); device=torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu"); model=_make_model(cfg,device); payload=torch.load(ROOT/a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload,dict) and "model_state_dict" in payload else payload); model.eval()
    rows=[]
    for i,(regime,file) in enumerate(REGIMES): rows.extend(evaluate(model,ROOT/a.probe_dir/file,cfg,device,99000+i*1000,regime))
    summaries,corrs,bins=summarize(rows); out.mkdir(parents=True,exist_ok=True); write_csv(out/"sample_pair_gradient_metrics.csv",rows); write_csv(out/"summary.csv",summaries); write_csv(out/"correlations.csv",corrs); write_csv(out/"error_bin_summary.csv",bins); plot(rows,bins,out)
    ck=ROOT/a.checkpoint; prov={"checkpoint":a.checkpoint,"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"training_performed":False,"optimizer_step_called":False,"loss": "diagonal multivariate Student-t NLL + lambda*pair-level evidence regularizer","lambda_reg":float(cfg["paper_specified"]["lambda_reg"]),"probes":a.probe_dir}
    (out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)); (out/"config.json").write_text(json.dumps({"regimes":[r[0] for r in REGIMES],"snr_db":15.0,"ng":16,"lambda_reg":prov["lambda_reg"]},indent=2)); (out/"results.json").write_text(json.dumps({"summary":summaries,"correlations":corrs,"error_bins":bins,"provenance":prov},indent=2,sort_keys=True)); print(json.dumps({"output_dir":str(out),"device":str(device),"gpu":prov["gpu"],"summary":summaries,"correlations":corrs},indent=2))


if __name__=="__main__": main()
