#!/usr/bin/env python3
"""Distribution-only ID versus 120 ns Epistemic analysis for epoch3."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
CKPT=ROOT/'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt'
DATA=ROOT/'runs/baseline_reproduction/step2_delay_sweep_repro/generated_data'
K=1024; NGS=[4,8,16,32]; ID_DELAYS=[10,20,40,60,80,100]; OOD_DELAY=120

def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def auc(x,y):
    z=np.concatenate([x,y]); r=rankdata(z,method='average'); return float((r[len(x):].sum()-len(y)*(len(y)+1)/2)/(len(x)*len(y)))

def overlap(x,y,bins=80):
    z=np.log10(np.concatenate([x[x>0],y[y>0]])); edges=np.linspace(z.min(),z.max(),bins+1); hx,_=np.histogram(np.log10(x[x>0]),bins=edges,density=True); hy,_=np.histogram(np.log10(y[y>0]),bins=edges,density=True); widths=np.diff(edges); return float(np.sum(np.minimum(hx,hy)*widths))

def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    m=_make_model(load_config('configs/current_valid_baseline_100k1_seed_20260819.json'),device); p=torch.load(CKPT,map_location=device,weights_only=False); s=p['model_state_dict'] if isinstance(p,dict) and 'model_state_dict' in p else p; m.load_state_dict(s); m.eval(); return m

def eval_batch(model,cfr_np,ng,seed,device):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    set_seeds(seed); b=torch.from_numpy(cfr_np).to(device); mask=torch.zeros((len(b),K),device=device); mask[:,::ng]=1; x,target,_=build_noisy_sparse_input(b,mask,15.0)
    with torch.inference_mode():
        o=model(x); ale_map=o.psi/(o.nu_expanded-2*K-1); epi_map=ale_map/o.kappa_expanded; omitted=1-mask
        pa=ale_map.reshape(len(b),2,4,K).permute(0,2,1,3).sum(2).mean(1); pe=epi_map.reshape(len(b),2,4,K).permute(0,2,1,3).sum(2).mean(1); count=omitted.sum(1).clamp_min(1); ale=(pa*omitted).sum(1)/count; epi=(pe*omitted).sum(1)/count
        nmse=10*torch.log10((((o.gamma-target).square()*omitted[:,None,:]).sum((1,2))/(target.square()*omitted[:,None,:]).sum((1,2)).clamp_min(1e-12)).clamp_min(1e-12))
        # Per-sample parameter summaries over omitted components.
        om=omitted[:,None,:].expand_as(ale_map); params={
            'aleatoric':(ale_map*om).sum((1,2))/om.sum((1,2)).clamp_min(1),
            'kappa':(o.kappa_expanded*om).sum((1,2))/om.sum((1,2)).clamp_min(1),
            'nu_margin':((o.nu_expanded-(2*K+1))*om).sum((1,2))/om.sum((1,2)).clamp_min(1),
            'psi':(o.psi*om).sum((1,2))/om.sum((1,2)).clamp_min(1),}
    out={'epistemic':epi.cpu().numpy(),'aleatoric':params['aleatoric'].cpu().numpy(),'kappa':params['kappa'].cpu().numpy(),'nu_margin':params['nu_margin'].cpu().numpy(),'psi':params['psi'].cpu().numpy(),'nmse':nmse.cpu().numpy()}
    del b,mask,x,target,o,ale_map,epi_map,omitted
    if device.type=='cuda': torch.cuda.empty_cache()
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--samples-per-regime',type=int,default=200); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--seed',type=int,default=20262000); a=ap.parse_args(); out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT)); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    model=load_model(device); records=[]; arrays={}
    all_regimes=[(d,f'{d} ns') for d in ID_DELAYS]+[(OOD_DELAY,'OOD-Near 120 ns')]
    for ri,(delay,label) in enumerate(all_regimes):
        with np.load(DATA/f'test_delay_{delay}_ns.npz') as d: cfr=np.asarray(d['cfr'][:a.samples_per_regime],dtype=np.complex64)
        for ng in NGS:
            parts=[]
            for start in range(0,len(cfr),a.batch_size): parts.append(eval_batch(model,cfr[start:start+a.batch_size],ng,a.seed+ri*100000+ng+start,device))
            values={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}; arrays[(delay,ng)]=values
            for i in range(len(cfr)): records.append({'delay_ns':delay,'regime':label,'ng':ng,'sample':i,**{k:float(v[i]) for k,v in values.items()}})
    write_csv(out/'per_sample.csv',records)
    summary=[]
    for ng in NGS:
        idv=np.concatenate([arrays[(d,ng)]['epistemic'] for d in ID_DELAYS]); ood=arrays[(OOD_DELAY,ng)]['epistemic']; tau=float(np.quantile(idv,.99)); row={'ng':ng,'id_count':len(idv),'ood_count':len(ood),'id_mean':float(np.mean(idv)),'id_median':float(np.median(idv)),'id_q95':float(np.quantile(idv,.95)),'id_q99':tau,'id_max':float(np.max(idv)),'ood_mean':float(np.mean(ood)),'ood_median':float(np.median(ood)),'ood_q95':float(np.quantile(ood,.95)),'ood_q99':float(np.quantile(ood,.99)),'ood_max':float(np.max(ood)),'auroc_id_vs_120ns':auc(idv,ood),'log10_overlap_coefficient':overlap(idv,ood),'id_q99_threshold':tau,'id_fpr_at_id_q99':float(np.mean(idv>tau)),'ood_tpr_at_id_q99':float(np.mean(ood>tau)),'id_nmse_median':float(np.median(np.concatenate([arrays[(d,ng)]['nmse'] for d in ID_DELAYS]))),'ood_nmse_median':float(np.median(arrays[(OOD_DELAY,ng)]['nmse']))}
        for name in ['aleatoric','kappa','nu_margin','psi']:
            iv=np.concatenate([arrays[(d,ng)][name] for d in ID_DELAYS]); ov=arrays[(OOD_DELAY,ng)][name]; row[f'id_{name}_median']=float(np.median(iv)); row[f'ood_{name}_median']=float(np.median(ov))
        summary.append(row)
    write_csv(out/'ng_distribution_summary.csv',summary)
    fig,axes=plt.subplots(2,2,figsize=(12,8));
    for ax,ng in zip(axes.flat,NGS):
        iv=np.concatenate([arrays[(d,ng)]['epistemic'] for d in ID_DELAYS]); ov=arrays[(OOD_DELAY,ng)]['epistemic']; positive=np.concatenate([iv[iv>0],ov[ov>0]]); bins=np.linspace(np.log10(positive.min()),np.log10(positive.max()),60); ax.hist(np.log10(iv[iv>0]),bins=bins,density=True,alpha=.55,label='ID pooled'); ax.hist(np.log10(ov[ov>0]),bins=bins,density=True,alpha=.55,label='120 ns'); ax.set_title(f'Ng={ng}'); ax.set_xlabel('log10 Epistemic'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'ng_epistemic_id_vs_120ns.png',dpi=180); plt.close(fig)
    (out/'results.json').write_text(json.dumps({'checkpoint':str(CKPT),'device':str(device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'ngs':NGS,'id_delays_ns':ID_DELAYS,'ood_delay_ns':OOD_DELAY,'samples_per_regime':a.samples_per_regime,'batch_size':a.batch_size,'no_training':True,'OOD_used_for_threshold':False,'protocol':['direct-CFR AWGN 15 dB','uniform periodic mask','Eq.(12)/(13) omitted aggregation'],'implementation_assumptions':['distribution analysis only; no controller/threshold modification','log10 histogram overlap coefficient used as descriptive overlap metric']},indent=2))

if __name__=='__main__': main()
