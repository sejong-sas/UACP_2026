#!/usr/bin/env python3
"""Frozen-checkpoint static Ng x delay sweep; no training or threshold tuning."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]
DELAYS=[10,20,40,60,80,120]; NGS=[1,4,8,16,32,64,128]
def q(v):
    a=np.asarray(v,float); return {'mean':float(a.mean()),'median':float(np.median(a)),'p05':float(np.quantile(a,.05)),'p25':float(np.quantile(a,.25)),'p75':float(np.quantile(a,.75)),'p95':float(np.quantile(a,.95))}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--samples-per-combination',type=int,default=200); p.add_argument('--batch-size',type=int,default=32); a=p.parse_args()
    out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT))
    from scripts.train_predictor import load_config,set_seeds
    from scripts.diagnose_predictor import _make_model
    from scripts.fig11_dynamic_runtime_validation import aggregate_omitted_scores,make_mask
    from src.training.data import build_noisy_sparse_input,cfr_to_real_imag
    cfg=load_config('runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json'); d=torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    m=_make_model(cfg,d); pay=torch.load(ROOT/a.checkpoint,map_location=d,weights_only=False); m.load_state_dict(pay.get('model_state_dict',pay) if isinstance(pay,dict) else pay); m.eval()
    rows=[]; raw_rows=[]; seed=20262000; finite_total=nonfinite_total=0
    for di,delay in enumerate(DELAYS):
        fn='test_delay_1_ms.npz' if delay==1000000 else f'test_delay_{delay}_ns.npz'
        with np.load(ROOT/'runs/baseline_reproduction/step2_delay_sweep_repro/generated_data'/fn) as z: cfr_np=z['cfr'][:a.samples_per_combination]
        for ng in NGS:
            allv=[]; omv=[]; alv=[]; epv=[]; nuf=[]; observed=1024//ng + (1 if 1024%ng else 0)
            for start in range(0,len(cfr_np),a.batch_size):
                set_seeds(seed+di*100000+ng*1000+start); c=torch.from_numpy(cfr_np[start:start+a.batch_size]).to(d); mask=make_mask(c.shape[0],ng,d); x,target,_=build_noisy_sparse_input(c,mask,15.0)
                with torch.inference_mode(): o=m(x); err=(o.predicted-target).square(); all_num=err.sum((1,2)); all_den=target.square().sum((1,2)).clamp_min(1e-12); omitted=(1-mask)[:,None,:].expand_as(target); om_num=(err*omitted).sum((1,2)); om_den=(target.square()*omitted).sum((1,2)).clamp_min(1e-12); allv.extend((10*torch.log10((all_num/all_den).clamp_min(1e-12))).cpu().numpy().tolist()); omv.extend((10*torch.log10((om_num/om_den).clamp_min(1e-12))).cpu().numpy().tolist())
                with torch.inference_mode(): ale=o.aleatoric; epi=o.epistemic; pair_ale=ale.reshape(ale.shape[0],2,4,1024).sum(1).mean(1); pair_epi=epi.reshape(epi.shape[0],2,4,1024).sum(1).mean(1); om=(1-mask); cnt=om.sum(1).clamp_min(1); alv.extend(((pair_ale*om).sum(1)/cnt).cpu().numpy().tolist()); epv.extend(((pair_epi*om).sum(1)/cnt).cpu().numpy().tolist()); nu=o.nu_expanded; nf=int((~torch.isfinite(nu)).sum().cpu()); ff=int(torch.isfinite(nu).sum().cpu()); finite_total+=ff; nonfinite_total+=nf; nuf.append(nf)
                del c,mask,x,target,o
            row={'delay_ns':delay,'ng':ng,'observed_subcarriers':observed,'samples':len(allv),'all_nmse_db':q(allv),'omitted_nmse_db':q(omv),'aleatoric':q(alv),'epistemic':q(epv),'nu_nonfinite_components':sum(nuf),'nu_nonfinite':bool(sum(nuf))}
            rows.append(row)
            raw_rows.extend({'delay_ns':delay,'ng':ng,'sample':i,'all_nmse_db':allv[i],'omitted_nmse_db':omv[i],'aleatoric':alv[i],'epistemic':epv[i]} for i in range(len(allv)))
    def flat(r):
        z={'delay_ns':r['delay_ns'],'ng':r['ng'],'observed_subcarriers':r['observed_subcarriers'],'samples':r['samples'],'nu_nonfinite_components':r['nu_nonfinite_components'],'nu_nonfinite':r['nu_nonfinite']}
        for key in ['all_nmse_db','omitted_nmse_db','aleatoric','epistemic']:
            for stat,val in r[key].items(): z[f'{key}_{stat}']=val
        return z
    flatrows=[flat(r) for r in rows]
    with (out/'sweep_summary.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(flatrows[0])); w.writeheader(); w.writerows(flatrows)
    with (out/'per_sample.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(raw_rows[0])); w.writeheader(); w.writerows(raw_rows)
    # Training coverage is analytically uniform over the configured four factors;
    # offsets are uniform within each selected factor.
    coverage={'training_mask_function':'src/training/data.py:183-190 random_grouping_mask','configured_factors':[4,8,16,32],'selection_probability_each':{str(x):.25 for x in [4,8,16,32]},'offset_policy':'uniform integer offset in [0, factor-1]','observed_subcarriers_by_factor':{'4':256,'8':128,'16':64,'32':32},'ng_exposed_by_training':[4,8,16,32],'ng_not_exposed_by_training':[1,64,128],'note':'Each training sample independently selects one factor; 100k×5 gives 500,000 draws, approximately 125,000 per factor in expectation.'}
    (out/'training_mask_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    delays=np.array(DELAYS); plt.figure(figsize=(8,5))
    for ng in [4,8,16,32,64,128]:
        z=[r for r in rows if r['ng']==ng]; plt.plot(delays,[r['aleatoric']['mean'] for r in z],marker='o',label=f'Ng={ng}')
    plt.xlabel('Delay spread (ns)'); plt.ylabel('Aleatoric score'); plt.grid(alpha=.25); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(out/'aleatoric_vs_delay.png',dpi=220); plt.savefig(out/'aleatoric_vs_delay.pdf'); plt.close()
    plt.figure(figsize=(8,5))
    for delay in [10,20,40,60,80]:
        z=[r for r in rows if r['delay_ns']==delay]; plt.plot(NGS,[r['epistemic']['mean'] for r in z],marker='o',label=f'{delay} ns')
    plt.xlabel('Ng'); plt.ylabel('Epistemic score'); plt.xscale('symlog',linthresh=1); plt.grid(alpha=.25); plt.legend(ncol=2); plt.tight_layout(); plt.savefig(out/'epistemic_vs_ng_id_delays.png',dpi=220); plt.savefig(out/'epistemic_vs_ng_id_delays.pdf'); plt.close()
    result={'checkpoint':a.checkpoint,'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'delays_ns':DELAYS,'ng_candidates':NGS,'samples_per_combination':a.samples_per_combination,'batch_size':a.batch_size,'training_performed':False,'finite_nu_components':finite_total,'nonfinite_nu_components':nonfinite_total,'assumptions':['IMPLEMENTATION-ASSUMPTION: delay-sweep files are used as static evaluation samples','same 15 dB reported-CFR complex AWGN and clean target pipeline','uniform grouping mask with fixed offset zero for static Ng','all-NMSE and omitted-NMSE are both stored; no threshold tuning']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
