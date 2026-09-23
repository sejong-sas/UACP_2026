#!/usr/bin/env python3
"""Measure frozen-checkpoint inference and controller latency only."""
from __future__ import annotations
import argparse, csv, json, sys, threading, subprocess, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def gpu_util():
    try:
        x=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
        return float(x)
    except Exception: return None
class Sampler:
    def __enter__(self):
        self.x=[]; self.stop=threading.Event()
        def run():
            while not self.stop.is_set(): self.x.append(gpu_util()); self.stop.wait(.2)
        self.t=threading.Thread(target=run,daemon=True); self.t.start(); return self
    def __exit__(self,*_): self.stop.set(); self.t.join(2)
def sync(d): torch.cuda.synchronize(d)
def stats(v):
    a=np.asarray(v,float); return {'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'p95_ms':float(np.quantile(a,.95)),'p99_ms':float(np.quantile(a,.99))}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--repeats',type=int,default=200); a=p.parse_args()
    out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(ROOT))
    from scripts.train_predictor import load_config, set_seeds
    from scripts.diagnose_predictor import _make_model
    from scripts.fig11_dynamic_runtime_validation import aggregate_omitted_scores, make_mask, select_next_ng
    from src.training.data import build_noisy_sparse_input
    cfg=load_config('runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json'); d=torch.device('cuda:0')
    model=_make_model(cfg,d); pay=torch.load(ROOT/a.checkpoint,map_location=d,weights_only=False); state=pay.get('model_state_dict',pay) if isinstance(pay,dict) else pay; model.load_state_dict(state); model.eval()
    with np.load(ROOT/'runs/baseline_reproduction/step2_delay_sweep_repro/generated_data/test_delay_20_ns.npz') as z: cfr=torch.from_numpy(z['cfr'][0:1]).to(d)
    mask=make_mask(1,16,d); set_seeds(20262000); x,target,_=build_noisy_sparse_input(cfr,mask,15.0)
    with torch.inference_mode():
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16): raw=model(x)
        from src.models.evidential import EvidentialOutput
        warm=EvidentialOutput(gamma=raw.gamma.float(),kappa=raw.kappa.float(),psi=raw.psi.float(),nu=raw.nu.float(),num_subcarriers=raw.num_subcarriers)
    ale_target=.14156769961118698; epi_threshold=.07602761693298817; ale_delta=.1*ale_target
    # Warm up both precision paths; no optimizer or checkpoint update is performed.
    for _ in range(20):
        with torch.inference_mode():
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16): model(x)
    sync(d); torch.cuda.reset_peak_memory_stats(d); fp32=[]; bf16=[]; unc=[]; ctrl=[]; end=[]; utils=[]
    with Sampler() as sampler:
        with torch.inference_mode():
            for _ in range(a.repeats):
                sync(d); t=time.perf_counter(); model(x); sync(d); fp32.append((time.perf_counter()-t)*1000)
                sync(d); t=time.perf_counter()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16): o=model(x)
                o=EvidentialOutput(gamma=o.gamma.float(),kappa=o.kappa.float(),psi=o.psi.float(),nu=o.nu.float(),num_subcarriers=o.num_subcarriers); sync(d); bf16.append((time.perf_counter()-t)*1000)
                sync(d); t=time.perf_counter(); ale=o.psi/(o.nu_expanded-2049.0); epi=ale/o.kappa_expanded; av,ev=aggregate_omitted_scores(ale,epi,mask); sync(d); unc.append((time.perf_counter()-t)*1000)
                t=time.perf_counter(); select_next_ng(16,av,ev,epi_threshold,ale_target,ale_delta); ctrl.append((time.perf_counter()-t)*1000)
                sync(d); t=time.perf_counter(); cc=cfr.to(d,non_blocking=True); mm=make_mask(1,16,d); xx,_,_=build_noisy_sparse_input(cc,mm,15.0)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16): oo=model(xx)
                oo=EvidentialOutput(gamma=oo.gamma.float(),kappa=oo.kappa.float(),psi=oo.psi.float(),nu=oo.nu.float(),num_subcarriers=oo.num_subcarriers); aa=oo.psi/(oo.nu_expanded-2049.0); ee=aa/oo.kappa_expanded; aav,eev=aggregate_omitted_scores(aa,ee,mm); select_next_ng(16,aav,eev,epi_threshold,ale_target,ale_delta); sync(d); end.append((time.perf_counter()-t)*1000)
                utils.extend([z for z in sampler.x[-2:] if z is not None])
    rows=[]
    for name,val in [('model_forward_fp32',fp32),('model_forward_bf16',bf16),('uncertainty_calculation',unc),('controller_decision',ctrl),('end_to_end_bf16',end)]: rows.append({'stage':name,**stats(val)})
    with (out/'runtime_summary.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result={'checkpoint':a.checkpoint,'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'repeats':a.repeats,'warmup_iterations':20,'no_training':True,'runtime_precision':'BF16 backbone with FP32 evidential output/aggregation','stages':rows,'gpu_utilization_mean_percent':float(np.mean(utils)) if utils else None,'gpu_utilization_median_percent':float(np.median(utils)) if utils else None,'gpu_utilization_p95_percent':float(np.quantile(utils,.95)) if utils else None,'peak_allocated_mib':torch.cuda.max_memory_allocated(d)/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved(d)/2**20,'assumptions':['IMPLEMENTATION-ASSUMPTION: BF16 AMP inference for the operational baseline','Eq.(12)/(13) controller copied from existing Fig.11 implementation','latency excludes plot/file I/O and optimizer updates','BER/EVM are not measured because no repository PHY/precoding/postcoding pipeline exists']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
