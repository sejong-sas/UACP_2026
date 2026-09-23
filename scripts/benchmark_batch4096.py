#!/usr/bin/env python3
"""Non-training benchmark for direct batch-4096 on the canonical 100k archive."""
from __future__ import annotations
import argparse, csv, json, math, subprocess, threading, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915"
CONFIG = BASE / "config.json"
TRAIN = BASE / "generated_data/train_5k_pilot.npz"

def snap():
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        return float(line)
    except Exception:
        return None

class Sampler:
    def __init__(self): self.x=[]; self.stop=threading.Event()
    def __enter__(self):
        def run():
            while not self.stop.is_set(): self.x.append(snap()); self.stop.wait(.2)
        self.t=threading.Thread(target=run, daemon=True); self.t.start(); return self
    def __exit__(self,*_): self.stop.set(); self.t.join(2)

def sync(dev): torch.cuda.synchronize(dev)

def fp32_output(o):
    from src.models.evidential import EvidentialOutput
    return EvidentialOutput(gamma=o.gamma.float(), kappa=o.kappa.float(), psi=o.psi.float(), nu=o.nu.float(), num_subcarriers=o.num_subcarriers)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--precision',choices=['fp32','bf16'],required=True); p.add_argument('--steps',type=int,default=20); p.add_argument('--batch-size',type=int,default=4096); p.add_argument('--warmup-steps',type=int,default=3); a=p.parse_args()
    out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'refusing overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.train_predictor import load_config, set_seeds
    from scripts.diagnose_predictor import _make_model
    from src.models.evidential import evidential_loss
    from src.training.data import CFRNPZDataset, build_noisy_sparse_input, random_grouping_mask
    cfg=load_config(CONFIG); dev=torch.device('cuda:0'); set_seeds(20260819)
    ds=CFRNPZDataset(TRAIN); n=len(ds); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(20260819),num_workers=0,pin_memory=True,drop_last=False)
    gate={'unique_training_cfr_count':n,'batch_size':a.batch_size,'drop_last':False,'dataloader_len':len(loader),'steps_per_epoch':len(loader),'ceil_steps':math.ceil(n/a.batch_size),'epochs_target':150,'total_optimizer_steps':len(loader)*150,'total_sample_exposure':n*150,'gpu':torch.cuda.get_device_name(0),'device':'cuda:0','precision':a.precision,'full_training_started':False}
    (out/'preflight.json').write_text(json.dumps(gate,indent=2)+'\n')
    if n != 100000 or len(loader) != math.ceil(n/a.batch_size): raise RuntimeError(f'gate failed: {gate}')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    model=_make_model(cfg,dev); opt=torch.optim.Adam(model.parameters(),lr=float(cfg['paper_specified']['learning_rate'])); model.train()
    rows=[]; it=iter(loader); utils=[]
    try:
        with Sampler() as samp:
            for step in range(a.steps+a.warmup_steps):
                sync(dev); t0=time.perf_counter(); batch=next(it); t1=time.perf_counter()
                cfr=batch['cfr']; cfr=cfr.to(dev,non_blocking=True); sync(dev); t2=time.perf_counter()
                mask=random_grouping_mask(cfr.shape[0],1024,[4,8,16,32],dev)
                i0=time.perf_counter(); x,target,_=build_noisy_sparse_input(cfr,mask,15.0); sync(dev); i1=time.perf_counter()
                e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True); e0.record()
                if a.precision=='bf16':
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16): raw=model(x)
                    o=fp32_output(raw)
                else: o=model(x)
                e1.record(); e1.synchronize(); forward=e0.elapsed_time(e1)/1000
                e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True); e0.record()
                loss=evidential_loss(o,target.float(),float(cfg['paper_specified']['lambda_reg']),nll_mode=cfg['implementation_assumption']['nll_mode'],reg_mode=cfg['implementation_assumption']['reg_mode'])['total']
                e1.record(); e1.synchronize(); loss_s=e0.elapsed_time(e1)/1000
                e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True); e0.record(); opt.zero_grad(set_to_none=True); loss.backward(); e1.record(); e1.synchronize(); backward=e0.elapsed_time(e1)/1000
                e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True); e0.record(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); e1.record(); e1.synchronize(); optim=e0.elapsed_time(e1)/1000
                sync(dev); wall=time.perf_counter()-t0
                if step>=a.warmup_steps: rows.append({'step':step-a.warmup_steps+1,'data_loading_s':t1-t0,'h2d_s':t2-t1,'input_build_s':i1-i0,'forward_s':forward,'loss_s':loss_s,'backward_s':backward,'optimizer_s':optim,'total_s':wall,'loss':float(loss.detach().cpu())})
            utils=[x for x in samp.x if x is not None]
    except RuntimeError as exc:
        if 'out of memory' in str(exc).lower() or 'cuda error' in str(exc).lower():
            result={**gate,'status':'OOM','error':f'{type(exc).__name__}: {exc}','peak_allocated_mib':torch.cuda.max_memory_allocated(dev)/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved(dev)/2**20}
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return
        raise
    result={**gate,'status':'success','warmup_steps':a.warmup_steps,'measured_steps':len(rows),'mean_sec_per_step':float(np.mean([r['total_s'] for r in rows])),'median_sec_per_step':float(np.median([r['total_s'] for r in rows])),'p95_sec_per_step':float(np.quantile([r['total_s'] for r in rows],.95)),'stage_mean_sec':{k:float(np.mean([r[k] for r in rows])) for k in ['data_loading_s','h2d_s','input_build_s','forward_s','loss_s','backward_s','optimizer_s']},'peak_allocated_mib':torch.cuda.max_memory_allocated(dev)/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved(dev)/2**20,'gpu_utilization_mean_percent':float(np.mean(utils)) if utils else None,'gpu_utilization_median_percent':float(np.median(utils)) if utils else None,'gpu_utilization_p95_percent':float(np.quantile(utils,.95)) if utils else None,'assumptions':['Dataset is RAM-cached by CFRNPZDataset; num_workers=0 avoids worker copies.','pin_memory=True and non_blocking=True.','No per-step validation/checkpoint/log file write.','BF16 condition is IMPLEMENTATION-ASSUMPTION; evidential output/loss stays FP32.','Direct batch only; no gradient accumulation.']}
    with (out/'step_timings.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
