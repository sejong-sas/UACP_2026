#!/usr/bin/env python3
"""Minimal post-hoc Ng=64/128 coverage recovery; max two extra epochs."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
BASE_CFG=ROOT/'runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/config_used.json'
TRAIN=ROOT/'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/generated_data/train_5k_pilot.npz'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--resume-checkpoint',required=True); p.add_argument('--epochs',type=int,default=1); a=p.parse_args()
 if a.epochs not in (1,2): raise ValueError('recovery epochs must be 1 or 2')
 out=ROOT/a.output_dir
 if out.exists() and any(out.iterdir()): raise FileExistsError(out)
 out.mkdir(parents=True); import sys; sys.path.insert(0,str(ROOT))
 from scripts.train_predictor import load_config,set_seeds
 from scripts.diagnose_predictor import _make_model
 from src.training.data import CFRNPZDataset,build_noisy_sparse_input,random_grouping_mask
 from src.models.evidential import EvidentialOutput,evidential_loss
 from src.training.metrics import nmse_all_db,nmse_omitted_db
 cfg=load_config(BASE_CFG); cfg=copy.deepcopy(cfg); factors=[4,8,16,32,64,128]; cfg['implementation_assumption']['mask_grouping_factors']=factors; cfg['implementation_assumption']['recovery_epochs']=a.epochs; cfg['implementation_assumption']['mask_coverage_recovery']='IMPLEMENTATION-ASSUMPTION: uniform Ng sampling over {4,8,16,32,64,128}; Ng=1 excluded'
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 d=torch.device('cuda:0'); ds=CFRNPZDataset(TRAIN); loader=DataLoader(ds,batch_size=400,shuffle=True,generator=torch.Generator().manual_seed(20260819),num_workers=0,pin_memory=True,drop_last=False); steps=len(loader)
 pay=torch.load(ROOT/a.resume_checkpoint,map_location=d,weights_only=False); model=_make_model(cfg,d); state=pay.get('model_state_dict',pay) if isinstance(pay,dict) else pay; model.load_state_dict(state)
 has_opt=isinstance(pay,dict) and 'optimizer_state_dict' in pay; opt=torch.optim.Adam(model.parameters(),lr=float(cfg['paper_specified']['learning_rate']));
 if has_opt: opt.load_state_dict(pay['optimizer_state_dict'])
 gate={'resume_checkpoint':a.resume_checkpoint,'resume_has_optimizer_state':has_opt,'unique_training_cfr_count':len(ds),'epochs_added':a.epochs,'batch_size':400,'steps_per_epoch':steps,'optimizer_steps_added':steps*a.epochs,'sample_exposure_added':len(ds)*a.epochs,'mask_factors':factors,'selection_probability_each':{str(x):1/6 for x in factors},'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'training_started':True}
 (out/'config_used.json').write_text(json.dumps(cfg,indent=2)+'\n'); (out/'run_start.json').write_text(json.dumps({**gate,'train_archive_sha256':sha(TRAIN),'resume_checkpoint_sha256':sha(ROOT/a.resume_checkpoint),'optimizer_state_restored':has_opt,'optimizer_state_note':'Existing checkpoint is model-only; new Adam created with canonical LR.' if not has_opt else 'Optimizer state restored.'},indent=2)+'\n')
 hist=[]; set_seeds(20260819); started=time.perf_counter()
 for ep in range(1,a.epochs+1):
  model.train(); totals={k:0.0 for k in ['total','nll','reg','lambda_reg_x_reg','nmse_all_db','nmse_omitted_db']}; count=0; ep0=time.perf_counter()
  for bi,b in enumerate(loader,1):
   c=b['cfr'].to(d,non_blocking=True); mask=random_grouping_mask(c.shape[0],1024,factors,d); x,t,_=build_noisy_sparse_input(c,mask,15.0)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16): raw=model(x)
   o=EvidentialOutput(gamma=raw.gamma.float(),kappa=raw.kappa.float(),psi=raw.psi.float(),nu=raw.nu.float(),num_subcarriers=raw.num_subcarriers); ls=evidential_loss(o,t.float(),1e-3,nll_mode=cfg['implementation_assumption']['nll_mode'],reg_mode=cfg['implementation_assumption']['reg_mode']); opt.zero_grad(set_to_none=True); ls['total'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); n=c.shape[0]; count+=n
   for k in ['total','nll','reg','lambda_reg_x_reg']: totals[k]+=float(ls[k].detach().cpu())*n
   totals['nmse_all_db']+=float(nmse_all_db(o.predicted.detach(),t).cpu())*n; totals['nmse_omitted_db']+=float(nmse_omitted_db(o.predicted.detach(),t,mask).cpu())*n
   if bi%25==0 or bi==steps: (out/'progress.jsonl').open('a').write(json.dumps({'epoch':ep,'batch':bi,'batches_total':steps,'optimizer_steps_done':(ep-1)*steps+bi,'sample_exposure_done':(ep-1)*len(ds)+min(bi*400,len(ds))})+'\n')
  metrics={k:v/count for k,v in totals.items()}; metrics['epoch_seconds']=time.perf_counter()-ep0; metrics['epoch']=ep; hist.append(metrics); torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':cfg,'recovery_epoch':ep,'source_checkpoint':a.resume_checkpoint},out/f'recovery_checkpoint_epoch_{ep}.pt')
 final=out/f'recovery_checkpoint_{a.epochs}ep.pt'; torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':cfg,'recovery_epoch':a.epochs,'source_checkpoint':a.resume_checkpoint},final)
 result={'experiment':'mask coverage recovery only','checkpoint':str(final.relative_to(ROOT)),'history':hist,'training_seconds':time.perf_counter()-started,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated(d)/2**20,'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved(d)/2**20,**gate,'optimizer_state_restored':has_opt,'assumptions':['IMPLEMENTATION-ASSUMPTION: uniform Ng sampling over {4,8,16,32,64,128}','BF16 backbone with FP32 evidential transform/loss','Existing checkpoint had no optimizer state; canonical Adam recreated with LR=1e-4','No architecture/loss/data/noise/target/layout changes']}
 (out/'training_results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
