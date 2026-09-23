#!/usr/bin/env python3
"""Fixed last4 Partial FT with a frozen Pre ID-feature anchor.

This is a single RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION experiment.
The adaptation objective and optimizer are shared with partial_ft_adapt.py;
the only added term is a normalized anchor loss against a frozen Pre teacher.
"""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.partial_ft_adapt import (load_json, build_model, load_checkpoint, configure_trainable_scope,
    optimizer_for_trainable, scheduled_mask, make_adaptation_observation)
from src.training.data import CFRNPZDataset
from src.models.evidential import evidential_loss

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def feature(model,x):
    h=model.input_projection(x)
    for block in model.residual_blocks: h=block(h)
    return h
def anchor_loss(student, teacher, x, eps=1e-8):
    zs=feature(student,x); zt=feature(teacher,x).detach()
    return ((zs-zt).square().sum(dim=(1,2))/(zt.square().sum(dim=(1,2))+eps)).mean()
def smoke(model, teacher, batch, config, device, seed):
    cfr=batch['cfr'].to(device); mask=scheduled_mask(cfr.shape[0],1024,1,0,seed,device)
    x,target,_=make_adaptation_observation(cfr,mask,seed,device)
    opt=optimizer_for_trainable(model,1e-4); before={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt.zero_grad(set_to_none=True); out=model(x)
    base=evidential_loss(out,target,float(config['paper_specified']['lambda_reg']),nll_mode=config['implementation_assumption'].get('nll_mode','elementwise'),reg_mode=config['implementation_assumption'].get('reg_mode','elementwise'))['total']
    al=anchor_loss(model,teacher,x); loss=base+al
    loss.backward(); grads=sum(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()); opt.step()
    frozen=all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before)
    return {'base_loss':float(base.detach().cpu()),'anchor_loss':float(al.detach().cpu()),'total_loss':float(loss.detach().cpu()),'finite':bool(torch.isfinite(loss)),'trainable_gradient_tensors':int(grads),'frozen_unchanged':frozen,'device':str(device)}
def run(args):
    outdir=ROOT/args.output_dir
    if outdir.exists() and any(outdir.iterdir()): raise FileExistsError(outdir)
    outdir.mkdir(parents=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' or 'GB10' not in torch.cuda.get_device_name(0): raise RuntimeError('NVIDIA GB10/cuda:0 required')
    config=load_json(args.config); student=build_model(args.config,device); teacher=build_model(args.config,device)
    csha=load_checkpoint(student,ROOT/args.checkpoint,device); load_checkpoint(teacher,ROOT/args.checkpoint,device)
    teacher.eval();
    for p in teacher.parameters(): p.requires_grad=False
    configure_trainable_scope(student,'last_4_blocks_plus_head')
    train=CFRNPZDataset(ROOT/args.adapt_train_data); val=CFRNPZDataset(ROOT/args.adapt_val_data); anchor=CFRNPZDataset(ROOT/args.anchor_data)
    manifest={'research_extension':True,'paper_setting':False,'scope':'last_4_blocks_plus_head','anchor_lambda':1.0,'anchor_definition':'mean(||z_student-z_teacher||^2/(||z_teacher||^2+1e-8)); z is ResidualBlock31/head input','teacher':'frozen Pre checkpoint','checkpoint':args.checkpoint,'checkpoint_sha256':csha,'adapt_train_data':args.adapt_train_data,'adapt_val_data':args.adapt_val_data,'anchor_data':args.anchor_data,'anchor_data_sha256':sha(ROOT/args.anchor_data),'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'seed':args.seed,'device':str(device),'gpu':torch.cuda.get_device_name(0),'lambda_reg':config['paper_specified']['lambda_reg'],'precision':'FP32','one_variable_changed':'objective: add fixed ID feature anchor; scope and all optimizer/data settings unchanged'}
    if args.smoke:
        manifest['smoke']=smoke(student,teacher,next(iter(DataLoader(train,batch_size=1,shuffle=False))),config,device,args.seed)
        (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2)); return
    import numpy as np
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed),num_workers=0)
    anchor_loader=DataLoader(anchor,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed+900000),num_workers=0)
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    opt=optimizer_for_trainable(student,args.lr); history=[]; times=[]; steps=0; start=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,args.epochs+1):
        student.train(); teacher.eval(); tl=[]; abl=[]
        for bi,(b,ab) in enumerate(zip(train_loader,anchor_loader)):
            t=time.perf_counter(); cfr=b['cfr'].to(device); acfr=ab['cfr'].to(device)
            mask=scheduled_mask(cfr.shape[0],1024,epoch,bi,args.seed,device); amask=scheduled_mask(acfr.shape[0],1024,epoch,bi,args.seed+900000,device)
            x,target,_=make_adaptation_observation(cfr,mask,args.seed,device,epoch,bi); ax,_,_=make_adaptation_observation(acfr,amask,args.seed+900000,device,epoch,bi)
            opt.zero_grad(set_to_none=True); out=student(x)
            base=evidential_loss(out,target,float(config['paper_specified']['lambda_reg']),nll_mode=config['implementation_assumption'].get('nll_mode','elementwise'),reg_mode=config['implementation_assumption'].get('reg_mode','elementwise'))['total']
            al=anchor_loss(student,teacher,ax); loss=base+al
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss epoch {epoch}')
            loss.backward(); opt.step()
            if device.type=='cuda': torch.cuda.synchronize(device)
            times.append(time.perf_counter()-t); tl.append(float(base.detach().cpu())); abl.append(float(al.detach().cpu())); steps+=1
        student.eval(); vl=[]
        with torch.inference_mode():
            for bi,b in enumerate(val_loader):
                cfr=b['cfr'].to(device); mask=scheduled_mask(cfr.shape[0],1024,epoch,bi,args.seed+700000,device); x,target,_=make_adaptation_observation(cfr,mask,args.seed+700000,device,epoch,bi); val_output=student(x)
                vl.append(float(evidential_loss(val_output,target,float(config['paper_specified']['lambda_reg']),nll_mode=config['implementation_assumption'].get('nll_mode','elementwise'),reg_mode=config['implementation_assumption'].get('reg_mode','elementwise'))['total'].detach().cpu()))
        cp=outdir/f'adapted_epoch_{epoch}.pt'; torch.save(student.state_dict(),cp)
        history.append({'epoch':epoch,'train_base_loss':sum(tl)/len(tl),'train_anchor_loss':sum(abl)/len(abl),'train_total_loss':sum(tl)/len(tl)+sum(abl)/len(abl),'val_loss':sum(vl)/len(vl),'optimizer_steps':steps,'samples_seen':steps*args.batch_size,'checkpoint':str(cp.relative_to(ROOT))})
    torch.save(student.state_dict(),outdir/'adapted_model.pt')
    r={'history':history,'trainable_params':sum(p.numel() for p in student.parameters() if p.requires_grad),'training_seconds':time.perf_counter()-start,'seconds_per_step':sum(times)/len(times),'peak_vram_mib':torch.cuda.max_memory_allocated(device)/2**20,'frozen_parameter_policy':'all non-last4/head parameters requires_grad=False'}
    manifest['training']=r; (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--config',default='configs/current_valid_baseline_100k1_seed_20260819.json'); p.add_argument('--adapt-train-data',required=True); p.add_argument('--adapt-val-data',required=True); p.add_argument('--anchor-data',required=True); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,required=True); p.add_argument('--output-dir',required=True); p.add_argument('--smoke',action='store_true'); run(p.parse_args())
