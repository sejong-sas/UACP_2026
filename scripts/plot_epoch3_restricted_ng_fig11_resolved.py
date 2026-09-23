#!/usr/bin/env python3
"""Plot the saved restricted-Ng Fig.11 trace with channel-regime annotations.

This is the saved candidate-restriction result: adaptive candidates are
{4, 8, 16, 32}, and the trace contains no fallback trigger.  It is kept
separate from the later Ng-conditioned-q99 trace, which has a step-90 trigger.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "runs/current_valid_baseline/epoch3_restricted_ng_fig11_20260919/dynamic_trace.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_restricted_ng_fig11_20260919/annotated_report_20260923"
SEGMENTS = [(0,39,"20 ns","#dbeafe"),(40,79,"80 ns","#ffedd5"),(80,119,"10 ns","#dcfce7"),(120,159,"40 ns","#fef3c7"),(160,199,"60 ns","#fce7f3"),(200,239,"120 ns","#ede9fe")]
BOUNDARIES = [40,80,120,160,200]

def add_background(ax, xmax):
    for s,e,label,color in SEGMENTS:
        if s > xmax: continue
        ax.axvspan(s-.5, min(e+.5,xmax), color=color, alpha=.42, zorder=0)
        m=(s+e)/2
        if m <= xmax:
            ax.text(m,.98,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=12,fontweight='bold',color='#243047')
    for b in BOUNDARIES:
        if b <= xmax: ax.axvline(b,color='#475569',ls='--',lw=1.1,alpha=.75)

def num(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf],np.nan)

def make(df, out):
    x=num(df.time_step); ng=num(df.current_ng); epi=num(df.epistemic); tau=num(df.epistemic_threshold); nmse=num(df.nmse_db)
    fig,ax=plt.subplots(3,1,figsize=(16,12),sharex=True,gridspec_kw={'height_ratios':[.9,1.35,1.]})
    for a in ax:
        add_background(a,239); a.set_xlim(0,239); a.grid(alpha=.24); a.tick_params(labelsize=11)
    ax[0].step(x,ng,where='post',lw=2.4,color='#1f4e79',label='Selected/current Ng')
    ax[0].set_ylabel('Selected Ng',fontsize=14); ax[0].set_yticks([1,4,8,16,32,64,128]); ax[0].set_ylim(.5,140); ax[0].legend(loc='lower left',fontsize=11)
    ax[1].plot(x,epi,lw=2.0,color='#7c3aed',marker='o',ms=2.5,label='Epistemic')
    ax[1].plot(x,tau,lw=1.8,color='#b45309',ls='--',label='Saved controller threshold')
    ax[1].set_yscale('log',base=10,subs=[]); ax[1].minorticks_off(); ax[1].set_ylabel('Epistemic / threshold (log scale)',fontsize=14); ax[1].legend(loc='upper left',fontsize=10)
    ax[2].plot(x,nmse,lw=2.0,color='#111827',marker='o',ms=2.5,label='Omitted NMSE'); ax[2].set_ylabel('Omitted NMSE [dB]',fontsize=14); ax[2].set_xlabel('Streaming time step',fontsize=14); ax[2].legend(loc='lower left',fontsize=10)
    vals=nmse[np.isfinite(nmse)]
    if len(vals):
        lo,hi=float(vals.min()),float(vals.max()); p=max(1,.08*(hi-lo)); ax[2].set_ylim(lo-p,hi+p)
    fig.suptitle('Epoch3 Fig.11-style dynamic trace — restricted Ng candidates (no false fallback)',fontsize=18,fontweight='bold')
    fig.text(.5,.006,'Candidates restricted to Ng={4, 8, 16, 32}; saved trace has zero OOD triggers and no Ng=1 fallback. Shaded bands show channel regimes.',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.10,right=.98,top=.92,bottom=.09,hspace=.18); fig.savefig(out,dpi=300); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trace',type=Path,default=TRACE); p.add_argument('--output-dir',type=Path,default=OUT); a=p.parse_args()
    trace=a.trace if a.trace.is_absolute() else ROOT/a.trace; outdir=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir; outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/'epoch3_restricted_ng_fig11_no_false_fallback.png'
    if out.exists(): raise FileExistsError(out)
    df=pd.read_csv(trace)
    if df.ood_trigger.astype(str).str.lower().isin(['true','1']).any(): raise RuntimeError('Trace contains an OOD trigger; not the resolved restricted result.')
    make(df,out); print(f'trace={trace}'); print('false_trigger_count=0'); print(f'figure={out}')

if __name__=='__main__': main()
