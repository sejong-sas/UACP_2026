#!/usr/bin/env python3
"""Create the Experiment-A severity/scope and target-NMSE figures."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/uncertainty_guided_adaptation_20260923_v8'
OUT=BASE/'plots'; OUT.mkdir(exist_ok=True)

def main():
    h=pd.read_csv(BASE/'h1_plot_data.csv')
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    ax.plot(h.severity_median_z_primary,h.scope_ordinal,'o-',color='#1f4e79',lw=2,ms=8)
    for _,r in h.iterrows(): ax.annotate(f"{int(r.delay_ns)} ns",(r.severity_median_z_primary,r.scope_ordinal),xytext=(5,7),textcoords='offset points',fontsize=9)
    ax.set_xlabel('Pre-adaptation normalized Epistemic severity, median $z=\\log_{10}(U_{epi}/\\tau_{Ng})$')
    ax.set_ylabel('Minimum sufficient scope ordinal')
    ax.set_yticks([1,2,3,4],['3.16%','12.53%','25.03%','Full']); ax.grid(alpha=.25)
    ax.set_title('Experiment A: Epistemic severity vs minimum adaptation scope')
    fig.tight_layout(); fig.savefig(OUT/'figure_A_severity_vs_scope.png',dpi=300); plt.close(fig)
    ev=pd.read_csv(BASE/'evaluation_summary.csv'); q=ev[(ev.regime=='target')&(ev.metric=='nmse_omitted_db')&(ev.ng.isin([16,32]))]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,ng in zip(axs,[16,32]):
        for scope,color in [('partial_small','#1f77b4'),('partial_medium','#ff7f0e'),('partial_large','#2ca02c'),('full','#d62728')]:
            s=q[(q.ng==ng)&(q.scope==scope)].sort_values('delay_ns'); ax.plot(s.delay_ns,s['median'],'o-',label=scope,color=color)
        ax.set_title(f'Ng={ng}'); ax.set_xlabel('Target delay spread [ns]'); ax.grid(alpha=.25)
    axs[0].set_ylabel('Target omitted NMSE [dB]'); axs[1].legend(fontsize=8); fig.suptitle('Experiment A: target reconstruction by scope')
    fig.tight_layout(); fig.savefig(OUT/'figure_B_target_nmse_by_scope.png',dpi=300); plt.close(fig)

if __name__=='__main__': main()
