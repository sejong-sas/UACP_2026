#!/usr/bin/env python3
"""Post-process Experiment A without rerunning training or evaluation."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/uncertainty_guided_adaptation_20260923_v8"

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    thresholds=pd.read_csv(BASE/'fixed_thresholds.csv').set_index('ng')['tau_ng'].to_dict()
    pre=pd.read_csv(BASE/'pre_evaluation_samples.csv')
    pre['tau_ng']=pre['ng'].map(thresholds)
    pre['z_log10_epi_over_tau']=np.log10(pre['epistemic'].clip(lower=1e-30)/pre['tau_ng'])
    sev=pre.groupby(['delay_ns','ng'])['z_log10_epi_over_tau'].agg(
        median_z='median', mean_z='mean', q90_z=lambda x:x.quantile(.90),
        q95_z=lambda x:x.quantile(.95), q99_z=lambda x:x.quantile(.99)).reset_index()
    write_csv(BASE/'pre_adaptation_severity_summary_by_delay.csv',sev.to_dict('records'))
    primary=sev[sev.ng.isin([16,32])].groupby('delay_ns')['median_z'].median().reset_index(name='severity_median_z_primary')
    ev=pd.read_csv(BASE/'evaluation_summary.csv')
    q=ev[(ev.regime=='target')&(ev.metric=='nmse_omitted_db')&(ev.ng.isin([16,32]))]
    rows=[]
    for delay in sorted(q.delay_ns.unique()):
        full=q[(q.delay_ns==delay)&(q.scope=='full')].set_index('ng')['median']
        chosen='none'; ordinal=np.nan
        for name,ord_ in [('partial_small',1),('partial_medium',2),('partial_large',3),('full',4)]:
            a=q[(q.delay_ns==delay)&(q.scope==name)].set_index('ng')['median']
            if all(float(a.loc[ng]) <= float(full.loc[ng])+.5 for ng in [16,32]):
                chosen=name; ordinal=ord_; break
        rows.append({'delay_ns':delay,'severity_median_z_primary':float(primary.loc[primary.delay_ns==delay,'severity_median_z_primary'].iloc[0]),'minimum_scope':chosen,'scope_ordinal':ordinal,'criterion_db':.5})
    write_csv(BASE/'minimum_sufficient_scope_corrected.csv',rows)
    x=np.array([r['severity_median_z_primary'] for r in rows]); y=np.array([r['scope_ordinal'] for r in rows])
    rho=float(spearmanr(x,y).statistic)
    gate={'rho':rho,'conditions_with_scope':len(rows),'scope_ordinals':y.tolist(),'non_decreasing_by_delay':bool(np.all(np.diff(y)>=0)),
          'passes_rho':rho>=.7,'passes_condition_count':len(rows)>=4,'nontrivial_scope_variation':len(set(y))>1,
          'h1_supported':bool(rho>=.7 and len(rows)>=4 and len(set(y))>1 and np.all(np.diff(y)>=0)),
          'severity_source':'pre_evaluation_samples.csv; corrected delay-wise aggregation'}
    (BASE/'h1_gate_corrected.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    # compact data for plots/tables
    write_csv(BASE/'h1_plot_data.csv',rows)
    print(json.dumps(gate,indent=2))

if __name__=='__main__': main()
