#!/usr/bin/env python3
"""Post-process only: plots from already saved canonical Fig.8/Fig.9 CSVs."""
import csv, json, argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);a=p.parse_args();d=Path(a.input_dir)
 with (d/'fig8_distribution.csv').open() as f: dist=list(csv.DictReader(f))
 with (d/'fig9_calibration.csv').open() as f: cal=list(csv.DictReader(f))
 plt.figure(figsize=(8,5)); labels=[x['regime'] for x in dist]; means=[float(x['mean']) for x in dist]; p05=[float(x['p05']) for x in dist]; p95=[float(x['p95']) for x in dist]
 plt.errorbar(labels,means,yerr=[[m-l for m,l in zip(means,p05)],[h-m for h,m in zip(p95,means)]],fmt='o-',capsize=4);plt.ylabel('Epistemic uncertainty (raw Eq.12/13)');plt.xticks(rotation=20);plt.grid(alpha=.25);plt.tight_layout();plt.savefig(d/'fig8_epistemic_distribution.png',dpi=180);plt.close()
 plt.figure(figsize=(6,5));plt.plot([0,1],[0,1],'k--',label='ideal y=x')
 for pool in ('ID','OOD'):
  z=[x for x in cal if x['pool']==pool];plt.plot([float(x['nominal']) for x in z],[float(x['empirical']) for x in z],'o-',label=pool)
 plt.xlabel('Nominal coverage');plt.ylabel('Empirical coverage');plt.grid(alpha=.25);plt.legend();plt.tight_layout();plt.savefig(d/'fig9_calibration_curve.png',dpi=180);plt.close()
if __name__=='__main__':main()
