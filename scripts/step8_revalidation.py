#!/usr/bin/env python3
"""Summarize corrected lambda revalidation without retraining."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]

def load(path):
    return json.loads(Path(path).read_text())

def row(lam, regime, m):
    return {
        "lambda": lam, "regime": regime, "nmse_omitted": m["nmse_omitted_db"],
        "aleatoric": m["aleatoric_omitted"]["mean"], "epistemic": m["epistemic_omitted"]["mean"],
        "psi": m["psi_omitted"]["mean"], "kappa": m["kappa_omitted"]["mean"],
        "denom": m["df_cov_omitted"]["mean"], "err_ale_p": m["error_aleatoric_pearson"],
        "err_epi_p": m["error_epistemic_pearson"], "ratio": m.get("reg_to_nll_ratio", abs(m.get("lambda_reg_x_l_reg", 0.0)) / (abs(m.get("nll", 0.0)) + 1e-8)),
    }

def gaps(rows):
    by = {r["regime"]: r for r in rows}
    idmax = max(by[REGIMES[0]]["epistemic"], by[REGIMES[1]]["epistemic"])
    return {
        "lambda": rows[0]["lambda"],
        "ale_gap_80_20": by[REGIMES[1]]["aleatoric"] - by[REGIMES[0]]["aleatoric"],
        "psi_gap_80_20": by[REGIMES[1]]["psi"] - by[REGIMES[0]]["psi"],
        "denom_gap_80_20": by[REGIMES[1]]["denom"] - by[REGIMES[0]]["denom"],
        "kappa_gap_120_id": by[REGIMES[2]]["kappa"] - idmax,
        "kappa_gap_1ms_id": by[REGIMES[3]]["kappa"] - idmax,
        "epi_gap_120_id": by[REGIMES[2]]["epistemic"] - idmax,
        "epi_gap_1ms_id": by[REGIMES[3]]["epistemic"] - idmax,
        "mean_id_nmse": (by[REGIMES[0]]["nmse_omitted"] + by[REGIMES[1]]["nmse_omitted"]) / 2,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir', required=True); ap.add_argument('--corrected', nargs=3, required=True); ap.add_argument('--prefix', nargs=3, required=True); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    corrected_specs=[('0',args.corrected[0]),('1e-3',args.corrected[1]),('1e-2',args.corrected[2])]
    allrows=[]; allgaps=[]
    gap_series = {}
    for lam,path in corrected_specs:
        d=load(path); rows=[row(lam,r,d['epoch_probe_results']['10'][r]) for r in REGIMES]; allrows.extend(rows); allgaps.append(gaps(rows))
        series=[]
        for epoch in range(1, 11):
            epoch_rows=[row(lam,r,d['epoch_probe_results'][str(epoch)][r]) for r in REGIMES]
            series.append({'epoch':epoch, **gaps(epoch_rows)})
        gap_series[lam]=series
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(3,1,figsize=(7,9),sharex=True)
        for ax,key,title in zip(axes,('ale_gap_80_20','epi_gap_120_id','epi_gap_1ms_id'),('Ale80-20','Epi120-ID','Epi1ms-ID')):
            ax.plot([x['epoch'] for x in series],[x[key] for x in series],marker='o'); ax.axhline(0,color='black',linewidth=.7); ax.set_ylabel(title); ax.grid(alpha=.25)
        axes[-1].set_xlabel('Epoch'); fig.tight_layout(); fig.savefig(out.parent/('lambda_'+lam if lam!='1e-3' else 'lambda_1e-3_reference')/'epoch_vs_gaps.png',dpi=140); plt.close(fig)
        error_dir = out.parent / ('lambda_'+lam if lam!='1e-3' else 'lambda_1e-3_reference')
        error_dir.mkdir(parents=True, exist_ok=True)
        with (error_dir/'error_bin_calibration.csv').open('w',newline='') as h:
            bins=[]
            for r in REGIMES:
                for b in d['epoch_probe_results']['10'][r]['error_bin_calibration']:
                    bins.append({'lambda':lam,'regime':r,**b})
            w=csv.DictWriter(h,fieldnames=list(bins[0])); w.writeheader(); w.writerows(bins)
    for name, rows in [('corrected_lambda_comparison.csv',allrows),('gap_summary.csv',allgaps)]:
        with (out/name).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    prefix_rows=[]
    for lam,path in zip(('0','1e-3','1e-2'),args.prefix):
        d=load(path); rs=[row(lam,r,d['epoch_probe_results']['10'][r]) for r in REGIMES]; g=gaps(rs)
        c=next(x for x in allgaps if x['lambda']==lam)
        prefix_rows.append({'lambda':lam,'prefix_ale_gap':g['ale_gap_80_20'],'corrected_ale_gap':c['ale_gap_80_20'],'prefix_epi120_gap':g['epi_gap_120_id'],'corrected_epi120_gap':c['epi_gap_120_id'],'prefix_epi1ms_gap':g['epi_gap_1ms_id'],'corrected_epi1ms_gap':c['epi_gap_1ms_id']})
    with (out/'prefix_vs_corrected.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(prefix_rows[0])); w.writeheader(); w.writerows(prefix_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(3,1,figsize=(7,9),sharex=True)
    for lam,series in gap_series.items():
        for ax,key in zip(axes,('ale_gap_80_20','epi_gap_120_id','epi_gap_1ms_id')):
            ax.plot([x['epoch'] for x in series],[x[key] for x in series],marker='o',label=lam)
    for ax,title in zip(axes,('Ale80-20','Epi120-ID','Epi1ms-ID')): ax.axhline(0,color='black',linewidth=.7); ax.set_ylabel(title); ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel('Epoch'); fig.tight_layout(); fig.savefig(out/'epoch_vs_gaps_comparison.png',dpi=140); plt.close(fig)
    (out/'summary.md').write_text('# STEP 8A Corrected Lambda Revalidation\n\n' + '\n'.join(f"- lambda={g['lambda']}: Ale80-20={g['ale_gap_80_20']:.8g}, Epi120-ID={g['epi_gap_120_id']:.8g}, Epi1ms-ID={g['epi_gap_1ms_id']:.8g}, mean ID NMSE={g['mean_id_nmse']:.4f} dB" for g in allgaps)+'\n')
    print(json.dumps({'corrected_gaps':allgaps,'prefix_vs_corrected':prefix_rows},indent=2))
if __name__=='__main__': main()
