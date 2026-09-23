"""Post-process the fixed-grid final Partial FT validation outputs."""
from pathlib import Path
import csv, json, math
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ["last_block_plus_head", "last_4_blocks_plus_head", "last_8_blocks_plus_head", "full"]
LABELS = {"last_block_plus_head":"3.16%", "last_4_blocks_plus_head":"12.53%", "last_8_blocks_plus_head":"25.03%", "full":"Full"}
SEEDS = ["20260921", "20260922", "20260923"]
NGS = [4,8,16,32]
REGIMES = ["20ns","80ns","120ns","1ms"]

def read_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def write_csv(path, rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})
def fl(v): return float(v) if v not in (None, "") else float("nan")
def meanstd(vals):
    x=np.asarray([v for v in vals if np.isfinite(v)],float)
    return (float(np.mean(x)), float(np.std(x,ddof=1)) if len(x)>1 else 0.0)
def fmt(m,s): return f"{m:.6g} ± {s:.3g}"
def boot_delta(a,b,reps=2000,seed=1):
    a=np.asarray(a); b=np.asarray(b); rng=np.random.default_rng(seed); n=len(a)
    x=np.empty(reps)
    for i in range(reps):
        ix=rng.integers(0,n,n); x[i]=np.mean((a-b)[ix])
    return float(np.quantile(x,.025)),float(np.quantile(x,.975))

def main():
    ap=__import__('argparse').ArgumentParser(); ap.add_argument('--input-dir',required=True); args=ap.parse_args()
    d=ROOT/args.input_dir; traj=read_csv(d/'epoch_trajectory.csv'); unc=read_csv(d/'uncertainty_summary.csv'); eff=read_csv(d/'efficiency_summary.csv')
    rec=read_csv(d/'reconstruction_summary.csv'); auc=read_csv(d/'auroc_summary.csv')
    # Correct AUROC table to one row per seed/scope/ng/comparison.
    au={}
    for r in auc:
        if r.get('auroc','')!='': au[(r['seed'],r['scope'],r['ng'],r['comparison'])]=r
    write_csv(d/'auroc_summary_deduplicated.csv',list(au.values()))
    # Primary reconstruction/forgetting table, seed-level and aggregated.
    rmap={(r['seed'],r['scope'],int(r['ng']),r['regime']):r for r in rec}
    prim=[]
    for scope in SCOPES:
      for ng in NGS:
       full=[]
       for seed in SEEDS:
        row=rmap.get((seed,scope,ng,'120ns')); frow=rmap.get((seed,'full',ng,'120ns'))
        if row and frow:
         prim.append({'seed':seed,'scope':scope,'model':LABELS[scope],'ng':ng,'regime':'120ns','nmse_improvement':row['mean'],'full_retention_pct':100*fl(row['mean'])/fl(frow['mean']) if fl(frow['mean']) else ''})
    write_csv(d/'training_summary.csv',prim)
    agg=[]
    for scope in SCOPES:
      for ng in NGS:
       for metric,regime in [('120_nmse_improvement','120ns'),('20_ns_forgetting','20ns'),('80_ns_forgetting','80ns')]:
        vals=[]
        for seed in SEEDS:
         row=rmap.get((seed,scope,ng,regime)); vals.append(fl(row['mean']) if row else np.nan)
        m,s=meanstd(vals); agg.append({'scope':scope,'model':LABELS[scope],'ng':ng,'metric':metric,'mean':m,'std':s,'mean_pm_std':fmt(m,s)})
    write_csv(d/'final_seed_summary.csv',agg)
    # uncertainty primary and evidential mechanism ratios.
    umap={(r['seed'],r['scope'],int(r['ng']),r['regime'],r['quantity']):r for r in traj if r['epoch']=='10'}
    mech=[]
    for scope in SCOPES:
      for ng in NGS:
       for regime in REGIMES:
        for q in ['psi','nu_margin','kappa','aleatoric','epistemic','total_uncertainty']:
         vals=[]; ratios=[]
         for seed in SEEDS:
          r=umap.get((seed,scope,ng,regime,q)); p=next((x for x in traj if x['seed']=='all' and x['scope']=='Pre' and x['epoch']=='0' and int(x['ng'])==ng and x['regime']==regime and x['quantity']==q),None)
          if r: vals.append(fl(r['median']))
          if r and p: ratios.append(fl(r['median'])/fl(p['median']))
         m,s=meanstd(vals); rm,rs=meanstd(ratios); mech.append({'scope':scope,'model':LABELS[scope],'ng':ng,'regime':regime,'quantity':q,'median_mean':m,'median_std':s,'median_ratio_mean':rm,'median_ratio_std':rs})
    write_csv(d/'evidential_parameter_trajectory.csv',mech)
    # aggregate efficiency.
    er=[]
    for scope in SCOPES:
      rows=[r for r in eff if r['scope']==scope]
      out={'scope':scope,'model':LABELS[scope],'trainable_params':rows[0]['trainable_params'],'trainable_percent':rows[0]['trainable_percent']}
      for key in ['training_seconds','seconds_per_step','peak_vram_mib']:
        m,s=meanstd([fl(r[key]) for r in rows]); out[key+'_mean']=m; out[key+'_std']=s; out[key+'_mean_pm_std']=fmt(m,s)
      er.append(out)
    write_csv(d/'efficiency_summary_aggregated.csv',er)
    # Basic plots, using the raw trajectory medians.
    try:
      import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
      pdir=d/'plots'; pdir.mkdir(exist_ok=True)
      for ng in [16,32]:
       fig,ax=plt.subplots(figsize=(7,4))
       for scope in SCOPES:
        xs=[];ys=[]
        for ep in [0,1,3,5,10]:
         vals=[]
         for reg in ['1ms']:
          qrows=[r for r in traj if r['scope']==scope and r['epoch']==str(ep) and r['regime']==reg and int(r['ng'])==ng and r['quantity']=='epistemic'] if ep else [r for r in traj if r['scope']=='Pre' and r['epoch']=='0' and r['regime']==reg and int(r['ng'])==ng and r['quantity']=='epistemic']
          vals += [fl(r['median']) for r in qrows if r['seed'] in SEEDS or r['seed']=='all']
         if vals: xs.append(ep); ys.append(np.mean(vals))
        if ys: ax.plot(xs,ys,marker='o',label=LABELS[scope])
       ax.set(xlabel='epoch',ylabel='1 ms Epistemic median',title=f'Far-OOD Epistemic trajectory Ng={ng}'); ax.legend(); fig.tight_layout(); fig.savefig(pdir/f'epistemic_1ms_trajectory_ng{ng}.png',dpi=150); plt.close(fig)
    except Exception as e: (d/'plot_error.txt').write_text(str(e)+'\n')
    (d/'postprocess_manifest.json').write_text(json.dumps({'primary_endpoint':'epoch10','scopes':SCOPES,'seeds':SEEDS,'ngs':NGS,'regimes':REGIMES,'implementation_assumption':'All Partial FT scope/schedule/dataset counts are research extensions; omitted-NMSE is primary metric.'},indent=2)+'\n')
    print(json.dumps({'output':str(d.relative_to(ROOT)),'primary_rows':len(prim),'summary_rows':len(agg),'mechanism_rows':len(mech),'efficiency_rows':len(er)},indent=2))
if __name__=='__main__': main()
