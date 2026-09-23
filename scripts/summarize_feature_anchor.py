"""Summarize the fixed feature-anchor experiment against prior final-validation artifacts."""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922"
ANCH = OUT / "evaluation_retry1"
OLD = ROOT / "runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5"
AU = OUT / "auroc_retry1/auroc.csv"
SEEDS = ["20260921", "20260922", "20260923"]
SCOPES = {"3.16%":"last_block_plus_head", "last4":"last_4_blocks_plus_head", "25%":"last_8_blocks_plus_head", "Full":"full", "Anchored-last4":"anchored_last4"}

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def write(path, rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def f(x): return float(x)
def ms(x):
    a=np.asarray(x,float); return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.0

def main():
    old_tr=read(OLD/'epoch_trajectory.csv'); old_rec=read(OLD/'reconstruction_summary.csv'); old_eff=read(OLD/'efficiency_summary.csv')
    pre_nmse_rows=read(ROOT/'runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/primary_performance_epoch3.csv')
    pre_nmse={(r['ng'],r['regime']):float(r['nmse_omitted_db']) for r in pre_nmse_rows if r['model']=='pre'}
    an_tr=read(ANCH/'epoch_trajectory.csv'); an_rec=read(ANCH/'reconstruction_summary.csv'); an_dr=read(ANCH/'representation_drift.csv'); auc=read(AU)
    def old_row(scope, seed, ep, reg, ng, q):
        return next(r for r in old_tr if r['scope']==scope and r['seed']==str(seed) and int(r['epoch'])==ep and r['regime']==reg and int(r['ng'])==ng and r['quantity']==q)
    def pre(reg,ng,q):
        return next(r for r in old_tr if r['scope']=='Pre' and r['regime']==reg and int(r['ng'])==ng and r['quantity']==q)
    def an_row(seed,ep,reg,ng,q):
        return next(r for r in an_tr if r['seed']==str(seed) and int(r['epoch'])==ep and r['regime']==reg and int(r['ng'])==ng and r['quantity']==q)
    # Per-seed primary and forgetting metrics.
    rows=[]
    for seed in SEEDS:
        for ng in (16,32):
            full=float(next(r for r in old_rec if r['scope']=='full' and r['seed']==seed and r['regime']=='120ns' and int(r['ng'])==ng)['mean'])
            ar=float(next(r for r in an_rec if r['seed']==seed and r['regime']=='120ns' and int(r['ng'])==ng)['mean'])
            for reg in ('20ns','80ns'):
                improvement=float(next(r for r in an_rec if r['seed']==seed and r['regime']==reg and int(r['ng'])==ng)['mean'])
                post=pre_nmse[(str(ng),reg)]-improvement
                rows.append({'seed':seed,'ng':ng,'metric':f'{reg}_nmse_forgetting','value':post-pre_nmse[(str(ng),reg)],'scope':'Anchored-last4'})
            rows.append({'seed':seed,'ng':ng,'metric':'120_nmse_improvement','value':ar,'scope':'Anchored-last4'})
            rows.append({'seed':seed,'ng':ng,'metric':'120_full_retention_pct','value':100*ar/full,'scope':'Anchored-last4'})
    write(OUT/'reconstruction_summary.csv',rows)
    # Full comparative epoch-10 summary. Existing old artifacts are reused by protocol.
    comp=[]
    for label,scope in SCOPES.items():
        for ng in (16,32):
            for reg in ('20ns','80ns','120ns','1ms'):
                for q in ('nmse','aleatoric','epistemic','total_uncertainty'):
                    vals=[]
                    if label=='Anchored-last4':
                        if q=='nmse':
                            vals=[pre_nmse[(str(ng),reg)]-float(next(r for r in an_rec if r['seed']==s and r['regime']==reg and int(r['ng'])==ng)['mean']) for s in SEEDS]
                        else:
                            vals=[f(an_row(s,10,reg,ng,q)['median']) for s in SEEDS]
                    elif q=='nmse':
                        # The old reconstruction table stores pre-minus-post improvement.
                        vals=[pre_nmse[(str(ng),reg)]-f(next(r for r in old_rec if r['scope']==scope and r['seed']==s and r['regime']==reg and int(r['ng'])==ng)['mean']) for s in SEEDS]
                    else:
                        vals=[f(old_row(scope,s,10,reg,ng,q)['median']) for s in SEEDS]
                    m,s=ms(vals); comp.append({'model':label,'ng':ng,'regime':reg,'quantity':q,'mean':m,'std':s})
    write(OUT/'comparative_epoch10_summary.csv',comp)
    # Mechanism ratios and trajectory for anchored, plus compact seed means.
    mech=[]
    for ep in (1,3,5,10):
        for ng in (16,32):
            for q in ('psi','nu_margin','kappa','aleatoric','epistemic'):
                vals=[]
                for s in SEEDS:
                    vals.append(f(an_row(s,ep,'1ms',ng,q)['median']) / f(pre('1ms',ng,q)['median']))
                m,sd=ms(vals); mech.append({'scope':'Anchored-last4','epoch':ep,'regime':'1ms','ng':ng,'quantity':q,'pre_ratio_mean':m,'pre_ratio_std':sd})
    write(OUT/'evidential_trajectory.csv',mech)
    # Anchor representation drift summary, preserved at seed level and aggregated.
    write(OUT/'representation_drift.csv',an_dr)
    # Efficiency from logs/manifests.
    eff=[]
    for s in SEEDS:
        txt=(OUT/f'train_seed_{s}.training.log').read_text()
        d=json.loads(txt)['training']
        eff.append({'scope':'Anchored-last4','seed':s,'trainable_params':d['trainable_params'],'trainable_percent':100*d['trainable_params']/11815320,'training_seconds':d['training_seconds'],'seconds_per_step':d['seconds_per_step'],'peak_vram_mib':d['peak_vram_mib']})
    write(OUT/'efficiency_summary.csv',eff)
    # AUROC summary, with seed means/std.
    auc_sum=[]
    for ng in (16,32):
        for comparison in ('ID(20+80) vs 120ns','ID(20+80) vs 1ms'):
            vals=[f(r['auroc']) for r in auc if int(r['ng'])==ng and r['comparison']==comparison]
            m,sd=ms(vals); auc_sum.append({'model':'Anchored-last4','ng':ng,'comparison':comparison,'auroc_mean':m,'auroc_std':sd})
    write(OUT/'auroc_summary.csv',auc_sum)
    # Summary of the pre-specified gates.
    gates=[]
    old_eff_map={r['scope']:r for r in old_eff}
    for ng,gate in ((16,.17),(32,.18)):
        ar=[f(r['mean']) for r in comp if r['model']=='Anchored-last4' and r['ng']==ng and r['regime']=='1ms' and r['quantity']=='epistemic'][0]
        pr=f(pre('1ms',ng,'epistemic')['median'])
        old=[f(old_row('last_4_blocks_plus_head',s,10,'1ms',ng,'epistemic')['median'])/pr for s in SEEDS]
        anchored_ratio=ar/pr
        gates.append({'ng':ng,'anchored_1ms_epi_median_pre_ratio':anchored_ratio,'unregularized_last4_ratio_mean':float(np.mean(old)),'required_min_ratio':gate,'preservation_gate_pass':anchored_ratio>=gate,'full_retention_ge90':float(np.mean([f(r['value']) for r in rows if r['ng']==ng and r['metric']=='120_full_retention_pct']))>=90})
    (OUT/'gate_summary.json').write_text(json.dumps({'gates':gates,'interpretation':'Feature anchoring was fixed at lambda=1.0; no tuning or model selection using 1 ms data.','gpu':'NVIDIA GB10 / cuda:0'},indent=2)+'\n')
    # Plots.
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        pdir=OUT/'plots'; pdir.mkdir(exist_ok=True)
        for ng in (16,32):
            fig,ax=plt.subplots(figsize=(7,4))
            for q in ('epistemic','kappa'):
                ys=[]
                for ep in (1,3,5,10):
                    vals=[f(an_row(s,ep,'1ms',ng,q)['median']) for s in SEEDS]; ys.append(np.mean(vals))
                ax.plot([1,3,5,10],ys,marker='o',label=q)
            ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('median (log scale)'); ax.set_title(f'Anchored last4 1 ms trajectory Ng={ng}'); ax.legend(); fig.tight_layout(); fig.savefig(pdir/f'anchored_1ms_mechanism_ng{ng}.png',dpi=150); plt.close(fig)
        for ng in (16,32):
            fig,ax=plt.subplots(figsize=(7,4))
            for reg in ('20ns','120ns','1ms'):
                ys=[]
                for ep in (1,3,5,10): ys.append(np.mean([f(next(r for r in an_dr if r['seed']==s and int(r['epoch'])==ep and r['regime']==reg and int(r['ng'])==ng and r['feature']=='head_input')['median']) for s in SEEDS]))
                ax.plot([1,3,5,10],ys,marker='o',label=reg)
            ax.set_xlabel('epoch'); ax.set_ylabel('normalized head-input drift'); ax.set_title(f'Anchored last4 representation drift Ng={ng}'); ax.legend(); fig.tight_layout(); fig.savefig(pdir/f'anchored_drift_ng{ng}.png',dpi=150); plt.close(fig)
    except Exception as e: (OUT/'plot_error.txt').write_text(str(e)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'gates':gates,'efficiency':eff,'auroc':auc_sum},indent=2))

if __name__=='__main__': main()
