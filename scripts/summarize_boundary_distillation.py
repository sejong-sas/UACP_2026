"""Assemble boundary-distillation results with read-only prior baselines."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922';EVAL=OUT/'evaluation_retry1';OLD=ROOT/'runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5';SEEDS=['20260921','20260922','20260923']
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 fs=[]
 for r in rows:
  for k in r:
   if k not in fs:fs.append(k)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows(rows)
def fl(x):return float(x)
def ms(v):
 a=np.asarray(v,float);return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else 0.0
def auc(pos,neg):
 s=np.r_[neg,pos];lab=np.r_[np.zeros(len(neg)),np.ones(len(pos))];order=np.argsort(s,kind='mergesort');r=np.empty(len(s));o=s[order];i=0
 while i<len(s):
  j=i+1
  while j<len(s) and o[j]==o[i]:j+=1
  r[order[i:j]]=(i+1+j)/2;i=j
 pr=r[lab==1];return float((pr.sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def main():
 oldtr=read(OLD/'epoch_trajectory.csv');oldrec=read(OLD/'reconstruction_summary.csv');oldeff=read(OLD/'efficiency_summary.csv');pre_nmse={(r['ng'],r['regime']):fl(r['nmse_omitted_db']) for r in read(ROOT/'runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/primary_performance_epoch3.csv') if r['model']=='pre'}
 antr=read(EVAL/'epoch_trajectory.csv');anrec=read(EVAL/'reconstruction_summary.csv');andrift=read(EVAL/'representation_drift.csv');samples=read(EVAL/'sample_metrics.csv')
 def old(scope,seed,ep,reg,ng,q):return next(r for r in oldtr if r['scope']==scope and r['seed']==str(seed) and int(r['epoch'])==ep and r['regime']==reg and int(r['ng'])==ng and r['quantity']==q)
 def pre(reg,ng,q):return next(r for r in oldtr if r['scope']=='Pre' and r['regime']==reg and int(r['ng'])==ng and r['quantity']==q)
 def an(seed,ep,reg,ng,q):return next(r for r in antr if r['seed']==str(seed) and int(r['epoch'])==ep and r['regime']==reg and int(r['ng'])==ng and r['quantity']==q)
 # Reconstruction and forgetting per seed.
 recon=[]
 for seed in SEEDS:
  for ng in (16,32):
   full=fl(next(r for r in oldrec if r['scope']=='full' and r['seed']==seed and r['regime']=='120ns' and int(r['ng'])==ng)['mean'])
   for reg in ('20ns','80ns','120ns'):
    imp=fl(next(r for r in anrec if r['seed']==seed and r['regime']==reg and int(r['ng'])==ng)['mean'])
    recon.append({'seed':seed,'scope':'boundary_distilled_last4','ng':ng,'regime':reg,'nmse_improvement_pre_minus_post':imp,'full_retention_pct':100*imp/full if reg=='120ns' else ''})
 write(OUT/'reconstruction_summary.csv',recon)
 # Comparative epoch-10 post-NMSE and uncertainty statistics.
 labels={'last_block_plus_head':'3.16%','last_4_blocks_plus_head':'last4','last_8_blocks_plus_head':'25%','full':'Full'};comp=[]
 for label,scope in [('3.16%','last_block_plus_head'),('last4','last_4_blocks_plus_head'),('25%','last_8_blocks_plus_head'),('Full','full'),('boundary_distilled_last4','boundary_distilled_last4')]:
  for ng in (16,32):
   for reg in ('20ns','80ns','120ns','1ms'):
    for q in ('nmse','aleatoric','epistemic','total_uncertainty'):
     vals=[]
     if label=='boundary_distilled_last4':
      if q=='nmse':vals=[pre_nmse[str(ng),reg]-fl(next(r for r in anrec if r['seed']==s and r['regime']==reg and int(r['ng'])==ng)['mean']) for s in SEEDS]
      else:vals=[fl(an(s,10,reg,ng,q)['median']) for s in SEEDS]
     elif q=='nmse':vals=[pre_nmse[str(ng),reg]-fl(next(r for r in oldrec if r['scope']==scope and r['seed']==s and r['regime']==reg and int(r['ng'])==ng)['mean']) for s in SEEDS]
     else:vals=[fl(old(scope,s,10,reg,ng,q)['median']) for s in SEEDS]
     m,sd=ms(vals);comp.append({'model':label,'ng':ng,'regime':reg,'quantity':q,'mean':m,'std':sd})
 write(OUT/'comparative_epoch10_summary.csv',comp)
 write(OUT/'uncertainty_summary.csv',[r for r in comp if r['quantity'] in ('aleatoric','epistemic','total_uncertainty')])
 # Mechanism ratios and drift.
 mech=[]
 for ep in (1,3,5,10):
  for ng in (16,32):
   for q in ('psi','nu_margin','kappa','aleatoric','epistemic'):
    vals=[fl(an(s,ep,'1ms',ng,q)['median'])/fl(pre('1ms',ng,q)['median']) for s in SEEDS];m,sd=ms(vals);mech.append({'scope':'boundary_distilled_last4','epoch':ep,'regime':'1ms','ng':ng,'quantity':q,'pre_ratio_mean':m,'pre_ratio_std':sd})
 write(OUT/'evidential_trajectory.csv',mech);write(OUT/'representation_drift.csv',andrift)
 # AUROC, including fixed pooled ID versus near/far.
 aucrows=[]
 for s in SEEDS:
  for ng in (16,32):
   by={reg:np.asarray([fl(r['epistemic']) for r in samples if r['seed']==s and int(r['ng'])==ng and r['regime']==reg]) for reg in ('20ns','80ns','120ns','1ms')};ids=np.r_[by['20ns'],by['80ns']]
   for reg in ('120ns','1ms'):aucrows.append({'model':'boundary_distilled_last4','seed':s,'ng':ng,'comparison':f'ID(20+80) vs {reg}','id_mean':float(ids.mean()),'ood_mean':float(by[reg].mean()),'id_median':float(np.median(ids)),'ood_median':float(np.median(by[reg])),'auroc':auc(by[reg],ids)})
 write(OUT/'auroc_summary.csv',aucrows)
 # Efficiency.
 eff=[]
 for s in SEEDS:
  d=json.loads((OUT/f'train_seed_{s}/manifest.json').read_text())['training'];eff.append({'scope':'boundary_distilled_last4','seed':s,'trainable_params':d['trainable_params'],'trainable_percent':100*d['trainable_params']/11815320,'training_seconds':d['training_seconds'],'seconds_per_step':d['seconds_per_step'],'peak_vram_mib':d['peak_vram_mib']})
 write(OUT/'efficiency_summary.csv',eff)
 # Compact seed summary and gate manifest.
 summary=[]
 for ng in (16,32):
  imp=[fl(next(r for r in recon if r['seed']==s and int(r['ng'])==ng and r['regime']=='120ns')['nmse_improvement_pre_minus_post']) for s in SEEDS];ret=[fl(next(r for r in recon if r['seed']==s and int(r['ng'])==ng and r['regime']=='120ns')['full_retention_pct']) for s in SEEDS]
  epi=[fl(next(r for r in comp if r['model']=='boundary_distilled_last4' and int(r['ng'])==ng and r['regime']=='1ms' and r['quantity']=='epistemic')['mean'])]
  ratio=epi[0]/fl(pre('1ms',ng,'epistemic')['median']);oldratio=np.mean([fl(old('last_4_blocks_plus_head',s,10,'1ms',ng,'epistemic')['median'])/fl(pre('1ms',ng,'epistemic')['median']) for s in SEEDS]);gate=.17 if ng==16 else .18
  summary.append({'ng':ng,'nmse_improvement_mean':np.mean(imp),'nmse_improvement_std':np.std(imp,ddof=1),'full_retention_mean_pct':np.mean(ret),'epi_1ms_median_pre_ratio':ratio,'unreg_last4_epi_ratio':oldratio,'preservation_gate':gate,'preservation_pass':ratio>=gate})
 write(OUT/'seed_summary.csv',summary)
 write(OUT/'training_summary.csv',recon)
 (OUT/'integrity_summary.json').write_text(json.dumps({'pre_checkpoint_sha256':'d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986','scope':'last_4_blocks_plus_head','trainable_params':1480728,'one_ms_used_only_for_final_evaluation':True,'proxy_reconstruction_target_loss':False,'same_fixed_evaluation_grid':True,'gpu':'NVIDIA GB10','device':'cuda:0'},indent=2)+'\n')
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  curve=read(OUT/'boundary_curve_retry2/boundary_curve.csv');pdir=OUT/'plots';pdir.mkdir(exist_ok=True)
  for ng in (16,32):
   fig,ax=plt.subplots(figsize=(7,4))
   for s in SEEDS:
    rr=[r for r in curve if r['seed']==s and int(r['ng'])==ng];ax.plot([float(r['delay_ns']) for r in rr],[float(r['epistemic_median']) if r['epistemic_median']!='nan' else np.nan for r in rr],marker='o',label=str(s))
   ax.set_xscale('symlog',linthresh=20);ax.set_yscale('log');ax.axvspan(10,100,color='green',alpha=.08);ax.axvline(120,color='orange',ls='--');ax.axvspan(160,500,color='red',alpha=.06);ax.axvline(1000,color='black',ls=':');ax.set_xlabel('delay spread (ns)');ax.set_ylabel('Epistemic median');ax.set_title(f'Boundary-distilled confidence curve Ng={ng}');ax.legend();fig.tight_layout();fig.savefig(pdir/f'boundary_curve_ng{ng}.png',dpi=150);plt.close(fig)
 except Exception as exc:(OUT/'plot_error.txt').write_text(str(exc)+'\n')
 print(json.dumps({'output':str(OUT.relative_to(ROOT)),'seed_summary':summary,'efficiency':eff},indent=2))
if __name__=='__main__':main()
