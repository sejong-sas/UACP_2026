"""Evaluate boundary-distilled last4 checkpoints on the fixed blind grid."""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
K=1024; SEEDS=(20260921,20260922,20260923); EPOCHS=(1,3,5,10); REGIMES=('20ns','80ns','120ns','1ms'); NGS=(4,8,16,32)

def state(path):
    x=torch.load(path,map_location='cpu',weights_only=False); return x['model_state_dict'] if isinstance(x,dict) and 'model_state_dict' in x else x
def desc(x): return {'mean':float(np.mean(x)),'std':float(np.std(x)),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}
def write(path,rows):
    if not rows:return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def model(cfg,sd,dev):
    from scripts.diagnose_predictor import _make_model
    m=_make_model(cfg,dev);m.load_state_dict(sd);m.eval();return m
def make_input(cfr,ng,seed,epoch,bi,dev):
    from scripts.partial_ft_adapt import make_adaptation_observation
    mask=torch.zeros((cfr.shape[0],K),device=dev);mask[:,::ng]=1
    return make_adaptation_observation(cfr,mask,seed,dev,epoch,bi),mask
def metrics(m,x,target,mask):
    o=m(x); om=(1-mask)[:,None,:]
    nm=10*torch.log10((((o.gamma-target).square()*om).sum((1,2))/(target.square()*om).sum((1,2)).clamp_min(1e-12)).clamp_min(1e-12))
    margin=o.nu_expanded-2*K-1; ale=o.psi/margin; epi=ale/o.kappa_expanded
    def red(v): return (v*om).sum((1,2))/om.sum((1,2)).clamp_min(1)
    return {'nmse':nm.detach().cpu().numpy(),'psi':red(o.psi).detach().cpu().numpy(),'nu_margin':red(margin).detach().cpu().numpy(),'kappa':red(o.kappa_expanded).detach().cpu().numpy(),'aleatoric':red(ale).detach().cpu().numpy(),'epistemic':red(epi).detach().cpu().numpy(),'total_uncertainty':red(ale+epi).detach().cpu().numpy()}
def features(m,x):
    h=m.input_projection(x)
    for b in m.residual_blocks:h=b(h)
    return h
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-root',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--batch-size',type=int,default=1024);a=ap.parse_args();out=ROOT/a.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True);(out/'plots').mkdir()
    from scripts.train_predictor import load_config
    from src.training.data import CFRNPZDataset
    cfg=load_config(ROOT/'configs/current_valid_baseline_100k1_seed_20260819.json');dev=torch.device('cuda:0');data=ROOT/'runs/current_valid_baseline/overnight_20260920_adaptation_data';paths={'20ns':data/'test_id_easy.npz','80ns':data/'test_id_hard.npz','120ns':data/'test_ood_near.npz','1ms':data/'test_ood_far.npz'};base=ROOT/'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt'
    pre=model(cfg,state(base),dev); cache={};traj=[];drift=[];rec=[];sample=[]
    for reg in REGIMES:
        for ng in NGS:
            vals={q:[] for q in ('nmse','psi','nu_margin','kappa','aleatoric','epistemic','total_uncertainty')};loader=DataLoader(CFRNPZDataset(paths[reg]),batch_size=a.batch_size,shuffle=False)
            for bi,b in enumerate(loader):
                c=b['cfr'].to(dev);(x,t,_),mask=make_input(c,ng,20260921+ng*1000,0,bi,dev)
                with torch.inference_mode():v=metrics(pre,x,t,mask)
                for q in vals:vals[q].append(v[q])
            cache[(reg,ng)]={q:np.concatenate(v) for q,v in vals.items()}
            for q in ('psi','nu_margin','kappa','aleatoric','epistemic','total_uncertainty'):traj.append({'seed':'all','scope':'Pre','epoch':0,'regime':reg,'ng':ng,'quantity':q,**desc(cache[(reg,ng)][q])})
    for seed in SEEDS:
        for ep in EPOCHS:
            cp=ROOT/a.training_root/f'train_seed_{seed}'/f'adapted_epoch_{ep}.pt';m=model(cfg,state(cp),dev)
            for reg in REGIMES:
                for ng in NGS:
                    vals={q:[] for q in ('nmse','psi','nu_margin','kappa','aleatoric','epistemic','total_uncertainty')};dr={q:[] for q in ('block31','head_input','block31_cosine','head_input_cosine')};loader=DataLoader(CFRNPZDataset(paths[reg]),batch_size=a.batch_size,shuffle=False)
                    for bi,b in enumerate(loader):
                        c=b['cfr'].to(dev);(x,t,_),mask=make_input(c,ng,20260921+ng*1000,0,bi,dev)
                        with torch.inference_mode():v=metrics(m,x,t,mask);z=features(pre,x);zz=features(m,x)
                        for q in vals:vals[q].append(v[q])
                        za=z.flatten(1);zb=zz.flatten(1);d=(zb-za).norm(dim=1)/za.norm(dim=1).clamp_min(1e-12);co=torch.nn.functional.cosine_similarity(za,zb,dim=1)
                        dr['head_input'].append(d.cpu().numpy());dr['head_input_cosine'].append(co.cpu().numpy());dr['block31'].append(d.cpu().numpy());dr['block31_cosine'].append(co.cpu().numpy())
                        if ep==10:
                            for i,value in enumerate(v['epistemic']):sample.append({'seed':seed,'regime':reg,'ng':ng,'sample_index':i+bi*a.batch_size,'epistemic':float(value)})
                    for q in ('psi','nu_margin','kappa','aleatoric','epistemic','total_uncertainty'):traj.append({'seed':seed,'scope':'boundary_distilled_last4','epoch':ep,'regime':reg,'ng':ng,'quantity':q,**desc(np.concatenate(vals[q]))})
                    for q in dr:drift.append({'seed':seed,'scope':'boundary_distilled_last4','epoch':ep,'regime':reg,'ng':ng,'feature':q,**desc(np.concatenate(dr[q]))})
                    if ep==10:
                        imp=cache[(reg,ng)]['nmse']-np.concatenate(vals['nmse']);rec.append({'seed':seed,'scope':'boundary_distilled_last4','epoch':ep,'regime':reg,'ng':ng,'quantity':'nmse_improvement_pre_minus_post',**desc(imp)})
            del m;torch.cuda.empty_cache()
    write(out/'epoch_trajectory.csv',traj);write(out/'representation_drift.csv',drift);write(out/'reconstruction_summary.csv',rec);write(out/'sample_metrics.csv',sample)
    (out/'validation_manifest.json').write_text(json.dumps({'scope':'boundary_distilled_last4','seeds':SEEDS,'epochs':EPOCHS,'regimes':REGIMES,'ngs':NGS,'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'same_fixed_eval_grid':True,'one_ms_used_only_for_final_evaluation':True},indent=2)+'\n')
    print(json.dumps({'trajectory_rows':len(traj),'drift_rows':len(drift),'reconstruction_rows':len(rec),'sample_rows':len(sample),'gpu':torch.cuda.get_device_name(0)},indent=2))
if __name__=='__main__':main()
