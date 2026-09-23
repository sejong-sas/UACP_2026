"""Create the delay-spread Epistemic confidence-boundary curve."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));K=1024;SEEDS=(20260921,20260922,20260923);NGS=(16,32);DELAYS=(20,40,60,80,100,120,160,250,500,1000)
def state(p):
 x=torch.load(p,map_location='cpu',weights_only=False);return x['model_state_dict'] if isinstance(x,dict) and 'model_state_dict' in x else x
def write(p,r):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--training-root',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--samples-per-delay',type=int,default=200);ap.add_argument('--batch-size',type=int,default=256);a=ap.parse_args();out=ROOT/a.output_dir
 if out.exists() and any(out.iterdir()):raise FileExistsError(out)
 out.mkdir(parents=True)
 from scripts.train_predictor import load_config
 from scripts.diagnose_predictor import _make_model
 from scripts.partial_ft_adapt import make_adaptation_observation
 from scripts.generate_dataset import load_dataset_config
 from src.channel.sionna_channel import generate_cfr_for_delay_spreads
 from src.training.data import CFRNPZDataset
 cfg=load_config(ROOT/'configs/current_valid_baseline_100k1_seed_20260819.json');gen_cfg=load_dataset_config(ROOT/'configs/adaptation_120ns_protocol_20260921.json');dev=torch.device('cuda:0');data=ROOT/'runs/current_valid_baseline/overnight_20260920_adaptation_data';
 rows=[]
 for delay in DELAYS:
  if delay==1000:
   cfr=np.load(data/'test_ood_far.npz',allow_pickle=True)['cfr']; provenance='existing_blind_1ms_test'
  else:
   ds=np.full(a.samples_per_delay,float(delay),dtype=np.float32);cfr=generate_cfr_for_delay_spreads(gen_cfg,ds,seed=20260922+int(delay));provenance='new_curve_only_eval'
  ct=torch.from_numpy(cfr)
  for seed in SEEDS:
   cp=ROOT/a.training_root/f'train_seed_{seed}'/'adapted_epoch_10.pt';m=_make_model(cfg,dev);m.load_state_dict(state(cp));m.eval()
   for ng in NGS:
    scores=[]
    loader=DataLoader(torch.utils.data.TensorDataset(ct),batch_size=a.batch_size,shuffle=False)
    for bi,(b,) in enumerate(loader):
     c=b.to(dev);mask=torch.zeros((c.shape[0],K),device=dev);mask[:,::ng]=1;x,_,_=make_adaptation_observation(c,mask,20260921+ng*1000,dev,0,bi)
     with torch.inference_mode():o=m(x);margin=o.nu_expanded-2*K-1;epi=(o.psi/margin/o.kappa_expanded).mean(dim=(1,2));scores.append(epi.cpu().numpy())
    s=np.concatenate(scores); rows.append({'seed':seed,'delay_ns':delay,'ng':ng,'n_samples':len(s),'provenance':provenance, 'epistemic_mean':float(s.mean()),'epistemic_median':float(np.median(s)),'epistemic_p10':float(np.quantile(s,.1)),'epistemic_p90':float(np.quantile(s,.9))})
   del m;torch.cuda.empty_cache()
 write(out/'boundary_curve.csv',rows);(out/'manifest.json').write_text(json.dumps({'delays_ns':DELAYS,'samples_per_new_delay':a.samples_per_delay,'one_ms_source':'untouched existing test_ood_far.npz','one_ms_used_for_model_selection':False,'gpu':torch.cuda.get_device_name(0),'device':'cuda:0'},indent=2)+'\n');print(json.dumps({'rows':len(rows),'gpu':torch.cuda.get_device_name(0)},indent=2))
if __name__=='__main__':main()
