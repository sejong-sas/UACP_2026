"""Create a provenance-preserving deterministic ID anchor subset."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output',required=True); p.add_argument('--count',type=int,default=1000); p.add_argument('--seed',type=int,default=20260922); a=p.parse_args()
 out=ROOT/a.output
 if out.exists(): raise FileExistsError(out)
 z=np.load(ROOT/a.source,allow_pickle=True); n=len(z['cfr']); rng=np.random.default_rng(a.seed); ix=np.sort(rng.choice(n,size=a.count,replace=False))
 out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,cfr=z['cfr'][ix],delay_spread_ns=z['delay_spread_ns'][ix],regime_label=z['regime_label'][ix],metadata_json=np.array(json.dumps({'source':a.source,'source_sha256':hashlib.sha256((ROOT/a.source).read_bytes()).hexdigest(),'subset_count':a.count,'subset_seed':a.seed,'role':'ID-reference anchor only','leakage_policy':'no adaptation/evaluation/OOD samples'})))
 print(json.dumps({'output':str(out.relative_to(ROOT)),'source':a.source,'source_count':n,'count':a.count,'seed':a.seed,'delay_min':float(z['delay_spread_ns'][ix].min()),'delay_max':float(z['delay_spread_ns'][ix].max())},indent=2))
if __name__=='__main__': main()
