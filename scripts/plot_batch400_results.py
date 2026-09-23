#!/usr/bin/env python3
"""Create local PNG/PDF figures and a compact cross-run comparison CSV."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/current_valid_baseline/batch400_10k5ep_20260917_retry"
def readj(p): return json.loads((ROOT/p).read_text())
def main():
 fig8=readj("runs/current_valid_baseline/batch400_10k5ep_20260917_retry/fig8_eval_v2/results.json"); fig9=readj("runs/current_valid_baseline/batch400_10k5ep_20260917_retry/fig9_eval_bounded_20/results.json"); lite=readj("runs/current_valid_baseline/batch400_10k5ep_20260917_retry/fig11_lite_v2/results.json")
 rows=[]
 for x in fig8["fig8_distribution"]: rows.append({"regime":x["regime"],"epi_linear_mean":x["mean"],"epi_db_mean_10log10_mean":10*np.log10(max(x["mean"],1e-12))})
 with (OUT/"fig8_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 fig,ax=plt.subplots(figsize=(7,4)); labels=[x["regime"] for x in rows]; vals=[x["epi_db_mean_10log10_mean"] for x in rows]; ax.plot(labels,vals,"o-",color="#1f77b4",label="batch 400"); ax.set_ylabel("Epistemic uncertainty (dB)"); ax.set_xlabel("Regime"); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"fig8_batch400_paper_style.png",dpi=300); fig.savefig(OUT/"fig8_batch400_paper_style.pdf"); plt.close(fig)
 cal=fig9["fig9_calibration"]; pools=["ID","OOD"]; fig,ax=plt.subplots(figsize=(5.5,4)); ax.plot([0,1],[0,1],"k--",label="Ideal")
 for pool,color,style in [("ID","#1f77b4","-^"),("OOD","#ff7f0e","-o")]:
  z=[x for x in cal if x["pool"]==pool]; ax.plot([x["nominal"] for x in z],[x["empirical"] for x in z],style,color=color,label=f"{pool} (MAE={fig9['fig9_calibration_error_mae'][pool]:.4f})")
 ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage"); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"fig9_batch400_bounded_exact.png",dpi=300); fig.savefig(OUT/"fig9_batch400_bounded_exact.pdf"); plt.close(fig)
 with (OUT/"comparison_summary.csv").open("w",newline="") as f:
  fields=["run","train_condition","update_mode","nmse_20_80_120_1ms","fig8_auroc_pooled","fig8_auroc_near","fig8_auroc_far","fig9_id_mae","fig9_ood_mae","fig11_ng_nmse","nu_boundary_inf","runtime"]
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([
   {"run":"50k×1","train_condition":"50,000 unique CFR × 1 epoch","update_mode":"direct batch 8; 6,250 updates","nmse_20_80_120_1ms":"not in matched summary","fig8_auroc_pooled":0.79330779125,"fig8_auroc_near":0.6035666275,"fig8_auroc_far":0.983048955,"fig9_id_mae":0.0385883232,"fig9_ood_mae":0.2323865111,"fig11_ng_nmse":"not rerun","nu_boundary_inf":"not reported","runtime":"training 1,313 s"},
   {"run":"100k×1","train_condition":"100,000 unique CFR × 1 epoch","update_mode":"direct batch 8; 12,500 updates","nmse_20_80_120_1ms":"see canonical training results","fig8_auroc_pooled":0.79160282,"fig8_auroc_near":0.594332835,"fig8_auroc_far":0.988872805,"fig9_id_mae":0.0533595153,"fig9_ood_mae":0.2171871115,"fig11_ng_nmse":"existing canonical only","nu_boundary_inf":"not reported","runtime":"training 2,660 s"},
   {"run":"10k×5 batch8","train_condition":"10,000 unique CFR × 5 epochs","update_mode":"direct batch 8; 6,250 updates","nmse_20_80_120_1ms":"existing run summary","fig8_auroc_pooled":"available in gaps artifact only","fig8_auroc_near":"available in gaps artifact only","fig8_auroc_far":"available in gaps artifact only","fig9_id_mae":"not available","fig9_ood_mae":"not available","fig11_ng_nmse":"near gap +1.37e-4; far gap +4.91e-3","nu_boundary_inf":"not reported","runtime":"training 1,469 s"},
   {"run":"10k×5 batch400","train_condition":"10,000 unique CFR × 5 epochs","update_mode":"direct batch 400; 125 updates","nmse_20_80_120_1ms":"all-NMSE 3.016/3.297/3.236/3.320 dB; validation omitted -14.081 dB","fig8_auroc_pooled":fig8["fig8_auroc"]["overall_id_vs_ood"],"fig8_auroc_near":fig8["fig8_auroc"]["id_vs_near"],"fig8_auroc_far":fig8["fig8_auroc"]["id_vs_far"],"fig9_id_mae":fig9["fig9_calibration_error_mae"]["ID"],"fig9_ood_mae":fig9["fig9_calibration_error_mae"]["OOD"],"fig11_ng_nmse":"mean Ng 4.175→1; all-NMSE 3.016/3.297/3.236/3.338/3.223/3.320","nu_boundary_inf":"not observed in saved outputs; full-feedback omitted NMSE NaN","runtime":"training 2,166 s; Fig.8 ~15 min; Fig.11 52 s; Fig.9 bounded"},
  ])
 (OUT/"comparison_summary.json").write_text(json.dumps({"fig8":fig8,"fig9":fig9,"fig11_lite":lite},indent=2)+"\n")
if __name__=="__main__": main()
