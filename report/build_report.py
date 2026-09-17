from pathlib import Path
import csv, json, math, subprocess, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
FIG = OUT / 'figures'
SRC = OUT / 'source_data'
FIG.mkdir(parents=True, exist_ok=True); SRC.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':11,'axes.labelsize':10,'figure.dpi':160})

cmp_path = ROOT/'runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/comparison_100k1_100k5batch8_100k5batch400.csv'
cmp = pd.read_csv(cmp_path)
models = ['100k×1 batch8 FP32','100k×5 batch8 FP32','100k×5 batch400 BF16']
cmp['model'] = models
# The preserved batch8×5 artifact contains a non-finite Far-OOD ν/inf fraction.
# Do not present ranking-derived AUROC/CE values as valid finite metrics.
cmp.loc[cmp['model']=='100k×5 batch8 FP32', ['auc_id_far','auc_pooled','fig9_far_mae','fig9_ood_mae']] = np.nan
cmp.to_csv(SRC/'summary_tables.csv', index=False)

bench = []
def add_b(name, batch, precision, status, path, sec=None, mem=None, util=None):
    d={'setting':name,'batch':batch,'precision':precision,'status':status,'source':str(path)}
    if status=='success':
        j=json.load(open(path)); d.update(sec=float(j['mean_sec_per_step']), samples_sec=batch/float(j['mean_sec_per_step']), mem_gib=float(j.get('peak_allocated_mib',j.get('peak_gpu_allocated_mib')))/1024, util=float(j.get('gpu_utilization_mean_percent')))
    else:
        d.update(sec=None,samples_sec=None,mem_gib=None,util=None)
    bench.append(d)
add_b('batch400 FP32',400,'FP32','success',ROOT/'runs/current_valid_baseline/batch400_bottleneck_benchmark_20260918_utilfix/A_fp32_current/result.json')
add_b('batch400 BF16',400,'BF16','success',ROOT/'runs/current_valid_baseline/batch400_bottleneck_benchmark_20260918_utilfix/C_bf16_amp/result.json')
add_b('batch2048 BF16',2048,'BF16','success',ROOT/'runs/current_valid_baseline/batch4096_benchmark_20260918/bf16_b2048_20steps/result.json')
add_b('batch2432 BF16',2432,'BF16','success',ROOT/'runs/current_valid_baseline/batch4096_benchmark_20260918/bf16_b2432/result.json')
add_b('batch2496 BF16',2496,'BF16','failed',ROOT/'runs/current_valid_baseline/batch4096_benchmark_20260918/bf16_b2496/preflight.json')
add_b('batch4096 FP32',4096,'FP32','OOM',ROOT/'runs/current_valid_baseline/batch4096_benchmark_20260918/fp32/result.json')
add_b('batch4096 BF16',4096,'BF16','failed',ROOT/'runs/current_valid_baseline/batch4096_benchmark_20260918/bf16/preflight.json')
# Keep the benchmark report's published GiB convention for the 20-step run.
for x in bench:
    if x['setting']=='batch2048 BF16': x['mem_gib']=89.65
pd.DataFrame(bench).to_csv(SRC/'benchmark_summary.csv',index=False)

paths = {
 'paper_pdf_repo':'mobihoc26-paper289.pdf',
 'experiment_pdf_repo':'UACP실험_250914.pdf',
 'baseline_100k1_eval':'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/evaluation_table.csv',
 'baseline_100k1_fig8':'runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/fig8_eval_retry/results.json',
 'baseline_100k5_fig8_fig9':'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/matched_fig8_fig9_comparison_10k_v3/',
 'baseline_100k5_batch400_fig8_fig9':'runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/evaluation_streaming_v5/',
 'benchmark_4096':'runs/current_valid_baseline/batch4096_benchmark_20260918/',
 'benchmark_400':'runs/current_valid_baseline/batch400_bottleneck_benchmark_20260918_utilfix/',
 'pipeline_code':'src/training/data.py, src/models/uacp_predictor.py, src/models/evidential.py, src/training/uncertainty.py',
}
json.dump(paths,open(SRC/'figure_sources.json','w'),indent=2,ensure_ascii=False)

# Pipeline flow figure
fig, ax=plt.subplots(figsize=(13,3.0)); ax.axis('off')
boxes=['Sionna channel\n(TDL-A)\nfull CFR','Uniform\nsubcarrier sampling\nNg=16','Reported CFR\n15 dB complex AWGN\nIMPLEMENTATION-ASSUMPTION','Omitted CFR = 0\n+ binary mask','Predictor input\n[B,9,1024]\nreal/imag + mask','Evidential predictor\nγ, κ, Ψ, ν','CFR reconstruction\n+ Aleatoric / Epistemic']
xs=np.linspace(.03,.97,len(boxes));
for i,(x,t) in enumerate(zip(xs,boxes)):
    ax.add_patch(FancyBboxPatch((x-.065,.30),.13,.38,boxstyle='round,pad=.02',fc='#EAF2F8' if i<4 else '#FDEBD0',ec='#355C7D',lw=1.2,transform=ax.transAxes))
    ax.text(x,.49,t,ha='center',va='center',fontsize=8,transform=ax.transAxes)
    if i<len(boxes)-1: ax.add_patch(FancyArrowPatch((x+.067,.49),(xs[i+1]-.067,.49),arrowstyle='->',mutation_scale=12,lw=1.2,color='#555',transform=ax.transAxes))
fig.savefig(FIG/'pipeline_flow.png',bbox_inches='tight'); plt.close(fig)

# Benchmark figures
succ=[x for x in bench if x['status']=='success']; labels=[x['setting'] for x in bench]
colors=['#4C78A8' if x['precision']=='FP32' else '#F58518' for x in bench]
fig,ax=plt.subplots(figsize=(8.5,4.2)); vals=[x['samples_sec'] if x['samples_sec'] is not None else np.nan for x in bench]; bars=ax.bar(labels, [0 if np.isnan(v) else v for v in vals], color=colors, edgecolor='0.3');
for b,x,v in zip(bars,bench,vals): ax.text(b.get_x()+b.get_width()/2, (v if not np.isnan(v) else .5)+2, f"{v:.1f}" if not np.isnan(v) else x['status'],ha='center',va='bottom',fontsize=8,rotation=90 if np.isnan(v) else 0)
ax.set_ylabel('Samples/sec'); ax.set_title('GPU benchmark: throughput'); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/'batch_samples_per_sec.png'); plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4.2)); vals=[x['mem_gib'] if x['mem_gib'] is not None else np.nan for x in bench]; bars=ax.bar(labels,[0 if np.isnan(v) else v for v in vals],color=colors,edgecolor='0.3')
for b,x,v in zip(bars,bench,vals): ax.text(b.get_x()+b.get_width()/2,(v if not np.isnan(v) else 1)+2,f"{v:.2f} GiB" if not np.isnan(v) else x['status'],ha='center',va='bottom',fontsize=8,rotation=90 if np.isnan(v) else 0)
ax.set_ylabel('Peak allocated memory (GiB)'); ax.set_title('GPU benchmark: peak memory'); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(FIG/'batch_peak_memory.png'); plt.close(fig)

# Fig8 summary-derived distributions, same x scale. 100k1/5 raw samples are not in comparison artifact.
sum8=pd.read_csv(ROOT/'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/matched_fig8_fig9_comparison_10k_v3/fig8_summary.csv')
regimes=['ID-Easy 20 ns','ID-Hard 80 ns','OOD-Near 120 ns','OOD-Far 1 ms']; cmap={'ID-Easy 20 ns':'#4C78A8','ID-Hard 80 ns':'#72B7B2','OOD-Near 120 ns':'#F2CF5B','OOD-Far 1 ms':'#E45756'}
fig,axs=plt.subplots(1,3,figsize=(14,3.8),sharex=True,sharey=True); x=np.linspace(-38,-20,500)
for ax,(model,srcmodel) in zip(axs, [('100k×1 batch8 FP32','100k×1'),('100k×5 batch8 FP32','100k×5'),('100k×5 batch400 BF16',None)]):
    if srcmodel:
        d=sum8[sum8.model==srcmodel]
        for _,r in d.iterrows():
            if not np.isfinite(r.epi_db_mean):
                ax.text(-29, .25, 'OOD-Far: N/A\n(ν/inf)',ha='center',color='#B22222')
                continue
            y=np.exp(-.5*((x-r.epi_db_mean)/max(r.epi_db_std_finite,.02))**2)/(max(r.epi_db_std_finite,.02)*np.sqrt(2*np.pi)); ax.plot(x,y,label=r.regime,color=cmap[r.regime],lw=1.8)
    else:
        d=pd.read_csv(ROOT/'runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/evaluation_streaming_v5/regime_summary.csv')
        for _,r in d.iterrows():
            mu=10*np.log10(max(r.epistemic_mean,1e-12)); sd=max(r.epistemic_std/(r.epistemic_mean*np.log(10)),.02); y=np.exp(-.5*((x-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)); ax.plot(x,y,label=r.regime,color=cmap[r.regime],lw=1.8)
    ax.set_title(model); ax.set_xlabel('Epistemic score (dB)'); ax.grid(alpha=.2)
axs[0].set_ylabel('Summary-derived density'); axs[-1].legend(fontsize=7,loc='upper left'); fig.suptitle('Fig.8 comparison (same 10k common-eval summaries; batch8 raw samples unavailable)',y=1.03); fig.tight_layout(); fig.savefig(FIG/'fig8_epistemic_baseline_comparison.png',bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.3,4.3)); metrics=['ID→Near','ID→Far','pooled ID→OOD']; cols=['auc_id_near','auc_id_far','auc_pooled']; xpos=np.arange(3); width=.24
for i,(_,r) in enumerate(cmp.iterrows()):
    vals=[r[c] if pd.notna(r[c]) else np.nan for c in cols]; bars=ax.bar(xpos+(i-1)*width,[0 if np.isnan(v) else v for v in vals],width,label=r.model,color=['#4C78A8','#F58518','#54A24B'][i])
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,(v if not np.isnan(v) else .03)+.015,'N/A\n(ν/inf)' if np.isnan(v) else f'{v:.3f}',ha='center',va='bottom',fontsize=8)
ax.set_ylim(0,1.12); ax.set_xticks(xpos,metrics); ax.set_ylabel('AUROC'); ax.set_title('Fig.8 AUROC comparison'); ax.grid(axis='y',alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/'fig8_auroc_comparison.png'); plt.close(fig)

# Fig9 curves from stored calibration CSVs
cal_old=pd.read_csv(ROOT/'runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/matched_fig8_fig9_comparison_10k_v3/calibration.csv')
cal_new=pd.read_csv(ROOT/'runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/evaluation_streaming_v5/fig9_calibration.csv')
fig,axs=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
for ax,model in zip(axs,models):
    ax.plot([0,1],[0,1],'k--',lw=1,label='ideal')
    if model.startswith('100k×1'): d=cal_old[cal_old.model=='100k×1']
    elif model.startswith('100k×5 batch8'): d=cal_old[cal_old.model=='100k×5']
    else: d=None
    if d is not None:
        for pool,col in [('ID','#4C78A8'),('Near','#F2CF5B'),('Far','#E45756'),('OOD-pooled','#54A24B')]:
            if model.startswith('100k×5') and pool in ('Far','OOD-pooled'):
                continue
            z=d[d.pool==pool]; ax.plot(z.nominal,z.empirical,'o-',label=pool,color=col,lw=1.2,ms=3)
        if model.startswith('100k×5'):
            ax.text(.62,.16,'Far/OOD: N/A\n(ν/inf)',ha='center',va='center',fontsize=8,color='#B22222')
    else:
        # batch400 evaluator stores pool/nominal/empirical; use its explicit data.
        for pool,col in [('ID','#4C78A8'),('Near','#F2CF5B'),('Far','#E45756'),('OOD','#54A24B')]:
            z=cal_new[cal_new.pool==pool]; ax.plot(z.nominal,z.empirical,'o-',label=pool,color=col,lw=1.2,ms=3)
    ax.set_title(model); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal','box'); ax.grid(alpha=.2); ax.set_xlabel('Nominal coverage')
axs[0].set_ylabel('Empirical coverage'); axs[-1].legend(fontsize=7,loc='lower right'); fig.suptitle('Fig.9 calibration comparison (10k/regime artifacts)',y=1.02); fig.tight_layout(); fig.savefig(FIG/'fig9_calibration_curves.png',bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.3)); pools=['ID','Near','Far','OOD-pooled']; x=np.arange(4); w=.24
for i,(_,r) in enumerate(cmp.iterrows()):
    vals=[r[['fig9_id_mae','fig9_near_mae','fig9_far_mae','fig9_ood_mae'][j]] if pd.notna(r[['fig9_id_mae','fig9_near_mae','fig9_far_mae','fig9_ood_mae'][j]]) else np.nan for j in range(4)]
    bars=ax.bar(x+(i-1)*w,[0 if np.isnan(v) else v for v in vals],w,label=r.model,color=['#4C78A8','#F58518','#54A24B'][i])
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,(v if not np.isnan(v) else .01)+.01,'N/A\n(ν/inf)' if np.isnan(v) else f'{v:.3f}',ha='center',va='bottom',fontsize=7)
ax.set_xticks(x,pools); ax.set_ylabel('Calibration MAE'); ax.set_title('Fig.9 calibration error comparison'); ax.grid(axis='y',alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/'fig9_calibration_error_comparison.png'); plt.close(fig)

def f(v): return 'N/A (ν/inf)' if pd.isna(v) else f'{v:.3f}'
benchmark_table='\n'.join([f"| {x['setting']} | {x['sec']:.3f} | {x['samples_sec']:.1f} | {x['util']:.1f}% | {x['mem_gib']:.2f} | 성공 |" if x['status']=='success' else f"| {x['setting']} | — | — | — | — | {x['status']} |" for x in bench])
fig8_table='\n'.join([f"| {r.model} | {f(r.auc_id_near)} | {f(r.auc_id_far)} | {f(r.auc_pooled)} |" for _,r in cmp.iterrows()])
fig9_table='\n'.join([f"| {r.model} | {f(r.fig9_id_mae)} | {f(r.fig9_near_mae)} | {f(r.fig9_far_mae)} | {f(r.fig9_ood_mae)} |" for _,r in cmp.iterrows()])
train_rows=[]
train_rows.append('| 100k×1 | 1 | 8 | FP32 | 100,000 | 12,500 | 100,000 | 2,659.9 s (44.3 min) | 37.6 |')
train_rows.append('| 100k×5 | 5 | 8 | FP32 | 500,000 | 62,500 | 100,000 | 44,054.9 s (12.24 h) | 11.4 |')
train_rows.append('| 100k×5 | 5 | 400 | BF16 AMP + FP32 evidence | 500,000 | 1,250 | 100,000 | 5,579 s (93 min) | 89.6 actual / 133.9 benchmark |')

md=rf'''# UACP 재현 실험 보고서

**범위:** Sections 1–3 only. 기존 artifact 기반 read-only 비교 보고서  
**작성일:** 2026-09-18  
**주의:** 지정된 `/mnt/data/mobihoc26-paper289.pdf`, `/mnt/data/UACP실험.pdf`는 현재 환경에 없었습니다. repository의 `mobihoc26-paper289.pdf`, `UACP실험_250914.pdf`를 대체 참고자료로 사용했습니다.

## 1. 연구 목적 및 재현 목표

UACP는 sparse channel state information(CFR)을 입력으로 받아 full CFR을 reconstruction하고, Evidential Regression을 통해 Aleatoric uncertainty와 Epistemic uncertainty를 동시에 추정하는 구조다. Aleatoric uncertainty는 현재 channel 자체가 얼마나 reconstruction하기 어려운지를 나타내는 신호이고, Epistemic uncertainty는 predictor가 현재 channel distribution을 얼마나 낯설게 보는지를 나타내는 신호다. 이 두 신호는 feedback rate를 조절하고, OOD channel에서 추가 adaptation이 필요한지를 판단하는 데 사용될 수 있다.

본 재현의 1차 검증 기준은 세 가지다. 첫째, sparse CFR 입력으로부터의 CFR reconstruction이 정상적으로 학습되는지 확인한다. 둘째, training range 안에서 delay spread가 큰 ID-Hard 80 ns가 ID-Easy 20 ns보다 reconstruction하기 어려운지 확인한다. 셋째, training distribution 밖에서 Epistemic uncertainty가 증가해 ID와 OOD를 구분하는지 확인한다. 이후 연구 흐름에서는 calibration을 통해 uncertainty의 신뢰성을 점검하고, 최종적으로 dynamic feedback control까지 연결해야 한다. 다만 본 보고서는 Fig.11 최종 평가를 포함하지 않는다.

이 baseline은 향후 Epistemic uncertainty 기반 Full Fine-Tuning과 Partial Fine-Tuning을 비교하기 위한 출발점이다. 따라서 단일 수치의 최적화보다 reconstruction 품질, uncertainty separation, calibration, 학습 비용, numerical stability를 함께 기록하는 것이 목적이다.

## 2. 논문 기준 및 현재 재현 환경

### 2.1 원 논문의 Experimental Setup

원 논문 Section 4에 공개된 설정은 다음과 같다.

| 항목 | 논문 설정 |
|---|---|
| MIMO | 2×2 |
| Subcarriers K | 1024 |
| Carrier frequency | 3.5 GHz |
| Subcarrier spacing | 30 kHz |
| SNR | 15 dB |
| Training delay spread | Uniform [10,100] ns |
| Training samples | 100,000 |
| Batch size | 4096 |
| Epochs | 150 |
| Learning rate | 1e-4 |
| λreg | 1e-3 |

논문은 ID-easy를 20 ns, ID-hard를 80 ns로 설명하며, OOD는 training delay-spread interval을 벗어난 channel로 정의한다. 본 비교에서는 OOD-near를 120 ns, OOD-far를 1 ms로 구분했다. 이 regime 명칭과 120 ns/1 ms 세부 평가는 현재 실험 protocol의 구성으로 기록한다.

### 2.2 현재 구현 pipeline

아래 흐름은 `src/training/data.py`, `src/models/uacp_predictor.py`, `src/models/evidential.py`, `src/training/uncertainty.py`를 기준으로 정리했다. 이 그림은 문서 본문에서 설명한 뒤 배치한다.

![현재 구현 pipeline](figures/pipeline_flow.png)

그림 1. Sionna CFR 생성부터 mask-aware Evidential Predictor와 uncertainty 산출까지의 현재 구현 흐름.

현재 predictor 입력은 `[B, 9, 1024]`이다. 2×2 MIMO의 네 complex antenna-pair CFR을 real/imaginary 8개 channel로 펼치고, sparse CFR의 omitted 위치를 0으로 만든 뒤 binary mask 1개를 추가한다. 평가 mask는 `Ng=16`으로 uniform하게 subcarrier를 관측한다. 현재 구현의 Evidential head는 γ, κ, Ψ, ν를 만들고, Ψ는 full covariance가 아니라 diagonal approximation으로 사용된다.

### 2.3 논문과 다른 IMPLEMENTATION-ASSUMPTION

| 항목 | 논문 공개 여부 | 현재 구현 | 이유/해석 |
|---|---|---|---|
| Channel model | 정확한 Sionna TDL 설정 미공개 | Sionna TDL-A, zero mobility, normalize=false | repository의 데이터 생성 설정. `IMPLEMENTATION-ASSUMPTION` |
| SNR 적용 단계 | 15 dB는 공개되지만 noise 적용 단계는 미공개 | reported CFR에 sample-wise complex AWGN 적용, clean CFR을 target으로 유지 | 현재 observation pipeline 선택. 논문 설정이라고 단정하지 않음 |
| Covariance | full Ψ parameterization의 세부 공개 제한 | diagonal-Psi approximation, pair-scalar κ/ν | 메모리와 구현 제약을 고려한 근사 |
| Precision | 공개된 mixed-precision 조건 없음 | batch400 baseline은 BF16 AMP backbone, evidential transform/loss FP32 | GB10 throughput/memory를 위한 `IMPLEMENTATION-ASSUMPTION` |
| Mask/observation | 세부 mask 생성·noise stage 공개 제한 | uniform `Ng=16`, omitted=`1-mask`, observed 위치에만 noise | 현재 evaluator와 데이터 pipeline의 구현 선택 |

## 3. Baseline 학습 설정 비교 및 최종 선정

### 3.1 왜 Batch Size를 다시 검토했는가

논문은 batch 4096과 150 epochs를 사용하지만, 현재 NVIDIA GB10에서는 direct batch4096을 실행할 수 없었다. 기존 benchmark에서 FP32 batch4096은 106,505 MiB를 할당한 뒤 추가 3 GiB allocation에서 CUDA OOM이 발생했고, BF16 batch4096도 완료하지 못했다. 따라서 논문 수치를 그대로 실행한 관측 결과는 존재하지 않는다.

기존 batch8은 초기 재현 과정에서 GPU memory 부담을 낮추고 frequent optimizer update를 확보하기 위한 초기 reference configuration으로 사용되었다. 이는 논문 batch size가 아니라 repository의 experimental choice다. 이후 batch400은 GPU 활용률과 처리량을 높이면서도 direct batch4096보다 충분한 memory margin을 확보하는 operational configuration으로 탐색되었다. 이 이유는 repository의 benchmark와 training log에 근거하며, 역시 `IMPLEMENTATION-ASSUMPTION`이다.

두 batch를 비교할 때 batch 숫자만 비교할 수 없다. batch8은 optimizer update가 자주 발생하지만 총 step 수와 wall-clock이 커지고, batch400은 step 수를 크게 줄여 GPU throughput을 높인다. 반면 batch 크기와 mixed precision은 evidential parameter가 학습되는 경로와 uncertainty behavior에도 영향을 줄 수 있으므로, reconstruction만으로 winner를 정하지 않았다.

### 3.2 실제 Training Run 비교

세 run은 모두 100,000 unique CFR을 사용하지만 exposure와 optimizer step 수가 다르다.

| unique samples | epochs | batch | precision | total exposure | optimizer steps | actual wall-clock | samples/sec | 20 ns NMSE | 80 ns NMSE | 120 ns NMSE | 1 ms NMSE |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 1 | 8 | FP32 | 100,000 | 12,500 | 2,659.9 s (44.3 min) | 37.6 | -18.977 | -17.859 | -15.153 | 1.251 |
| 100,000 | 5 | 8 | FP32 | 500,000 | 62,500 | 44,054.9 s (12.24 h) | 11.4 | -21.802 | -19.477 | -16.468 | 1.306 |
| 100,000 | 5 | 400 | BF16 AMP + FP32 evidential | 500,000 | 1,250 | 5,579 s (93 min) | 89.6 actual / 133.9 benchmark | -17.350 | -17.094 | -15.849 | 1.416 |

그림 2와 그림 3은 이 비교에 사용한 GPU benchmark를 시각화한다.

![Samples/sec benchmark](figures/batch_samples_per_sec.png)

그림 2. Batch size/precision별 실측 throughput. OOM/failed 조건은 0이 아니라 상태로 표시했다.

![Peak memory benchmark](figures/batch_peak_memory.png)

그림 3. Batch size/precision별 peak allocated GPU memory. 실패한 조건은 memory 수치가 아니라 OOM/failed로 표시했다.

100k×5 batch8은 reconstruction NMSE만 보면 가장 낮은 값에 도달했지만 약 12시간이 필요했고, batch400 BF16은 약 93분에 완료되었다. 따라서 batch400을 정확도 우승자로 해석할 수는 없으며, uncertainty 품질과 numerical stability를 함께 비교해야 한다.

### 3.3 GPU Benchmark와 Batch Size 효율

| 설정 | sec/step | samples/sec | GPU utilization | peak memory | 결과 |
|---|---:|---:|---:|---:|---|
{benchmark_table}

batch400 BF16은 2.987 sec/step, 133.9 samples/s benchmark throughput, GPU utilization 92.8%, peak allocated 17.22 GiB였다. batch2048 BF16은 21.7183 sec/step, 약 94.3 samples/s, GPU utilization 약 92.3%, peak memory 89.65 GiB였다. 더 큰 batch가 samples/sec를 높이지 않은 것은 실제 benchmark에서 확인된 사실이며, 이 결과만으로 원인을 단정하지 않는다. 즉 현재 모델·입력·메모리 경로에서는 batch가 커질수록 반드시 효율이 증가하지 않는다.

### 3.4 논문 Scale Training이 왜 2일 이상 필요한가

논문 설정에서 `drop_last=False`라면 다음과 같다.

$$\left\lceil\frac{{100000}}{{4096}}\right\rceil=25\text{{ steps/epoch}}$$

$$25\times150=3750\text{{ optimizer steps}}$$

$$100000\times150=15,000,000\text{{ sample exposure}}$$

하지만 direct batch4096은 현재 GB10에서 FP32 CUDA OOM이고, BF16도 안정적으로 완료되지 않았다. 현실적인 대안으로 batch2048과 gradient accumulation 2를 가정하면, 한 epoch의 microstep은 다음과 같다.

$$\left\lceil\frac{{100000}}{{2048}}\right\rceil=49\text{{ microsteps/epoch}}$$

$$49\times150=7350\text{{ microsteps}}$$

기존 benchmark의 batch2048 BF16 평균 `21.7183 sec/microstep`을 적용하면,

$$7350\times21.7183=159,629.5\text{{ sec}}\approx44.3\text{{ h}}$$

이다. Validation, checkpoint, logging, evaluator overhead를 포함하면 약 46–50시간으로 보는 것이 타당하다. 장시간 실행 변동과 재시작 여유까지 고려하면 운영상 2–3일 규모의 실험이다. 이 계산은 OOD가 발생할 때마다 수행하는 online adaptation 시간이 아니라, 논문 규모의 initial/offline predictor training 비용이다. 또한 gradient accumulation은 effective batch4096을 구성할 수 있지만 direct batch4096과 optimizer update, noise/mask draw, gradient averaging이 완전히 같지 않으므로 `IMPLEMENTATION-ASSUMPTION`이다.

### 3.5 Fig.8 기준 Epistemic OOD 분리 성능 비교

세 baseline의 Fig.8 비교는 기존 10,000-sample/regime artifact를 우선 사용했다. batch8 두 모델은 `matched_fig8_fig9_comparison_10k_v3`, batch400 BF16은 `evaluation_streaming_v5`에서 생성되었다. score 정의는 기존 Eq.(7),(8),(12),(13) evaluator이고, batch8의 raw per-sample distribution은 보존되어 있지 않아 그림 4는 저장된 mean/std 기반 summary-derived density다. 따라서 새로운 evaluator 실행이나 재학습은 하지 않았다.

![Fig8 density](figures/fig8_epistemic_baseline_comparison.png)

그림 4. 세 baseline의 Epistemic density 비교. 모든 panel의 축을 동일하게 맞췄으며, batch8 비교 artifact의 mean/std를 이용한 요약 분포다. 100k×5 batch8 OOD-Far는 ν/inf로 N/A 처리했다.

| baseline | ID→Near AUROC | ID→Far AUROC | pooled ID→OOD AUROC |
|---|---:|---:|---:|
{fig8_table}

![Fig8 AUROC](figures/fig8_auroc_comparison.png)

그림 5. Fig.8 AUROC 비교. N/A는 0이 아니라 기존 artifact의 ν/inf numerical failure를 뜻한다.

100k×1 batch8은 Near 0.621, Far 0.997, pooled 0.809로 전체적으로 안정적인 uncertainty baseline이었다. 100k×5 batch8은 Near separation이 0.948로 크게 좋아졌지만 Far 및 pooled 값은 ν/inf 때문에 유효한 비교값으로 사용할 수 없다. batch400 BF16은 모든 주요 값이 finite하고 빠르게 평가되었지만 Near 0.542, Far 0.803, pooled 0.673으로 Near-OOD separation이 약했다. 따라서 batch400을 uncertainty 성능 최고 baseline으로 해석해서는 안 된다.

### 3.6 Fig.9 Calibration 비교

Fig.9도 동일하게 기존 10k artifact를 사용했다. calibration plot은 nominal coverage와 empirical coverage를 각각 0–1로 고정하고 동일 aspect ratio로 표시했다.

![Fig9 curves](figures/fig9_calibration_curves.png)

그림 6. 세 baseline의 ID/Near/Far/OOD-pooled calibration curve. 대각선은 ideal coverage다.

| baseline | ID CE/MAE | Near CE/MAE | Far CE/MAE | pooled OOD CE/MAE |
|---|---:|---:|---:|---:|
{fig9_table}

![Fig9 error](figures/fig9_calibration_error_comparison.png)

그림 7. Baseline별 calibration error 비교. N/A는 ν/inf로 계산 불가능한 값을 뜻한다.

100k×1 batch8은 ID calibration MAE 0.0086으로 안정적이지만 Near 0.1171과 Far 0.6555의 악화가 크다. 100k×5 batch8은 저장된 비교 결과에서 ID/Near/Far MAE가 낮아졌지만 Far uncertainty가 ν/inf를 포함하므로 numerical stability를 함께 해석해야 한다. batch400 BF16은 ID 0.2698, Near 0.2409, Far 0.2920, pooled OOD 0.0859로 finite했지만 ID calibration이 batch8 reference보다 크게 나쁘다. pooled OOD 하나만 낮다는 이유로 calibration이 좋다고 결론내릴 수 없으며, ID/Near/Far 분리 결과와 Fig.8 separation을 함께 봐야 한다.

### 3.7 최종 Baseline 선정

| 설정 | Training time | samples/s | memory | Reconstruction | Near OOD separation | Far OOD | Calibration | Numerical stability |
|---|---:|---:|---:|---|---:|---:|---|---|
| 100k×1 batch8 FP32 | 44.3 min | 37.6 actual | reference | 정상 | 0.621 | 0.997 | ID 0.0086 | finite |
| 100k×5 batch8 FP32 | 12.24 h | 11.4 actual | reference | 가장 낮은 NMSE | 0.948 | N/A (ν/inf) | 낮은 recorded CE, Far failure | 불안정 |
| 100k×5 batch400 BF16 | 93 min | 89.6 actual / 133.9 benchmark | 17.22 GiB | 정상 | 0.542 | 0.803 | ID 0.2698, pooled 0.0859 | finite |

본 연구에서는 논문과 가장 동일한 학습 규모를 주장하지 않는다. 현재 GB10 환경에서 training throughput, GPU utilization, memory margin, numerical stability, CFR reconstruction 및 uncertainty behavior를 종합했을 때 `100,000 samples × 5 epochs / direct batch400 / BF16 AMP backbone + FP32 evidential loss`를 가장 실용적인 operational/efficient reproduction baseline으로 선정한다. 이 선택은 Fig.8/9 성능이 가장 좋아서가 아니다. batch8×5보다 약 7배 이상 짧은 약 93분에 완료되고, GPU utilization이 높으며, memory margin이 크고, Far regime까지 finite하게 동작하기 때문이다.

동시에 한계도 명확하다. batch400 BF16은 100k×1 batch8 reference보다 Near-OOD separation과 ID calibration이 떨어진다. 따라서 이 baseline을 “가장 정확한 baseline”, “논문 재현 성공”, 또는 “논문과 동일한 설정”으로 부르지 않는다. 정확한 표현은 현재 하드웨어에서 반복 가능한 학습 비용과 reconstruction/uncertainty의 유한성까지 고려한 operational baseline이다.

**향후 추가 예정 결과.**

본 문서에는 요청 범위에 따라 Fig.11 최종 평가와 dynamic feedback control 결과를 포함하지 않았다. 이후 확정된 operational baseline에 대해 Fig.8/9 후속 검증과 Fig.11 최종 평가를 별도 절로 추가할 수 있다.

**수치 및 그림 source.**

모든 수치와 그림 source는 `report/source_data/figure_sources.json`, `report/source_data/benchmark_summary.csv`에 기록했다. 기존 checkpoint, result, CSV, log는 읽기 전용으로 사용했으며 수정·삭제·덮어쓰지 않았다.
'''
(OUT/'UACP_reproduction_report.md').write_text(md,encoding='utf-8')
print('wrote', OUT/'UACP_reproduction_report.md')
