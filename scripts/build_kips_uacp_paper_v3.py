#!/usr/bin/env python3
"""Build the compact KIPS V3 draft from existing artifacts only.

No training or evaluation is performed. V1/V2 documents and all raw artifacts
are preserved; this builder writes only V3 outputs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt

from build_kips_uacp_paper_v1 import ROOT, style_doc, add_md_to_doc

OUT = ROOT / "output"
FIGDIR = OUT / "KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V3_figures"
MD = ROOT / "docs/KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V3.md"
DOCX = OUT / "KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V3.docx"
COMM = ROOT / "runs/communication_surrogate_scope_20260926_v1_retry"
ROLE = ROOT / "runs/bf_sufficient_epi_role_analysis_20260926_v1_retry5"
V1MD = ROOT / "docs/KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V1.md"

BF = pd.read_csv(ROLE / "tables/table1_bf_only_required_scope.csv")
REF = pd.read_csv(COMM / "id_hard_reference.csv")
EFF = pd.read_csv(COMM / "tables/table6_efficiency.csv")
SEV = pd.read_csv(ROLE / "tables/table2_pre_ei_severity_vs_bf_scope.csv")

COLORS = {"small": "#4C78A8", "medium": "#F58518", "full": "#54A24B"}
LABELS = {"small": "Small", "medium": "Medium", "full": "Full"}


def make_figures():
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: compact model/scope schematic.
    fig, ax = plt.subplots(figsize=(9.6, 1.85), dpi=220)
    ax.axis("off")
    boxes = [
        (0.08, "Sparse CFR\n+ mask"),
        (0.27, "Input\nprojection"),
        (0.47, "Residual blocks ×32\n(backbone)"),
        (0.70, "CFR mean +\nγ, Ψ, κ, ν heads"),
        (0.91, "CFR prediction\n+ uncertainty"),
    ]
    for x, label in boxes:
        ax.text(x, .58, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.45", fc="#edf3f8", ec="#365f7d", lw=1.1))
    for x1, x2 in zip([.14, .33, .57, .78], [.21, .40, .64, .85]):
        ax.annotate("", xy=(x2, .58), xytext=(x1, .58),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"))
    ax.text(.36, .16, "Small: last block + heads   |   Medium: last 4 blocks + heads   |   Full: all parameters",
            ha="center", va="center", fontsize=8.5, color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIGDIR / "figure1_uacp_partial_scope.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "figure1_uacp_partial_scope.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: BF gain, seed points, means/std, seed-mean reference line.
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.25), sharey=True)
    delays = [160, 180, 200]
    for ax, ng in zip(axes, [16, 32]):
        ref = REF.loc[REF.ng == ng, "bf_mean"].mean()
        x = np.arange(3)
        for j, scope in enumerate(["small", "medium", "full"]):
            col = f"{scope}_bf_ng{ng}"
            means = [BF.loc[BF.delay_ns == d, col].mean() for d in delays]
            stds = [BF.loc[BF.delay_ns == d, col].std(ddof=1) for d in delays]
            xx = x + (j - 1) * .20
            ax.errorbar(xx, means, yerr=stds, fmt="o-", capsize=3, lw=1.3,
                        color=COLORS[scope], label=LABELS[scope])
            for d, xpos in zip(delays, xx):
                vals = BF.loc[BF.delay_ns == d, col].to_numpy()
                ax.scatter(np.full(len(vals), xpos), vals, s=20, color=COLORS[scope],
                           edgecolor="white", linewidth=.35, zorder=3)
        ax.axhline(ref, color="#333", ls="--", lw=1.1)
        ax.set_title(f"Ng={ng}", fontsize=10)
        ax.set_xticks(x, [f"{d} ns" for d in delays], fontsize=8)
        ax.set_ylim(.98, 1.001)
        ax.grid(axis="y", alpha=.25)
        ax.set_xlabel("OOD delay spread", fontsize=8.5)
    axes[0].set_ylabel("Full-band normalized BF gain", fontsize=8.5)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Medium is the first scope meeting the ID-Hard reference", fontsize=11)
    fig.text(.5, .005, "Dashed line: mean of seed-specific 80 ns ID-Hard references; pass uses each seed's reference.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=[0, .055, 1, .93])
    fig.savefig(FIGDIR / "figure2_bf_gain_reference.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "figure2_bf_gain_reference.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: severity increases, required scope remains Medium.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 4.0),
                                   gridspec_kw={"height_ratios": [2.1, 1]}, sharex=True)
    delays = [160, 180, 200]
    for i, d in enumerate(delays):
        vals = SEV.loc[SEV.delay_ns == d, "primary_z"].to_numpy()
        ax1.scatter(np.full(len(vals), i), vals, color="#4C78A8", s=34, zorder=3)
        ax1.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color="#1f4e79",
                     capsize=4, lw=1.3, zorder=4)
    ax1.plot(range(3), [SEV.loc[SEV.delay_ns == d, "primary_z"].mean() for d in delays],
             color="#1f4e79", lw=1.2, alpha=.8)
    ax1.set_ylabel("Pre-adaptation severity z")
    ax1.set_title("Severity increases while the required evaluated scope is unchanged", fontsize=11)
    ax1.grid(axis="y", alpha=.25)
    ax2.scatter(range(3), [2, 2, 2], s=230, marker="s", color=COLORS["medium"])
    ax2.set_yticks([1, 2, 3], ["Small", "Medium", "Full"])
    ax2.set_ylabel("Required scope")
    ax2.set_ylim(.5, 3.5)
    ax2.set_xticks(range(3), [f"{d} ns" for d in delays])
    ax2.grid(axis="y", alpha=.25)
    fig.text(.5, .01, "Top: individual seeds and mean±std. Bottom: BF-only minimum evaluated scope.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .055, 1, .94])
    fig.savefig(FIGDIR / "figure3_severity_scope_unchanged.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "figure3_severity_scope_unchanged.pdf", bbox_inches="tight")
    plt.close(fig)


def paper_markdown() -> str:
    return r'''# Evidential MIMO 채널 예측에서 부분 파라미터 업데이트를 이용한 OOD 적응 분석

## Parameter-Efficient OOD Adaptation in Evidential MIMO Channel Prediction

이유빈¹, 이강원²  \ ¹세종대학교 지능기전공학부 스마트기기전공 학부생, ²세종대학교 컴퓨터공학과 교수  \ 22011969@sju.ac.kr, kangwon.lee@sejong.ac.kr

### 요약

Evidential MIMO 채널 예측기가 학습하지 않은 채널 환경을 만나면 추가 적응이 필요할 수 있지만, predictor 전체를 Fine-Tuning하면 계산 비용과 GPU memory가 증가한다. 본 연구는 UACP(Uncertainty-Aware Channel Prediction) 기반 predictor를 Small, Medium, Full 세 update scope로 나누고, 80 ns ID-Hard 조건의 normalized beamforming gain을 회복하는 최소 범위를 비교하였다. 평가 대상은 160, 180, 200 ns OOD delay spread와 세 독립 seed의 9개 조건이다. Medium은 마지막 네 residual block과 evidential heads를 업데이트하며 1,480,728개, 전체의 12.5323% parameter를 학습한다. 모든 9개 조건에서 Medium이 ID-Hard 수준의 normalized beamforming gain을 회복하는 minimum evaluated sufficient scope로 나타났다. Medium은 Full보다 trainable parameter 87.47%, measured training time 45.92%, peak VRAM 72.57%를 줄였다. Pre-adaptation Epistemic severity는 160 ns에서 200 ns로 갈수록 증가했지만 required scope는 모두 Medium이었다. 따라서 평가한 조건에서는 Partial Fine-Tuning의 비용 절감 가능성을 확인했지만, uncertainty magnitude만으로 필요한 adaptation budget을 결정하는 관계는 확인하지 못했다.

**주제어:** MIMO, Channel State Information, Evidential Regression, Out-of-Distribution, Partial Fine-Tuning, Beamforming

## 1. 서론

Multiple-Input Multiple-Output(MIMO)은 여러 송·수신 안테나를 사용하여 무선 채널의 공간적 자유도를 활용하는 기술이다. 송신기가 여러 안테나에서 신호를 어떤 방향으로 전송할지 결정하려면 Channel State Information(CSI)이 필요하다. OFDM에서는 여러 주파수 구간인 subcarrier를 사용하므로, 각 subcarrier의 주파수 응답을 나타내는 Channel Frequency Response(CFR)를 정확히 예측하는 것이 중요하다. 모든 CFR을 feedback하면 전송 overhead가 커지기 때문에 일부 관측으로 전체 CFR을 복원하는 연구가 진행되어 왔다. CsiNet 계열 연구는 낮은 feedback overhead에서 CSI를 복원하고 beamforming 품질을 유지하는 방법을 제시하였다[1].

UACP는 sparse CFR observation에서 full CFR을 복원하는 predictor이며, Evidential Regression을 이용해 예측과 uncertainty를 함께 출력한다. Aleatoric uncertainty는 데이터 자체의 변동성에서 비롯되고, Epistemic uncertainty는 모델이 충분히 경험하지 못한 입력에서 증가할 수 있는 model uncertainty이다[2]. 원 UACP 연구는 Epistemic uncertainty를 이용해 학습 분포에서 벗어난 channel regime을 감지하고 feedback/adaptation 동작을 조절하는 구조를 다룬다[3]. Dataset shift에서는 예측 성능과 uncertainty를 별도로 평가해야 한다는 점도 보고되었다[4].

OOD(Out-of-Distribution) channel을 감지한 뒤 adaptation이 필요하다는 사실은 predictor 전체를 업데이트해야 한다는 뜻과 다르다. Full Fine-Tuning은 모든 parameter와 optimizer state를 관리하므로 비용이 증가할 수 있다. Adapter와 LoRA는 전체 pretrained parameter를 업데이트하지 않는 Parameter-Efficient Fine-Tuning(PEFT)의 대표적인 연구 방향이다[5,6]. 본 연구는 Adapter나 LoRA를 구현하지 않고, UACP predictor에서 업데이트하는 residual block 범위를 제한하는 Partial Fine-Tuning을 분석한다.

본 연구는 세 질문에 답한다. 첫째, OOD channel에서 일부 parameter만 업데이트해도 ID-Hard 수준의 beamforming-oriented CSI quality를 회복할 수 있는가? 둘째, 가능한 경우 Full Fine-Tuning보다 parameter, time, memory를 얼마나 줄일 수 있는가? 셋째, Fine-Tuning 전 Epistemic severity가 필요한 update scope를 결정하는 단서가 되는가? 이를 위해 동일 pretrained checkpoint와 paired clean CFR 조건에서 Small, Medium, Full을 비교하였다.

본 연구의 기여는 다음과 같다. (1) 160–200 ns OOD에서 normalized BF criterion으로 minimum evaluated update scope를 정의하였다. (2) Medium과 Full의 parameter, training time, peak VRAM을 비교하였다. (3) pre-adaptation Epistemic severity와 required scope의 관계를 분석하여 uncertainty magnitude와 adaptation budget이 같은 개념인지 검토하였다.

## 2. 실험 디자인

### 2.1 UACP 채널 예측 모델과 OOD adaptation

Predictor의 입력은 일부 subcarrier의 sparse CFR과 어떤 위치가 관측되었는지를 나타내는 mask이다. 입력은 feature extraction과 32개의 residual block을 거쳐 CFR mean과 evidential parameters를 출력한다. Residual block은 변환된 특징을 원래 입력 특징에 더하는 구조로, 깊은 network에서 특징을 단계적으로 학습하도록 돕는다. Prediction head는 backbone 특징을 이용해 최종 CFR과 uncertainty 관련 값을 출력한다. 현재 구현의 evidential output은 CFR mean γ와 uncertainty 계산에 사용되는 Ψ, κ, ν이다. 따라서 CFR prediction과 uncertainty는 서로 다른 모델이 아니라 같은 predictor의 출력에서 계산된다.

원 UACP 논문은 MIMO-OFDM, evidential regression, uncertainty 기반 feedback/adaptation을 설명하지만, 아래의 Partial Fine-Tuning layer 범위와 adaptation dataset 구성까지 공개한 것은 아니다. 그러므로 논문에 직접 명시되지 않은 현재 구현 항목은 `IMPLEMENTATION-ASSUMPTION`으로 표시한다.

![Figure 1. UACP predictor와 update scope](output/KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V3_figures/figure1_uacp_partial_scope.png)

**Figure 1.** Sparse CFR과 mask가 predictor에 입력되고 CFR 및 evidential output으로 이어지는 흐름이다. 색으로 구분한 scope는 마지막 block과 head만 업데이트하는 Small, 마지막 네 block과 head를 업데이트하는 Medium, 전체를 업데이트하는 Full의 차이를 나타낸다.

### 2.2 데이터 및 학습 설정

원래 training distribution은 RMS delay spread 10–100 ns이다. 80 ns는 학습 범위 안에서 상대적으로 어려운 ID-Hard reference이고, 160/180/200 ns는 100 ns를 넘으므로 OOD adaptation 대상이다. 모든 scope는 같은 clean CFR, sample order, mask/noise schedule을 사용하였다.

| 항목 | 설정 |
|---|---|
| MIMO / subcarrier | 2×2 / K=1024 |
| Training / adaptation | 10–100 ns / 160, 180, 200 ns |
| Input / target | sparse CFR + mask / clean full CFR |
| Observation noise | reported CFR에 15 dB complex AWGN |
| Ng / optimizer | {4, 8, 16, 32} balanced / Adam |
| LR / batch / epoch | 1e-4 / 8 / 3 |
| Precision / regularizer | FP32 / λreg=1e-3 |
| Seeds | 20260926, 20260927, 20260928 |

Sionna 2.0.1과 TDL-A channel generation, zero mobility, `normalize=false`, 관측 noise pipeline은 `IMPLEMENTATION-ASSUMPTION`이다. Clean CFR은 split과 seed별로 독립 생성되었고 train/validation/test 및 cross-seed hash overlap은 0이었다. 동일 seed의 scope 간에는 같은 CFR과 mask/noise schedule을 공유하였다.

### 2.3 Partial Fine-Tuning 범위

Fine-Tuning은 pretrained model parameter를 새로운 OOD 데이터에 맞추어 추가 업데이트하는 과정이다. Partial Fine-Tuning은 전체 parameter가 아니라 선택한 layer만 업데이트하는 방식이다. 본 연구의 scope는 다음과 같다.

| Scope | 업데이트 layer | Trainable parameters | 전체 대비 |
|---|---|---:|---:|
| Small | ResidualBlock 31 + 4 evidential heads | 373,656 | 3.1625% |
| Medium | ResidualBlock 28–31 + 4 evidential heads | 1,480,728 | 12.5323% |
| Full | 전체 predictor | 11,815,320 | 100% |

뒤쪽 residual block과 head를 업데이트하는 것은 현재 구현의 Partial FT 설계 선택이며 원 UACP가 공식적으로 제시한 adaptation 범위는 아니다. Partial run의 frozen parameter maximum difference는 0이었다.

### 2.4 Normalized Beamforming Gain 기반 성능 평가

NMSE만으로는 channel error가 실제 송신 방향 선택에 미치는 영향을 직접 알기 어렵다. 따라서 predicted CSI로 선택한 beamforming 방향을 clean channel에 적용했을 때의 gain을 평가하는 communication-aware surrogate를 사용하였다. 이는 BER/BLER, throughput, 실제 QoS 측정이 아니다.

각 subcarrier k에서 clean channel을 H_k, predicted channel을 H-hat_k라 한다. 먼저 predicted channel에서 신호가 가장 강하게 전달되는 송신 방향을 얻기 위해 SVD를 사용하고, 그 principal right singular vector를 v-hat_k로 둔다.

**Gₚᵣₑd(k) = ‖Hₖ v̂ₖ‖₂²**

이는 predicted CSI로 정한 방향을 실제 clean channel에 적용했을 때의 gain이다. Perfect CSI를 알고 있을 때의 최대 gain은 다음과 같다.

**Gₒᵣₐcₗₑ(k) = σₘₐₓ(Hₖ)²**

따라서 normalized beamforming gain은 다음과 같다.

**G_BF(k) = Gₚᵣₑd(k) / Gₒᵣₐcₗₑ(k)**

값이 1에 가까울수록 predicted CSI가 clean channel의 dominant spatial direction을 잘 보존한다. Full-band gain은 모든 subcarrier gain의 평균이다. Sanity test에서 perfect/oracle gain은 1.0, random unit beamformer 평균은 0.541518이었고 finite/range 검사를 통과하였다.

ID-Hard reference는 Ng16에서 0.994180, Ng32에서 0.993673이다. 동일 seed에서 두 Ng의 adaptation 후 평균 gain이 모두 seed별 ID-Hard reference 이상이면 pass로 정의하고, Small→Medium→Full 순서에서 처음 pass한 scope를 minimum evaluated performance-sufficient scope로 정의하였다.

### 2.5 Epistemic uncertainty 기반 OOD severity

이번 논문에서는 Fine-Tuning 전 Epistemic uncertainty만 사용한다. Ng마다 uncertainty의 기본 scale이 다르므로 raw uncertainty를 직접 비교하지 않고 ID 데이터의 Ng별 q99 threshold와 비교한다. OOD sample의 Epistemic uncertainty를 U_epi, 해당 Ng의 fixed ID q99 threshold를 τ_Ng라 하면 severity는 다음과 같다.

**z = log₁₀(Uₑₚᵢ / τₙ₉)**

z=0은 ID q99와 같은 크기이고, z>0은 ID 기준보다 높은 uncertainty이다. 예를 들어 z=1이면 threshold의 10배이다. 이 severity와 2.4에서 정의한 BF-only required scope를 3.3에서 비교한다.

## 3. 결과 분석

### 3.1 OOD 환경에서의 최소 Fine-Tuning 범위

2.4의 ID-Hard BF criterion을 사용해 세 scope를 Small→Medium→Full 순서로 확인하였다. Raw 결과는 다음과 같다.

| OOD delay | Small | Medium | Full | Minimum evaluated scope |
|---:|---:|---:|---:|---:|
| 160 ns | 0/3 | 3/3 | 3/3 | Medium |
| 180 ns | 0/3 | 3/3 | 2/3 | Medium |
| 200 ns | 0/3 | 3/3 | 2/3 | Medium |

Small은 특히 Ng32에서 ID-Hard reference보다 낮은 seed가 반복되어 joint criterion을 통과하지 못했다. Medium은 모든 delay의 세 seed에서 Ng16과 Ng32 reference를 함께 넘었다. Full이 180/200 ns에서 항상 통과하지 않은 사실도 확인되지만, fixed 3-epoch protocol에서의 관찰이며 원인은 본 실험에서 분석하지 않았다. 따라서 Partial FT가 Full보다 본질적으로 우수하거나 Full이 overfit된다고 해석하지 않는다.

![Figure 2. ID-Hard reference와 OOD BF gain](output/KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V3_figures/figure2_bf_gain_reference.png)

**Figure 2.** 점은 세 seed, 오차 막대는 mean±std이다. 점선은 seed-specific ID-Hard reference의 평균을 나타내며 실제 pass 판정은 seed별 reference를 사용한다. Ng16과 Ng32를 모두 확인하면 Small은 기준 아래에 남지만 Medium은 세 delay에서 처음으로 기준을 만족한다.

### 3.2 Partial Fine-Tuning의 계산 효율

3.1에서 Medium이 minimum evaluated sufficient scope로 확인되었으므로, Full 대신 Medium을 사용할 때의 실제 비용을 비교하였다.

| Scope | Parameters / ratio | Training time | Peak VRAM | Full 대비 감소 |
|---|---:|---:|---:|---|
| Small | 373,656 / 3.1625% | 125.44 s | 161.0 MiB | time 51.47%, VRAM 80.26% |
| Medium | 1,480,728 / 12.5323% | 139.80 s | 223.7 MiB | time 45.92%, VRAM 72.57% |
| Full | 11,815,320 / 100% | 258.48 s | 815.7 MiB | 기준 |

Medium은 Full 대비 trainable parameter 87.47%를 줄이고, measured adaptation training time 45.92%, peak allocated VRAM 72.57%를 줄였다. Small이 더 저렴하지만 3.1의 BF criterion을 만족하지 못했으므로, Medium은 비용이 가장 작은 scope가 아니라 성능 기준을 만족한 후보 중 가장 작은 scope이다.

### 3.3 OOD severity와 필요한 Fine-Tuning 범위

검증할 가설은 OOD severity가 증가할수록 더 큰 Fine-Tuning scope가 필요할 수 있다는 것이다. Pre-adaptation severity의 seed별 범위는 다음과 같다.

| OOD delay | Pre-Epistemic severity z | 상대적 severity | Required scope |
|---:|---:|---|---|
| 160 ns | 0.540385–0.568568 | 낮음 | Medium (3/3) |
| 180 ns | 0.650400–0.664611 | 중간 | Medium (3/3) |
| 200 ns | 0.738193–0.745868 | 높음 | Medium (3/3) |

세 delay 사이에서 severity는 160 ns < 180 ns < 200 ns 순서로 증가했다. 그러나 required scope는 모든 조건에서 Medium이었다. 따라서 현재 평가 범위에서는 Epistemic magnitude가 distribution shift의 정도를 반영했지만, 필요한 parameter budget과 단조롭게 연결된다는 근거는 확인되지 않았다. Required scope variation이 없으므로 Spearman correlation은 계산하지 않았고, 이 결과만으로 자동 scope selection policy를 주장하지 않는다.

![Figure 3. Severity와 required scope](output/KIPS_UACP_PARTIAL_FINETUNING_PAPER_DRAFT_V3_figures/figure3_severity_scope_unchanged.png)

**Figure 3.** 위쪽은 seed별 pre-adaptation z와 평균±표준편차, 아래쪽은 BF-only minimum evaluated scope를 나타낸다. Delay가 증가할수록 severity는 상승하지만 scope는 Medium으로 유지된다는 점을 한 그림에서 확인할 수 있다.

## 4. 결론

평가한 160–200 ns OOD 환경에서는 전체 predictor를 업데이트하지 않고 12.5323%를 업데이트하는 Medium scope만으로 ID-Hard 수준의 normalized beamforming gain 기준을 만족할 수 있었다. 세 delay와 세 독립 seed의 9개 조건에서 Medium이 모두 minimum evaluated sufficient scope였다.

Medium은 Full보다 trainable parameter 87.47%, measured training time 45.92%, peak allocated VRAM 72.57%를 줄였다. 따라서 현재 조건에서는 Full Fine-Tuning이 아닌 Partial Fine-Tuning으로 비용을 줄일 수 있는 사례를 확인하였다.

Pre-adaptation Epistemic severity는 delay 증가에 따라 높아졌지만 required scope는 모두 Medium이었다. 따라서 uncertainty magnitude만으로 필요한 adaptation budget을 결정하는 관계는 현재 평가 범위에서 확인되지 않았다. 이 결론은 2×2 MIMO, K=1024, 현재 channel simulation, 3-epoch protocol, Small/Medium/Full 후보에 한정된다. Normalized BF gain은 BER/BLER, throughput, 실제 QoS가 아닌 communication-aware surrogate이다. 향후에는 더 다양한 OOD 조건과 세분화된 update scope를 이용해 severity와 adaptation budget의 관계를 추가 검증할 예정이다.

## 참고문헌

[1] C.-K. Wen, W.-T. Shih, S. Jin, “Deep Learning for Massive MIMO CSI Feedback,” *IEEE Wireless Communications Letters*, vol. 7, no. 5, pp. 748–751, 2018, doi: 10.1109/LWC.2018.2818160.

[2] A. Amini, W. Schwarting, A. Soleimany, D. Rus, “Deep Evidential Regression,” *Advances in Neural Information Processing Systems 33*, 2020.

[3] Anonymous authors, “UACP: Uncertainty-Aware Channel Prediction in MIMO Networks,” submitted to *ACM MobiHoc ’26*, Tokyo, Japan, 2026. 공개 제출본에 저자·페이지·DOI가 없어 추정하지 않았다.

[4] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. V. Dillon, B. Lakshminarayanan, J. Snoek, “Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,” *Advances in Neural Information Processing Systems 32*, pp. 13969–13980, 2019.

[5] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attarian, S. Gelly, “Parameter-Efficient Transfer Learning for NLP,” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, pp. 2790–2799, 2019.

[6] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” *International Conference on Learning Representations*, 2022. 본 연구는 LoRA를 구현하지 않았으며 PEFT 배경으로만 인용한다.

'''


def build():
    make_figures()
    MD.write_text(paper_markdown(), encoding="utf-8")
    doc = Document()
    style_doc(doc)
    add_md_to_doc(doc, MD.read_text(encoding="utf-8"), ROOT)
    # Compact 2–3 page manuscript typography; source Markdown remains full-sized.
    doc.styles['Normal'].font.size = Pt(8.8)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.0
    doc.styles['Normal'].paragraph_format.space_after = Pt(2)
    doc.save(DOCX)
    print(MD)
    print(DOCX)
    print(FIGDIR)


if __name__ == "__main__":
    build()
