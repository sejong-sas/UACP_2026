#!/usr/bin/env python3
"""Read-only audit of the already-computed 100k Fig.8 scores.

No model forward pass is performed. The script compares the four requested
linear/dB aggregation variants from the saved sample-level scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/fig8_eval_retry/per_sample.csv"
OUTDIR = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/score_definition_audit"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def summarize(df: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for regime in REGIMES:
        x = df.loc[df.regime == regime, column].to_numpy(float)
        rows.append({"regime": regime, "mean": float(x.mean()), "median": float(np.median(x)),
                     "std": float(x.std()), "q05": float(np.quantile(x, .05)),
                     "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
                     "q95": float(np.quantile(x, .95)), "min": float(x.min()), "max": float(x.max())})
    return rows


def aucs(df: pd.DataFrame, column: str) -> dict:
    id_easy = df.loc[df.regime == REGIMES[0], column].to_numpy(float)
    id_hard = df.loc[df.regime == REGIMES[1], column].to_numpy(float)
    near = df.loc[df.regime == REGIMES[2], column].to_numpy(float)
    far = df.loc[df.regime == REGIMES[3], column].to_numpy(float)
    ident = np.concatenate([id_easy, id_hard])
    return {"id_vs_ood_pooled": auc(np.concatenate([near, far]), ident),
            "id_vs_near": auc(near, ident), "id_vs_far": auc(far, ident),
            "id_hard_80_vs_near_120": auc(near, id_hard)}


def main() -> None:
    df = pd.read_csv(INFILE)
    if len(df) != 40000 or set(df.regime) != set(REGIMES):
        raise RuntimeError(f"Unexpected saved Fig.8 rows: {len(df)}")
    linear = df.epistemic.to_numpy(float)
    df["A_sample_10log10"] = 10.0 * np.log10(np.maximum(linear, 1e-12))
    df["B_sample_20log10"] = 20.0 * np.log10(np.maximum(linear, 1e-12))
    # C is a regime-level linear mean followed by 10log10; it is not a
    # sample-level distribution, but is recorded to expose the distinction.
    regime_mean = df.groupby("regime").epistemic.transform("mean").to_numpy(float)
    df["C_regime_mean_10log10"] = 10.0 * np.log10(np.maximum(regime_mean, 1e-12))
    df["D_mean_of_sample_10log10"] = df["A_sample_10log10"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("A_sample_10log10", "B_sample_20log10", "C_regime_mean_10log10", "D_mean_of_sample_10log10"):
        for row in summarize(df, variant):
            row["variant"] = variant
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTDIR / "conversion_variants.csv", index=False)
    pd.DataFrame([{"variant": v, **aucs(df, v)} for v in ("A_sample_10log10", "B_sample_20log10", "C_regime_mean_10log10", "D_mean_of_sample_10log10")]).to_csv(OUTDIR / "conversion_auroc.csv", index=False)
    df.to_csv(OUTDIR / "sample_scores_with_conversion_variants.csv", index=False)
    result = {
        "input": str(INFILE), "rows": len(df), "samples_per_regime": 10000,
        "variants": ["A_sample_10log10", "B_sample_20log10", "C_regime_mean_10log10", "D_mean_of_sample_10log10"],
        "current_fig8_primary": "A_sample_10log10",
        "current_evaluator_formula": "reshape(B,2,4,K).sum(real_imag).mean(pair).masked_mean(omitted)",
        "paper_formula_mapping": {
            "eq7": "Sigma_ale = Psi / (nu - 2K - 1)",
            "eq8": "Sigma_epi = Sigma_ale / kappa",
            "eq12": "U_epi[k] = (1/(Nr Nt)) sum_(r,t) tr(P_k Sigma_epi_(r,t) P_k^T)",
            "eq13": "U_epi(Sbar) = (1/|Sbar|) sum_(k in Sbar) U_epi[k]",
        },
        "assumptions": [
            "IMPLEMENTATION-ASSUMPTION: paper does not state a Fig.8 dB conversion formula in the disclosed text; 10log10 is used because the score is variance/covariance trace-like.",
            "IMPLEMENTATION-ASSUMPTION: the diagonal-Psi implementation supplies only diagonal covariance entries; Eq.12 is evaluated as its diagonal specialization.",
        ],
        "source_checks": {
            "common_eval_rows": 40000,
            "mask_polarity": "omitted = 1 - mask",
            "omitted_only": True,
            "pair_mapping": "(0,4),(1,5),(2,6),(3,7)",
            "kappa_nu": "[B,4,1] pair-level scalars expanded over 2K channels",
            "dimension": "2K=2048; nu denominator nu-2K-1",
        },
    }
    (OUTDIR / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(rows).query("variant in ['A_sample_10log10','B_sample_20log10']").to_string(index=False))


if __name__ == "__main__":
    main()
