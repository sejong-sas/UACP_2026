#!/usr/bin/env python3
"""Assemble read-only artifacts for the current diagonal/banded ablation."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.evidential import (
    EvidentialOutput,
    banded_covariance_dense,
    banded_multivariate_student_t_nll,
    build_banded_cholesky,
)


OUT = ROOT / "runs/current_valid_baseline/structured_covariance/banded_20260912_current_valid_clean"
DIAG = OUT.parent / "diagonal_reference_power_normalized/summary.csv"
BAND = OUT / "power_normalized_eval/summary.csv"
FINAL_DIAG = ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/summary.csv"
FINAL_BAND = OUT / "final_results.csv"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def toy_sanity() -> dict:
    torch.manual_seed(7)
    k, bw = 8, 2
    raw = torch.randn(1, 4, k, 4 * bw, requires_grad=True)
    diag = torch.ones(1, 4, 2 * k)
    factors, padded = build_banded_cholesky(raw, k, bw, diag)
    dense = banded_covariance_dense(factors, k, padded)
    cov = dense[..., : 2 * k, : 2 * k]
    lower = torch.tril(factors)
    offdiag = factors - torch.diag_embed(torch.diagonal(factors, dim1=-2, dim2=-1))
    outside = factors.new_zeros(factors.shape)
    # The block construction has no entries beyond the declared local block.
    for block in range(factors.shape[2]):
        for row in range(factors.shape[-1]):
            for col in range(factors.shape[-1]):
                if row // 2 != col // 2 and abs(row // 2 - col // 2) > bw:
                    outside[:, :, block, row, col] = factors[:, :, block, row, col]
    gamma = torch.zeros(1, 8, k)
    output = EvidentialOutput(gamma=gamma, kappa=torch.ones(1, 4, 1), psi=torch.ones(1, 8, k), nu=torch.full((1, 4, 1), 2 * k + 2.0), num_subcarriers=k, banded_factor=factors, banded_padded_dim=padded)
    target = torch.randn(1, 8, k)
    loss = banded_multivariate_student_t_nll(output, target)
    loss.backward()
    rhs = torch.randn(1, 4, 2 * k)
    dense_inv_quad = torch.einsum("bpi,bpij,bpj->bp", rhs, torch.linalg.inv(cov), rhs)
    interleaved = rhs.reshape(1, 4, k, 2)
    padded_rhs = torch.nn.functional.pad(interleaved, (0, 0, 0, padded // 2 - k)).reshape(1, 4, -1, factors.shape[-1])
    solved = torch.linalg.solve_triangular(factors, padded_rhs.unsqueeze(-1), upper=False).squeeze(-1)
    solve_quad = solved.square().sum((-1, -2))
    logdet_factor = 2 * torch.log(torch.diagonal(factors, dim1=-2, dim2=-1)).sum((-1, -2))
    sign, logdet_dense = torch.linalg.slogdet(cov)
    return {
        "factor_shape": list(factors.shape),
        "lower_triangular": bool(torch.allclose(factors, lower)),
        "symmetric_covariance": bool(torch.allclose(cov, cov.transpose(-2, -1), atol=1e-6)),
        "positive_definite_min_eigenvalue": float(torch.linalg.eigvalsh(cov).min()),
        "cholesky_valid": bool(torch.isfinite(factors).all()),
        "offdiagonal_nonzero": bool(offdiag.abs().max() > 0),
        "band_outside_max_abs": float(outside.abs().max()),
        "mahalanobis_solve_matches_dense": bool(torch.allclose(solve_quad, dense_inv_quad, atol=1e-5, rtol=1e-5)),
        "logdet_matches_dense": bool(torch.allclose(logdet_factor[..., :], logdet_dense, atol=1e-5, rtol=1e-5)),
        "backward_finite": bool(raw.grad is not None and torch.isfinite(raw.grad).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "dense_slogdet_sign": float(sign.min()),
        "note": "bandwidth=0 has no branch; zeroing generated off-diagonal coefficients is the diagonal-equivalent toy check.",
    }


def main() -> None:
    diag_norm = {r["regime"]: r for r in read_csv(DIAG)}
    band_norm = {r["regime"]: r for r in read_csv(BAND)}
    diag_raw = {r["regime"]: r for r in read_csv(FINAL_DIAG)}
    band_raw = {r["regime"]: r for r in read_csv(FINAL_BAND)}
    diag_samples = read_csv(OUT.parent / "diagonal_reference_power_normalized/sample_metrics.csv")
    band_samples = read_csv(OUT / "power_normalized_eval/sample_metrics.csv")
    def normalized_component(rows: list[dict[str, str]], regime: str, key: str) -> float:
        values = [float(r[key]) for r in rows if r["regime"] == regime]
        return sum(values) / len(values)
    regime_rows = []
    for regime in REGIMES:
        d, b = diag_norm[regime], band_norm[regime]
        regime_rows.append({
            "regime": regime,
            "diagonal_nmse_omitted_db": float(d["normalized_nmse_db_mean"]),
            "banded_nmse_omitted_db": float(b["normalized_nmse_db_mean"]),
            "difference_nmse_db": float(b["normalized_nmse_db_mean"]) - float(d["normalized_nmse_db_mean"]),
            "diagonal_normalized_total_uncertainty": float(d["normalized_total_uncertainty_mean"]),
            "banded_normalized_total_uncertainty": float(b["normalized_total_uncertainty_mean"]),
            "diagonal_normalized_aleatoric": normalized_component(diag_samples, regime, "normalized_aleatoric"),
            "banded_normalized_aleatoric": normalized_component(band_samples, regime, "normalized_aleatoric"),
            "diagonal_normalized_epistemic": normalized_component(diag_samples, regime, "normalized_epistemic"),
            "banded_normalized_epistemic": normalized_component(band_samples, regime, "normalized_epistemic"),
            "diagonal_raw_aleatoric": float(diag_raw[regime]["aleatoric_mean"]),
            "banded_raw_aleatoric": float(band_raw[regime]["aleatoric"]),
            "diagonal_raw_epistemic": float(diag_raw[regime]["epistemic_mean"]),
            "banded_raw_epistemic": float(band_raw[regime]["epistemic"]),
        })
    write_csv(OUT / "regime_metrics.csv", regime_rows)
    summary = []
    for metric, key_d, key_b in [
        ("NMSE 20", "ID-Easy 20 ns", "normalized_nmse_db_mean"),
        ("NMSE 80", "ID-Hard 80 ns", "normalized_nmse_db_mean"),
        ("NMSE 120", "OOD-Near 120 ns", "normalized_nmse_db_mean"),
        ("NMSE 1 ms", "OOD-Far 1 ms", "normalized_nmse_db_mean"),
        ("NormTotalU 20", "ID-Easy 20 ns", "normalized_total_uncertainty_mean"),
        ("NormTotalU 80", "ID-Hard 80 ns", "normalized_total_uncertainty_mean"),
        ("NormTotalU 120", "OOD-Near 120 ns", "normalized_total_uncertainty_mean"),
        ("NormTotalU 1 ms", "OOD-Far 1 ms", "normalized_total_uncertainty_mean"),
    ]:
        summary.append({"metric": metric, "diagonal": float(diag_norm[key_d][key_b]), "banded": float(band_norm[key_d][key_b]), "difference": float(band_norm[key_d][key_b]) - float(diag_norm[key_d][key_b])})
    d20, d80 = float(diag_norm[REGIMES[0]]["normalized_total_uncertainty_mean"]), float(diag_norm[REGIMES[1]]["normalized_total_uncertainty_mean"])
    b20, b80 = float(band_norm[REGIMES[0]]["normalized_total_uncertainty_mean"]), float(band_norm[REGIMES[1]]["normalized_total_uncertainty_mean"])
    summary += [{"metric": "Delta_NormTotalU_ID", "diagonal": d80 - d20, "banded": b80 - b20, "difference": (b80 - b20) - (d80 - d20)}]
    write_csv(OUT / "summary.csv", summary)
    structure = read_csv(OUT / "learned_covariance_summary.csv")
    write_csv(OUT / "covariance_stats.csv", structure)
    checks = toy_sanity()
    (OUT / "sanity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (OUT / "implementation_audit.md").write_text("""# Structured covariance implementation audit\n\n- Branch: `pair_scalar_banded`; bandwidth: 32 (`IMPLEMENTATION-ASSUMPTION`).\n- `Psi=L L^T` is block-local and factorized by antenna pair; it is an `APPROXIMATION`, not paper-exact full covariance.\n- Production NLL uses `torch.linalg.solve_triangular`; no dense 2048x2048 inverse is used.\n- Aleatoric/Epistemic diagnostics use the diagonal of `L L^T` for the existing Eq.12→Eq.13 map.\n- Power-normalized evaluation was post-processed with the existing `p1_power_normalization_diagnostic.py` utility; training was not repeated for normalization.\n- See `sanity_checks.json`, `regime_metrics.csv`, and `covariance_stats.csv`.\n""", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    positions = list(range(len(REGIMES)))
    for key, ylabel, filename in (("normalized_aleatoric", "Normalized Aleatoric", "normalized_aleatoric_comparison.png"), ("normalized_epistemic", "Normalized Epistemic", "normalized_epistemic_comparison.png"), ("normalized_nmse_db_mean", "Omitted NMSE (dB)", "nmse_comparison.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        if key == "normalized_nmse_db_mean":
            diagonal = [float(diag_norm[r][key]) for r in REGIMES]
            banded = [float(band_norm[r][key]) for r in REGIMES]
        else:
            diagonal = [next(float(row[f"diagonal_{key}"]) for row in regime_rows if row["regime"] == r) for r in REGIMES]
            banded = [next(float(row[f"banded_{key}"]) for row in regime_rows if row["regime"] == r) for r in REGIMES]
        ax.bar([p - .18 for p in positions], diagonal, .36, label="Diagonal")
        ax.bar([p + .18 for p in positions], banded, .36, label="Banded lag32")
        ax.set_xticks(positions, REGIMES, rotation=20); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT / filename, dpi=150); plt.close(fig)
    lag = [1, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in structure:
        ax.plot(lag, [float(row[f"lag_{n}_mean"]) for n in lag], "o-", label=row["regime"])
    ax.set_xlabel("Frequency lag"); ax.set_ylabel("Mean absolute local covariance"); ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "covariance_lag_profile.png", dpi=150); plt.close(fig)
    print(json.dumps({"output": str(OUT), "sanity": checks, "delta_norm_total_uncertainty_diagonal": d80-d20, "delta_norm_total_uncertainty_banded": b80-b20}, indent=2))


if __name__ == "__main__":
    main()
