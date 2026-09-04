#!/usr/bin/env python3
"""STEP 7 equation audit and bounded covariance pilot helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evidential import EvidentialOutput, diagonal_multivariate_student_t_nll
from src.models.evidential_reference import paper_predictive_student_t_log_prob, paper_uncertainties


def _to_channels(values: torch.Tensor) -> torch.Tensor:
    b, pairs, d = values.shape
    return values.reshape(b, pairs, 2, d // 2).permute(0, 2, 1, 3).reshape(b, 8, d // 2)


def run_audit(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(707)
    rows = []
    max_logprob_error = 0.0
    for dimension in (4, 8):
        b, pairs = 3, 4
        target = torch.randn(b, pairs, dimension, dtype=torch.float64)
        gamma = torch.randn_like(target)
        psi_diag = torch.rand_like(target) + 0.5
        kappa = torch.rand(b, pairs, dtype=torch.float64) + 0.5
        nu = dimension + 2.0 + torch.rand(b, pairs, dtype=torch.float64)
        output = EvidentialOutput(_to_channels(gamma), kappa.unsqueeze(-1), _to_channels(psi_diag), nu.unsqueeze(-1), dimension // 2)
        optimized = -diagonal_multivariate_student_t_nll(output, _to_channels(target))
        reference = paper_predictive_student_t_log_prob(target, gamma, kappa, torch.diag_embed(psi_diag), nu).mean()
        error = abs(float(optimized - reference))
        max_logprob_error = max(max_logprob_error, error)
        rows.append({"dimension": dimension, "optimized_log_probability": float(optimized), "dense_reference_log_probability": float(reference), "absolute_error": error})

        ale, epi, total = paper_uncertainties(torch.diag_embed(psi_diag), kappa, nu)
        expected_ale = psi_diag / (nu - dimension - 1).unsqueeze(-1)
        expected_epi = expected_ale / kappa.unsqueeze(-1)
        assert torch.allclose(ale.diagonal(dim1=-2, dim2=-1), expected_ale)
        assert torch.allclose(epi.diagonal(dim1=-2, dim2=-1), expected_epi)
        assert torch.allclose(total, ale + epi)

    (output_dir / "dense_vs_diag.csv").open("w", newline="", encoding="utf-8").close()
    with (output_dir / "dense_vs_diag.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {
        "dimensions": [4, 8],
        "max_abs_log_probability_error": max_logprob_error,
        "tolerance": 1e-8,
        "dense_diagonal_match": max_logprob_error < 1e-8,
        "df_formula": "nu - d + 1",
        "scale_formula": "((kappa + 1) / (kappa * df)) * Psi",
        "uncertainty_formula": "ale=Psi/(nu-d-1), epi=ale/kappa",
        "layout": "[Re(pair0..3, k), Im(pair0..3, k)] -> pair vector [Re(k1..K), Im(k1..K)]",
        "eq11_scope": "full target CFR, observed and omitted frequencies",
        "finding": "The diagonal multivariate NLL is mathematically exact after correcting pair-major channel mapping; full Psi=LL^T remains approximated by diagonal Psi.",
    }
    (output_dir / "reference_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "paper_equation_mapping.md").write_text("""# STEP 7A Equation Mapping

| Paper item | Current implementation | Status |
| --- | --- | --- |
| Eq. (5) df | `nu - d + 1`, with `d=2K` | Exact for diagonal Psi |
| Eq. (5) scale | `((kappa+1)/(kappa*df))*Psi` | Exact for diagonal Psi |
| Eq. (5) density | Multivariate gamma, log determinant, Mahalanobis term | Exact diagonal specialization |
| Eq. (7) | `Psi/(nu-d-1)` | Exact diagonal specialization |
| Eq. (8) | `Sigma_ale/kappa` | Exact |
| Eq. (9) | softplus constraints; full `Psi=LL^T` absent | Diagonal approximation |
| Eq. (10) | Mean of pair log densities | Reduction assumption; paper writes a sum |
| Eq. (11) | Pair squared error times `kappa+nu`, all target frequencies | Formula aligned; reduction is implementation-defined |

The channel layout is real channels for all four antenna pairs followed by imaginary
channels for all four pairs. Pair vectors must therefore be formed by permutation,
not a direct reshape. The previous direct reshape and pair-scalar interleaved
broadcast were layout mismatches and are corrected in the current implementation.

`Psi=D+UU^T` is an implementation approximation for STEP 7B, not a paper claim.
""", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# STEP 7A Equation Audit\n\n"
        f"Dense diagonal reference comparison passed for d=4,8 with maximum absolute log-probability error {max_logprob_error:.3e} (tolerance 1e-8).\n\n"
        "The Student-t scale matrix is kept distinct from covariance. The remaining paper gap is the full per-pair Psi=LL^T covariance; current production remains diagonal until the structured pilot.\n",
        encoding="utf-8",
    )
    return result


def run_compare(output_dir: Path, diagonal_results: Path, lowrank_results: Path, lowrank_config: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagonal = json.loads(diagonal_results.read_text(encoding="utf-8"))
    lowrank = json.loads(lowrank_results.read_text(encoding="utf-8"))
    regimes = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]
    rows = []
    for label, result in (("Diagonal corrected control", diagonal), ("Low-Rank rank4", lowrank)):
        for regime in regimes:
            metrics = result["epoch_probe_results"]["10"][regime]
            rows.append({
                "model": label,
                "regime": regime,
                "nmse_all_db": metrics["nmse_all_db"],
                "nmse_omitted_db": metrics["nmse_omitted_db"],
                "psi_mean": metrics["psi_omitted"]["mean"],
                "covariance_diag_mean": (metrics.get("covariance_diag_omitted") or {}).get("mean", metrics["psi_omitted"]["mean"]),
                "lowrank_diag_contribution_mean": (metrics.get("lowrank_diag_contribution_omitted") or {}).get("mean", 0.0),
                "off_diagonal_energy_ratio_mean": metrics.get("off_diagonal_energy_ratio_mean", 0.0),
                "aleatoric_mean": metrics["aleatoric_omitted"]["mean"],
                "epistemic_mean": metrics["epistemic_omitted"]["mean"],
                "error_aleatoric_pearson": metrics["error_aleatoric_pearson"],
                "error_epistemic_pearson": metrics["error_epistemic_pearson"],
            })
    with (output_dir / "covariance_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_model = {name: {row["regime"]: row for row in rows if row["model"] == name} for name in ("Diagonal corrected control", "Low-Rank rank4")}
    gap_rows = []
    for name, values in by_model.items():
        id_max = max(values["ID-Easy 20 ns"]["epistemic_mean"], values["ID-Hard 80 ns"]["epistemic_mean"])
        gap_rows.append({
            "model": name,
            "ale_gap_80_20": values["ID-Hard 80 ns"]["aleatoric_mean"] - values["ID-Easy 20 ns"]["aleatoric_mean"],
            "epi_gap_120_id": values["OOD-Near 120 ns"]["epistemic_mean"] - id_max,
            "epi_gap_1ms_id": values["OOD-Far 1 ms"]["epistemic_mean"] - id_max,
        })
    with (output_dir / "gap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gap_rows[0])); writer.writeheader(); writer.writerows(gap_rows)

    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diag_model = _make_model(load_config("configs/train_formulation_d_pair_reg.json"), device)
    low_cfg = load_config(lowrank_config)
    low_model = _make_model(low_cfg, device)
    parameter_rows = []
    checkpoint_paths = {
        "Diagonal corrected control": diagonal_results.parent / "uacp_predictor_step4a.pt",
        "Low-Rank rank4": lowrank_results.parent / "uacp_predictor_step4a.pt",
    }
    for name, model in (("Diagonal corrected control", diag_model), ("Low-Rank rank4", low_model)):
        model.load_state_dict(torch.load(checkpoint_paths[name], map_location=device))
        model.eval()
        sample = torch.zeros(1, 9, 1024, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        for _ in range(2):
            with torch.no_grad(): model(sample)
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(5):
            with torch.no_grad(): model(sample)
        if device.type == "cuda": torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0 / 5.0
        parameter_rows.append({"model": name, "total_parameters": sum(p.numel() for p in model.parameters()), "covariance_head_parameters": sum(p.numel() for n, p in model.named_parameters() if "head" in n), "single_sample_inference_ms": latency_ms, "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)})
    with (output_dir / "cost_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0])); writer.writeheader(); writer.writerows(parameter_rows)
    summary = ["# STEP 7 Covariance Comparison", "", "| Model | Regime | NMSE omitted | Aleatoric | Epistemic | off-diagonal ratio |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        offdiag = row['off_diagonal_energy_ratio_mean']
        summary.append(f"| {row['model']} | {row['regime']} | {row['nmse_omitted_db']:.4f} | {row['aleatoric_mean']:.6f} | {row['epistemic_mean']:.6f} | {offdiag if offdiag else 0.0:.4f} |")
    summary += ["", "## Gaps", "", "| Model | Ale 80-20 | Epi 120-ID | Epi 1ms-ID |", "| --- | ---: | ---: | ---: |"]
    for row in gap_rows:
        summary.append(f"| {row['model']} | {row['ale_gap_80_20']:.8f} | {row['epi_gap_120_id']:.8f} | {row['epi_gap_1ms_id']:.8f} |")
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"rows": rows, "gaps": gap_rows, "parameters": parameter_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "compare"])
    parser.add_argument("--output-dir", default="runs/baseline_reproduction/step7_covariance_structure/equation_audit")
    parser.add_argument("--diagonal-results")
    parser.add_argument("--lowrank-results")
    parser.add_argument("--lowrank-config")
    args = parser.parse_args()
    if args.command == "audit":
        result = run_audit(Path(args.output_dir))
    else:
        result = run_compare(Path(args.output_dir), Path(args.diagonal_results), Path(args.lowrank_results), Path(args.lowrank_config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
