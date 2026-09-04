#!/usr/bin/env python3
"""논문 수식과 현재 evidential 구현의 의미가 같은지 검사한다.

실험 목적:
    active diagonal Student-t 경로의 식, reduction, parameter coupling,
    diagonal Psi 근사가 Aleatoric 의미에 미치는 영향을 읽기 전용으로 비교한다.

입력과 출력:
    current valid checkpoint와 대표 probe를 읽어 식 대응표, gradient 비교,
    parameter 범위, 작은 차원 reference 결과를 새 폴더에 저장한다.

주의:
    optimizer.step()과 checkpoint 수정은 절대 수행하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import (  # noqa: E402
    channels_to_pair_vectors,
    constrain_pair_scalar_evidential,
    diagonal_multivariate_student_t_nll,
    pair_level_evidence_regularizer,
    student_t_nll,
)
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402


BASELINE = ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected"
CONFIG_PATH = BASELINE / "config.json"
CHECKPOINT = BASELINE / "checkpoint_with_provenance.pt"
OUT = ROOT / "runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v3"
REGIMES = [
    ("ID-Easy 20 ns", "data/prototype/test_id_easy.npz"),
    ("ID-Hard 80 ns", "data/prototype/test_id_hard.npz"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)

    return pearson(ranks(x), ranks(y))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().float().reshape(-1).cpu().numpy()
    qs = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
    }


def model_raw_forward(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Replicate pair-scalar forward while retaining pre-softplus heads."""
    h = model.input_projection(x)
    for block in model.residual_blocks:
        h = block(h)
    pooled = model.pool(h)
    raw = {
        "gamma": model.gamma_head(h),
        "psi": model.psi_head(h),
        "kappa": model.kappa_head(pooled),
        "nu": model.nu_head(pooled),
    }
    output = constrain_pair_scalar_evidential(
        gamma_raw=raw["gamma"],
        psi_raw=raw["psi"],
        kappa_raw=raw["kappa"],
        nu_raw=raw["nu"],
        num_subcarriers=model.num_subcarriers,
    )
    return output, raw


def loss_components(output, target: torch.Tensor) -> dict[str, torch.Tensor]:
    active = diagonal_multivariate_student_t_nll(output, target)
    # The old elementwise implementation is retained in production for
    # historical compatibility, but is not active in the baseline config.
    elementwise = student_t_nll(output, target)
    residual = channels_to_pair_vectors(target - output.gamma)
    pair_reg = (residual.square().sum(-1, keepdim=True) * (output.kappa + output.nu)).squeeze(-1)
    return {"nll_pair": active, "nll_elementwise": elementwise, "reg_pair": pair_reg}


def grad_norm(value: torch.Tensor | None) -> float:
    return 0.0 if value is None else float(value.detach().norm().cpu())


def gradients_for(scalar: torch.Tensor, params: tuple[torch.Tensor, ...]) -> list[float]:
    grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return [grad_norm(g) for g in grads]


def reduction_audit(model: torch.nn.Module, probes: dict[str, tuple[torch.Tensor, torch.Tensor]], cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    reg_rows: list[dict] = []
    lambda_reg = float(cfg["paper_specified"]["lambda_reg"])
    for regime, (x, target) in probes.items():
        model.zero_grad(set_to_none=True)
        output, raw = model_raw_forward(model, x)
        components = loss_components(output, target)
        params = (model.gamma_head.weight, raw["psi"], raw["kappa"], raw["nu"])
        for variant, scalar in (
            ("paper_pair_sum", components["nll_pair"] * (x.shape[0] * 4)),
            ("current_pair_mean", components["nll_pair"]),
            ("elementwise_mean_reference", components["nll_elementwise"]),
        ):
            norms = gradients_for(scalar, params)
            rows.append({
                "regime": regime, "variant": variant, "scalar_nll": float(scalar.detach().cpu()),
                "gamma_head_grad_norm": norms[0], "raw_psi_grad_norm": norms[1],
                "raw_kappa_grad_norm": norms[2], "raw_nu_grad_norm": norms[3],
                "active_baseline_variant": variant == "current_pair_mean",
            })
        pair_reg = components["reg_pair"]
        for variant, scalar in (
            ("paper_pair_sum", pair_reg.sum()),
            ("current_pair_mean", pair_reg.mean()),
        ):
            norms = gradients_for(scalar, params)
            weighted = lambda_reg * scalar
            weighted_norms = gradients_for(weighted, params)
            nll_for_variant = components["nll_pair"] * (x.shape[0] * 4) if variant == "paper_pair_sum" else components["nll_pair"]
            reg_rows.append({
                "regime": regime, "variant": variant,
                "raw_l_reg": float(scalar.detach().cpu()),
                "lambda_reg": lambda_reg, "lambda_reg_x_l_reg": float(weighted.detach().cpu()),
                "raw_nll_magnitude": abs(float(nll_for_variant.detach().cpu())),
                "loss_ratio_abs_lambda_reg_over_nll": abs(float(weighted.detach().cpu())) / (abs(float(nll_for_variant.detach().cpu())) + 1e-12),
                "gamma_head_grad_norm": norms[0], "raw_psi_grad_norm": norms[1],
                "raw_kappa_grad_norm": norms[2], "raw_nu_grad_norm": norms[3],
                "weighted_gamma_grad_norm": weighted_norms[0], "weighted_psi_grad_norm": weighted_norms[1],
                "weighted_kappa_grad_norm": weighted_norms[2], "weighted_nu_grad_norm": weighted_norms[3],
                "active_baseline_variant": variant == "current_pair_mean",
            })
    return rows, reg_rows


def sensitivity_audit(model: torch.nn.Module, probes: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> list[dict]:
    rows: list[dict] = []
    for regime, (x, _) in probes.items():
        model.zero_grad(set_to_none=True)
        output, raw = model_raw_forward(model, x)
        psi_pair = channels_to_pair_vectors(output.psi)
        # Pair average of diagonal Eq.7 covariance, before Eq.12 aggregation.
        denom = output.nu - 2 * 1024 - 1
        ale_pair = psi_pair / denom
        ale_score = ale_pair.mean(dim=-1)
        auto_psi, auto_nu, auto_kappa = torch.autograd.grad(
            ale_score.sum(), (raw["psi"], raw["nu"], raw["kappa"]), retain_graph=True, allow_unused=True
        )
        raw_psi_pair = channels_to_pair_vectors(raw["psi"])
        raw_nu_pair = raw["nu"]
        dimension = float(2 * 1024)
        analytic_psi = torch.sigmoid(raw_psi_pair) / denom / dimension
        analytic_nu = -(psi_pair.mean(dim=-1, keepdim=True) * torch.sigmoid(raw_nu_pair)) / denom.square()
        rows.append({
            "regime": regime,
            "mean_abs_dA_d_rawPsi_autograd": float(auto_psi.detach().abs().mean().cpu()),
            "mean_abs_dA_d_rawNu_autograd": float(auto_nu.detach().abs().mean().cpu()),
            "mean_abs_dA_d_rawKappa_autograd": 0.0 if auto_kappa is None else float(auto_kappa.detach().abs().mean().cpu()),
            "mean_abs_dA_d_rawPsi_analytic": float(analytic_psi.detach().mean().cpu()),
            "mean_abs_dA_d_rawNu_analytic": float(analytic_nu.detach().abs().mean().cpu()),
            "max_abs_autograd_minus_analytic_psi": 0.0,
            "psi_to_nu_sensitivity_ratio": float(auto_psi.detach().abs().mean().cpu() / (auto_nu.detach().abs().mean().cpu() + 1e-12)),
        })
        # Explicit flattened comparison avoids relying on channel ordering in the report.
        auto_psi_pair = channels_to_pair_vectors(auto_psi.detach())
        rows[-1]["max_abs_autograd_minus_analytic_psi"] = float((auto_psi_pair - analytic_psi).abs().max().detach().cpu())
        auto_nu_pair = auto_nu.detach()
        rows[-1]["max_abs_autograd_minus_analytic_nu"] = float((auto_nu_pair - analytic_nu).abs().max().detach().cpu())
    return rows


def operating_ranges(model: torch.nn.Module, probes: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    raw_rows: list[dict] = []
    for regime, batches in probes.items():
        collected: dict[str, list[torch.Tensor]] = {k: [] for k in ("psi", "kappa", "nu", "denom", "df", "predictive_scale", "aleatoric", "epistemic", "raw_psi", "raw_kappa", "raw_nu")}
        for x, _ in batches:
            with torch.no_grad():
                output, raw = model_raw_forward(model, x)
            d = 2048
            degrees = output.nu - d + 1
            scale = ((output.kappa + 1) / (output.kappa * degrees)) * channels_to_pair_vectors(output.psi)
            collected["psi"].append(channels_to_pair_vectors(output.psi))
            collected["kappa"].append(output.kappa)
            collected["nu"].append(output.nu)
            collected["denom"].append(output.nu - d - 1)
            collected["df"].append(degrees)
            collected["predictive_scale"].append(scale)
            collected["aleatoric"].append(channels_to_pair_vectors(output.aleatoric))
            collected["epistemic"].append(channels_to_pair_vectors(output.epistemic))
            collected["raw_psi"].append(channels_to_pair_vectors(raw["psi"]))
            collected["raw_kappa"].append(raw["kappa"])
            collected["raw_nu"].append(raw["nu"])
        for quantity, values in collected.items():
            stat = tensor_stats(torch.cat(values, dim=0))
            rows.append({"regime": regime, "quantity": quantity, **stat})
            if quantity.startswith("raw_"):
                raw_rows.append({"regime": regime, "quantity": quantity, **stat})
    return rows, raw_rows


def full_student_t_nll(target: torch.Tensor, gamma: torch.Tensor, kappa: torch.Tensor, psi: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    d = target.numel()
    df = nu - d + 1
    factor = (kappa + 1) / (kappa * df)
    scale = factor * psi
    sign, logdet = torch.linalg.slogdet(scale)
    if sign <= 0:
        raise ValueError("Toy covariance is not positive definite")
    residual = target - gamma
    q = residual @ torch.linalg.solve(scale, residual)
    logp = torch.lgamma((df + d) / 2) - torch.lgamma(df / 2)
    logp = logp - 0.5 * (d * torch.log(df * torch.pi) + logdet)
    logp = logp - ((df + d) / 2) * torch.log1p(q / df)
    return -logp


def toy_audit() -> list[dict]:
    rows: list[dict] = []
    torch.manual_seed(731)
    for d in (4, 8):
        raw_diag = torch.log(torch.expm1(torch.ones(d))).requires_grad_()
        raw_nu_diag = torch.tensor(float(math.log(math.expm1(5.0))), requires_grad=True)
        raw_nu_full = raw_nu_diag.detach().clone().requires_grad_()
        gamma = torch.zeros(d)
        target = torch.linspace(-0.7, 1.1, d)
        diag = torch.diag(torch.nn.functional.softplus(raw_diag) + 1e-6)
        corr = torch.eye(d) * 0.7 + torch.ones(d, d) * 0.3
        full = corr * diag.diag().sqrt().unsqueeze(0) * diag.diag().sqrt().unsqueeze(1)
        kappa = torch.tensor(2.0)
        nll_diag = full_student_t_nll(target, gamma, kappa, diag, d + 1 + torch.nn.functional.softplus(raw_nu_diag) + 1e-6)
        nll_full = full_student_t_nll(target, gamma, kappa, full, d + 1 + torch.nn.functional.softplus(raw_nu_full) + 1e-6)
        grad_diag = torch.autograd.grad(nll_diag, (raw_diag, raw_nu_diag), retain_graph=True)
        grad_full = torch.autograd.grad(nll_full, (raw_diag, raw_nu_full), retain_graph=False)
        rows.extend([
            {"dimension": d, "case": "diagonal", "nll": float(nll_diag.detach()), "grad_raw_psi_norm": float(grad_diag[0].norm()), "grad_raw_nu_abs": float(grad_diag[1].abs()), "offdiag_correlation": 0.0},
            {"dimension": d, "case": "correlated_full_same_marginal", "nll": float(nll_full.detach()), "grad_raw_psi_norm": float(grad_full[0].norm()), "grad_raw_nu_abs": float(grad_full[1].abs()), "offdiag_correlation": 0.3},
        ])
    return rows


def equation_mapping() -> str:
    return """# Paper Eq.5~11 vs Current Evidential Implementation

This is a read-only semantic audit. No optimizer step or checkpoint write was performed.

## Status vocabulary

- **PAPER-SPECIFIED**: stated by the supplied paper text/equations.
- **IMPLEMENTATION-ASSUMPTION**: selected in this prototype where the paper does not specify details.
- **APPROXIMATION**: a deliberate computational simplification of the paper formulation.
- **UNKNOWN IN PAPER**: not exposed by the supplied paper.

## Mapping

| Item | Paper formulation | Current implementation | Status |
| --- | --- | --- | --- |
| Latent dimension | `d=2K`, one pair vector in `R^d` | `d=2048`, `K=1024` | PAPER-SPECIFIED / configuration fixed |
| `gamma_p` | pair CFR mean, `[d]` | channels `[8,K]`, converted to pair `[4,2K]` | PAPER-SPECIFIED semantics, layout is implementation |
| `kappa_p` | one scalar per antenna pair | `[B,4,1]`, softplus constrained | PAPER-SPECIFIED semantics |
| `Psi_p` | full `[d,d]`, `Psi=L L^T` | diagonal `[B,4,d]` | PAPER-SPECIFIED full structure; diagonal is APPROXIMATION |
| `nu_p` | one scalar per antenna pair, `nu>d+1` | `[B,4,1]`, `d+1+softplus(raw_nu)+eps` | PAPER-SPECIFIED semantics, transform implementation |
| Predictive df | `nu-d+1` | same in active diagonal NLL | MATCH |
| Predictive scale | `((kappa+1)/(kappa*df))*Psi` | same, diagonal specialization | MATCH under diagonal approximation |
| Eq.7 | `Sigma_ale=Psi/(nu-d-1)` | same diagonal marginal | MATCH under diagonal approximation |
| Eq.8 | `Sigma_epi=Sigma_ale/kappa` | same, pair scalar broadcast | MATCH |
| Eq.9 | positive covariance, `Psi=L L^T` | positive diagonal via softplus | APPROXIMATION |
| Eq.10 | pair predictive log-density aggregation | mean over batch and pairs | IMPLEMENTATION-ASSUMPTION / reduction difference |
| Eq.11 scope | pair error norm times `Phi=kappa+nu`, observed + omitted | full clean target, pair norm, pair evidence | MATCH in scope; reduction is assumption |
| Eq.11 reduction | sum notation in paper | mean over batch and pairs | IMPLEMENTATION-ASSUMPTION |

## Eq.5 expansion

For one pair, `r=h-gamma`, `S=((kappa+1)/(kappa*df))*Psi`, `df=nu-d+1`:

`log p(h) = lgamma((df+d)/2) - lgamma(df/2) - 0.5*(d*log(df*pi)+log|S|) - ((df+d)/2)*log(1 + r^T S^-1 r / df)`.

For diagonal `Psi`, `log|S| = sum_i log(S_i)` and `r^T S^-1 r = sum_i r_i^2/S_i`. The active code implements these as `scale_diag`, `torch.log(scale_diag).sum(dim=-1)`, and `torch.sum(residual.square()/scale_diag, dim=-1)` in `diagonal_multivariate_student_t_nll`. The returned value is the negative mean log density.

The Student-t scale `S` is not the covariance `Sigma_ale`. Eq.7 is the aleatoric covariance component and uses the different denominator `nu-d-1`.

## Reduction and semantic cautions

Paper sum versus current mean changes the absolute scalar and gradient magnitude by the number of batch/pair terms. If NLL and regularizer use the same reduction, their common factor cancels in a simple ratio; however, mixing reductions or using the inactive elementwise path changes effective balance. The active baseline uses diagonal multivariate NLL plus pair regularizer.

The pair regularizer contains `kappa+nu` but no `Psi`, so `d L_reg / d raw_Psi = 0` is intrinsic to this Eq.11 implementation, not a numerical accident. Aleatoric can therefore be changed by `Psi` through NLL and by `nu` through both NLL and regularization; `kappa` affects Epistemic directly but not Aleatoric directly.

The paper's full `Psi` can encode frequency-domain covariance in the quadratic and determinant. Diagonal `Psi` retains marginal variances but removes those off-diagonal terms. This is a semantic risk when delay spread changes frequency correlation, but the audit does not establish it as the sole cause.

## Paper unknowns and prototype assumptions

The supplied paper does not expose the exact network head layout, noise placement, TDL-A choice, 5k/10 epoch pilot scale, or reduction convention used here. Those are IMPLEMENTATION-ASSUMPTIONS, not paper claims.
"""


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUT}")
    start = time.perf_counter()
    cfg = load_config(CONFIG_PATH)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    requested = cfg["implementation_assumption"].get("device", "cuda:0")
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    checkpoint_sha = sha256(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload)
    model.eval()

    probes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    range_probes: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for index, (regime, relpath) in enumerate(REGIMES):
        dataset = CFRNPZDataset(ROOT / relpath)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        fixed_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        set_seeds(int(cfg["implementation_assumption"]["seed"]) + 71000 + index)
        for batch in loader:
            cfr = batch["cfr"].to(device)
            mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
            x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
            fixed_batches.append((x, target))
        probes[regime] = fixed_batches[0]
        range_probes[regime] = fixed_batches

    reduction_rows, reg_rows = reduction_audit(model, probes, cfg)
    sensitivity_rows = sensitivity_audit(model, probes)
    range_rows, raw_range_rows = operating_ranges(model, range_probes)
    toy_rows = toy_audit()

    paper_text = equation_mapping()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paper_code_mapping.md").write_text(paper_text, encoding="utf-8")
    write_csv(OUT / "reduction_gradient_comparison.csv", reduction_rows)
    write_csv(OUT / "regularizer_gradient_comparison.csv", reg_rows)
    write_csv(OUT / "parameter_sensitivity.csv", sensitivity_rows)
    write_csv(OUT / "parameter_operating_range.csv", range_rows)
    write_csv(OUT / "raw_parameter_operating_range.csv", raw_range_rows)
    write_csv(OUT / "toy_full_vs_diagonal.csv", toy_rows)

    for name, rows, xkey, ykey in (
        ("reduction_gradient_norms.png", reduction_rows, "variant", "raw_psi_grad_norm"),
        ("regularizer_gradient_norms.png", reg_rows, "variant", "weighted_nu_grad_norm"),
    ):
        plt.figure(figsize=(8, 4))
        labels = [f"{r['regime']}\n{r[xkey]}" for r in rows]
        vals = [float(r[ykey]) for r in rows]
        plt.bar(np.arange(len(vals)), vals)
        plt.xticks(np.arange(len(vals)), labels, rotation=35, ha="right", fontsize=7)
        plt.ylabel(ykey)
        plt.tight_layout()
        plt.savefig(OUT / name, dpi=160)
        plt.close()

    provenance = {
        "audit": "Paper Eq.5~11 vs Current Evidential Implementation Semantic Consistency Audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "optimizer_step_called": False,
        "baseline": str(BASELINE),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256_before": checkpoint_sha,
        "checkpoint_sha256_after": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG_PATH),
        "code_sha256": {p: sha256(ROOT / p) for p in ("src/models/evidential.py", "src/models/uacp_predictor.py", "src/training/data.py", "src/training/uncertainty.py", "scripts/evidential_formula_semantic_audit.py")},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False).stdout,
        "seed": cfg["implementation_assumption"]["seed"],
        "device_requested": requested,
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "finite_outputs": finite_tree({"reduction": reduction_rows, "regularizer": reg_rows, "sensitivity": sensitivity_rows, "ranges": range_rows, "toy": toy_rows}),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    audit_json = {
        "paper_source": "mobihoc26-paper289.pdf",
        "active_config": cfg["implementation_assumption"],
        "eq5_active_path": "diagonal_multivariate_student_t_nll",
        "eq11_active_path": "pair_level_evidence_regularizer",
        "reduction_rows": reduction_rows,
        "regularizer_rows": reg_rows,
        "sensitivity_rows": sensitivity_rows,
        "operating_range_rows": range_rows,
        "toy_rows": toy_rows,
        "findings": {
            "active_diagonal_formula": "algebraically consistent with diagonal specialization of Eq.5/7/8",
            "regularizer_raw_psi_gradient": "exactly zero by construction because Eq.11 implementation contains kappa+nu but no Psi",
            "diagonal_semantic_risk": "off-diagonal frequency covariance is not represented; existing empirical lag separation makes this a plausible risk, not a proof",
            "reduction_status": "current mean is an implementation assumption; numerical scale is reported in the tables",
        },
        "provenance": provenance,
    }
    (OUT / "equation_audit.json").write_text(json.dumps(audit_json, indent=2, sort_keys=True), encoding="utf-8")
    summary = [
        "# Evidential Formula / Semantic Consistency Audit",
        "",
        "Read-only audit of the current valid baseline. No training or optimizer step was performed.",
        "",
        f"- Device: `{device}`; GPU: `{provenance['gpu_name']}`",
        f"- Checkpoint SHA-256 preserved: `{provenance['checkpoint_sha256_before'] == provenance['checkpoint_sha256_after']}`",
        f"- All audit outputs finite: `{provenance['finite_outputs']}`",
        "",
        "## Main conclusions",
        "",
        "1. The active diagonal multivariate Student-t path matches the diagonal specialization of the paper's Eq.5, Eq.7, and Eq.8 algebraically.",
        "2. The current pair mean reduction is an implementation assumption relative to paper sum notation; exact values and gradient norms are in the CSV files.",
        "3. The active Eq.11 regularizer has zero direct raw-Psi gradient because its implemented factor is `(kappa+nu)`. Its direct pressure is on kappa/nu and gamma.",
        "4. Full-Psi frequency covariance is paper-specified, while diagonal Psi is an approximation. Existing empirical lag-correlation separation makes semantic loss of off-diagonal information plausible, but this audit does not prove causality.",
        "",
        "## Files",
        "",
        "- `paper_code_mapping.md`",
        "- `equation_audit.json`",
        "- `reduction_gradient_comparison.csv`",
        "- `regularizer_gradient_comparison.csv`",
        "- `parameter_sensitivity.csv`",
        "- `parameter_operating_range.csv`",
        "- `raw_parameter_operating_range.csv`",
        "- `toy_full_vs_diagonal.csv`",
    ]
    (OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "device": str(device), "gpu": provenance["gpu_name"], "finite": provenance["finite_outputs"], "seconds": provenance["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
