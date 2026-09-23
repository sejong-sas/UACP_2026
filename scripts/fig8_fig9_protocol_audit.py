#!/usr/bin/env python3
"""Read-only Fig.8/Fig.9 protocol sensitivity audit on the frozen 100k×1 model."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, t as student_t

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
COMMON = ROOT / "runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval"
FIG8_SAVED = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/fig8_eval_retry/per_sample.csv"
FIG9_AUDIT = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig9_audit_20260915_final"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]
NOMINALS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
PROTOCOLS = ["omitted_only", "all_subcarrier", "observed_only"]
K = 1024


def uncertainty_protocol_scores(epi: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Eq.(12) trace map followed by omitted/all/observed sample aggregation."""
    if epi.ndim != 3 or epi.shape[1] != 8 or mask.shape != (epi.shape[0], epi.shape[2]):
        raise ValueError("expected epistemic [B,8,K] and matching mask [B,K]")
    b = epi.shape[0]
    num_subcarriers = epi.shape[2]
    pair_map = epi.reshape(b, 2, 4, num_subcarriers).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)
    masks = {"omitted_only": 1.0 - mask, "all_subcarrier": torch.ones_like(mask), "observed_only": mask}
    result = {}
    for key, selected in masks.items():
        count = selected.sum(dim=1)
        result[key] = (pair_map * selected).sum(dim=1) / count.clamp_min(1.0)
    return result


def coverage_count(covered: torch.Tensor, mask: torch.Tensor) -> dict[str, tuple[int, int]]:
    """Count component-wise coverage over all, observed, and omitted CFR components."""
    if covered.ndim != 3 or covered.shape[1] != 8:
        raise ValueError("covered must have shape [B,8,K]")
    observed = mask[:, None, :].expand_as(covered).bool()
    all_components = torch.ones_like(observed)
    masks = {"all_subcarrier": all_components, "observed_only": observed,
             "omitted_only": ~observed}
    return {name: (int((covered & keep).sum().item()), int(keep.sum().item()))
            for name, keep in masks.items()}


def auc_from_scores(negative: np.ndarray, positive: np.ndarray) -> float:
    negative = np.asarray(negative, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    both = np.concatenate([negative, positive])
    ranks = rankdata(both, method="average")
    pos_rank_sum = ranks[len(negative):].sum()
    return float((pos_rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(negative) * len(positive)))


def pooled_coverage(items: list[tuple[int, int]]) -> tuple[int, int, float]:
    covered = sum(x[0] for x in items)
    count = sum(x[1] for x in items)
    return covered, count, covered / count


def read_saved_fig8() -> dict[str, np.ndarray]:
    out = {name: [] for name in REGIMES}
    with FIG8_SAVED.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["regime"] in out:
                out[row["regime"]].append(float(row["epistemic"]))
    return {key: np.asarray(values, dtype=np.float64) for key, values in out.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(device: torch.device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    model = _make_model(cfg, device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def read_regime_cfr(label: str, count: int) -> np.ndarray:
    path = COMMON / (label.replace(" ", "_").replace("/", "_") + ".npz")
    with np.load(path) as data:
        cfr = np.asarray(data["cfr"][:count], dtype=np.complex64)
    if len(cfr) != count:
        raise RuntimeError(f"{label}: expected {count}, found {len(cfr)}")
    return cfr


def fig8_scores(model, device: torch.device, count: int, batch_size: int) -> tuple[dict, dict]:
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
    scores = {name: {protocol: [] for protocol in PROTOCOLS} for name in REGIMES}
    max_abs_saved_difference = 0.0
    for ri, label in enumerate(REGIMES):
        cfr = read_regime_cfr(label, count)
        for start in range(0, count, batch_size):
            batch = torch.from_numpy(cfr[start:start + batch_size]).to(device)
            mask = uniform_grouping_mask(len(batch), K, 16, device)
            set_seeds(20261509 + ri * 100000 + start)
            x, _, _ = build_noisy_sparse_input(batch, mask, 15.0)
            with torch.inference_mode():
                output = model(x)
                batch_scores = uncertainty_protocol_scores(output.epistemic, mask)
                for protocol, value in batch_scores.items():
                    scores[label][protocol].extend(value.detach().cpu().numpy().astype(np.float64).tolist())
            del batch, mask, x, output, batch_scores
        del cfr
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    saved = read_saved_fig8()
    for label in REGIMES:
        current = np.asarray(scores[label]["omitted_only"])
        if len(current) != len(saved[label]):
            raise RuntimeError(f"saved Fig.8 count mismatch for {label}")
        max_abs_saved_difference = max(max_abs_saved_difference, float(np.max(np.abs(current - saved[label]))))
    result_rows = []
    aurocs = {}
    for protocol in PROTOCOLS:
        values = {name: np.asarray(scores[name][protocol], dtype=np.float64) for name in REGIMES}
        id_pool = np.concatenate([values[REGIMES[0]], values[REGIMES[1]]])
        near, far = values[REGIMES[2]], values[REGIMES[3]]
        pooled_ood = np.concatenate([near, far])
        comparisons = {
            "ID_pooled_vs_Near": (id_pool, near),
            "ID_pooled_vs_Far": (id_pool, far),
            "ID_pooled_vs_OOD_pooled": (id_pool, pooled_ood),
            "ID_Hard_80_vs_Near_120": (values[REGIMES[1]], near),
            "ID_Easy_20_vs_Near_120": (values[REGIMES[0]], near),
        }
        for name, (negative, positive) in comparisons.items():
            auc_linear = auc_from_scores(negative, positive)
            auc_db = auc_from_scores(10 * np.log10(negative), 10 * np.log10(positive))
            result_rows.append({"protocol": protocol, "comparison": name,
                                "n_negative": len(negative), "n_positive": len(positive),
                                "auroc_linear": auc_linear, "auroc_db": auc_db,
                                "linear_db_auc_abs_difference": abs(auc_linear - auc_db)})
            aurocs[(protocol, name)] = auc_linear
    baseline = read_saved_fig8()
    id_saved = np.concatenate([baseline[REGIMES[0]], baseline[REGIMES[1]]])
    saved_near = baseline[REGIMES[2]]
    saved_far = baseline[REGIMES[3]]
    saved_expected = auc_from_scores(id_saved, saved_near)
    saved_expected_far = auc_from_scores(id_saved, saved_far)
    return result_rows, {"omitted_max_abs_difference_vs_saved_fig8": max_abs_saved_difference,
                         "saved_artifact_near_auroc_recomputed": saved_expected,
                         "saved_artifact_far_auroc_recomputed": saved_expected_far,
                         "score_mean_by_protocol_regime": {p: {r: float(np.mean(scores[r][p])) for r in REGIMES} for p in PROTOCOLS}}


def fig9_coverage(model, device: torch.device, count: int, batch_size: int) -> list[dict]:
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
    counts = {(regime, protocol, float(nominal)): [0, 0]
              for regime in REGIMES for protocol in PROTOCOLS for nominal in NOMINALS}
    probs = np.asarray([(1.0 + nominal) / 2.0 for nominal in NOMINALS], dtype=np.float64)[:, None, None, None]
    for ri, label in enumerate(REGIMES):
        cfr = read_regime_cfr(label, count)
        for start in range(0, count, batch_size):
            batch = torch.from_numpy(cfr[start:start + batch_size]).to(device)
            mask = uniform_grouping_mask(len(batch), K, 16, device)
            set_seeds(20262000 + ri * 100000 + start)
            x, target, _ = build_noisy_sparse_input(batch, mask, 15.0)
            with torch.inference_mode():
                output = model(x)
                df_pair = output.nu - 2 * K + 1.0
                scale_sq = ((output.kappa_expanded + 1.0) / (output.kappa_expanded * output.nu_expanded.sub(2 * K).add(1.0))) * output.psi
                scale = torch.sqrt(scale_sq.clamp_min(1e-12))
                df_np = df_pair.detach().cpu().numpy()
                q_np = student_t.ppf(probs, df_np[None, ...])
                q_pair = torch.as_tensor(q_np, dtype=scale.dtype, device=device)
                q = torch.cat((q_pair.expand(-1, -1, -1, K), q_pair.expand(-1, -1, -1, K)), dim=2)
                covered = (target - output.gamma).abs().unsqueeze(0) <= q * scale.unsqueeze(0)
                keep = torch.stack((torch.ones_like(mask), mask, 1.0 - mask), dim=0).bool()
                covered_counts = (covered.unsqueeze(1) & keep[None, :, :, None, :]).sum(dim=(2, 3, 4)).detach().cpu().numpy()
                component_counts = keep[:, :, None, :].expand(-1, -1, 8, -1).sum(dim=(1, 2, 3)).detach().cpu().numpy()
                for ni, nominal in enumerate(NOMINALS):
                    for pi, protocol in enumerate(("all_subcarrier", "observed_only", "omitted_only")):
                        item = counts[(label, protocol, float(nominal))]
                        item[0] += int(covered_counts[ni, pi])
                        item[1] += int(component_counts[pi])
                del df_np, q_np, q_pair, q, covered, keep, covered_counts, component_counts
            del batch, mask, x, target, output, df_pair, scale_sq, scale
        del cfr
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_rows = []
    pools = {"ID": REGIMES[:2], "OOD-Near": [REGIMES[2]], "OOD-Far": [REGIMES[3]],
             "OOD-pooled": REGIMES[2:]}
    for protocol in PROTOCOLS:
        for pool, regimes in pools.items():
            errors = []
            for nominal in NOMINALS:
                covered, total = 0, 0
                for regime in regimes:
                    n, d = counts[(regime, protocol, float(nominal))]
                    covered += n
                    total += d
                empirical = covered / total
                errors.append(abs(empirical - nominal))
                out_rows.append({"protocol": protocol, "pool": pool, "nominal": nominal,
                                 "covered": covered, "component_count": total,
                                 "empirical": empirical, "abs_error": abs(empirical - nominal),
                                 "ce_mae_full_grid": ""})
            ce = float(np.mean(errors))
            for row in out_rows:
                if row["protocol"] == protocol and row["pool"] == pool:
                    row["ce_mae_full_grid"] = ce
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-regime", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not all(p.exists() for p in [CHECKPOINT, FIG8_SAVED, FIG9_AUDIT / "calibration_curve.csv", FIG9_AUDIT / "results.json"]):
        raise FileNotFoundError("canonical checkpoint or prior audit artifacts missing")
    sys.path.insert(0, str(ROOT))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = load_model(device)
    started = time.perf_counter()
    fig8_rows, fig8_check = fig8_scores(model, device, args.samples_per_regime, args.batch_size)
    fig9_rows = fig9_coverage(model, device, args.samples_per_regime, args.batch_size)
    write_csv(out / "fig8_protocol_comparison.csv", fig8_rows)
    write_csv(out / "fig9_protocol_comparison.csv", fig9_rows)

    prior_curve = []
    with (FIG9_AUDIT / "calibration_curve.csv").open(newline="", encoding="utf-8") as f:
        prior_curve = list(csv.DictReader(f))
    prior_id = {(float(r["nominal"]), r["pool"]): float(r["empirical"]) for r in prior_curve}
    current_omitted = {(float(r["nominal"]), r["pool"]): float(r["empirical"])
                       for r in fig9_rows if r["protocol"] == "omitted_only"}
    pool_alias = {"OOD": "OOD-pooled"}
    comparison_diffs = {f"{nominal:.2f}_{pool}": abs(current_omitted[(nominal, pool_alias.get(pool, pool))] - value)
                        for (nominal, pool), value in prior_id.items()
                        if (nominal, pool_alias.get(pool, pool)) in current_omitted
                        and pool in {"ID", "OOD", "OOD-Near", "OOD-Far"}}
    prior_results = json.loads((FIG9_AUDIT / "results.json").read_text(encoding="utf-8"))
    ce_table = {r["pool"]: {"legacy_q10_q90": float(r["ce_mae_0.1_0.9"]), "all_grid": float(r["ce_mae_all_grid"])}
                for r in prior_results["ce"]}
    fig8_by_protocol = {}
    for row in fig8_rows:
        fig8_by_protocol.setdefault(row["protocol"], {})[row["comparison"]] = row["auroc_linear"]
    epi_reference = fig8_by_protocol["omitted_only"]
    fig9_summary = {}
    for protocol in PROTOCOLS:
        rows = [r for r in fig9_rows if r["protocol"] == protocol]
        fig9_summary[protocol] = {
            pool: {"ce_mae_full_grid": next(float(r["ce_mae_full_grid"]) for r in rows if r["pool"] == pool),
                   "coverage_at_0.9": next(float(r["empirical"]) for r in rows if r["pool"] == pool and float(r["nominal"]) == .9),
                   "above_ideal_count": sum(float(r["empirical"]) > float(r["nominal"]) for r in rows if r["pool"] == pool)}
            for pool in ("ID", "OOD-Near", "OOD-Far", "OOD-pooled")}
    result = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "common_eval_dir": str(COMMON.relative_to(ROOT)),
        "samples_per_regime": args.samples_per_regime, "batch_size": args.batch_size,
        "training_performed": False, "device": str(device), "gpu": gpu,
        "fig8": {"score_formula": "current output.epistemic = (Psi/(nu-2K-1))/kappa; Eq.(12) Re/Imag pair trace and four-pair mean; aggregation by omitted/all/observed subcarriers",
                 "aggregation_status": {"omitted_only": "paper Eq.(13) current primary", "all_subcarrier": "RESEARCH-DIAGNOSTIC", "observed_only": "RESEARCH-DIAGNOSTIC"},
                 "pooling": "ID is equal CFR counts from 20+80 ns; OOD pooled is equal CFR counts from 120 ns+1ms; comparisons use sample-level scores (IMPLEMENTATION-ASSUMPTION: paper exact pooling unpublished)",
                 "results": fig8_rows, "checks": fig8_check,
                 "near_auc_range_across_aggregations": [min(fig8_by_protocol[p]["ID_pooled_vs_Near"] for p in PROTOCOLS), max(fig8_by_protocol[p]["ID_pooled_vs_Near"] for p in PROTOCOLS)],
                 "omitted_only_reference_auroc": epi_reference},
        "fig9": {"formula": "df=nu-2K+1; scale²=((kappa+1)/(kappa*df))*Psi; scale=sqrt(scale²); q=t_df^-1((1+c)/2); interval=gamma±q*scale",
                 "student_t_synthetic_sanity_pass": json.loads((FIG9_AUDIT / "synthetic_student_t_sanity.json").read_text())["pass"],
                 "coverage_protocols": {"omitted_only": "canonical existing evaluator", "all_subcarrier": "RESEARCH-DIAGNOSTIC", "observed_only": "RESEARCH-DIAGNOSTIC"},
                 "pooled_ood": "count-weighted 120ns+1ms; because each has equal 10k samples and each mask has equal components this is equal-regime pooling; exact paper mixture unpublished",
                 "results_summary": fig9_summary, "coverage_rows": len(fig9_rows),
                 "max_abs_difference_omitted_vs_existing_audit": max(comparison_diffs.values()),
                 "differences_by_nominal_pool": comparison_diffs,
                 "existing_audit_ce": ce_table},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2),
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "output": str(out.relative_to(ROOT)),
    }
    (out / "protocol_audit_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "fig8_near_by_aggregation": {p: fig8_by_protocol[p]["ID_pooled_vs_Near"] for p in PROTOCOLS},
                      "fig9_cov90": {p: {pool: fig9_summary[p][pool]["coverage_at_0.9"] for pool in fig9_summary[p]} for p in PROTOCOLS},
                      "audit_max_difference": result["fig9"]["max_abs_difference_omitted_vs_existing_audit"],
                      "device": str(device), "gpu": gpu, "peak_rss_gib": result["peak_rss_gib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
