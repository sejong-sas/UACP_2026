#!/usr/bin/env python3
"""현재 valid baseline의 sample/pair calibration과 covariance를 검사한다.

실험 목적:
    sample별 error와 evidence 관계, 그리고 clean CFR의 주파수 상관 구조가
    diagonal Psi가 놓치는 정보인지 확인한다.

주의:
    학습 없이 평가만 수행하며 checkpoint를 덮어쓰지 않는다.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402
from src.training.metrics import nmse_all_db, nmse_omitted_db  # noqa: E402
from src.training.uncertainty import paper_omitted_uncertainty_score  # noqa: E402


ROOT_BASELINE = ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected"
OUT_ROOT = ROOT / "runs/current_valid_baseline/diagnostics"
EXP_A = OUT_ROOT / "expA_sample_pair_evidence_calibration"
EXP_B = OUT_ROOT / "expB_diagonal_psi_covariance_diagnostic"
CONFIG_PATH = ROOT_BASELINE / "config.json"
SETUP_PATH = ROOT_BASELINE / "evaluation_setup.json"
CHECKPOINT_PATH = ROOT_BASELINE / "checkpoint_with_provenance.pt"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]
PAIR_NAMES = ["Pair0", "Pair1", "Pair2", "Pair3"]
LAGS = [1, 2, 4, 8, 16, 32]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr(np.argsort(np.argsort(np.asarray(x).reshape(-1))), np.argsort(np.argsort(np.asarray(y).reshape(-1))))


def load_model(cfg: dict, device: torch.device):
    model = _make_model(cfg, device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def shared_probes(cfg: dict, device: torch.device) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], dict]:
    setup = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
    seed = int(cfg["implementation_assumption"]["seed"])
    probes = {}
    manifest = {"seed_by_regime": {}, "paths": setup["paths"], "ng": 16, "noise_snr_db": 15.0, "same_in_memory_probes_for_exp_a_and_model_trace": True}
    for index, regime in enumerate(REGIMES):
        set_seeds(seed + 81000 + index)
        manifest["seed_by_regime"][regime] = seed + 81000 + index
        loader = DataLoader(CFRNPZDataset(setup["paths"][regime]), batch_size=16)
        batches = []
        for batch in loader:
            cfr = batch["cfr"].to(device)
            mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
            x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
            batches.append((x, target, mask))
        probes[regime] = batches
    return probes, manifest


def error_bin_rows(records: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    result = []
    groups = sorted({tuple(row[key] for key in group_keys) for row in records})
    for group in groups:
        selected = [row for row in records if tuple(row[key] for key in group_keys) == group]
        error = np.array([row["error_omitted_sq"] for row in selected])
        boundaries = np.quantile(error, [1 / 3, 2 / 3])
        for bin_index, chosen in enumerate((error <= boundaries[0], (error > boundaries[0]) & (error <= boundaries[1]), error > boundaries[1])):
            rows = [row for row, keep in zip(selected, chosen) if keep]
            result.append({**{key: value for key, value in zip(group_keys, group)}, "error_bin": bin_index, "count": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in ("error_omitted_sq", "error_full_sq", "kappa", "nu", "evidence", "denom", "psi_mean", "aleatoric_score", "epistemic_score")}})
    return result


def run_experiment_a(probes: dict, cfg: dict, out: Path, device: torch.device) -> dict:
    model = load_model(cfg, device)
    records = []
    final_rows = []
    for regime in REGIMES:
        pred_parts, target_parts, mask_parts = [], [], []
        regime_records = []
        for x, target, mask in probes[regime]:
            with torch.no_grad():
                output = model(x)
            target_pair = channels_to_pair_vectors(target)
            gamma_pair = channels_to_pair_vectors(output.gamma)
            psi_pair = channels_to_pair_vectors(output.psi)
            denom = output.nu - 2 * 1024 - 1
            ale_pair = psi_pair / denom
            epi_pair = ale_pair / output.kappa
            omitted_pair = torch.cat((mask, mask), dim=1)[:, None, :].expand_as(ale_pair)
            for sample in range(x.shape[0]):
                for pair in range(4):
                    residual = target_pair[sample, pair] - gamma_pair[sample, pair]
                    omitted = omitted_pair[sample, pair]
                    record = {
                        "regime": regime,
                        "pair": pair,
                        "error_full_sq": float(residual.square().sum().cpu()),
                        "error_omitted_sq": float((residual.square() * omitted).sum().cpu()),
                        "kappa": float(output.kappa[sample, pair, 0].cpu()),
                        "nu": float(output.nu[sample, pair, 0].cpu()),
                        "evidence": float((output.kappa[sample, pair, 0] + output.nu[sample, pair, 0]).cpu()),
                        "denom": float(denom[sample, pair, 0].cpu()),
                        "psi_mean": float(psi_pair[sample, pair].mean().cpu()),
                        "psi_median": float(psi_pair[sample, pair].median().cpu()),
                        "psi_trace": float(psi_pair[sample, pair].sum().cpu()),
                        "aleatoric_score": float((ale_pair[sample, pair] * omitted).sum().div(omitted.sum().clamp_min(1.0)).cpu()),
                        "epistemic_score": float((epi_pair[sample, pair] * omitted).sum().div(omitted.sum().clamp_min(1.0)).cpu()),
                    }
                    records.append(record)
                    regime_records.append(record)
            pred_parts.append(output.predicted.detach()); target_parts.append(target); mask_parts.append(mask)
        pred = torch.cat(pred_parts); target = torch.cat(target_parts); mask = torch.cat(mask_parts)
        with torch.no_grad():
            # Re-run only the model on the same cached inputs to obtain the validated Eq.12->13 score.
            ale_maps, epi_maps = [], []
            for x, _, _ in probes[regime]:
                output = model(x)
                ale_maps.append(output.aleatoric.detach()); epi_maps.append(output.epistemic.detach())
            ale_map = torch.cat(ale_maps); epi_map = torch.cat(epi_maps)
        final_rows.append({
            "regime": regime,
            "nmse_all_db": float(nmse_all_db(pred, target).cpu()),
            "nmse_omitted_db": float(nmse_omitted_db(pred, target, mask).cpu()),
            "u_ale_omitted": float(paper_omitted_uncertainty_score(ale_map, mask).cpu()),
            "u_epi_omitted": float(paper_omitted_uncertainty_score(epi_map, mask).cpu()),
        })
    corr_rows = []
    for regime in REGIMES:
        for pair in range(4):
            selected = [row for row in records if row["regime"] == regime and row["pair"] == pair]
            selected_all = [row for row in records if row["regime"] == regime]
            for scope, values in (("pair", selected), ("all_pairs", selected_all)):
                error = np.array([row["error_omitted_sq"] for row in values])
                for name in ("kappa", "nu", "evidence", "denom", "aleatoric_score", "epistemic_score"):
                    corr_rows.append({"regime": regime, "pair": pair if scope == "pair" else "all", "scope": scope, "variable": name, "pearson": corr(error, np.array([row[name] for row in values])), "spearman": spearman(error, np.array([row[name] for row in values]))})
                if scope == "all_pairs":
                    break
    bins = error_bin_rows(records, ("regime",)) + error_bin_rows(records, ("regime", "pair"))
    result = {"experiment": "A Sample / Pair-level Evidence Calibration", "baseline": str(ROOT_BASELINE), "primary_error": "omitted pair-vector squared error", "records": len(records), "final_rows": final_rows, "correlations": corr_rows, "gaps": {"ale_gap_80_20": final_rows[1]["u_ale_omitted"] - final_rows[0]["u_ale_omitted"], "epi_gap_120_id": final_rows[2]["u_epi_omitted"] - max(final_rows[0]["u_epi_omitted"], final_rows[1]["u_epi_omitted"]), "epi_gap_1ms_id": final_rows[3]["u_epi_omitted"] - max(final_rows[0]["u_epi_omitted"], final_rows[1]["u_epi_omitted"])}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"baseline": str(ROOT_BASELINE), "checkpoint": str(CHECKPOINT_PATH), "fixed_setup": str(SETUP_PATH), "psi": "diagonal approximation", "pair_mapping": {f"pair{i}": [i, i + 4] for i in range(4)}}, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out / "sample_pair_records.csv", records)
    write_csv(out / "correlations.csv", corr_rows)
    write_csv(out / "error_bin_summary.csv", bins)
    write_csv(out / "final_summary.csv", final_rows)
    make_a_plots(out, records)
    (out / "summary.md").write_text("# Experiment A - Sample / Pair Evidence Calibration\n\n" + "\n".join(f"- {row['regime']}: NMSE omitted={row['nmse_omitted_db']:.6f} dB, U_ale={row['u_ale_omitted']:.9f}, U_epi={row['u_epi_omitted']:.9f}" for row in final_rows) + f"\n\n- Ale80-20: `{result['gaps']['ale_gap_80_20']:.9g}`\n- Epi120-ID: `{result['gaps']['epi_gap_120_id']:.9g}`\n- Epi1ms-ID: `{result['gaps']['epi_gap_1ms_id']:.9g}`\n", encoding="utf-8")
    return result


def make_a_plots(out: Path, records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for regime in REGIMES:
        selected = [row for row in records if row["regime"] == regime]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, key, title in zip(axes, ("evidence", "aleatoric_score", "epistemic_score"), ("Evidence", "Aleatoric", "Epistemic")):
            for pair in range(4):
                rows = [row for row in selected if row["pair"] == pair]
                axis.scatter([row["error_omitted_sq"] for row in rows], [row[key] for row in rows], s=8, alpha=.45, label=f"P{pair}")
            axis.set_xlabel("Omitted pair error squared"); axis.set_ylabel(title); axis.grid(alpha=.2)
        axes[0].legend(); fig.suptitle(regime); fig.tight_layout(); fig.savefig(out / f"{regime.replace(' ', '_').replace('/', '_')}_scatter.png", dpi=140); plt.close(fig)
    grouped = error_bin_rows(records, ("regime",))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, title in zip(axes, ("evidence", "aleatoric_score", "epistemic_score"), ("Evidence", "Aleatoric", "Epistemic")):
        for index, regime in enumerate(REGIMES):
            rows = [row for row in grouped if row["regime"] == regime]
            axis.plot([0, 1, 2], [row[key] for row in rows], marker="o", label=regime)
        axis.set_title(title); axis.set_xlabel("Error tertile"); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "error_bin_trends.png", dpi=140); plt.close(fig)


def run_experiment_b(cfg: dict, out: Path) -> dict:
    setup = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
    model_device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, model_device)
    energy_rows, lag_rows, learned_rows = [], [], []
    for regime in REGIMES:
        with np.load(setup["paths"][regime], allow_pickle=False) as data:
            cfr = data["cfr"]
        learned_psi = [[] for _ in range(4)]
        # Use the same deterministic noisy probe protocol to trace learned diagonal Psi.
        set_seeds(int(cfg["implementation_assumption"]["seed"]) + 81000 + REGIMES.index(regime))
        loader = DataLoader(CFRNPZDataset(setup["paths"][regime]), batch_size=16)
        with torch.no_grad():
            for batch in loader:
                cfr_t = batch["cfr"].to(model_device); mask = uniform_grouping_mask(cfr_t.shape[0], 1024, 16, model_device)
                from src.training.data import build_noisy_sparse_input
                x, _, _ = build_noisy_sparse_input(cfr_t, mask, 15.0)
                output = model(x); psi_pair = channels_to_pair_vectors(output.psi)
                for pair in range(4): learned_psi[pair].extend(psi_pair[:, pair].mean(dim=-1).cpu().tolist())
        for pair in range(4):
            series = np.stack([cfr[:, :, pair // 2, pair % 2].real, cfr[:, :, pair // 2, pair % 2].imag], axis=1)
            vector = np.concatenate((series[:, 0], series[:, 1]), axis=1).astype(np.float64)
            centered = vector - vector.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(1, vector.shape[0] - 1)
            diagonal = np.diag(covariance)
            diag_energy = float(np.square(diagonal).sum())
            total_energy = float(np.square(covariance).sum())
            off_energy = total_energy - diag_energy
            energy_rows.append({"regime": regime, "pair": pair, "diag_energy": diag_energy, "offdiag_energy": off_energy, "total_energy": total_energy, "offdiag_total_ratio": off_energy / max(total_energy, 1e-12), "diag_variance_sum": float(diagonal.sum()), "learned_psi_mean": float(np.mean(learned_psi[pair]))})
            complex_series = cfr[:, :, pair // 2, pair % 2].astype(np.complex128)
            complex_centered = complex_series - complex_series.mean(axis=0, keepdims=True)
            variances = np.mean(np.abs(complex_centered) ** 2, axis=0)
            for lag in LAGS:
                cross = np.mean(complex_centered[:, :-lag] * np.conj(complex_centered[:, lag:]), axis=0)
                denom = np.sqrt(variances[:-lag] * variances[lag:]).clip(min=1e-15)
                lag_rows.append({"regime": regime, "pair": pair, "lag": lag, "correlation_magnitude": float(np.mean(np.abs(cross / denom))), "correlation_real": float(np.mean(np.real(cross / denom)))})
        for lag in LAGS:
            values = [row["correlation_magnitude"] for row in lag_rows if row["regime"] == regime and row["lag"] == lag]
            # Pair-average is the same lag statistic used in the plotted summary.
            learned_rows.append({"regime": regime, "lag": lag, "pair_averaged_correlation_magnitude": float(np.mean(values))})
    result = {"experiment": "B Diagonal Psi Approximation Diagnostic", "covariance_definition": "sample covariance of [Re(H pair, all K), Im(H pair, all K)] across samples", "complex_lag_definition": "mean over k of abs(E[(H_k-mean) conjugate(H_k+lag-mean)] / sqrt(var_k var_k+lag))", "energy_rows": energy_rows, "lag_rows": lag_rows, "pair_averaged_lag_rows": learned_rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"baseline": str(ROOT_BASELINE), "checkpoint": str(CHECKPOINT_PATH), "fixed_setup": str(SETUP_PATH), "psi": "learned diagonal approximation", "empirical_covariance": "clean ground-truth CFR; no predictor/noise"}, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out / "covariance_energy.csv", energy_rows); write_csv(out / "lag_correlation.csv", lag_rows); write_csv(out / "pair_averaged_lag_correlation.csv", learned_rows)
    make_b_plots(out, energy_rows, lag_rows, learned_rows)
    (out / "summary.md").write_text("# Experiment B - Diagonal Psi Covariance Diagnostic\n\n" + "Empirical covariance uses clean CFR across samples; complex lag correlation uses normalized complex covariance magnitude averaged over frequency.\n\n" + "\n".join(f"- {r['regime']} Pair{r['pair']}: off/total={r['offdiag_total_ratio']:.6f}, learned Psi mean={r['learned_psi_mean']:.6f}" for r in energy_rows) + "\n", encoding="utf-8")
    return result


def make_b_plots(out: Path, energy_rows: list[dict], lag_rows: list[dict], mean_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime in REGIMES:
        rows = [row for row in mean_rows if row["regime"] == regime]
        ax.plot([row["lag"] for row in rows], [row["pair_averaged_correlation_magnitude"] for row in rows], marker="o", label=regime)
    ax.set_xlabel("Frequency lag"); ax.set_ylabel("Mean normalized complex correlation magnitude"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "correlation_decay_pair_averaged.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for pair, axis in enumerate(axes.flat):
        for regime in REGIMES:
            rows = [row for row in lag_rows if row["regime"] == regime and row["pair"] == pair]
            axis.plot([row["lag"] for row in rows], [row["correlation_magnitude"] for row in rows], marker="o", label=regime)
        axis.set_title(f"Pair {pair}"); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "correlation_decay_by_pair.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(REGIMES)); width=.18
    for pair in range(4):
        vals = [next(row["offdiag_total_ratio"] for row in energy_rows if row["regime"] == regime and row["pair"] == pair) for regime in REGIMES]
        ax.bar(positions + (pair - 1.5) * width, vals, width, label=f"Pair {pair}")
    ax.set_xticks(positions, REGIMES, rotation=20); ax.set_ylabel("Off-diagonal / total covariance energy"); ax.legend(); fig.tight_layout(); fig.savefig(out / "offdiag_ratio_by_regime.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime in REGIMES:
        rows = [row for row in energy_rows if row["regime"] == regime]
        ax.scatter([row["offdiag_total_ratio"] for row in rows], [row["learned_psi_mean"] for row in rows], label=regime, s=45)
    ax.set_xlabel("Empirical off-diagonal / total energy"); ax.set_ylabel("Learned diagonal Psi mean"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out / "learned_psi_vs_offdiag_ratio.png", dpi=140); plt.close(fig)


def main() -> None:
    for path in (EXP_A, EXP_B):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    cfg = load_config(CONFIG_PATH)
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    probes, manifest = shared_probes(cfg, device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "shared_observation_manifest_ab.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    exp_a = run_experiment_a(probes, cfg, EXP_A, device)
    exp_b = run_experiment_b(cfg, EXP_B)
    (OUT_ROOT / "ab_run_summary.json").write_text(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seconds": time.perf_counter() - start, "exp_a_gaps": exp_a["gaps"], "exp_b_energy_rows": len(exp_b["energy_rows"])}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seconds": time.perf_counter() - start, "exp_a_gaps": exp_a["gaps"], "exp_b_energy_rows": len(exp_b["energy_rows"])}, indent=2))


if __name__ == "__main__":
    main()
