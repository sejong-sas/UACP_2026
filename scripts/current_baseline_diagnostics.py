#!/usr/bin/env python3
"""현재 valid baseline의 reconstruction과 parameter를 읽기 전용으로 진단한다.

실험 목적:
    Predictor를 쓰지 않는 linear interpolation과 antenna-pair evidential
    parameter trace를 같은 평가 조건에서 비교한다.

주의:
    baseline checkpoint와 기존 결과를 수정하지 않는다.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model, _mean_std  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors  # noqa: E402
from src.training.data import (  # noqa: E402
    CFRNPZDataset,
    build_noisy_sparse_input,
    linear_interpolate_real_imag,
    uniform_grouping_mask,
)
from src.training.metrics import nmse_all_db, nmse_omitted_db  # noqa: E402
from src.training.uncertainty import paper_omitted_uncertainty_score, paper_subcarrier_uncertainty_map  # noqa: E402


BASELINE = ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected"
DIAGNOSTICS_ROOT = ROOT / "runs/current_valid_baseline/diagnostics"
EXP1 = DIAGNOSTICS_ROOT / "exp1_linear_interpolation_no_predictor"
EXP2 = DIAGNOSTICS_ROOT / "exp2_evidential_parameter_trace"
CONFIG = BASELINE / "config.json"
SETUP = BASELINE / "evaluation_setup.json"
CHECKPOINT = BASELINE / "checkpoint_with_provenance.pt"
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_nmse_db(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pair: int) -> float:
    channels = torch.tensor([pair, pair + 4], device=pred.device)
    omitted = (1.0 - mask)[:, None, :]
    error = (pred.index_select(1, channels) - target.index_select(1, channels)).square() * omitted
    power = target.index_select(1, channels).square() * omitted
    ratio = error.sum(dim=(1, 2)) / power.sum(dim=(1, 2)).clamp_min(1e-12)
    return float((10.0 * torch.log10(ratio.mean().clamp_min(1e-12))).cpu())


def load_model(cfg: dict, device: torch.device):
    model = _make_model(cfg, device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_shared_probes(cfg: dict, device: torch.device) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]], dict]:
    setup = json.loads(SETUP.read_text(encoding="utf-8"))
    seed = int(cfg["implementation_assumption"]["seed"])
    probes = {}
    snr_summary = {}
    for index, regime in enumerate(REGIMES):
        set_seeds(seed + 70000 + index)
        dataset = CFRNPZDataset(setup["paths"][regime])
        batches = []
        snr_values = []
        loader = DataLoader(dataset, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
        for batch in loader:
            cfr = batch["cfr"].to(device)
            mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
            x, target, snr = build_noisy_sparse_input(cfr, mask, 15.0)
            batches.append((x, target, mask, snr))
            snr_values.append(float(snr["measured_snr_db_mean"]))
        probes[regime] = batches
        snr_summary[regime] = {"requested_snr_db": 15.0, "measured_snr_db_mean": sum(snr_values) / len(snr_values), "batches": len(batches)}
    return probes, snr_summary


def run_exp1(probes: dict, out: Path) -> dict:
    rows = []
    pair_rows = []
    for regime in REGIMES:
        all_error = []
        all_target = []
        pair_values = {pair: [] for pair in range(4)}
        for x, target, mask, _ in probes[regime]:
            pred = linear_interpolate_real_imag(x[:, :8], mask)
            all_error.append(pred)
            all_target.append(target)
            for pair in range(4):
                pair_values[pair].append(pair_nmse_db(pred, target, mask, pair))
        pred = torch.cat(all_error)
        target = torch.cat(all_target)
        mask = torch.cat([item[2] for item in probes[regime]])
        row = {"regime": regime, "nmse_all_db": float(nmse_all_db(pred, target).cpu()), "nmse_omitted_db": float(nmse_omitted_db(pred, target, mask).cpu())}
        for pair in range(4):
            row[f"pair{pair}_nmse_omitted_db"] = sum(pair_values[pair]) / len(pair_values[pair])
            pair_rows.append({"regime": regime, "pair": pair, "nmse_omitted_db": row[f"pair{pair}_nmse_omitted_db"]})
        rows.append(row)
    order = [row["nmse_omitted_db"] for row in rows]
    result = {"experiment": "Experiment 1 No-Predictor Linear Reconstruction", "method": "frequency-axis real/imag linear interpolation; edge extrapolation uses np.interp endpoint values", "rows": rows, "pair_rows": pair_rows, "hypothesis_h1": bool(order[0] < order[1] < order[2]), "shared_observation": True}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"baseline": str(BASELINE), "evaluation_setup": str(SETUP), "method": result["method"]}, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out / "overall_summary.csv", rows)
    write_csv(out / "pair_summary.csv", pair_rows)
    (out / "summary.md").write_text("# Experiment 1 - No-Predictor Linear Reconstruction\n\n" + "\n".join(f"- {r['regime']}: NMSE all={r['nmse_all_db']:.6f} dB, omitted={r['nmse_omitted_db']:.6f} dB" for r in rows) + f"\n\nH1 ordering PASS: `{result['hypothesis_h1']}`\n", encoding="utf-8")
    return result


def run_exp2(probes: dict, cfg: dict, out: Path, device: torch.device) -> dict:
    model = load_model(cfg, device)
    parameter_rows = []
    distribution_rows = []
    final_rows = []
    for regime in REGIMES:
        pair_acc = {pair: {key: [] for key in ("gamma_nmse", "psi_mean", "psi_median", "psi_trace", "psi_std", "kappa", "nu", "denom", "aleatoric", "epistemic")} for pair in range(4)}
        final_ale = []
        final_epi = []
        all_pred = []
        all_target = []
        all_mask = []
        err_values, ale_values, epi_values = [], [], []
        for x, target, mask, _ in probes[regime]:
            with torch.no_grad():
                output = model(x)
            gamma_pair = channels_to_pair_vectors(output.gamma)
            psi_pair = channels_to_pair_vectors(output.psi)
            denom = output.nu - 2 * 1024 - 1
            ale_pair = psi_pair / denom
            epi_pair = ale_pair / output.kappa
            omitted_pair = torch.cat((mask, mask), dim=1)[:, None, :].expand_as(ale_pair)
            pair_score_ale = ((ale_pair * omitted_pair).sum(dim=-1) / omitted_pair.sum(dim=-1).clamp_min(1.0))
            pair_score_epi = ((epi_pair * omitted_pair).sum(dim=-1) / omitted_pair.sum(dim=-1).clamp_min(1.0))
            final_ale.extend(pair_score_ale.detach().cpu().reshape(-1).tolist())
            final_epi.extend(pair_score_epi.detach().cpu().reshape(-1).tolist())
            final_ale.append(float(paper_omitted_uncertainty_score(output.aleatoric, mask).cpu()))
            final_epi.append(float(paper_omitted_uncertainty_score(output.epistemic, mask).cpu()))
            for pair in range(4):
                pair_error = (gamma_pair[:, pair] - channels_to_pair_vectors(target)[:, pair]).square()
                pair_omitted = omitted_pair[:, pair]
                pair_nmse = pair_error.mul(pair_omitted).sum(dim=1) / channels_to_pair_vectors(target)[:, pair].square().mul(pair_omitted).sum(dim=1).clamp_min(1e-12)
                for i in range(x.shape[0]):
                    psi_values = psi_pair[i, pair]
                    pair_acc[pair]["gamma_nmse"].append(float(10 * torch.log10(pair_nmse[i].clamp_min(1e-12)).cpu()))
                    pair_acc[pair]["psi_mean"].append(float(psi_values.mean().cpu()))
                    pair_acc[pair]["psi_median"].append(float(psi_values.median().cpu()))
                    pair_acc[pair]["psi_trace"].append(float(psi_values.sum().cpu()))
                    pair_acc[pair]["psi_std"].append(float(psi_values.std(unbiased=False).cpu()))
                    pair_acc[pair]["kappa"].append(float(output.kappa[i, pair, 0].cpu()))
                    pair_acc[pair]["nu"].append(float(output.nu[i, pair, 0].cpu()))
                    pair_acc[pair]["denom"].append(float(denom[i, pair, 0].cpu()))
                    pair_acc[pair]["aleatoric"].append(float(pair_score_ale[i, pair].cpu()))
                    pair_acc[pair]["epistemic"].append(float(pair_score_epi[i, pair].cpu()))
                    distribution_rows.append({"regime": regime, "pair": pair, **{key: pair_acc[pair][key][-1] for key in pair_acc[pair]}})
            all_pred.append(output.predicted.detach())
            all_target.append(target)
            all_mask.append(mask)
            omitted = (1.0 - mask)[:, None, :].expand_as(output.predicted)
            err_values.append((output.predicted - target).square()[omitted > 0.5].detach().cpu())
            ale_values.append(output.aleatoric[omitted > 0.5].detach().cpu())
            epi_values.append(output.epistemic[omitted > 0.5].detach().cpu())
        pred = torch.cat(all_pred); target = torch.cat(all_target); mask = torch.cat(all_mask)
        # Reuse the same validated helper on the final collected forward maps for the primary score.
        with torch.no_grad():
            maps_ale, maps_epi = [], []
            for x, _, m, _ in probes[regime]:
                o = model(x)
                maps_ale.append(o.aleatoric.detach())
                maps_epi.append(o.epistemic.detach())
            map_ale = torch.cat(maps_ale); map_epi = torch.cat(maps_epi)
        primary_ale = float(paper_omitted_uncertainty_score(map_ale, mask).cpu())
        primary_epi = float(paper_omitted_uncertainty_score(map_epi, mask).cpu())
        for pair in range(4):
            values = pair_acc[pair]
            parameter_rows.append({"regime": regime, "pair": pair, **{f"{key}_mean": sum(vals) / len(vals) for key, vals in values.items()}, **{f"{key}_std": float(torch.tensor(vals).std(unbiased=False)) for key, vals in values.items()}})
        final_rows.append({"regime": regime, "nmse_all_db": float(nmse_all_db(pred, target).cpu()), "nmse_omitted_db": float(nmse_omitted_db(pred, target, mask).cpu()), "u_ale_omitted": primary_ale, "u_epi_omitted": primary_epi, "pair_ale_mean": sum(final_ale[: len(final_ale) - len(probes[regime])]) / max(1, len(final_ale) - len(probes[regime])), "pair_epi_mean": sum(final_epi[: len(final_epi) - len(probes[regime])]) / max(1, len(final_epi) - len(probes[regime]))})
    idmax = max(final_rows[0]["u_epi_omitted"], final_rows[1]["u_epi_omitted"])
    result = {"experiment": "Experiment 2 Antenna-Pair Evidential Parameter Trace", "parameter_rows": parameter_rows, "final_rows": final_rows, "gaps": {"ale_gap_80_20": final_rows[1]["u_ale_omitted"] - final_rows[0]["u_ale_omitted"], "epi_gap_120_id": final_rows[2]["u_epi_omitted"] - idmax, "epi_gap_1ms_id": final_rows[3]["u_epi_omitted"] - idmax}, "paper_aggregation_helper": "paper_subcarrier_uncertainty_map -> paper_omitted_uncertainty_score", "shared_observation": True}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"baseline": str(BASELINE), "checkpoint": str(CHECKPOINT), "mapping": {f"pair{i}": [i, i + 4] for i in range(4)}, "psi": "diagonal approximation", "score": result["paper_aggregation_helper"]}, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out / "pair_parameter_summary.csv", parameter_rows)
    write_csv(out / "sample_pair_distributions.csv", distribution_rows)
    write_csv(out / "final_uncertainty_summary.csv", final_rows)
    lines = ["# Experiment 2 - Antenna-Pair Evidential Parameter Trace", "", "Psi is the current diagonal approximation, not the paper's full covariance.", "", "| Regime | NMSE omitted | U ale omitted | U epi omitted |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {r['regime']} | {r['nmse_omitted_db']:.6f} | {r['u_ale_omitted']:.9f} | {r['u_epi_omitted']:.9f} |" for r in final_rows)
    lines.extend(["", f"- Aleatoric gap 80-20: `{result['gaps']['ale_gap_80_20']:.9g}`", f"- Epistemic gap 120-ID: `{result['gaps']['epi_gap_120_id']:.9g}`", f"- Epistemic gap 1ms-ID: `{result['gaps']['epi_gap_1ms_id']:.9g}`"])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    if EXP1.exists() and (EXP1 / "results.json").exists():
        exp1 = json.loads((EXP1 / "results.json").read_text(encoding="utf-8"))
    elif EXP1.exists() and any(EXP1.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {EXP1}")
    else:
        exp1 = None
    if EXP2.exists() and any(EXP2.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {EXP2}")
    cfg = load_config(CONFIG)
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    probes, snr = make_shared_probes(cfg, device)
    if exp1 is None:
        exp1 = run_exp1(probes, EXP1)
    exp2 = run_exp2(probes, cfg, EXP2, device)
    DIAGNOSTICS_ROOT.mkdir(parents=True, exist_ok=True)
    (DIAGNOSTICS_ROOT / "shared_observation_manifest.json").write_text(json.dumps({"baseline": str(BASELINE), "seed": cfg["implementation_assumption"]["seed"], "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "snr": snr, "regimes": REGIMES, "ng": 16, "mask_indices": list(range(0, 1024, 16)), "note": "Exp1 and Exp2 consumed the same in-memory noisy sparse probes; no model training."}, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seconds": time.perf_counter() - start, "exp1_h1": exp1["hypothesis_h1"], "exp2_gaps": exp2["gaps"]}, indent=2))


if __name__ == "__main__":
    main()
