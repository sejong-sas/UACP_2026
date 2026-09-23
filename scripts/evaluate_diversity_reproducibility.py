#!/usr/bin/env python3
"""Common 10k-evaluation and seed reproducibility audit for diversity runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.channel.sionna_channel import generate_cfr_batch, load_sectioned_config  # noqa: E402
from src.training.data import build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]


def bootstrap(values: np.ndarray, seed: int, fn, draws: int = 2000) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        out[i] = fn(values[rng.integers(0, values.shape[0], values.shape[0])])
    return {"mean": float(out.mean()), "std": float(out.std(ddof=1)), "ci95_low": float(np.quantile(out, .025)), "ci95_high": float(np.quantile(out, .975)), "draws": draws, "seed": seed}


def make_common_eval(channel_cfg: dict, out: Path, count: int, batch_size: int, seed: int) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for index, (label, delay) in enumerate(REGIMES):
        chunks = []
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            _, cfr = generate_cfr_batch(channel_cfg, delay, n, seed + index * 100_000 + start)
            chunks.append(cfr.cpu().numpy().astype(np.complex64, copy=False))
        path = out / (label.replace(" ", "_").replace("/", "_") + ".npz")
        np.savez(path, cfr=np.concatenate(chunks, axis=0), delay_spread_ns=np.full(count, delay, dtype=np.float32), metadata_json=np.array(json.dumps({"common_eval": True, "samples": count, "seed": seed, "regime": label, "assumption": "IMPLEMENTATION-ASSUMPTION: TDL-A/zero mobility/normalize=false"})))
        paths[label] = str(path)
    return paths


@torch.no_grad()
def evaluate_model(model, paths: dict[str, str], device: torch.device, count: int, batch_size: int, eval_seed: int, model_name: str) -> list[dict[str, float | int | str]]:
    rows = []
    for ri, (label, _) in enumerate(REGIMES):
        cfr_np = np.load(paths[label])["cfr"]
        for start in range(0, count, batch_size):
            cfr = torch.from_numpy(cfr_np[start:start + batch_size]).to(device)
            n = cfr.shape[0]
            mask = uniform_grouping_mask(n, 1024, 16, device)
            set_seeds(eval_seed + ri * 100_000 + start)
            x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
            output = model(x)
            omitted = (1.0 - mask)
            omitted_count = omitted.sum(dim=1).clamp_min(1.0)
            ale_map = output.aleatoric.reshape(n, 2, 4, 1024).sum(dim=2).mean(dim=1)
            epi_map = output.epistemic.reshape(n, 2, 4, 1024).sum(dim=2).mean(dim=1)
            ale = ((ale_map * omitted).sum(dim=1) / omitted_count).cpu().numpy()
            epi = ((epi_map * omitted).sum(dim=1) / omitted_count).cpu().numpy()
            error = (output.predicted - target).square()
            err_om = (error * omitted[:, None, :]).reshape(n, -1).sum(dim=1)
            tar_om = (target.square() * omitted[:, None, :]).reshape(n, -1).sum(dim=1)
            nmse = (10 * torch.log10((err_om / tar_om.clamp_min(1e-12)).clamp_min(1e-12))).cpu().numpy()
            for i in range(n):
                rows.append({"model": model_name, "regime": label, "sample": start + i, "nmse_omitted_db": float(nmse[i]), "aleatoric": float(ale[i]), "epistemic": float(epi[i])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/current_valid_baseline_condition_c.json")
    parser.add_argument("--channel-config", default="configs/dataset_prototype.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-regime", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--common-eval-dir", default=None)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config); channel_cfg = load_sectioned_config(ROOT / args.channel_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_dir = ROOT / args.common_eval_dir if args.common_eval_dir else out / "common_eval"
    paths = make_common_eval(channel_cfg, common_dir, args.samples_per_regime, args.batch_size, seed=20260914) if not common_dir.exists() else {label: str(common_dir / (label.replace(" ", "_").replace("/", "_") + ".npz")) for label, _ in REGIMES}
    models = [
        ("A 5k x 10", cfg, ROOT / "runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/model_state_dict.pt"),
        ("B1 50k x 1 seed20260819", cfg, ROOT / "runs/current_valid_baseline/diversity_ablation/unique_50k_1ep/uacp_predictor_step4a.pt"),
        ("B2 50k x 1 seed20260915", load_config("configs/current_valid_baseline_seed_20260915.json"), ROOT / "runs/current_valid_baseline/diversity_ablation/seed_20260915/uacp_predictor_step4a.pt"),
        ("B3 50k x 1 seed20260916", load_config("configs/current_valid_baseline_seed_20260916.json"), ROOT / "runs/current_valid_baslation/seed_20260916/uacp_predictor_step4a.pt"),
    ]
    # Correct the explicit B3 path while keeping the table above easy to audit.
    models[-1] = (models[-1][0], models[-1][1], ROOT / "runs/current_valid_baseline/diversity_ablation/seed_20260916/uacp_predictor_step4a.pt")
    rows = []
    for name, model_cfg, checkpoint in models:
        model = _make_model(model_cfg, device); payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload); model.eval()
        rows.extend(evaluate_model(model, paths, device, args.samples_per_regime, args.batch_size, 20261001, name))
    summaries = []
    for name in [m[0] for m in models]:
        for label, _ in REGIMES:
            subset = [r for r in rows if r["model"] == name and r["regime"] == label]
            summaries.append({"model": name, "regime": label, "samples": len(subset), **{
                f"{key}_{stat}": float((np.median(values) if stat == "median" else getattr(values, stat)()))
                for key in ("nmse_omitted_db", "aleatoric", "epistemic")
                for stat, values in ((stat, np.asarray([r[key] for r in subset])) for stat in ("mean", "median", "std"))
            }})
    by={(r["model"],r["regime"]):r for r in summaries}; gaps=[]; b_names=[m[0] for m in models][1:]
    for name in [m[0] for m in models]:
        a20=np.asarray([r["aleatoric"] for r in rows if r["model"]==name and r["regime"]=="ID-Easy 20 ns"]); a80=np.asarray([r["aleatoric"] for r in rows if r["model"]==name and r["regime"]=="ID-Hard 80 ns"])
        e20=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="ID-Easy 20 ns"]); e80=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="ID-Hard 80 ns"]); en=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="OOD-Near 120 ns"]); ef=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="OOD-Far 1 ms"])
        id_mean=max(e20.mean(),e80.mean()); gaps.append({"model":name,"delta_ale_80_minus_20":float(a80.mean()-a20.mean()),"near_epi_gap_vs_max_id":float(en.mean()-id_mean),"far_epi_gap_vs_max_id":float(ef.mean()-id_mean),"nmse_80_gt_20":by[(name,"ID-Hard 80 ns")]["nmse_omitted_db_mean"]>by[(name,"ID-Easy 20 ns")]["nmse_omitted_db_mean"]})
    boot={}
    for name in [m[0] for m in models]:
        a20=np.asarray([r["aleatoric"] for r in rows if r["model"]==name and r["regime"]=="ID-Easy 20 ns"]); a80=np.asarray([r["aleatoric"] for r in rows if r["model"]==name and r["regime"]=="ID-Hard 80 ns"]); e20=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="ID-Easy 20 ns"]); e80=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="ID-Hard 80 ns"]); en=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="OOD-Near 120 ns"]); ef=np.asarray([r["epistemic"] for r in rows if r["model"]==name and r["regime"]=="OOD-Far 1 ms"])
        boot[name]={"delta_ale":bootstrap(np.column_stack([a20,a80]),20261010,lambda z:z[:,1].mean()-z[:,0].mean()),"near_gap":bootstrap(np.column_stack([e20,e80,en]),20261011,lambda z:z[:,2].mean()-max(z[:,0].mean(),z[:,1].mean())),"far_gap":bootstrap(np.column_stack([e20,e80,ef]),20261012,lambda z:z[:,2].mean()-max(z[:,0].mean(),z[:,1].mean()))}
    for fn,data in [("per_sample.csv",rows),("summary.csv",summaries),("gaps.csv",gaps)]:
        with (out/fn).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    result={"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"samples_per_regime":args.samples_per_regime,"common_eval_paths":paths,"summaries":summaries,"gaps":gaps,"bootstrap":boot,"training_seeds":{"B1":20260819,"B2":20260915,"B3":20260916},"updates":{"A":6250,"B1":6250,"B2":6250,"B3":6250}}
    (out/"results.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
