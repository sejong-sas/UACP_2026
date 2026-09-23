"""Sample-level AUROC supplement for anchored-last4 epoch-10 checkpoints."""
from pathlib import Path
import argparse, csv, sys
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K = 1024
SEEDS = (20260921, 20260922, 20260923)
REGIMES = ("20ns", "80ns", "120ns", "1ms")
NGS = (16, 32)

def state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj

def auc(pos, neg):
    scores = np.r_[neg, pos]
    labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and ordered[j] == ordered[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pr = ranks[labels == 1]
    return float((pr.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    from scripts.train_predictor import load_config
    from scripts.diagnose_predictor import _make_model
    from scripts.partial_ft_adapt import make_adaptation_observation
    from src.training.data import CFRNPZDataset
    cfg = load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json")
    dev = torch.device("cuda:0")
    data = ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data"
    paths = {"20ns": data/"test_id_easy.npz", "80ns": data/"test_id_hard.npz", "120ns": data/"test_ood_near.npz", "1ms": data/"test_ood_far.npz"}
    rows = []
    for seed in SEEDS:
        cp = ROOT / args.training_root / f"train_seed_{seed}" / "adapted_epoch_10.pt"
        model = _make_model(cfg, dev)
        model.load_state_dict(state(cp)); model.eval()
        for regime in REGIMES:
            for ng in NGS:
                loader = DataLoader(CFRNPZDataset(paths[regime]), batch_size=args.batch_size, shuffle=False)
                for bi, batch in enumerate(loader):
                    cfr = batch["cfr"].to(dev)
                    mask = torch.zeros((cfr.shape[0], K), device=dev); mask[:, ::ng] = 1
                    x, target, _ = make_adaptation_observation(cfr, mask, 20260921 + ng * 1000, dev, epoch=0, batch_index=bi)
                    with torch.inference_mode():
                        o = model(x)
                        margin = o.nu_expanded - 2*K - 1
                        epi = o.psi / margin / o.kappa_expanded
                        score = epi.mean(dim=(1, 2)).detach().cpu().numpy()
                    for i, value in enumerate(score):
                        rows.append({"seed": seed, "regime": regime, "ng": ng, "sample_index": i + bi*args.batch_size, "epistemic": float(value)})
        del model; torch.cuda.empty_cache()
    with (out / "sample_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    auc_rows = []
    for seed in SEEDS:
        for ng in NGS:
            by = {r: np.asarray([x["epistemic"] for x in rows if x["seed"] == seed and int(x["ng"]) == ng and x["regime"] == r]) for r in REGIMES}
            ids = np.r_[by["20ns"], by["80ns"]]
            for regime in ("120ns", "1ms"):
                auc_rows.append({"seed": seed, "ng": ng, "comparison": f"ID(20+80) vs {regime}", "id_mean": float(ids.mean()), "ood_mean": float(by[regime].mean()), "id_median": float(np.median(ids)), "ood_median": float(np.median(by[regime])), "auroc": auc(by[regime], ids)})
    with (out / "auroc.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(auc_rows[0])); writer.writeheader(); writer.writerows(auc_rows)
    print({"rows": len(rows), "auroc_rows": len(auc_rows), "gpu": torch.cuda.get_device_name(0)})

if __name__ == "__main__":
    main()
