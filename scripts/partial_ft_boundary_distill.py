#!/usr/bin/env python3
"""Boundary-distilled last4 Partial FT.

This is a RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION. The fixed last4
scope is trained on 120 ns sparse-to-clean reconstruction while separate ID
and proxy-OOD batches provide uncertainty-only auxiliary losses. The 1 ms
evaluation set is never loaded by this script.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.partial_ft_adapt import (
    load_json, build_model, load_checkpoint, configure_trainable_scope,
    optimizer_for_trainable, scheduled_mask, make_adaptation_observation,
)
from src.training.data import CFRNPZDataset
from src.models.evidential import evidential_loss

EPS = 1e-8
K = 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uncertainty_vectors(output):
    margin = output.nu_expanded - 2 * K - 1
    aleatoric = output.psi / margin
    epistemic = aleatoric / output.kappa_expanded
    reduce_dims = tuple(range(1, epistemic.ndim))
    return {
        "psi": output.psi.mean(dim=reduce_dims),
        "nu_margin": margin.mean(dim=reduce_dims),
        "kappa": output.kappa_expanded.mean(dim=reduce_dims),
        "aleatoric": aleatoric.mean(dim=reduce_dims),
        "epistemic": epistemic.mean(dim=reduce_dims),
    }


def id_uncertainty_distillation(student_epi, student_ale, teacher_epi, teacher_ale, eps=EPS):
    return torch.mean((torch.log(student_epi + eps) - torch.log(teacher_epi + eps)) ** 2) + torch.mean((torch.log(student_ale + eps) - torch.log(teacher_ale + eps)) ** 2)


def one_sided_boundary_loss(student_epi, teacher_epi, eps=EPS):
    return torch.mean(torch.relu(torch.log(teacher_epi + eps) - torch.log(student_epi + eps)) ** 2)


class InitialMagnitudeNormalizer:
    """Normalize one auxiliary loss by its first detached smoke-batch magnitude."""
    def __init__(self):
        self.initial = None

    def __call__(self, value):
        if self.initial is None:
            self.initial = float(value.detach().abs().clamp_min(EPS).cpu())
        return value / self.initial


def resolve_zero_initial_scale(normalizer, reference):
    """Use the ID auxiliary scale when the one-sided boundary starts at zero."""
    if normalizer.initial is None or normalizer.initial <= EPS:
        normalizer.initial = float(reference.initial)
    return normalizer


def make_batch(dataset_batch, seed, epoch, batch_index, device):
    cfr = dataset_batch["cfr"].to(device)
    mask = scheduled_mask(cfr.shape[0], K, epoch, batch_index, seed, device)
    x, target, _ = make_adaptation_observation(cfr, mask, seed, device, epoch, batch_index)
    return x, target, mask


def aux_terms(student, teacher, x, boundary_x):
    student_id = uncertainty_vectors(student(x))
    with torch.no_grad():
        teacher_id = uncertainty_vectors(teacher(x))
        teacher_boundary = uncertainty_vectors(teacher(boundary_x))
    student_boundary = uncertainty_vectors(student(boundary_x))
    id_loss = id_uncertainty_distillation(student_id["epistemic"], student_id["aleatoric"], teacher_id["epistemic"], teacher_id["aleatoric"])
    boundary_loss = one_sided_boundary_loss(student_boundary["epistemic"], teacher_boundary["epistemic"])
    return id_loss, boundary_loss


def smoke(model, teacher, adapt_batch, id_batch, proxy_batch, config, device, seed):
    ax, target, _ = make_batch(adapt_batch, seed, 1, 0, device)
    ix, _, _ = make_batch(id_batch, seed + 900000, 1, 0, device)
    px, _, _ = make_batch(proxy_batch, seed + 910000, 1, 0, device)
    optimizer = optimizer_for_trainable(model, 1e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    base = evidential_loss(model(ax), target, float(config["paper_specified"]["lambda_reg"]), nll_mode=config["implementation_assumption"].get("nll_mode", "elementwise"), reg_mode=config["implementation_assumption"].get("reg_mode", "elementwise"))["total"]
    id_loss, boundary_loss = aux_terms(model, teacher, ix, px)
    normal_id, normal_boundary = InitialMagnitudeNormalizer(), InitialMagnitudeNormalizer()
    normalized_id = normal_id(id_loss)
    normal_boundary(boundary_loss)
    resolve_zero_initial_scale(normal_boundary, normal_id)
    total = base + normalized_id + boundary_loss / normal_boundary.initial
    total.backward()
    gradient_tensors = sum(bool(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters())
    optimizer.step()
    frozen_unchanged = all(torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)
    return {"base_loss": float(base.detach().cpu()), "id_loss": float(id_loss.detach().cpu()), "boundary_loss": float(boundary_loss.detach().cpu()), "normalized_total_loss": float(total.detach().cpu()), "id_initial_magnitude": normal_id.initial, "boundary_initial_magnitude": normal_boundary.initial, "finite": bool(torch.isfinite(total)), "trainable_gradient_tensors": int(gradient_tensors), "frozen_unchanged": frozen_unchanged, "teacher_parameters_require_grad": any(p.requires_grad for p in teacher.parameters()), "device": str(device)}


def run(args):
    outdir = ROOT / args.output_dir
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(outdir)
    outdir.mkdir(parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10/cuda:0 required")
    config = load_json(args.config)
    student = build_model(args.config, device)
    teacher = build_model(args.config, device)
    checkpoint_sha = load_checkpoint(student, ROOT / args.checkpoint, device)
    load_checkpoint(teacher, ROOT / args.checkpoint, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    configure_trainable_scope(student, "last_4_blocks_plus_head")
    adapt = CFRNPZDataset(ROOT / args.adapt_train_data)
    validation = CFRNPZDataset(ROOT / args.adapt_val_data)
    old_id = CFRNPZDataset(ROOT / args.id_data)
    proxy = CFRNPZDataset(ROOT / args.proxy_data)
    manifest = {
        "research_extension": True,
        "paper_setting": False,
        "scope": "last_4_blocks_plus_head",
        "teacher": "frozen Pre checkpoint",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "adapt_train_data": args.adapt_train_data,
        "adapt_val_data": args.adapt_val_data,
        "old_id_data": args.id_data,
        "proxy_ood_data": args.proxy_data,
        "one_ms_data_loaded": False,
        "proxy_reconstruction_target_loss": False,
        "id_loss": "MSE(log(Epistemic_student+eps),log(Epistemic_teacher+eps))+MSE(log(Aleatoric_student+eps),log(Aleatoric_teacher+eps))",
        "boundary_loss": "mean(ReLU(log(Epistemic_teacher+eps)-log(Epistemic_student+eps))^2)",
        "auxiliary_normalization": "each auxiliary loss divided by detached first smoke-batch magnitude",
        "lambda_id": 1.0,
        "lambda_boundary": 1.0,
        "eps": EPS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "lambda_reg": config["paper_specified"]["lambda_reg"],
        "precision": "FP32",
        "implementation_assumptions": ["proxy delay Uniform(160,500) ns", "all auxiliary weights fixed after one smoke-batch normalization", "one-sided boundary is zero at identical Pre initialization; its fixed scale falls back to the ID auxiliary initial magnitude", "Partial FT layer policy and adaptation schedule are not paper-specified"],
    }
    if args.smoke:
        manifest["smoke"] = smoke(student, teacher, next(iter(DataLoader(adapt, batch_size=1, shuffle=False))), next(iter(DataLoader(old_id, batch_size=1, shuffle=False))), next(iter(DataLoader(proxy, batch_size=1, shuffle=False))), config, device, args.seed)
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2)); return
    import numpy as np
    adapt_loader = DataLoader(adapt, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    id_loader = DataLoader(old_id, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 900000), num_workers=0)
    proxy_loader = DataLoader(proxy, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 910000), num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = optimizer_for_trainable(student, args.lr)
    id_norm, boundary_norm = InitialMagnitudeNormalizer(), InitialMagnitudeNormalizer()
    history, step_times, steps = [], [], 0
    start = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        student.train(); teacher.eval(); base_values=[]; id_values=[]; boundary_values=[]; total_values=[]
        for batch_index, (adapt_batch, id_batch, proxy_batch) in enumerate(zip(adapt_loader, id_loader, proxy_loader)):
            tick = time.perf_counter()
            ax, target, _ = make_batch(adapt_batch, args.seed, epoch, batch_index, device)
            ix, _, _ = make_batch(id_batch, args.seed + 900000, epoch, batch_index, device)
            px, _, _ = make_batch(proxy_batch, args.seed + 910000, epoch, batch_index, device)
            optimizer.zero_grad(set_to_none=True)
            base = evidential_loss(student(ax), target, float(config["paper_specified"]["lambda_reg"]), nll_mode=config["implementation_assumption"].get("nll_mode", "elementwise"), reg_mode=config["implementation_assumption"].get("reg_mode", "elementwise"))["total"]
            id_loss, boundary_loss = aux_terms(student, teacher, ix, px)
            normalized_id = id_norm(id_loss)
            boundary_norm(boundary_loss)
            resolve_zero_initial_scale(boundary_norm, id_norm)
            total = base + normalized_id + boundary_loss / boundary_norm.initial
            if not torch.isfinite(total): raise FloatingPointError(f"nonfinite total loss at epoch {epoch}, step {batch_index}")
            total.backward(); optimizer.step()
            if device.type == "cuda": torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - tick); steps += 1
            base_values.append(float(base.detach().cpu())); id_values.append(float(id_loss.detach().cpu())); boundary_values.append(float(boundary_loss.detach().cpu())); total_values.append(float(total.detach().cpu()))
        student.eval(); val_values=[]
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                vx, vt, _ = make_batch(batch, args.seed + 700000, epoch, batch_index, device)
                val_values.append(float(evidential_loss(student(vx), vt, float(config["paper_specified"]["lambda_reg"]), nll_mode=config["implementation_assumption"].get("nll_mode", "elementwise"), reg_mode=config["implementation_assumption"].get("reg_mode", "elementwise"))["total"].detach().cpu()))
        checkpoint = outdir / f"adapted_epoch_{epoch}.pt"
        torch.save(student.state_dict(), checkpoint)
        history.append({"epoch": epoch, "train_base_loss": float(np.mean(base_values)), "train_id_loss_raw": float(np.mean(id_values)), "train_boundary_loss_raw": float(np.mean(boundary_values)), "train_total_loss": float(np.mean(total_values)), "val_adaptation_loss": float(np.mean(val_values)), "optimizer_steps": steps, "samples_seen": steps * args.batch_size, "checkpoint": str(checkpoint.relative_to(ROOT))})
    torch.save(student.state_dict(), outdir / "adapted_model.pt")
    manifest["training"] = {"history": history, "id_initial_magnitude": id_norm.initial, "boundary_initial_magnitude": boundary_norm.initial, "trainable_params": sum(p.numel() for p in student.parameters() if p.requires_grad), "training_seconds": time.perf_counter() - start, "seconds_per_step": float(np.mean(step_times)), "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / 2**20), "frozen_parameter_policy": "all non-last4/head parameters requires_grad=False"}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--config", default="configs/current_valid_baseline_100k1_seed_20260819.json")
    parser.add_argument("--adapt-train-data", required=True); parser.add_argument("--adapt-val-data", required=True); parser.add_argument("--id-data", required=True); parser.add_argument("--proxy-data", required=True)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
