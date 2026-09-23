#!/usr/bin/env python3
"""Partial fine-tuning readiness tool and one-step smoke runner.

This file prepares and validates research-extension adaptation scopes. It never
starts a multi-epoch adaptation run unless the caller explicitly omits
``--dry-run``; overnight usage intentionally invokes only ``--dry-run``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEAD_PREFIXES = ("gamma_head.", "psi_head.", "kappa_head.", "nu_head.")
SCOPES = ("head_only", "last_block_plus_head", "last_4_blocks_plus_head", "last_8_blocks_plus_head", "full")


def scheduled_mask(batch_size: int, num_subcarriers: int, epoch: int, batch_index: int,
                   seed: int, device: torch.device) -> torch.Tensor:
    """Deterministic canonical periodic masks for a shared adaptation schedule."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed + epoch * 1_000_003 + batch_index * 10_007 + 11))
    factors = (4, 8, 16, 32)
    mask = torch.zeros((batch_size, num_subcarriers), dtype=torch.float32)
    for row in range(batch_size):
        factor = factors[int(torch.randint(0, len(factors), (1,), generator=generator).item())]
        offset = int(torch.randint(0, factor, (1,), generator=generator).item())
        mask[row, offset::factor] = 1.0
    return mask.to(device)


def scheduled_noise_base(batch_size: int, num_subcarriers: int, epoch: int, batch_index: int,
                         seed: int, device: torch.device) -> torch.Tensor:
    """Deterministic complex-standard-normal base noise, independent of model scope."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed + epoch * 1_000_003 + batch_index * 10_007 + 29))
    real = torch.randn((batch_size, num_subcarriers, 2, 2), generator=generator)
    imag = torch.randn((batch_size, num_subcarriers, 2, 2), generator=generator)
    return torch.complex(real, imag).to(device)


def make_adaptation_observation(cfr: torch.Tensor, mask: torch.Tensor, seed: int,
                                device: torch.device, epoch: int = 1, batch_index: int = 0,
                                snr_db: float = 15.0):
    """Create sparse noisy input and clean full-CFR target for adaptation."""
    from src.training.data import build_sparse_input, cfr_to_real_imag

    noise_base = scheduled_noise_base(cfr.shape[0], cfr.shape[1], epoch, batch_index, seed, device)
    mask_complex = mask[:, :, None, None].to(dtype=cfr.real.dtype)
    observation_count = mask_complex.expand_as(cfr.real).sum(dim=(1, 2, 3)).clamp_min(1.0)
    signal_power = (cfr.abs().square() * mask_complex).sum(dim=(1, 2, 3)) / observation_count
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = noise_base * torch.sqrt(noise_power[:, None, None, None] / 2.0) * mask_complex
    observed_cfr = cfr + noise
    x, _ = build_sparse_input(observed_cfr, mask)
    target = cfr_to_real_imag(cfr)
    return x, target, noise


def adaptation_schedule_digest(sample_count: int, batch_size: int, epochs: int, seed: int) -> dict:
    """Digest the scope-independent order/mask/noise schedule for provenance."""
    digest = hashlib.sha256()
    batches = (sample_count + batch_size - 1) // batch_size
    for epoch in range(1, epochs + 1):
        order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(seed + epoch - 1))
        digest.update(order.numpy().tobytes())
        for batch_index in range(batches):
            current = min(batch_size, sample_count - batch_index * batch_size)
            mask = scheduled_mask(current, 1024, epoch, batch_index, seed, torch.device("cpu"))
            noise = scheduled_noise_base(current, 1024, epoch, batch_index, seed, torch.device("cpu"))
            digest.update(mask.numpy().tobytes())
            digest.update(noise.numpy().tobytes())
    return {"sha256": digest.hexdigest(), "sample_count": sample_count, "batch_size": batch_size,
            "epochs": epochs, "seed": seed, "ng_candidates": [4, 8, 16, 32],
            "noise": "15 dB complex AWGN on observed reported CFR only"}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_model(config_path: str | Path, device: torch.device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config

    return _make_model(load_config(config_path), device)


def load_checkpoint(model, checkpoint: str | Path, device: torch.device) -> str:
    path = Path(checkpoint)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_trainable_scope(model, scope: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"unknown trainable scope {scope!r}; choose from {SCOPES}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in HEAD_PREFIXES):
            parameter.requires_grad = True
        elif scope == "last_block_plus_head" and name.startswith("residual_blocks.31."):
            parameter.requires_grad = True
        elif scope == "last_4_blocks_plus_head" and name.startswith("residual_blocks."):
            block = int(name.split(".")[1])
            if block >= 28:
                parameter.requires_grad = True
        elif scope == "last_8_blocks_plus_head" and name.startswith("residual_blocks."):
            block = int(name.split(".")[1])
            if block >= 24:
                parameter.requires_grad = True


def optimizer_for_trainable(model, lr: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("scope selected zero trainable parameters")
    return torch.optim.Adam(parameters, lr=float(lr))


def parameter_inventory(model) -> tuple[list[dict], list[dict]]:
    total = sum(parameter.numel() for parameter in model.parameters())
    rows = []
    cumulative = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        cumulative += count
        rows.append({"module_name": name, "tensor_shape": "x".join(map(str, parameter.shape)),
                     "parameter_count": count, "cumulative_parameter_count": cumulative,
                     "model_percent": 100.0 * count / total})
    scope_rows = []
    for scope in SCOPES:
        configure_trainable_scope(model, scope)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        frozen = total - trainable
        scope_rows.append({"scope": scope, "trainable_params": trainable, "frozen_params": frozen,
                           "trainable_percent": 100.0 * trainable / total,
                           "adam_state_bytes_fp32": trainable * 2 * 4,
                           "adam_state_mib_fp32": trainable * 2 * 4 / 2**20})
    return rows, scope_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def smoke_one_step(model, data_path: str | Path, config: dict, device: torch.device, lr: float, seed: int) -> dict:
    from scripts.train_predictor import set_seeds
    from src.models.evidential import evidential_loss
    from src.training.data import CFRNPZDataset

    set_seeds(int(config["implementation_assumption"]["seed"]))
    dataset = CFRNPZDataset(data_path)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    cfr = batch["cfr"].to(device)
    mask = scheduled_mask(cfr.shape[0], int(config["paper_specified"]["num_subcarriers"]), 1, 0, seed, device)
    x, target, noise = make_adaptation_observation(cfr, mask, seed, device)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    optimizer = optimizer_for_trainable(model, lr)
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    losses = evidential_loss(output, target, float(config["paper_specified"]["lambda_reg"]),
                             nll_mode=config["implementation_assumption"].get("nll_mode", "elementwise"),
                             reg_mode=config["implementation_assumption"].get("reg_mode", "elementwise"))
    loss = losses["total"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite smoke loss: {loss.item()}")
    loss.backward()
    grad_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None and torch.isfinite(parameter.grad).all()]
    optimizer.step()
    frozen_unchanged = all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters() if name in before)
    return {"batch_size": 1, "input_is_sparse": True, "target_is_clean_full_cfr": True,
            "loss": float(loss.detach().cpu()),
            "loss_finite": True, "trainable_gradient_tensors": len(grad_names),
            "frozen_parameters_unchanged": frozen_unchanged,
            "sparse_observation_nonzero_fraction": float((x[:, :8] != 0).float().mean().detach().cpu()),
            "schedule_seed": seed, "device": str(device)}


def train_adaptation(model, train_path: str | Path, val_path: str | Path, config: dict,
                     device: torch.device, lr: float, epochs: int, batch_size: int, seed: int,
                     checkpoint_dir: Path) -> dict:
    from scripts.train_predictor import set_seeds
    from src.models.evidential import evidential_loss
    from src.training.data import CFRNPZDataset

    set_seeds(seed)
    train_loader = DataLoader(CFRNPZDataset(train_path), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(CFRNPZDataset(val_path), batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = optimizer_for_trainable(model, lr)
    history = []
    total_steps = 0
    step_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for batch_index, batch in enumerate(train_loader):
            started = time.perf_counter()
            cfr = batch["cfr"].to(device)
            mask = scheduled_mask(cfr.shape[0], int(config["paper_specified"]["num_subcarriers"]), epoch, batch_index, seed, device)
            x, target, _ = make_adaptation_observation(cfr, mask, seed, device, epoch, batch_index)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            losses = evidential_loss(output, target, float(config["paper_specified"]["lambda_reg"]),
                                     nll_mode=config["implementation_assumption"].get("nll_mode", "elementwise"),
                                     reg_mode=config["implementation_assumption"].get("reg_mode", "elementwise"))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite adaptation loss at epoch {epoch}")
            losses["total"].backward(); optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - started)
            total_steps += 1
            train_losses.append(float(losses["total"].detach().cpu()))
        model.eval(); val_losses = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                cfr = batch["cfr"].to(device)
                mask = scheduled_mask(cfr.shape[0], int(config["paper_specified"]["num_subcarriers"]), epoch, batch_index, seed + 700_000, device)
                x, target, _ = make_adaptation_observation(cfr, mask, seed + 700_000, device, epoch, batch_index)
                output = model(x)
                losses = evidential_loss(output, target, float(config["paper_specified"]["lambda_reg"]),
                                         nll_mode=config["implementation_assumption"].get("nll_mode", "elementwise"),
                                         reg_mode=config["implementation_assumption"].get("reg_mode", "elementwise"))
                val_losses.append(float(losses["total"].detach().cpu()))
        checkpoint = checkpoint_dir / f"adapted_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint)
        history.append({"epoch": epoch, "train_loss": float(sum(train_losses) / len(train_losses)),
                        "val_loss": float(sum(val_losses) / len(val_losses)), "optimizer_steps": total_steps,
                        "samples_seen": total_steps * batch_size, "checkpoint": str(checkpoint.relative_to(ROOT))})
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 2**20
    else:
        peak_vram = None
    return {"history": history, "optimizer_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "optimizer_steps": total_steps, "step_seconds_mean": float(sum(step_times) / len(step_times)),
            "step_seconds_total": float(sum(step_times)), "peak_allocated_vram_mib": peak_vram}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/current_valid_baseline_100k1_seed_20260819.json")
    parser.add_argument("--trainable-scope", choices=SCOPES, required=True)
    parser.add_argument("--adapt-train-data", required=True)
    parser.add_argument("--adapt-val-data", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size, and lr must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"expected NVIDIA GB10/cuda:0, found {device} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'}")
    config = load_json(args.config)
    from src.training.data import CFRNPZDataset
    train_count = len(CFRNPZDataset(ROOT / args.adapt_train_data))
    model = build_model(args.config, device)
    checkpoint_sha256 = load_checkpoint(model, ROOT / args.checkpoint, device)
    inventory, scopes = parameter_inventory(model)
    _write_csv(out / "parameter_inventory.csv", inventory)
    _write_csv(out / "scope_inventory.csv", scopes)
    configure_trainable_scope(model, args.trainable_scope)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    manifest = {"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha256,
                "scope": args.trainable_scope, "adapt_train_data": args.adapt_train_data,
                "adapt_val_data": args.adapt_val_data, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "seed": args.seed, "device": str(device), "gpu": torch.cuda.get_device_name(0),
                "research_extension": True, "paper_setting": False, "full_training_started": False,
                "trainable_parameter_names": trainable_names}
    manifest["adaptation_schedule"] = adaptation_schedule_digest(train_count, args.batch_size, args.epochs, args.seed)
    if args.inventory_only:
        (out / "readiness_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return
    started = time.perf_counter()
    if args.dry_run:
        result = smoke_one_step(model, ROOT / args.adapt_train_data, config, device, args.lr, args.seed)
    else:
        result = train_adaptation(model, ROOT / args.adapt_train_data, ROOT / args.adapt_val_data,
                                  config, device, args.lr, args.epochs, args.batch_size, args.seed, out)
        torch.save(model.state_dict(), out / "adapted_model.pt")
        result["checkpoint"] = str((out / "adapted_model.pt").relative_to(ROOT))
    result["elapsed_seconds"] = time.perf_counter() - started
    result["dry_run_only"] = bool(args.dry_run)
    manifest["full_training_started"] = not args.dry_run
    manifest["dry_run_result"] = result
    (out / "readiness_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
