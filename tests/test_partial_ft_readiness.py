from pathlib import Path

import torch


def test_scope_selection_freezes_only_real_model_modules():
    from scripts.partial_ft_adapt import build_model, configure_trainable_scope

    model = build_model(Path("configs/current_valid_baseline_100k1_seed_20260819.json"), torch.device("cpu"))
    configure_trainable_scope(model, "last_block_plus_head")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("residual_blocks.31.") or any(name.startswith(prefix) for prefix in (
        "gamma_head.", "psi_head.", "kappa_head.", "nu_head."
    )) for name in trainable)
    assert not any(name.startswith("residual_blocks.30.") for name in trainable)


def test_optimizer_excludes_frozen_parameters():
    from scripts.partial_ft_adapt import build_model, configure_trainable_scope, optimizer_for_trainable

    model = build_model(Path("configs/current_valid_baseline_100k1_seed_20260819.json"), torch.device("cpu"))
    configure_trainable_scope(model, "head_only")
    optimizer = optimizer_for_trainable(model, 1e-4)
    optimizer_names = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_names == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}


def test_last_four_blocks_scope_selects_blocks_28_through_31_and_heads():
    from scripts.partial_ft_adapt import build_model, configure_trainable_scope

    model = build_model(Path("configs/current_valid_baseline_100k1_seed_20260819.json"), torch.device("cpu"))
    configure_trainable_scope(model, "last_4_blocks_plus_head")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(
        (name.startswith("residual_blocks.") and 28 <= int(name.split(".")[1]) <= 31)
        or any(name.startswith(prefix) for prefix in ("gamma_head.", "psi_head.", "kappa_head.", "nu_head."))
        for name in trainable
    )
    assert not any(name.startswith("residual_blocks.27.") for name in trainable)
