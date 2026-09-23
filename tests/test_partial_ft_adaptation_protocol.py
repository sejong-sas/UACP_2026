import torch


def test_adaptation_observation_is_sparse_and_targets_clean_full_cfr():
    from scripts.partial_ft_adapt import make_adaptation_observation, scheduled_mask

    cfr = torch.ones((2, 8, 2, 2), dtype=torch.complex64)
    mask = scheduled_mask(batch_size=2, num_subcarriers=8, epoch=1, batch_index=0, seed=20260921, device=torch.device("cpu"))
    x, target, noise_base = make_adaptation_observation(cfr, mask, seed=20260921, device=torch.device("cpu"))

    assert target.shape == (2, 8, 8)
    assert torch.equal(target[0, :4, 0], torch.ones(4))
    assert torch.equal(target[0, 4:, 0], torch.zeros(4))
    observed = mask.bool().unsqueeze(1).expand_as(x[:, :8])
    assert torch.all(x[:, :8][~observed] == 0)
    assert torch.all(x[:, 8:][~mask.bool().unsqueeze(1)] == 0)
    assert torch.allclose(noise_base[~mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(noise_base)], torch.zeros_like(noise_base[~mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(noise_base)]))


def test_adaptation_schedule_is_scope_invariant():
    from scripts.partial_ft_adapt import scheduled_mask, scheduled_noise_base

    kwargs = dict(batch_size=4, num_subcarriers=16, epoch=2, batch_index=3, seed=20260921, device=torch.device("cpu"))
    mask_a = scheduled_mask(**kwargs)
    mask_b = scheduled_mask(**kwargs)
    noise_a = scheduled_noise_base(**kwargs)
    noise_b = scheduled_noise_base(**kwargs)
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(noise_a, noise_b)
