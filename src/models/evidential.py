"""Evidential regression utilities for the prototype UACP predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def channels_to_pair_vectors(values: torch.Tensor) -> torch.Tensor:
    """Convert [Re(pair-major), Im(pair-major), frequency] to [pair, Re/Im]."""
    batch_size, channels, num_subcarriers = values.shape
    if channels != 8:
        raise ValueError("Expected eight real/imag channels.")
    return values.reshape(batch_size, 2, 4, num_subcarriers).permute(0, 2, 1, 3).reshape(batch_size, 4, 2 * num_subcarriers)


def pair_vectors_to_channels(values: torch.Tensor, num_subcarriers: int) -> torch.Tensor:
    batch_size, num_pairs, dimension = values.shape[:3]
    if num_pairs != 4 or dimension != 2 * num_subcarriers:
        raise ValueError("Expected four antenna pairs and 2K vector dimension.")
    return values.reshape(batch_size, 4, 2, num_subcarriers).permute(0, 2, 1, 3).reshape(batch_size, 8, num_subcarriers)


def _pair_vectors_to_interleaved(values: torch.Tensor) -> torch.Tensor:
    """Convert [pair, Re(0..K), Im(0..K)] to [pair, k, (Re, Im)]."""
    batch_size, pairs, dimension = values.shape
    return values.reshape(batch_size, pairs, 2, dimension // 2).permute(0, 1, 3, 2)


def _interleaved_to_pair_vectors(values: torch.Tensor) -> torch.Tensor:
    batch_size, pairs, num_subcarriers, _ = values.shape
    return values.permute(0, 1, 3, 2).reshape(batch_size, pairs, 2 * num_subcarriers)


def build_banded_cholesky(
    raw_factor: torch.Tensor,
    num_subcarriers: int,
    bandwidth: int,
    diagonal: torch.Tensor,
    factor_scale: float = 0.05,
) -> tuple[torch.Tensor, int]:
    """Build block-banded lower L without constructing a d x d covariance.

    The block coordinates are interleaved (Re(k), Im(k)). Blocks contain
    ``bandwidth + 1`` frequencies, so this bounded pilot has no cross-block
    covariance. ``Psi = L L^T`` is therefore positive definite when the
    diagonal is positive. The final block is padded with an identity.
    """
    if raw_factor.ndim != 4 or raw_factor.shape[2] != num_subcarriers:
        raise ValueError("raw_factor must have shape [B, pairs, K, 4*bandwidth].")
    if raw_factor.shape[3] != 4 * bandwidth:
        raise ValueError("raw_factor channel dimension must be 4*bandwidth.")
    if diagonal.shape != raw_factor.shape[:2] + (2 * num_subcarriers,):
        raise ValueError("diagonal must have shape [B, pairs, 2*K].")
    if bandwidth < 1:
        raise ValueError("bandwidth must be positive.")

    batch_size, pairs = raw_factor.shape[:2]
    block_size = bandwidth + 1
    num_blocks = (num_subcarriers + block_size - 1) // block_size
    block_dimension = 2 * block_size
    padded_subcarriers = num_blocks * block_size
    interleaved_diag = _pair_vectors_to_interleaved(diagonal)
    factors = raw_factor.new_zeros(batch_size, pairs, num_blocks, block_dimension, block_dimension)

    valid_frequency = torch.arange(num_subcarriers, device=raw_factor.device)
    block_index = torch.div(valid_frequency, block_size, rounding_mode="floor")
    local_index = valid_frequency.remainder(block_size)
    diagonal_index = 2 * local_index
    for component in range(2):
        indices = diagonal_index + component
        factors[:, :, block_index, indices, indices] = torch.sqrt(torch.clamp(interleaved_diag[:, :, :, component], min=1e-8))

    # The padded coordinates do not affect the real d-dimensional density.
    padded_frequency = torch.arange(num_subcarriers, padded_subcarriers, device=raw_factor.device)
    if padded_frequency.numel():
        padded_blocks = torch.div(padded_frequency, block_size, rounding_mode="floor")
        padded_local = padded_frequency.remainder(block_size)
        padded_index = 2 * padded_local
        for component in range(2):
            indices = padded_index + component
            factors[:, :, padded_blocks, indices, indices] = 1.0

    offdiag = torch.tanh(raw_factor.reshape(batch_size, pairs, num_subcarriers, bandwidth, 2, 2)) * factor_scale
    for lag in range(1, bandwidth + 1):
        valid = local_index >= lag
        if not torch.any(valid):
            continue
        rows = 2 * local_index[valid]
        cols = 2 * (local_index[valid] - lag)
        blocks = block_index[valid]
        for row in range(2):
            for col in range(2):
                factors[:, :, blocks, rows + row, cols + col] = offdiag[:, :, valid, lag - 1, row, col]
    return factors, 2 * padded_subcarriers


def banded_covariance_dense(factors: torch.Tensor, num_subcarriers: int, padded_dim: int | None = None) -> torch.Tensor:
    """Dense reference helper for tests and tiny dimensions only."""
    if padded_dim is None:
        padded_dim = factors.shape[-1]
    dense = factors.new_zeros(factors.shape[0], factors.shape[1], padded_dim, padded_dim)
    block_dimension = factors.shape[-1]
    for block in range(factors.shape[2]):
        start = block * block_dimension
        dense[:, :, start : start + block_dimension, start : start + block_dimension] = factors[:, :, block] @ factors[:, :, block].transpose(-2, -1)
    return dense


@dataclass
class EvidentialOutput:
    gamma: torch.Tensor
    kappa: torch.Tensor
    psi: torch.Tensor
    nu: torch.Tensor
    num_subcarriers: int = 1024
    covariance_factor: torch.Tensor | None = None
    banded_factor: torch.Tensor | None = None
    banded_padded_dim: int | None = None

    @property
    def predicted(self) -> torch.Tensor:
        return self.gamma

    @property
    def kappa_expanded(self) -> torch.Tensor:
        return expand_pair_scalar(self.kappa, self.gamma)

    @property
    def nu_expanded(self) -> torch.Tensor:
        return expand_pair_scalar(self.nu, self.gamma)

    @property
    def aleatoric(self) -> torch.Tensor:
        # Eq.7: Psi를 자유도 항으로 나누어 데이터 자체의 변동을 계산한다.
        return self.covariance_diag / (self.nu_expanded - 2 * self.num_subcarriers - 1)

    @property
    def epistemic(self) -> torch.Tensor:
        # Eq.8: 평균에 대한 evidence(kappa)가 적을수록 모델 불확실성이 커진다.
        return self.aleatoric / self.kappa_expanded

    @property
    def covariance_diag(self) -> torch.Tensor:
        if self.banded_factor is not None:
            diagonal = self.banded_factor.square().sum(dim=-1)
            dimension = 2 * self.num_subcarriers
            diagonal = diagonal.reshape(diagonal.shape[0], diagonal.shape[1], -1)[..., :dimension]
            interleaved = diagonal.reshape(diagonal.shape[0], diagonal.shape[1], self.num_subcarriers, 2)
            return pair_vectors_to_channels(_interleaved_to_pair_vectors(interleaved), self.num_subcarriers)
        if self.covariance_factor is None:
            return self.psi
        lowrank_diag = self.covariance_factor.square().sum(dim=-1)
        return pair_vectors_to_channels(channels_to_pair_vectors(self.psi) + lowrank_diag, self.num_subcarriers)

    @property
    def lowrank_diag_contribution(self) -> torch.Tensor | None:
        if self.covariance_factor is None:
            return None
        return self.covariance_factor.square().sum(dim=-1)


def expand_pair_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape == reference.shape:
        return value
    if value.ndim == 3 and value.shape[1] == 4 and value.shape[2] == 1 and reference.shape[1] == 8:
        expanded = value.expand(-1, -1, reference.shape[2])
        return torch.cat((expanded, expanded), dim=1)
    raise ValueError(f"Cannot expand evidential tensor shape {tuple(value.shape)} to {tuple(reference.shape)}")


def constrain_evidential(raw: torch.Tensor, num_output_channels: int = 8, num_subcarriers: int = 1024) -> EvidentialOutput:
    gamma, raw_kappa, raw_psi, raw_nu = torch.split(raw, num_output_channels, dim=1)
    eps = 1e-6
    return EvidentialOutput(
        gamma=gamma,
        kappa=F.softplus(raw_kappa) + eps,
        psi=F.softplus(raw_psi) + eps,
        nu=(2 * num_subcarriers + 1) + F.softplus(raw_nu) + eps,
        num_subcarriers=num_subcarriers,
    )


def constrain_pair_scalar_evidential(
    gamma_raw: torch.Tensor,
    psi_raw: torch.Tensor,
    kappa_raw: torch.Tensor,
    nu_raw: torch.Tensor,
    num_subcarriers: int = 1024,
) -> EvidentialOutput:
    eps = 1e-6
    return EvidentialOutput(
        gamma=gamma_raw,
        kappa=F.softplus(kappa_raw) + eps,
        psi=F.softplus(psi_raw) + eps,
        nu=(2 * num_subcarriers + 1) + F.softplus(nu_raw) + eps,
        num_subcarriers=num_subcarriers,
    )


def student_t_nll(output: EvidentialOutput, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    degrees = output.nu_expanded - 2 * output.num_subcarriers + 1
    scale_sq = ((output.kappa_expanded + 1.0) / (output.kappa_expanded * degrees)) * output.psi
    scale_sq = torch.clamp(scale_sq, min=eps)
    residual_sq = (target - output.gamma).square()
    nll = (
        torch.lgamma((degrees + 1.0) / 2.0)
        - torch.lgamma(degrees / 2.0)
        - 0.5 * torch.log(degrees * torch.pi * scale_sq)
        - ((degrees + 1.0) / 2.0) * torch.log1p(residual_sq / (degrees * scale_sq))
    )
    return -nll.mean()


def diagonal_multivariate_student_t_nll(output: EvidentialOutput, target: torch.Tensor) -> torch.Tensor:
    # pair 하나의 2048차원 CFR를 독립 scalar들의 평균이 아닌 하나의
    # diagonal multivariate Student-t density로 계산한다.
    eps = 1e-8
    batch_size, channels, num_subcarriers = target.shape
    if channels != 8:
        raise ValueError("Diagonal multivariate NLL expects 8 real/imag channels for 2x2 MIMO.")
    dimension = 2 * num_subcarriers
    residual = channels_to_pair_vectors(target - output.gamma)
    psi_diag = torch.clamp(channels_to_pair_vectors(output.psi), min=eps)
    kappa = output.kappa if output.kappa.shape[1:] == (4, 1) else channels_to_pair_vectors(output.kappa).mean(dim=-1, keepdim=True)
    nu = output.nu if output.nu.shape[1:] == (4, 1) else channels_to_pair_vectors(output.nu).mean(dim=-1, keepdim=True)
    degrees = torch.clamp(nu - dimension + 1, min=eps)
    scale_diag = torch.clamp(((kappa + 1.0) / (kappa * degrees)) * psi_diag, min=eps)
    mahalanobis = torch.sum(residual.square() / scale_diag, dim=-1)
    logdet = torch.sum(torch.log(scale_diag), dim=-1)
    degrees_squeezed = degrees.squeeze(-1)
    log_prob = (
        torch.lgamma((degrees_squeezed + dimension) / 2.0)
        - torch.lgamma(degrees_squeezed / 2.0)
        - 0.5 * (dimension * torch.log(degrees_squeezed * torch.pi) + logdet)
        - ((degrees_squeezed + dimension) / 2.0) * torch.log1p(mahalanobis / degrees_squeezed)
    )
    return -log_prob.mean()


def lowrank_multivariate_student_t_nll(output: EvidentialOutput, target: torch.Tensor) -> torch.Tensor:
    """Multivariate Student-t NLL for Psi = diag(D) + U U^T.

    The determinant lemma and Woodbury identity avoid constructing a d x d
    covariance matrix. The returned density is one d-dimensional density per
    antenna pair, matching the diagonal multivariate implementation.
    """
    eps = 1e-8
    batch_size, channels, num_subcarriers = target.shape
    if channels != 8 or output.covariance_factor is None:
        raise ValueError("Low-rank NLL expects 8 channels and a covariance factor.")
    dimension = 2 * num_subcarriers
    residual = channels_to_pair_vectors(target - output.gamma)
    diagonal = torch.clamp(channels_to_pair_vectors(output.psi), min=eps)
    factor = output.covariance_factor
    rank = factor.shape[-1]
    kappa = output.kappa if output.kappa.shape[1:] == (4, 1) else channels_to_pair_vectors(output.kappa).mean(dim=-1, keepdim=True)
    nu = output.nu if output.nu.shape[1:] == (4, 1) else channels_to_pair_vectors(output.nu).mean(dim=-1, keepdim=True)
    degrees = torch.clamp(nu - dimension + 1, min=eps)
    scale_factor = torch.clamp((kappa + 1.0) / (kappa * degrees), min=eps)

    d_inv_factor = factor / diagonal.unsqueeze(-1)
    eye = torch.eye(rank, dtype=factor.dtype, device=factor.device).reshape(1, 1, rank, rank)
    small = eye + torch.matmul(factor.transpose(-2, -1), d_inv_factor)
    sign, logdet_small = torch.linalg.slogdet(small)
    if torch.any(sign <= 0):
        raise FloatingPointError("Low-rank determinant is not positive.")
    logdet_psi = torch.log(diagonal).sum(dim=-1) + logdet_small

    d_inv_residual = residual / diagonal
    projected = torch.matmul(factor.transpose(-2, -1), d_inv_residual.unsqueeze(-1)).squeeze(-1)
    solved = torch.linalg.solve(small, projected.unsqueeze(-1)).squeeze(-1)
    base_quadratic = (residual * d_inv_residual).sum(dim=-1)
    correction = (projected * solved).sum(dim=-1)
    quadratic_psi = torch.clamp(base_quadratic - correction, min=0.0)
    logdet_scale = dimension * torch.log(scale_factor.squeeze(-1)) + logdet_psi
    quadratic_scale = quadratic_psi / scale_factor.squeeze(-1)
    degrees_squeezed = degrees.squeeze(-1)
    log_prob = (
        torch.lgamma((degrees_squeezed + dimension) / 2.0)
        - torch.lgamma(degrees_squeezed / 2.0)
        - 0.5 * (dimension * torch.log(degrees_squeezed * torch.pi) + logdet_scale)
        - ((degrees_squeezed + dimension) / 2.0) * torch.log1p(quadratic_scale / degrees_squeezed)
    )
    return -log_prob.mean()


def banded_multivariate_student_t_nll(output: EvidentialOutput, target: torch.Tensor) -> torch.Tensor:
    """Exact Student-t NLL for the block-banded ``Psi=L L^T`` pilot.

    The solve and log determinant operate independently on small frequency
    blocks. No dense 2048-by-2048 matrix is formed in this production path.
    """
    eps = 1e-8
    if output.banded_factor is None or target.shape[1] != 8:
        raise ValueError("Banded NLL expects eight real/imag channels and a banded factor.")
    dimension = 2 * output.num_subcarriers
    residual = _pair_vectors_to_interleaved(channels_to_pair_vectors(target - output.gamma))
    factor = output.banded_factor
    block_dimension = factor.shape[-1]
    padded_subcarriers = factor.shape[-3] * (block_dimension // 2)
    residual = F.pad(residual, (0, 0, 0, padded_subcarriers - output.num_subcarriers))
    residual = residual.reshape(residual.shape[0], residual.shape[1], -1, block_dimension)
    solved = torch.linalg.solve_triangular(factor, residual.unsqueeze(-1), upper=False).squeeze(-1)
    quadratic_psi = solved.square().sum(dim=(-1, -2))
    logdet_psi = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1).clamp_min(eps)).sum(dim=(-1, -2))
    kappa = output.kappa
    nu = output.nu
    degrees = torch.clamp(nu - dimension + 1.0, min=eps)
    scale_factor = torch.clamp((kappa + 1.0) / (kappa * degrees), min=eps)
    logdet_scale = dimension * torch.log(scale_factor.squeeze(-1)) + logdet_psi
    quadratic_scale = quadratic_psi / scale_factor.squeeze(-1)
    degrees_squeezed = degrees.squeeze(-1)
    log_prob = (
        torch.lgamma((degrees_squeezed + dimension) / 2.0)
        - torch.lgamma(degrees_squeezed / 2.0)
        - 0.5 * (dimension * torch.log(degrees_squeezed * torch.pi) + logdet_scale)
        - ((degrees_squeezed + dimension) / 2.0) * torch.log1p(quadratic_scale / degrees_squeezed)
    )
    return -log_prob.mean()


def evidence_regularizer(output: EvidentialOutput, target: torch.Tensor) -> torch.Tensor:
    evidence = output.kappa_expanded + output.nu_expanded
    return ((target - output.gamma).square() * evidence).mean()


def pair_level_evidence_regularizer(output: EvidentialOutput, target: torch.Tensor) -> torch.Tensor:
    # 관측·미관측 전체 CFR pair error와 scalar evidence를 함께 사용해,
    # 크게 틀린 예측에 높은 자신감을 주는 것을 줄인다.
    batch_size, channels, num_subcarriers = target.shape
    residual = channels_to_pair_vectors(target - output.gamma)
    pair_error_norm_sq = residual.square().sum(dim=-1, keepdim=True)
    if output.kappa.shape[1:] == (4, 1):
        evidence = output.kappa + output.nu
    else:
        evidence = channels_to_pair_vectors(output.kappa + output.nu).mean(dim=-1, keepdim=True)
    return (pair_error_norm_sq * evidence).mean()


def evidential_loss(
    output: EvidentialOutput,
    target: torch.Tensor,
    lambda_reg: float,
    nll_mode: str = "elementwise",
    reg_mode: str = "elementwise",
) -> dict[str, torch.Tensor]:
    if nll_mode == "elementwise":
        nll = student_t_nll(output, target)
    elif nll_mode == "diagonal_multivariate":
        nll = diagonal_multivariate_student_t_nll(output, target)
    elif nll_mode == "lowrank_multivariate":
        nll = lowrank_multivariate_student_t_nll(output, target)
    elif nll_mode == "banded_multivariate":
        nll = banded_multivariate_student_t_nll(output, target)
    else:
        raise ValueError(f"Unknown nll_mode: {nll_mode}")

    if reg_mode == "elementwise":
        reg = evidence_regularizer(output, target)
    elif reg_mode == "pair":
        reg = pair_level_evidence_regularizer(output, target)
    else:
        raise ValueError(f"Unknown reg_mode: {reg_mode}")

    lambda_reg_x_reg = float(lambda_reg) * reg
    total = nll + lambda_reg_x_reg
    return {"total": total, "nll": nll, "reg": reg, "lambda_reg_x_reg": lambda_reg_x_reg}
