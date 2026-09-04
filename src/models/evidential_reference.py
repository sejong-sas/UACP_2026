"""Small dense reference formulas for the paper-equation audit."""

from __future__ import annotations

import torch


def paper_predictive_student_t_log_prob(
    target: torch.Tensor,
    gamma: torch.Tensor,
    kappa: torch.Tensor,
    psi: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    dimension = target.shape[-1]
    degrees = nu - dimension + 1.0
    scale = ((kappa + 1.0) / (kappa * degrees)).unsqueeze(-1).unsqueeze(-1) * psi
    sign, logdet = torch.linalg.slogdet(scale)
    if torch.any(sign <= 0):
        raise FloatingPointError("Reference scale matrix is not positive definite.")
    residual = (target - gamma).unsqueeze(-1)
    quadratic = torch.matmul(residual.transpose(-2, -1), torch.linalg.solve(scale, residual)).squeeze(-1).squeeze(-1)
    return (
        torch.lgamma((degrees + dimension) / 2.0)
        - torch.lgamma(degrees / 2.0)
        - 0.5 * (dimension * torch.log(degrees * torch.pi) + logdet)
        - ((degrees + dimension) / 2.0) * torch.log1p(quadratic / degrees)
    )


def paper_uncertainties(psi: torch.Tensor, kappa: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimension = psi.shape[-1]
    aleatoric = psi / (nu - dimension - 1.0).unsqueeze(-1).unsqueeze(-1)
    epistemic = aleatoric / kappa.unsqueeze(-1).unsqueeze(-1)
    return aleatoric, epistemic, aleatoric + epistemic


def paper_pair_regularizer(target: torch.Tensor, gamma: torch.Tensor, kappa: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    return ((target - gamma).square().sum(dim=-1) * (kappa + nu)).mean()
