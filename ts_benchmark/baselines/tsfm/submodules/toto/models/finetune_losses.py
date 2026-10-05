from __future__ import annotations

from typing import Protocol

import torch


class LossFn(Protocol):
    def __call__(self, distr: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor: ...


class NegativeLogLikelihood(LossFn):
    def __call__(self, distr: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
        return -distr.log_prob(target)


class GeneralRobustLoss(LossFn):
    def __init__(self, delta: float = 1.0, alpha: float = 1.0):
        assert alpha <= 2, "Alpha must be in the range (-inf, 2]"
        self.delta = delta
        self.alpha = alpha

    def __call__(self, distr: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
        prediction = distr.mean.contiguous()
        x = prediction - target

        alpha = torch.tensor(self.alpha, device=target.device, dtype=target.dtype)
        delta = torch.tensor(self.delta, device=target.device, dtype=target.dtype)

        if self.alpha == 2:
            return 0.5 * ((x / delta) ** 2)
        if self.alpha == 0:
            return torch.log1p(0.5 * ((x / delta) ** 2))
        if self.alpha == float("-inf"):
            return 1 - torch.exp(-0.5 * ((x / delta) ** 2))

        abs_alpha_minus_two = torch.abs(alpha - 2)
        coef = abs_alpha_minus_two / alpha
        return coef * ((((x / delta) ** 2 / abs_alpha_minus_two) + 1) ** (alpha / 2) - 1)


class CombinedLoss(LossFn):
    def __init__(self, lambda_nll: float = 0.575, delta: float = 0.1, alpha: float = 0.0):
        self.nll_loss = NegativeLogLikelihood()
        self.general_robust_loss = GeneralRobustLoss(delta=delta, alpha=alpha)
        self.lambda_nll = lambda_nll

    def __call__(self, distr: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
        nll_loss = self.nll_loss(distr, target)
        general_robust_loss = self.general_robust_loss(distr, target)
        return self.lambda_nll * nll_loss + (1 - self.lambda_nll) * general_robust_loss
