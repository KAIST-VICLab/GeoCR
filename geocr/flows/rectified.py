"""Rectified flow with t = 0 noise and t = 1 data: ``z_t = (1 - t) eps + t z1``, ``u* = z1 - eps``."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_t(batch: int, device, generator: torch.Generator | None = None,
             scheme: str = "logit_normal", shift: float = 1.0) -> Tensor:
    """Training-time draw ``logit(t) = N(0, 1) - log(shift)``: ``shift`` multiplies the odds of noise
    and applies to this draw only (``euler_sample`` integrates an unshifted uniform grid)."""
    if scheme != "logit_normal":
        raise ValueError(f"unknown time scheme {scheme!r}")
    n = torch.randn(batch, device=device, generator=generator)
    return torch.sigmoid(n - math.log(shift))


def interpolate(z1: Tensor, eps: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    tb = t.view(-1, *([1] * (z1.ndim - 1)))
    return (1.0 - tb) * eps + tb * z1, z1 - eps


def flow_loss(v: Tensor, u_star: Tensor, ms_weight: float = 1.0) -> tuple[Tensor, dict]:
    """Channel-grouped MSE: [:128] = z_rgb (weight 1), [128:] = z_ms."""
    err = (v - u_star) ** 2
    rgb = err[:, :128].mean()
    ms = err[:, 128:].mean()
    total = (rgb + ms_weight * ms) / (1.0 + ms_weight) if ms_weight > 0 else rgb
    return total, {"loss_rgb": rgb.detach(), "loss_ms": ms.detach()}


@torch.no_grad()
def euler_sample(model_fn, shape: tuple, nfe: int, device,
                 generator: torch.Generator | None = None, z0: Tensor | None = None) -> Tensor:
    """Integrate t: 0 -> 1 in ``nfe`` Euler steps on a uniform grid; ``model_fn(z, t) -> v``.

    ``z0`` is the starting noise when the caller owns it (per-sample noise); otherwise it is drawn here.
    """
    z = (z0.to(device) if z0 is not None
         else torch.randn(shape, device=device, generator=generator))
    ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    for i in range(nfe):
        t = ts[i].expand(shape[0])
        v = model_fn(z, t)
        z = z + (ts[i + 1] - ts[i]) * v
    return z
