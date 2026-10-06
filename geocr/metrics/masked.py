"""Masked reflectance metrics for Sentinel-2 stacks: reflectance clipped to [0,1], averaged over
the pixels where ``mask`` is 1, and reported separately for the RGB bands, the non-RGB bands and
all 13 bands.

SSIM is the 11x11 Gaussian (sigma 1.5) map averaged over the mask without erosion, so
neighbourhoods outside the mask enter border windows; every method is treated the same way.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as tF
from torch import Tensor

from geocr.normalize import MS_IDX, RGB_IDX


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _gauss_kernel(win: int = 11, sigma: float = 1.5, device=None) -> Tensor:
    g = torch.exp(-((torch.arange(win, device=device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.T @ g).view(1, 1, win, win)


def masked_psnr(pred: Tensor, gt: Tensor, mask: Tensor) -> float:
    mse = _masked_mean((pred - gt) ** 2, mask)
    return float(10.0 * math.log10(1.0 / max(mse.item(), 1e-12)))


def masked_sam_deg(pred: Tensor, gt: Tensor, mask: Tensor) -> float:
    """Spectral angle over the band axis in degrees, masked mean, computed in float64.
    ``[B,C,H,W]``."""
    pred, gt, mask = pred.double(), gt.double(), mask.double()
    dot = (pred * gt).sum(dim=1)
    denom = pred.norm(dim=1) * gt.norm(dim=1)
    ang = torch.acos((dot / denom.clamp_min(1e-8)).clamp(-1.0, 1.0))
    return float(_masked_mean(ang, mask.squeeze(1)) * 180.0 / math.pi)


def masked_ssim(pred: Tensor, gt: Tensor, mask: Tensor) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    k = _gauss_kernel(device=pred.device).to(pred.dtype)
    b, c, h, w = pred.shape
    p = pred.reshape(b * c, 1, h, w)
    g = gt.reshape(b * c, 1, h, w)
    pad = 5
    mu_p = tF.conv2d(p, k, padding=pad)
    mu_g = tF.conv2d(g, k, padding=pad)
    var_p = tF.conv2d(p * p, k, padding=pad) - mu_p ** 2
    var_g = tF.conv2d(g * g, k, padding=pad) - mu_g ** 2
    cov = tF.conv2d(p * g, k, padding=pad) - mu_p * mu_g
    ssim = ((2 * mu_p * mu_g + c1) * (2 * cov + c2)
            / ((mu_p ** 2 + mu_g ** 2 + c1) * (var_p + var_g + c2)))
    m = mask.expand(b, c, h, w).reshape(b * c, 1, h, w)
    return float(_masked_mean(ssim, m))


def masked_cc(pred: Tensor, gt: Tensor, mask: Tensor) -> float:
    """Pearson correlation per band over masked pixels, averaged over bands.
    ``[B,C,H,W]``; a constant band (zero variance) contributes 0, not NaN."""
    m = mask.expand_as(pred)
    n = m.sum(dim=(0, 2, 3)).clamp_min(1.0)
    mp = (pred * m).sum(dim=(0, 2, 3)) / n
    mg = (gt * m).sum(dim=(0, 2, 3)) / n
    dp = (pred - mp.view(1, -1, 1, 1)) * m
    dg = (gt - mg.view(1, -1, 1, 1)) * m
    cov = (dp * dg).sum(dim=(0, 2, 3))
    denom = (dp.pow(2).sum(dim=(0, 2, 3)) * dg.pow(2).sum(dim=(0, 2, 3))).sqrt()
    cc = torch.where(denom > 1e-12, cov / denom.clamp_min(1e-12),
                     torch.zeros_like(cov))
    return float(cc.mean())


def masked_ergas(pred: Tensor, gt: Tensor, mask: Tensor, ratio: float = 1.0) -> float:
    """ERGAS = 100 * ratio * sqrt(mean_b((RMSE_b / mean_gt_b)^2)), masked.
    ``ratio`` = h/l resolution ratio; 1 for same-resolution cloud removal."""
    m = mask.expand_as(pred)
    n = m.sum(dim=(0, 2, 3)).clamp_min(1.0)
    rmse = (((pred - gt) ** 2 * m).sum(dim=(0, 2, 3)) / n).sqrt()
    mean_gt = ((gt * m).sum(dim=(0, 2, 3)) / n).clamp_min(1e-6)
    return float(100.0 * ratio * ((rmse / mean_gt) ** 2).mean().sqrt())


def masked_lpips_rgb(pred_rgb_disp: Tensor, gt_rgb_disp: Tensor) -> float:
    """LPIPS (alex) on display-range RGB in [-1,1], full frame (unmasked). Imports ``lpips``
    lazily; the first call downloads the network weights."""
    import lpips as _lpips
    if not hasattr(masked_lpips_rgb, "_net"):
        masked_lpips_rgb._net = _lpips.LPIPS(net="alex", verbose=False).eval()
    net = masked_lpips_rgb._net.to(pred_rgb_disp.device)
    with torch.no_grad():
        return float(net(pred_rgb_disp.clamp(-1, 1), gt_rgb_disp.clamp(-1, 1)).mean())


def strip_ms_scope(metrics: dict[str, float]) -> dict[str, float]:
    """Drop every ``*_all`` / ``*_nonrgb`` key: when the z_ms half is not predicted those
    numbers would score a constant plane, not the model."""
    return {k: v for k, v in metrics.items()
            if not k.endswith(("_all", "_nonrgb"))}


def masked_metrics(pred_refl: Tensor, gt_refl: Tensor, mask: Tensor) -> dict[str, float]:
    """``pred/gt [B,13,H,W]`` reflectance [0,1]; ``mask [B,1,H,W]`` 1 = scored."""
    if pred_refl.shape[1] != 13:
        raise ValueError(f"expected 13 bands, got {pred_refl.shape[1]}")
    out: dict[str, float] = {}
    groups = {"all": list(range(13)), "rgb": list(RGB_IDX), "nonrgb": list(MS_IDX)}
    for name, idx in groups.items():
        p, g = pred_refl[:, idx].clamp(0, 1), gt_refl[:, idx].clamp(0, 1)
        m = mask.expand(-1, len(idx), -1, -1)
        out[f"psnr_{name}"] = masked_psnr(p, g, m)
        out[f"mae_{name}"] = float(_masked_mean((p - g).abs(), m))
        out[f"rmse_{name}"] = float(_masked_mean((p - g) ** 2, m).sqrt())
        out[f"ssim_{name}"] = masked_ssim(p, g, mask)
    out["sam_all"] = masked_sam_deg(pred_refl.clamp(0, 1), gt_refl.clamp(0, 1), mask)
    out["cc_all"] = masked_cc(pred_refl.clamp(0, 1), gt_refl.clamp(0, 1), mask)
    out["ergas_all"] = masked_ergas(pred_refl.clamp(0, 1), gt_refl.clamp(0, 1), mask)
    return out
