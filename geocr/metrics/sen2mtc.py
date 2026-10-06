"""Sen2_MTC_New metrics, following the DiffCR / EMRDM evaluation code: reflectance -> DN
(x1e4, round, clamp [0, 10000]) -> clip [0, 2000] -> per-image global min-max stretch to [0,1]
(prediction and reference stretched independently), RGB bands only. PSNR with data range 1;
SSIM = 11x11 Gaussian (sigma 1.5), zero padding, C1 = 0.01^2, C2 = 0.03^2; LPIPS (alex) on the
stretched image mapped to [-1,1]. Per-image metrics, averaged over the set.
"""
from __future__ import annotations

import torch
from torch import Tensor

from geocr.metrics.masked import masked_psnr, masked_ssim


def mtc_stretch(refl_rgb: Tensor) -> Tensor:
    """``[3,H,W]`` reflectance (R,G,B) -> [0,1]: integer DN, clip [0,2000], global min-max.
    A constant image maps to ones."""
    v = (refl_rgb * 10000.0 + 0.5).clamp(0, 10000).floor().clamp(0, 2000)
    v = v - v.min()
    m = v.max()
    return torch.ones_like(v) if m == 0 else v / m


def sen2mtc_metrics(pred_refl_rgb: Tensor, gt_refl_rgb: Tensor,
                    lpips: bool = False) -> dict[str, float]:
    """``[B,3,H,W]`` reflectance in R,G,B order. Returns means of per-image
    ``mtc_psnr``, ``mtc_ssim`` (+ ``mtc_lpips`` on request)."""
    psnrs, ssims, lps = [], [], []
    ones = torch.ones(1, 1, *pred_refl_rgb.shape[-2:], device=pred_refl_rgb.device)
    for bi in range(pred_refl_rgb.shape[0]):
        p = mtc_stretch(pred_refl_rgb[bi]).unsqueeze(0)
        g = mtc_stretch(gt_refl_rgb[bi]).unsqueeze(0)
        psnrs.append(masked_psnr(p, g, ones.expand_as(p)))
        ssims.append(masked_ssim(p, g, ones))
        if lpips:
            from geocr.metrics.masked import masked_lpips_rgb
            lps.append(masked_lpips_rgb(p * 2.0 - 1.0, g * 2.0 - 1.0))
    out = {"mtc_psnr": sum(psnrs) / len(psnrs), "mtc_ssim": sum(ssims) / len(ssims)}
    if lpips:
        out["mtc_lpips"] = sum(lps) / len(lps)
    return out
