"""Metrics for the 8-bit benchmarks, following each benchmark's evaluation code.

* CUHK-CR: quantise [0,1] floats with a truncating uint8 cast; PSNR = skimage over all bands
  jointly (data range 255); SSIM = mean over bands of grey skimage SSIM on the luma projection
  of the band replicated to 3 channels; LPIPS (alex) per band on replicated 0..255 floats.
  ``cuhk_metrics`` takes R,G,B,NIR; RGB-only scoring applies ``_cuhk_one`` to three bands.
* T-CLOUD: float [0,1], no quantisation or clamp; PSNR = 10 log10(1/mse) over 3xHxW;
  SSIM = skimage multichannel, data range 1.

All functions take ``[C,H,W]`` or ``[B,C,H,W]`` tensors in the display [0,1] domain
(png/255 for these datasets). skimage is imported lazily.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

#: Luma weights of the CUHK-CR evaluation code (R,G,B), applied verbatim.
_LUMA = np.array([0.298912, 0.586611, 0.114478], dtype=np.float64)


def _hwc_u8(x01: Tensor) -> np.ndarray:
    """[C,H,W] float [0,1] -> HWC uint8 with a truncating cast."""
    return np.clip(x01.detach().cpu().numpy() * 255.0, 0, 255) \
        .astype(np.uint8).transpose(1, 2, 0)


def _ssim_gray(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=data_range))


def _batched(fn, pred: Tensor, gt: Tensor, **kw) -> dict[str, float]:
    if pred.ndim == 3:
        pred, gt = pred[None], gt[None]
    outs = [fn(pred[i], gt[i], **kw) for i in range(pred.shape[0])]
    return {k: sum(o[k] for o in outs) / len(outs) for k in outs[0]}


def _cuhk_one(pred01: Tensor, gt01: Tensor, lpips: bool) -> dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio
    a, b = _hwc_u8(pred01), _hwc_u8(gt01)
    out = {"cuhk_psnr": float(peak_signal_noise_ratio(b, a, data_range=255))}
    ssims, lps = [], []
    for c in range(a.shape[-1]):
        a3 = np.repeat(a[..., c:c + 1], 3, axis=-1).astype(np.float64)
        b3 = np.repeat(b[..., c:c + 1], 3, axis=-1).astype(np.float64)
        ssims.append(_ssim_gray(a3 @ _LUMA, b3 @ _LUMA, 255))
        if lpips:
            import lpips as _lp
            if not hasattr(_cuhk_one, "_net"):
                _cuhk_one._net = _lp.LPIPS(net="alex", verbose=False).eval()
            with torch.no_grad():
                lps.append(float(_cuhk_one._net(
                    torch.from_numpy(a3.transpose(2, 0, 1)).float()[None],
                    torch.from_numpy(b3.transpose(2, 0, 1)).float()[None])))
    out["cuhk_ssim"] = sum(ssims) / len(ssims)
    if lpips:
        out["cuhk_lpips"] = sum(lps) / len(lps)
    return out


def cuhk_metrics(pred01_4band: Tensor, gt01_4band: Tensor,
                 lpips: bool = False) -> dict[str, float]:
    if pred01_4band.shape[-3] != 4:
        raise ValueError("cuhk_metrics expects 4 bands (R,G,B,NIR)")
    return _batched(_cuhk_one, pred01_4band, gt01_4band, lpips=lpips)


def _tcloud_one(pred01: Tensor, gt01: Tensor) -> dict[str, float]:
    from skimage.metrics import structural_similarity
    a = pred01.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
    b = gt01.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    return {
        "tcloud_psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
        "tcloud_ssim": float(structural_similarity(
            a, b, data_range=1, channel_axis=-1)),
    }


def tcloud_metrics(pred01: Tensor, gt01: Tensor) -> dict[str, float]:
    return _batched(_tcloud_one, pred01, gt01)
