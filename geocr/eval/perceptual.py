"""FID, KID, DISTS and LPIPS on RGB views of predictions and references."""
from __future__ import annotations

import contextlib
import math
import sys

import numpy as np
import torch

MAX_FEATURE_N = 3000


def feature_ids(ids: list[str]) -> list[str]:
    """Every k-th id with k = ceil(N / 3000): the images the feature metrics are computed on."""
    return ids[::max(1, math.ceil(len(ids) / MAX_FEATURE_N))]


def to_rgb01(x: torch.Tensor, lineage: str) -> torch.Tensor:
    """``[C,H,W]`` canonical-domain image -> ``[3,H,W]`` RGB view in [0, 1]."""
    t = x.to(torch.float32)
    if lineage == "sen2mtc":
        from geocr.metrics.sen2mtc import mtc_stretch
        return mtc_stretch(t[:3]).clamp(0, 1)
    if lineage == "masked13":
        from geocr.normalize import RGB_IDX, RGBVariantNorm
        return RGBVariantNorm().to_u01(t[list(RGB_IDX)]).clamp(0, 1)
    return t[:3].clamp(0, 1)


@torch.no_grad()
def perceptual(pred: torch.Tensor, gt: torch.Tensor, metrics, device="cpu", batch: int = 8,
               kid_seed: int = 0) -> dict[str, float]:
    """``pred``/``gt``: ``[N,3,H,W]`` RGB views in [0, 1]. LPIPS (AlexNet, v0.1) and DISTS
    average per-image scores. FID and KID use 2048-d Inception pool-3 features of the 8-bit
    images; KID averages 50 subsets of size min(1000, N/2) drawn with ``kid_seed``."""
    out, n = {}, len(pred)
    for name in ("lpips", "dists"):
        if name in metrics:
            import pyiqa
            with contextlib.redirect_stdout(sys.stderr):
                net = pyiqa.create_metric(name, device=device)
            vals = []
            for i in range(0, n, batch):
                vals += net(pred[i:i + batch].to(device), gt[i:i + batch].to(device)).flatten().tolist()
            out[name] = float(np.mean(vals))
    nets = {}
    if "fid" in metrics:
        from torchmetrics.image.fid import FrechetInceptionDistance
        nets["fid"] = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    if "kid" in metrics:
        from torchmetrics.image.kid import KernelInceptionDistance
        nets["kid"] = KernelInceptionDistance(feature=2048, subsets=50, normalize=False,
                                              subset_size=max(2, min(1000, n // 2))).to(device)
    for real, x in ((True, gt), (False, pred)):
        u8 = (x * 255).round().clamp(0, 255).to(torch.uint8)
        for i in range(0, n, batch):
            for m in nets.values():
                m.update(u8[i:i + batch].to(device), real=real)
    if "fid" in nets:
        out["fid"] = float(nets["fid"].compute())
    if "kid" in nets:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(kid_seed)
            kid, kid_std = nets["kid"].compute()
        out["kid"], out["kid_std"] = float(kid), float(kid_std)
    return out
