"""DINO similarity: cosine between spatially aligned patch tokens of the frozen DINOv3-SAT
ViT-L/16, excluding CLS and register tokens, averaged over patches and images.

The SAT-493M weights are distributed by Meta under the DINOv3 License. Pass the downloaded
``.pth`` file under its original name (``dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth``): the
name selects the SAT variant of the architecture.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

SAT_MEAN, SAT_STD = (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)


def load_dinov3_sat(weights, repo=None, device="cpu"):
    """ViT-L/16 from the DINOv3 code (``repo``: a clone of facebookresearch/dinov3, unless the
    ``dinov3`` package is installed) with the SAT-493M checkpoint strictly loaded; frozen, bf16."""
    if repo and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vitl16
    model = dinov3_vitl16(pretrained=False, weights=str(weights))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


@torch.no_grad()
def patch_tokens(model, x01: torch.Tensor) -> torch.Tensor:
    """``[B,3,H,W]`` RGB in [0, 1] -> ``[B, HW/256, 1024]`` patch tokens."""
    p = next(model.parameters())
    x = x01.to(device=p.device, dtype=torch.float32)
    m = torch.tensor(SAT_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(SAT_STD, device=x.device).view(1, 3, 1, 1)
    return model.forward_features(((x - m) / s).to(p.dtype))["x_norm_patchtokens"].float()


def dino_cos(pred: torch.Tensor, gt: torch.Tensor, model, batch: int = 8) -> float:
    """``pred``/``gt``: ``[N,3,H,W]`` RGB views in [0, 1]."""
    cos = []
    for i in range(0, len(pred), batch):
        fp = patch_tokens(model, pred[i:i + batch]).cpu()
        fg = patch_tokens(model, gt[i:i + batch]).cpu()
        cos.append(F.cosine_similarity(fp, fg, dim=-1).mean(dim=1))
    return float(torch.cat(cos).numpy().mean())
