"""Batch -> target latent and observation-set tensors for ``GeoCRDiT``.

Slot layout ``[K S2 frames | used SAR slots]``; every slot is one ``gh x gw`` grid of 256 latent
channels. A SAR slot that no batch element uses is skipped; elements without SAR in a kept slot are
marked ``MOD_NULL`` and replaced by the null token inside the DiT.
"""
from __future__ import annotations

import torch
from torch import Tensor

from geocr.models.obs_embed import MOD_NULL, MOD_S1, MOD_S2
from geocr.models.s2_codec import DualS2Codec


@torch.no_grad()
def encode_target(codec: DualS2Codec, batch: dict) -> Tensor:
    return codec.encode_s2(batch["target_rgb"], batch["target_ms"],
                           stem=batch.get("ms_stem"))


@torch.no_grad()
def assemble_obs(codec: DualS2Codec, batch: dict) -> dict[str, Tensor]:
    rgb = batch["frames_rgb"]                       # [B,K,3,H,W]
    b, k = rgb.shape[:2]
    s1, s1_valid = batch["s1"], batch["s1_valid"]   # [B,S1,2,H,W], [B,S1]
    slots_lat, slots_mod = [], []

    for j in range(k):
        z = codec.encode_s2(rgb[:, j], batch["frames_ms"][:, j],
                            stem=batch.get("ms_stem"))
        slots_lat.append(z)
        slots_mod.append(torch.full((b,), MOD_S2, device=z.device, dtype=torch.long))

    for j in range(s1.shape[1]):
        if not s1_valid[:, j].any():
            continue
        z = codec.encode_sar(s1[:, j])
        slots_lat.append(z)
        slots_mod.append(torch.where(s1_valid[:, j].to(z.device),
                                     torch.tensor(MOD_S1, device=z.device),
                                     torch.tensor(MOD_NULL, device=z.device)))

    return {
        "obs_lat": torch.stack(slots_lat, dim=1),
        "obs_mod": torch.stack(slots_mod, dim=1),
    }
