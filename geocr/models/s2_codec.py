"""Dual-latent S2 codec: ``[z_rgb | z_ms]`` on one token grid.

RGB (B4/B3/B2) goes through the frozen FLUX.2 autoencoder; the non-RGB bands and SAR go through
``MultiStemAE`` stems around the same frozen trunk. The codec is frozen while the generator trains.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from geocr.models.flux2_ae import Flux2AE
from geocr.models.multistem_ae import MultiStemAE, stem_width

LATENT_C = Flux2AE.latent_channels          # 128 per half


class DualS2Codec(nn.Module):
    """``ms_stems`` are the non-RGB stem keys; the SAR stem ``2`` is always built, even when
    ``ms_enabled`` is False (then ``z_ms`` is zero for every S2 input)."""

    def __init__(self, ae: Flux2AE, ms_enabled: bool = True, ms_stems: tuple[int | str, ...] = (10,)):
        super().__init__()
        ae.requires_grad_(False).eval()
        self.ae = [ae]                       # out of the module tree, like MultiStemAE
        self.ms_enabled = ms_enabled
        self.ms_stems = tuple(ms_stems)
        counts = (*self.ms_stems, 2) if ms_enabled else (2,)
        self.stems = MultiStemAE(ae, channel_counts=counts)

    @torch.no_grad()
    def encode_s2(self, rgb: Tensor, ms: Tensor, stem: int | str | None = None) -> Tensor:
        """``[B,3,H,W]``, ``[B,n,H,W]`` -> ``[B,256,gh,gw]``. ``stem`` selects the non-RGB stem
        (``None``: the stem keyed by ``ms.shape[1]``); a zero-channel ``ms`` gives a zero ``z_ms``."""
        z_rgb = self.ae[0].encode(rgb)
        if self.ms_enabled and ms.shape[1] > 0:
            z_ms = self.stems.encode(ms, ms.shape[1] if stem is None else stem)
        else:
            z_ms = torch.zeros_like(z_rgb)
        return torch.cat([z_rgb, z_ms], dim=1)

    @torch.no_grad()
    def decode_s2(self, z: Tensor, stem: int | str = 10) -> tuple[Tensor, Tensor]:
        z_rgb, z_ms = z[:, :LATENT_C], z[:, LATENT_C:]
        rgb = self.ae[0].decode(z_rgb)
        ms = self.stems.decode(z_ms, stem) if self.ms_enabled \
            else rgb.new_zeros(rgb.shape[0], stem_width(stem), *rgb.shape[-2:])
        return rgb, ms

    @torch.no_grad()
    def encode_sar(self, s1: Tensor) -> Tensor:
        """``[B,2,H,W]`` -> ``[B,256,gh,gw]``: SAR fills the ``z_rgb`` slot and ``z_ms`` is zero."""
        z = self.stems.encode(s1, 2)
        return torch.cat([z, torch.zeros_like(z)], dim=1)

    def load_stems(self, path: str) -> None:
        """Load ``stems.safetensors``: the stem tensors, with their keys listed in the
        ``channel_counts`` metadata. The frozen trunk is taken from this codec's autoencoder."""
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            counts = (f.metadata() or {}).get("channel_counts", "")
            state = {k: f.get_tensor(k) for k in f.keys()}
        if counts.split(",") != [str(k) for k in self.stems.channel_counts]:
            raise ValueError(f"{path} has channel_counts {counts!r}, "
                             f"this codec expects {self.stems.channel_counts}")
        ae = self.ae[0]
        state.update({f"trunk_enc.{k}": v for k, v in ae.encoder.state_dict().items()
                      if not k.startswith("conv_in.")})
        state.update({f"trunk_dec.{k}": v for k, v in ae.decoder.state_dict().items()
                      if not k.startswith("conv_out.")})
        self.stems.load_state_dict(state)
