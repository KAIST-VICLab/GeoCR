# Adapted from Black Forest Labs' FLUX.2 (Apache-2.0); modified for GeoCR.
"""Input/output stems around the shared frozen FLUX.2 autoencoder trunk.

Each stem key owns one ``conv_in`` and one ``conv_out``; everything between them is the frozen trunk.
A key is ``<width>[letter]``: ``"10"`` (ten-band TOA), ``"10b"`` (ten-band BOA), ``"1"`` (NIR) and
``"2"`` (VV/VH SAR). Stems are initialised in closed form from the RGB projections, which reproduces
the RGB codec exactly on channel-replicated input:

* ``encoder.conv_in`` sums over its input channels, so ``W_N[:, i] = W_3.sum(dim=1) / N`` (bias unchanged);
* ``decoder.conv_out`` reads each output channel independently, so ``W_N[i] = W_3.mean(dim=0)`` (bias mean).

``decoder.conv_in`` is not the partner of ``encoder.conv_in``; the symmetric pair is
``encoder.conv_in`` / ``decoder.conv_out``.
"""
from __future__ import annotations

import copy
import re

import torch
from torch import Tensor, nn

from geocr.models.flux2_ae import Flux2AE

__all__ = ["MultiStemAE", "init_stem_in", "init_stem_out", "stem_width"]


def stem_width(key: int | str) -> int:
    """Channel width of a stem key ``<width>[letter]``, e.g. ``"10b"`` -> 10."""
    m = re.match(r"(\d+)[a-z]?$", str(key))
    if not m:
        raise ValueError(f"stem key {key!r} is not '<width>' or '<width><letter>'")
    return int(m.group(1))


def init_stem_in(w3: Tensor, b3: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """``[ch,3,k,k] -> [ch,n,k,k]``: sum over input channels, split evenly."""
    if w3.shape[1] != 3:
        raise ValueError(f"expected a 3-channel input kernel, got {tuple(w3.shape)}")
    return w3.sum(dim=1, keepdim=True).repeat(1, n, 1, 1) / n, b3.clone()


def init_stem_out(w3: Tensor, b3: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """``[3,ch,k,k] -> [n,ch,k,k]``: every output channel starts as the RGB mean."""
    if w3.shape[0] != 3:
        raise ValueError(f"expected a 3-channel output kernel, got {tuple(w3.shape)}")
    return w3.mean(dim=0, keepdim=True).repeat(n, 1, 1, 1), b3.mean().repeat(n)


class MultiStemAE(nn.Module):
    """Stems ``stem_in[key]``/``stem_out[key]`` around a private copy of the frozen trunk.

    ``conv_in`` is the first operation of ``Encoder.forward`` and ``conv_out`` the last of
    ``Decoder.forward``, so replacing both by ``nn.Identity`` in the copy and applying the stem outside
    is exact. Latent packing and the BatchNorm statistics come from the frozen autoencoder, which is
    kept out of the module tree.
    """

    def __init__(self, ae: Flux2AE, channel_counts: tuple[int | str, ...] = (1, 2)):
        super().__init__()
        if not channel_counts:
            raise ValueError("channel_counts must not be empty")

        w_in = ae.encoder.conv_in.weight.detach().float()
        b_in = ae.encoder.conv_in.bias.detach().float()
        w_out = ae.decoder.conv_out.weight.detach().float()
        b_out = ae.decoder.conv_out.bias.detach().float()
        ch, _, k, _ = w_in.shape
        block_in = w_out.shape[1]

        self.trunk_enc = copy.deepcopy(ae.encoder)
        self.trunk_enc.conv_in = nn.Identity()
        self.trunk_dec = copy.deepcopy(ae.decoder)
        self.trunk_dec.conv_out = nn.Identity()
        self.trunk_enc.requires_grad_(False).eval()
        self.trunk_dec.requires_grad_(False).eval()

        self.stem_in = nn.ModuleDict()
        self.stem_out = nn.ModuleDict()
        for key in channel_counts:
            n = stem_width(key)
            si = nn.Conv2d(n, ch, kernel_size=k, stride=1, padding=k // 2)
            so = nn.Conv2d(block_in, n, kernel_size=k, stride=1, padding=k // 2)
            with torch.no_grad():
                si.weight.copy_(init_stem_in(w_in, b_in, n)[0])
                si.bias.copy_(b_in)
                w, b = init_stem_out(w_out, b_out, n)
                so.weight.copy_(w)
                so.bias.copy_(b)
            self.stem_in[str(key)] = si
            self.stem_out[str(key)] = so

        self.ae = [ae]                       # list: kept out of the module tree
        self.ps = tuple(ae.ps)
        self.spatial_factor = int(ae.spatial_factor)
        self.channel_counts = tuple(channel_counts)

    def _check(self, x: Tensor, key: int | str, what: str) -> None:
        if str(key) not in self.stem_in:
            raise ValueError(f"no stem {key!r}; have {self.channel_counts}")
        if x.ndim != 4:
            raise ValueError(f"{what} expects a 4-D tensor, got {tuple(x.shape)}")

    def encode(self, x: Tensor, key: int | str) -> Tensor:
        """``[B,n,H,W]`` in ``[-1,1]`` -> normalized packed latent ``[B,128,H/16,W/16]``."""
        self._check(x, key, "encode")
        n = stem_width(key)
        if x.shape[1] != n:
            raise ValueError(f"encode({key!r}) expects [B,{n},H,W], got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % self.spatial_factor or w % self.spatial_factor:
            raise ValueError(f"encode requires H and W divisible by {self.spatial_factor}, got {h}x{w}")
        feat = self.stem_in[str(key)](x)
        moments = self.trunk_enc(feat.to(next(self.trunk_enc.parameters()).dtype))
        mean = torch.chunk(moments, 2, dim=1)[0]
        return self.ae[0].normalize(Flux2AE.pack(mean, self.ps))

    def decode(self, z: Tensor, key: int | str) -> Tensor:
        """Normalized packed latent -> ``[B,n,H,W]`` in ``[-1,1]``."""
        self._check(z, key, "decode")
        h = self.trunk_dec(Flux2AE.unpack(self.ae[0].inv_normalize(z), self.ps))
        return self.stem_out[str(key)](h.to(next(self.stem_out[str(key)].parameters()).dtype))
