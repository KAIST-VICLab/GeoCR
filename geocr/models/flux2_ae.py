# Adapted from Black Forest Labs' FLUX.2 (Apache-2.0); modified for GeoCR.
"""Frozen FLUX.2 autoencoder.

``ae.safetensors`` loads with ``strict=True`` (251 tensors). ``quant_conv`` lives inside ``encoder.*``
and ``post_quant_conv`` inside ``decoder.*``. Latents are normalised by an ``affine=False``
BatchNorm under ``bn.*`` whose running statistics ship in the checkpoint; ``normalize`` and
``inv_normalize`` force ``bn.eval()``, and ``inv_normalize`` reuses ``bn_eps`` so the pair
round-trips exactly. The encoder returns the posterior mean, packed 2x2 space-to-depth:
3x256x256 -> 128x16x16 (spatial factor 16). Callers wrap ``encode``/``decode`` in ``torch.no_grad()``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(self, ch: int, out_ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Flux2AE(nn.Module):
    """Frozen FLUX.2 autoencoder; submodule names (``encoder``, ``decoder``, ``bn``) match the checkpoint."""

    latent_channels: int = 128
    spatial_factor: int = 16

    def __init__(self, params: AutoEncoderParams | None = None):
        super().__init__()
        params = params or AutoEncoderParams()
        self.params = params
        self.encoder = Encoder(
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        if math.prod(self.ps) * params.z_channels != self.latent_channels:
            raise ValueError(f"packed latent channels {math.prod(self.ps) * params.z_channels} "
                             f"!= Flux2AE.latent_channels {self.latent_channels}")

    @staticmethod
    def pack(z: Tensor, ps: tuple[int, int] = (2, 2)) -> Tensor:
        """``[..., C, H, W] -> [..., C*pi*pj, H/pi, W/pj]``."""
        return rearrange(z, "... c (i pi) (j pj)  -> ... (c pi pj) i j", pi=ps[0], pj=ps[1])

    @staticmethod
    def unpack(z: Tensor, ps: tuple[int, int] = (2, 2)) -> Tensor:
        """Exact inverse of :meth:`pack`."""
        return rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=ps[0], pj=ps[1])

    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        """``[B,3,H,W]`` in ``[-1,1]`` -> ``[B,128,H/16,W/16]`` (posterior mean)."""
        if x.ndim != 4 or x.shape[1] != self.params.in_channels:
            raise ValueError(f"Flux2AE.encode expects [B,{self.params.in_channels},H,W], got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % self.spatial_factor or w % self.spatial_factor:
            raise ValueError(f"Flux2AE.encode requires H and W divisible by {self.spatial_factor}, got {h}x{w}")

        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = self.pack(mean, tuple(self.ps))
        return self.normalize(z)

    def decode(self, z: Tensor) -> Tensor:
        """``[B,128,h,w]`` -> ``[B,3,16h,16w]`` in ``[-1,1]``-ish."""
        if z.ndim != 4 or z.shape[1] != self.latent_channels:
            raise ValueError(f"Flux2AE.decode expects [B,{self.latent_channels},h,w], got {tuple(z.shape)}")
        z = self.inv_normalize(z)
        z = self.unpack(z, tuple(self.ps))
        return self.decoder(z)

    @classmethod
    def from_pretrained(cls, path: str, device: str | torch.device | None = None,
                        dtype: torch.dtype = torch.float32) -> "Flux2AE":
        """Load ``ae.safetensors`` with ``strict=True``; frozen, in eval mode, on ``device`` (default CPU)."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"FLUX.2 autoencoder weights not found at: {path}")

        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
        model = cls()
        model.load_state_dict(state, strict=True)
        model.to(device=device or "cpu", dtype=dtype)
        model.eval()
        model.requires_grad_(False)
        return model

    def train(self, mode: bool = True) -> "Flux2AE":
        """The autoencoder is frozen: never leave eval mode."""
        return super().train(False)
