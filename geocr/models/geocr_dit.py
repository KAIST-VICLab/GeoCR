"""GeoCR flow transformer: FLUX.2 [klein] 4B geometry conditioned on a set of observation latents.

Time convention: t = 0 noise, t = 1 data, and the model predicts ``u* = z1 - eps``. The Klein
checkpoint uses the opposite convention, so ``time_in`` receives ``1 - t`` and the output is negated.
``img_in`` projects the noisy target ``[z_rgb | z_ms]``; cloudy S2 observations use the same tensor
(``cond_cloud``) and SAR its own ``cond_sar`` on the ``z_rgb`` slot. Padded slots and unconditional
samples are replaced by a learned null token.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn

from geocr.models import flux2 as F2
from geocr.models.obs_embed import MOD_NULL, MOD_S1, ObsEmbedder

_KLEIN_GEOM = dict(hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=20)


@dataclass
class GeoCRParams:
    target_channels: int = 256
    cond_channels: int = 256
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0


class GeoCRDiT(nn.Module):
    """``forward(z_t, t, obs_lat, obs_mod, uncond) -> v`` with v matching ``u* = z1 - eps``."""

    def __init__(self, params: GeoCRParams | None = None):
        super().__init__()
        p = params or GeoCRParams()
        self.params = p
        h, nh = p.hidden_size, p.num_heads
        pe_dim = h // nh
        if sum(p.axes_dim) != pe_dim:
            raise ValueError(f"axes_dim {p.axes_dim} must sum to head_dim {pe_dim}")

        self.img_in = nn.Linear(p.target_channels, h, bias=False)
        self.sar_channels = p.cond_channels // 2
        self.cond_sar = nn.Linear(self.sar_channels, h, bias=False)
        self.time_in = F2.MLPEmbedder(in_dim=256, hidden_dim=h, disable_bias=True)
        self.pe_embedder = F2.EmbedND(dim=pe_dim, theta=p.theta, axes_dim=p.axes_dim)
        self.obs_embed = ObsEmbedder(h)
        self.null_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)

        self.double_blocks = nn.ModuleList(
            [F2.DoubleStreamBlock(h, nh, mlp_ratio=p.mlp_ratio) for _ in range(p.depth)])
        self.single_blocks = nn.ModuleList(
            [F2.SingleStreamBlock(h, nh, mlp_ratio=p.mlp_ratio) for _ in range(p.depth_single_blocks)])
        self.double_stream_modulation_img = F2.Modulation(h, double=True, disable_bias=True)
        self.double_stream_modulation_txt = F2.Modulation(h, double=True, disable_bias=True)
        self.single_stream_modulation = F2.Modulation(h, double=False, disable_bias=True)
        self.final_layer = F2.LastLayer(h, p.target_channels)
        self._pe_cache: dict = {}
        self._pe_cache_ids: dict = {}

    @property
    def cond_cloud(self) -> nn.Linear:
        """Cloudy-S2 conditions share the target projection: one tensor, one state-dict key."""
        return self.img_in

    @staticmethod
    def tokenize(z: Tensor) -> Tensor:
        return rearrange(z, "b c h w -> b (h w) c")

    @staticmethod
    def untokenize(v: Tensor, gh: int, gw: int) -> Tensor:
        return rearrange(v, "b (h w) c -> b c h w", h=gh, w=gw)

    def _ids(self, t_axis: int, gh: int, gw: int, device) -> Tensor:
        t = torch.full((1,), float(t_axis), device=device)
        h = torch.arange(gh, device=device, dtype=torch.float32)
        w = torch.arange(gw, device=device, dtype=torch.float32)
        lvl = torch.zeros(1, device=device)
        return torch.cartesian_prod(t, h, w, lvl)          # [gh*gw, 4]

    def _pe(self, t_axis: int, gh: int, gw: int, batch: int, device) -> Tensor:
        key = (t_axis, gh, gw, batch, str(device))
        if key not in self._pe_cache:
            ids = self._ids(t_axis, gh, gw, device)[None].expand(batch, -1, -1)
            self._pe_cache[key] = self.pe_embedder(ids)
        return self._pe_cache[key]

    def _ids_obs(self, b: int, s: int, gh: int, gw: int, device) -> Tensor:
        """ids ``[B, S*gh*gw, 4]``: spatial axes shared with the target ``(0, h, w, 0)``, 4th axis = 1.
        Every observation block gets the same ids, so blocks are told apart by content and modality."""
        key = ("obs", b, s, gh, gw, str(device))
        if key not in self._pe_cache_ids:
            h = torch.arange(gh, device=device, dtype=torch.float32)
            w = torch.arange(gw, device=device, dtype=torch.float32)
            hh, ww = torch.meshgrid(h, w, indexing="ij")
            spat = torch.stack([hh, ww], dim=-1).reshape(1, 1, gh * gw, 2).expand(b, s, -1, -1)
            zeros = torch.zeros(b, s, gh * gw, 1, device=device)
            lvl = torch.ones(b, s, gh * gw, 1, device=device)
            self._pe_cache_ids[key] = torch.cat(
                [zeros, spat, lvl], dim=-1).reshape(b, s * gh * gw, 4)
        return self._pe_cache_ids[key]

    def _pe_null(self, batch: int, device) -> Tensor:
        key = ("null", batch, str(device))
        if key not in self._pe_cache:
            ids = torch.zeros(batch, 1, 4, device=device)
            self._pe_cache[key] = self.pe_embedder(ids)
        return self._pe_cache[key]

    def forward(self, z_t: Tensor, t: Tensor, obs_lat: Tensor, obs_mod: Tensor,
                uncond: Tensor | None = None) -> Tensor:
        """``z_t [B,256,gh,gw]``, ``t [B]`` (0 = noise, 1 = data), ``obs_lat [B,S,256,gh,gw]``,
        ``obs_mod [B,S]`` (``MOD_NULL`` = padded), ``uncond [B]`` bool -> velocity ``[B,256,gh,gw]``."""
        b, _c, gh, gw = z_t.shape
        dev = z_t.device
        vec = self.time_in(F2.timestep_embedding(1.0 - t, 256))     # Klein's time convention

        img = self.img_in(self.tokenize(z_t))
        pe_img = self._pe(0, gh, gw, b, dev)
        n_tgt = img.shape[1]

        emb = self.obs_embed(obs_mod)                                # [B,S,h]
        dead_all = obs_mod == MOD_NULL
        if uncond is not None:
            dead_all = dead_all | uncond[:, None]
        # Stream assignment comes from obs_mod, not slot order (one host sync).
        is_sar = (obs_mod == MOD_S1).any(dim=0).tolist()
        s2_toks, sar_toks = [], []
        for j in range(obs_lat.shape[1]):
            lat = self.tokenize(obs_lat[:, j])
            # Both projections run and are selected per element, so cond_sar is in every graph (DDP).
            tok = torch.where((obs_mod[:, j] == MOD_S1)[:, None, None],
                              self.cond_sar(lat[..., :self.sar_channels]),
                              self.cond_cloud(lat)) + emb[:, j][:, None]
            tok = torch.where(dead_all[:, j][:, None, None],
                              self.null_token.to(tok.dtype), tok)
            # The fp32 modality embedding promotes tok under bf16 autocast; match the target tokens.
            (sar_toks if is_sar[j] else s2_toks).append(tok.to(img.dtype))
        if s2_toks:
            img = torch.cat([img] + s2_toks, dim=1)
            pe_img = torch.cat(
                [pe_img, self.pe_embedder(self._ids_obs(b, len(s2_toks), gh, gw, dev))], dim=2)
        if sar_toks:
            cond = torch.cat(sar_toks, dim=1)
            pe_cond = self.pe_embedder(self._ids_obs(b, len(sar_toks), gh, gw, dev))
        else:                                                        # no SAR: the null token
            cond = self.null_token.expand(b, 1, -1).to(img.dtype)
            pe_cond = self._pe_null(b, dev)
        n_cond = cond.shape[1]

        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)
        for blk in self.double_blocks:
            img, cond = blk(img, cond, pe_img, pe_cond, mod_img, mod_txt)

        x = torch.cat((cond, img), dim=1)
        pe = torch.cat((pe_cond, pe_img), dim=2)
        mod_s = self.single_stream_modulation(vec)[0]
        for blk in self.single_blocks:
            x = blk(x, pe, mod_s, n_cond)
        tgt = x[:, n_cond:n_cond + n_tgt]
        return -self.untokenize(self.final_layer(tgt, vec), gh, gw)  # Klein predicts eps - z1

    @torch.no_grad()
    def load_klein(self, state: dict) -> dict:
        """Initialise from a FLUX.2 [klein] 4B state dict.

        ``img_in`` (= ``cond_cloud``) RGB columns and ``final_layer.linear`` RGB rows come from Klein and
        the ``z_ms`` columns/rows start at zero, so the model is function-preserving at initialisation
        only. ``cond_sar`` takes Klein's ``img_in`` verbatim. Second-stream block weights and modulation
        are cloned from the first stream; Klein's ``txt_in`` is not used.
        """
        p = self.params
        if {k: getattr(p, k) for k in _KLEIN_GEOM} != _KLEIN_GEOM:
            raise ValueError(f"load_klein requires Klein-4B geometry {_KLEIN_GEOM}")
        own = dict(self.named_parameters())
        loaded, cloned = [], []

        rgb = slice(0, 128)
        self.img_in.weight.zero_()
        self.img_in.weight[:, rgb] = state["img_in.weight"]
        self.cond_sar.weight.copy_(state["img_in.weight"])
        loaded += ["img_in.weight(cols 0:128) [= cond_cloud]", "cond_sar.weight(verbatim)"]

        self.final_layer.linear.weight.zero_()
        self.final_layer.linear.weight[rgb] = state["final_layer.linear.weight"]
        loaded.append("final_layer.linear.weight(rows 0:128)")

        for name, param in own.items():
            if name.startswith(("img_in.", "cond_sar.", "obs_embed.", "null_token")):
                continue
            if name == "final_layer.linear.weight":
                continue
            if ".txt_" in name or name.startswith("double_stream_modulation_txt."):
                cloned.append(name)
            elif name in state and state[name].shape == param.shape:
                param.copy_(state[name])
                loaded.append(name)
        for name in cloned:
            src = name.replace(".txt_", ".img_") if ".txt_" in name else \
                name.replace("double_stream_modulation_txt.", "double_stream_modulation_img.", 1)
            own[name].copy_(own[src])
        random = [n for n in own
                  if n not in set(loaded) and n not in set(cloned)
                  and not n.startswith(("obs_embed.", "null_token"))
                  and n not in ("img_in.weight", "cond_sar.weight", "final_layer.linear.weight")]
        fresh = ["obs_embed", "null_token",
                 "img_in z_ms columns (zero; = cond_cloud)",
                 "final_layer z_ms rows (zero)"]
        return {"loaded": len(loaded), "cloned_txt_stream": len(cloned),
                "randomly_initialised": random,
                "fresh_modules": fresh}
