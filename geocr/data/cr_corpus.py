"""Pretraining mixture loader.

One batch = one dataset (and one K), drawn deterministically from (seed, step), so tensors
stack without padding and every batch is a pure function of (seed, step, rank). The members are
the ``benchmarks`` providers plus the AllClear sampler; this module picks between them and
cuts each batch to its band family:

- ``ms10`` (AllClear, SEN12MS-CR, WHUS2-CRv): 13 bands -> rgb(3) + ms(10).
- ``nir1`` (Sen2_MTC_New, CUHK-CR): rgb(3) + ms(1) = B8 only, cut from the provider's
  zero-filled 10-band tensor before the codec sees it.
- ``nir1_cond_only`` (Sen2_MTC_Old): NIR on the conditions only; the target is RGB.
- ``none`` (T-CLOUD, RICE): no multispectral observation; ms has 0 channels, the codec writes
  z_ms = 0 and ``ms_loss_w`` = 0 removes the z_ms half from the loss.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from geocr.config import DataCfg
from geocr.data.benchmarks import PROVIDERS
from geocr.data.dataset import AllClearObsDataset
from geocr.data.index import AllClearIndex
from geocr.data.sampler import ObservationBatchSampler
from geocr.normalize import MS_IDX

#: Band family per member. "cuhk_cr2" and "rice2" are separate members because their
#: providers take one sub-dataset per root.
FAMILY: dict[str, str] = {
    "allclear": "ms10", "sen12mscr": "ms10", "whus2crv": "ms10",
    "sen2mtc_new": "nir1", "cuhk_cr": "nir1", "cuhk_cr2": "nir1",
    "sen2mtc_old": "nir1_cond_only",
    "tcloud": "none", "rice": "none", "rice2": "none",
}
#: Member name -> benchmarks provider KIND ("allclear" uses the AllClear sampler).
PROVIDER_OF: dict[str, str] = {
    n: n for n in FAMILY if n != "allclear"} | {"cuhk_cr2": "cuhk_cr",
                                               "rice2": "rice"}
#: MS stem key per member, by width and radiometry; RGB-only members have none.
STEM_KEY: dict[str, str | None] = {
    "allclear": "10", "sen12mscr": "10", "whus2crv": "10b",
    "sen2mtc_new": "1", "cuhk_cr": "1", "cuhk_cr2": "1", "sen2mtc_old": "1",
    "tcloud": None, "rice": None, "rice2": None,
}
#: Position of B8 (13-band index 7) inside the 10-band ms tensor.
NIR_POS = MS_IDX.index(7)

#: Keys with a leading K (frame) axis, sliced together by the K-ramp cap.
_FRAME_KEYS = ("frames_rgb", "frames_ms")
#: Keys whose trailing two dims are the image grid, cropped together.
_HW_KEYS = ("frames_rgb", "frames_ms", "s1",
            "target_rgb", "target_ms", "target_refl", "eval_mask")


def stem_key_for_source(source: str) -> str | None:
    """MS stem key for a single-benchmark ``data.source`` (a provider KIND): the key the
    pretraining used for that corpus, or None (no member maps to it, or it has no stem)."""
    keys = {STEM_KEY[n] for n, kind in PROVIDER_OF.items() if kind == source}
    if len(keys) > 1:
        raise ValueError(f"provider kind {source!r} serves cr_corpus members "
                         f"with different stem keys {sorted(map(str, keys))}")
    return keys.pop() if keys else None


def family_for_source(source: str) -> str:
    """Band family for a single-benchmark ``data.source``; kinds that are not mixture members
    (the RGB-only renderings) are "none"."""
    fams = {FAMILY[n] for n, kind in PROVIDER_OF.items() if kind == source}
    if len(fams) > 1:
        raise ValueError(f"provider kind {source!r} serves cr_corpus members "
                         f"with different families {sorted(fams)}")
    return fams.pop() if fams else "none"


def apply_family_slice(batch: dict, fam: str) -> dict:
    """Cut ``frames_ms``/``target_ms`` to the family's real MS channels and set ``ms_loss_w``.
    Shared by pretraining and finetuning. Mutates and returns ``batch``."""
    if fam == "nir1":
        batch["frames_ms"] = batch["frames_ms"][:, :, [NIR_POS]]
        batch["target_ms"] = batch["target_ms"][:, [NIR_POS]]
        batch["ms_loss_w"] = 1.0
    elif fam == "nir1_cond_only":
        batch["frames_ms"] = batch["frames_ms"][:, :, [NIR_POS]]
        batch["target_ms"] = batch["target_ms"][:, :0]
        batch["ms_loss_w"] = 0.0
    elif fam == "none":
        batch["frames_ms"] = batch["frames_ms"][:, :, :0]
        batch["target_ms"] = batch["target_ms"][:, :0]
        batch["ms_loss_w"] = 0.0
    elif fam == "ms10":
        batch["ms_loss_w"] = 1.0
    else:
        raise ValueError(f"unhandled band family {fam!r}")
    return batch


def _dataset_seed(seed: int, name: str) -> int:
    """Per-member seed derived from the member NAME, so adding a member does not reshuffle
    the others."""
    return (seed ^ zlib.crc32(name.encode())) & 0x7FFFFFFF


class _AllClearLoader:
    """Adapts the (sampler, dataset) pair to the providers' ``load(step)``."""

    def __init__(self, root: str, split: str, batch_size: int, seed: int,
                 cfg: DataCfg, draws_per_opt_step: int):
        index = AllClearIndex(root)
        self.ds = AllClearObsDataset(index)
        self.sampler = ObservationBatchSampler(
            index, index.split_rois(split, cfg.split_dir), cfg,
            batch_size, seed, draws_per_opt_step)
        self.s1_max = cfg.s1_max

    def load(self, step: int) -> dict:
        return self.ds.load_batch(self.sampler.batch(step), self.s1_max)


class CRCorpus:
    """``load(step)`` over a weighted union of CR datasets.

    The dataset draw uses its own PCG64 stream, independent of every member's stream; each
    member gets a seed derived from its name. During the K ramp the frame count is capped
    exactly as in ``ObservationBatchSampler`` (counted in optimizer steps).
    """

    _DRAW_TAG = 0xC0FFEE

    def __init__(self, roots: dict[str, str], weights: dict[str, float],
                 split: str, batch_size: int, seed: int, cfg: DataCfg,
                 draws_per_opt_step: int = 1):
        if set(roots) != set(weights) or not roots:
            raise ValueError(f"roots {sorted(roots)} and weights "
                             f"{sorted(weights)} must be the same non-empty set")
        unknown = sorted(set(roots) - set(FAMILY))
        if unknown:
            raise ValueError(f"unknown corpus dataset(s) {unknown}")
        self.cfg = cfg
        self.seed = seed
        self.draws_per_opt_step = max(1, int(draws_per_opt_step))
        self.names = sorted(roots)
        w = np.array([float(weights[n]) for n in self.names])
        if (w <= 0).any():
            raise ValueError("corpus weights must be > 0")
        self.probs = w / w.sum()
        self.loaders = {}
        for n in self.names:
            ds_seed = _dataset_seed(seed, n)
            if n == "allclear":
                self.loaders[n] = _AllClearLoader(
                    roots[n], split, batch_size, ds_seed, cfg,
                    self.draws_per_opt_step)
            else:
                self.loaders[n] = PROVIDERS[PROVIDER_OF[n]](
                    roots[n], split, batch_size, ds_seed, cfg)

    def _rng(self, step: int) -> np.random.Generator:
        s = (self.seed ^ self._DRAW_TAG) & 0xFFFFFFFF
        return np.random.default_rng(np.random.PCG64((s << 32) ^ (step & 0xFFFFFFFF)))

    def _k_cap(self, step: int) -> int:
        """Highest K the ramp allows at this draw."""
        cfg = self.cfg
        if not cfg.k_ramp_steps:
            return max(cfg.k_choices)
        opt_step = step // self.draws_per_opt_step
        hi = 1 + int((len(cfg.k_choices) - 1)
                     * min(1.0, opt_step / cfg.k_ramp_steps))
        return cfg.k_choices[hi - 1]

    def _crop(self, batch: dict, rng: np.random.Generator) -> dict:
        """Random crop to ``cfg.resolution`` for members with larger tiles (WHUS2-CRv, 384 px):
        one window per batch element, shared by that element's tensors."""
        res = self.cfg.resolution
        h, w = batch["target_rgb"].shape[-2:]
        if (h, w) == (res, res):
            return batch
        if h < res or w < res:
            raise ValueError(f"tile {h}x{w} smaller than resolution {res}")
        b = batch["target_rgb"].shape[0]
        ys = rng.integers(0, h - res + 1, size=b)
        xs = rng.integers(0, w - res + 1, size=b)
        for key in _HW_KEYS:
            if key not in batch or batch[key].shape[-2:] != (h, w):
                continue
            t = batch[key]
            batch[key] = torch.stack(
                [t[i, ..., ys[i]:ys[i] + res, xs[i]:xs[i] + res]
                 for i in range(b)])
        return batch

    def _apply_family(self, batch: dict, name: str, step: int) -> dict:
        batch = apply_family_slice(batch, FAMILY[name])
        cap = self._k_cap(step)
        if batch["frames_rgb"].shape[1] > cap:
            for key in _FRAME_KEYS:
                batch[key] = batch[key][:, :cap]
            batch["k"] = cap
        batch["ds_name"] = name
        if STEM_KEY[name]:
            batch["ms_stem"] = STEM_KEY[name]
        return batch

    def load(self, step: int) -> dict:
        rng = self._rng(step)
        name = str(rng.choice(self.names, p=self.probs))
        batch = self.loaders[name].load(step)
        if name != "allclear":
            batch = self._crop(batch, rng)
        return self._apply_family(batch, name, step)
