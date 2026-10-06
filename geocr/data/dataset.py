"""SampleSpec -> normalised tensors. Memmaps are opened lazily per roi and cached.

``load_batch`` stacks frames directly because the sampler fixes K per batch; SAR is padded to
``s1_max`` with a validity mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from geocr.data.index import AllClearIndex
from geocr.data.sampler import BatchSpec, SampleSpec
from geocr.normalize import Norms, reflectance_from_dn


class AllClearObsDataset:
    def __init__(self, index: AllClearIndex, norms: Norms | None = None):
        self.index = index
        self.norms = norms or Norms()
        self._mm: dict[tuple[str, str], np.memmap | None] = {}

    def _arr(self, roi: str, name: str) -> np.memmap | None:
        key = (roi, name)
        if key not in self._mm:
            path: Path = self.index.roi_dir(roi) / f"{name}.npy"
            self._mm[key] = np.load(path, mmap_mode="r") if path.is_file() else None
        return self._mm[key]

    def s2_refl(self, roi: str, i: int) -> Tensor:
        arr = self._arr(roi, "s2")
        if arr is None:
            raise FileNotFoundError(f"roi {roi}: s2.npy missing")
        return reflectance_from_dn(torch.from_numpy(np.array(arr[i])))

    def cld(self, roi: str, i: int) -> tuple[Tensor, Tensor]:
        """(cloud probability in [0,1], evaluation mask: 1 = cloud- and shadow-free)."""
        arr = self._arr(roi, "cld")
        if arr is None:
            raise FileNotFoundError(f"roi {roi}: cld.npy missing")
        a = torch.from_numpy(np.array(arr[i]))
        return a[0].float() / 200.0, 1.0 - a[1].float().clamp(0, 1)

    def s1_db(self, roi: str, i: int) -> Tensor:
        arr = self._arr(roi, "s1")
        if arr is None:
            raise FileNotFoundError(f"roi {roi}: s1.npy missing but spec references it")
        return torch.from_numpy(np.array(arr[i])).float() / 100.0

    def load_sample(self, spec: SampleSpec, k: int, s1_max: int) -> dict:
        # The target scene must never be one of its own condition frames.
        if spec.target_i in spec.frame_is:
            raise ValueError(f"roi {spec.roi}: target scene i={spec.target_i} "
                             "appears among its own condition frames")
        t = self.index.tables[spec.roi]
        n = self.norms

        tgt = self.s2_refl(spec.roi, spec.target_i).unsqueeze(0)
        tgt_rgb, tgt_ms = n.s2.split(tgt)
        _, eval_mask = self.cld(spec.roi, spec.target_i)

        f_rgb = torch.zeros(k, 3, *tgt.shape[-2:])
        f_ms = torch.zeros(k, 10, *tgt.shape[-2:])
        # Frames are packed in date order (the model treats them as a set).
        for j, i in enumerate(sorted(spec.frame_is,
                                     key=lambda i: (int(t.s2_date[i]), i))):
            rgb, ms = n.s2.split(self.s2_refl(spec.roi, i).unsqueeze(0))
            f_rgb[j], f_ms[j] = rgb[0], ms[0]

        s1 = torch.zeros(s1_max, 2, *tgt.shape[-2:])
        s1_valid = torch.zeros(s1_max, dtype=torch.bool)
        for j, i in enumerate(spec.s1_is[:s1_max]):
            s1[j] = n.s1.to_model(self.s1_db(spec.roi, i))
            s1_valid[j] = True

        return {
            "frames_rgb": f_rgb, "frames_ms": f_ms,
            "s1": s1, "s1_valid": s1_valid,
            "target_rgb": tgt_rgb[0], "target_ms": tgt_ms[0],
            "target_refl": tgt[0].clamp(0.0, 1.0),   # metric domain
            "eval_mask": eval_mask,
            "uncond": torch.tensor(spec.uncond),
        }

    def load_batch(self, bspec: BatchSpec, s1_max: int) -> dict:
        samples = [self.load_sample(s, bspec.k, s1_max) for s in bspec.specs]
        out = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        out["k"] = bspec.k
        return out
