"""Pixel-range maps between stored values and the model's [-1, 1] range.

* Sentinel-2: reflectance x = DN / 1e4, model input 2 * clip(x, 0, 1) - 1 for the RGB bands
  (B4, B3, B2) and for the ten non-RGB bands.
* 8-bit imagery: model input 2u/255 - 1 (``benchmarks._png13`` divides by ``S2Norm.rgb_gain``).
* RGB-only renderings of reflectance datasets: u = round(255 * clip(2x, 0, 1)), then as 8-bit.
* SAR: dB clipped to [-30, 0] and mapped linearly to [-1, 1].

Every constant is defined here once; all readers, exporters and metrics use these classes.
``flow.shift`` was chosen for the latent scale these maps produce, so changing ``ms_clip`` or
``rgb_gain`` invalidates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

#: Sentinel-2 band order of the 13-band stacks.
S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7",
            "B8", "B8A", "B9", "B10", "B11", "B12")
#: Indices into S2_BANDS. RGB is B4/B3/B2; MS is the remaining ten, in order.
RGB_IDX = (3, 2, 1)
MS_IDX = tuple(i for i in range(13) if i not in RGB_IDX)


def reflectance_from_dn(dn: Tensor) -> Tensor:
    """uint16 DN -> float reflectance (x 1e-4). No clipping here."""
    return dn.float() / 10000.0


@dataclass(frozen=True)
class S2Norm:
    """RGB: ``clip(gain * x, 0, 1) * 2 - 1``; MS: ``clip(x, 0, c) / c * 2 - 1``.
    The inverses are exact on the unclipped range."""

    ms_clip: float = 1.0
    rgb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def _gain(self, ref: Tensor) -> Tensor:
        return torch.tensor(self.rgb_gain, device=ref.device,
                            dtype=ref.dtype).view(1, 3, 1, 1)

    def split(self, refl13: Tensor) -> tuple[Tensor, Tensor]:
        """``[B,13,H,W]`` reflectance -> (rgb ``[B,3]``, ms ``[B,10]``), model range."""
        if refl13.shape[1] != 13:
            raise ValueError(f"expected 13 bands, got {refl13.shape[1]}")
        rgb = self.rgb_to_model(refl13[:, list(RGB_IDX)])
        ms = self.ms_to_model(refl13[:, list(MS_IDX)])
        return rgb, ms

    def rgb_to_model(self, rgb_refl: Tensor) -> Tensor:
        return (rgb_refl * self._gain(rgb_refl)).clamp(0.0, 1.0) * 2.0 - 1.0

    def rgb_to_refl(self, y: Tensor) -> Tensor:
        return ((y + 1.0) / 2.0).clamp(0.0, 1.0) / self._gain(y)

    def ms_to_model(self, ms_refl: Tensor) -> Tensor:
        return ms_refl.clamp(0.0, self.ms_clip) / self.ms_clip * 2.0 - 1.0

    def ms_to_refl(self, y: Tensor) -> Tensor:
        return ((y + 1.0) / 2.0).clamp(0.0, 1.0) * self.ms_clip

    def merge_refl(self, rgb_refl: Tensor, ms_refl: Tensor) -> Tensor:
        """(rgb ``[B,3]``, ms ``[B,10]``) reflectance -> ``[B,13]`` in S2 band order."""
        b = rgb_refl.shape[0]
        out = rgb_refl.new_empty(b, 13, *rgb_refl.shape[2:])
        for k, i in enumerate(RGB_IDX):
            out[:, i] = rgb_refl[:, k]
        for k, i in enumerate(MS_IDX):
            out[:, i] = ms_refl[:, k]
        return out


@dataclass(frozen=True)
class S1Norm:
    """One window for both polarisations and every dataset: clip to [-30, 0] dB."""

    db_min: float = -30.0
    db_max: float = 0.0

    def to_model(self, db: Tensor) -> Tensor:
        half = (self.db_max - self.db_min) / 2.0
        # No-data (NaN) -> db_min; clamp alone would propagate NaN.
        db = db.nan_to_num(nan=self.db_min, posinf=self.db_max, neginf=self.db_min)
        return (db.clamp(self.db_min, self.db_max) - self.db_min) / half - 1.0

    def to_db(self, y: Tensor) -> Tensor:
        half = (self.db_max - self.db_min) / 2.0
        return (y + 1.0) * half + self.db_min


@dataclass(frozen=True)
class RGBVariantNorm:
    """Defines the pixels of the RGB-only datasets rendered from SEN12MS-CR, WHUS2-CRv and
    Sen2_MTC_New: ``u8 = round(255 * clip(reflectance * gain, 0, 1))``. Linear, no per-image
    stretch, identical for cloudy inputs and clear targets. Changing ``gain`` changes the
    benchmark itself."""

    gain: float = 2.0

    def to_u01(self, rgb_refl: Tensor) -> Tensor:
        """RGB reflectance -> [0,1]; multiply by 255 and round to write u8."""
        return (rgb_refl * self.gain).clamp(0.0, 1.0)


@dataclass(frozen=True)
class Norms:
    s2: S2Norm = field(default_factory=S2Norm)
    s1: S1Norm = field(default_factory=S1Norm)
