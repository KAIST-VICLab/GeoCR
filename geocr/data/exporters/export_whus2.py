"""WHUS2-CRv 13-band fuser shared by every loader that reads WHUS2-CRv (GeoCR's provider applies
the same band map and replication, from ``WHUS2CRvProvider.BAND_POS``/``UPSAMPLE``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geocr.data.exporters._common import geocr_import

FUSED_SHAPE = (13, 384, 384)


def fuse13(root: Path, sid: str, role: str, g: dict | None = None) -> np.ndarray:
    """``uint16 [13,384,384]`` DN in Sentinel-2 band order for one clip.

    ``sid`` is a clear 10 m clip id as listed in ``<root>/splits/<split>.txt``
    (``<dir>/clearDNclips/10m/<scene>/<N>.tif``); ``role`` is ``"clear"`` or ``"cloud"``.
    The 20 m and 60 m rasters are replicated x2 and x6 (nearest neighbour) onto the 10 m grid.
    """
    g = g or geocr_import()
    W = g["whus2"]
    out = np.zeros(FUSED_SHAPE, dtype=np.uint16)
    for res, dst in W.BAND_POS.items():
        p = Path(root) / sid.replace("clearDNclips", f"{role}DNclips") \
                            .replace("/10m/", f"/{res}/")
        arr = g["read_tif"](p)
        if arr.shape[0] != len(dst):
            raise ValueError(f"{p}: expected {len(dst)} bands, got {arr.shape[0]}")
        f = W.UPSAMPLE[res]
        if f > 1:
            arr = np.repeat(np.repeat(arr, f, axis=-2), f, axis=-1)
        if arr.shape[-2:] != FUSED_SHAPE[-2:]:
            raise ValueError(f"{p}: upsampled to {arr.shape[-2:]}, "
                             f"expected {FUSED_SHAPE[-2:]}")
        out[list(dst)] = arr.astype(np.uint16, copy=False)
    return out
