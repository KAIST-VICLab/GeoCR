"""Index over the transcoded AllClear layout written by ``geocr.data.transcode``::

    index.parquet                 one row per scene
    splits/{train,val,test}.txt   one roi id per line (digits only; geocr.data.make_splits)
    rois/roi{ID}/s2.npy           [N,13,256,256] uint16   round(DN), NaN -> 0
    rois/roi{ID}/cld.npy          [N, 2,256,256] uint8    cloud prob x2 (0-200); cloud|shadow mask
    rois/roi{ID}/s1.npy           [M, 2,256,256] int16    round(dB x 100); absent without S1

``index.parquet`` columns: ``roi``, ``sensor`` ('s2'|'s1'), ``i`` (row in the roi array,
date-sorted), ``date`` (days since 2022-01-01), ``doy``, ``cloud30``, ``shadow30``,
``nan_pct`` (percent; s2 rows), plus transcode provenance columns on s2 rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCENE_COLUMNS = ("roi", "sensor", "i", "date", "cloud30", "shadow30", "nan_pct")


@dataclass
class RoiTable:
    """Per-ROI, date-sorted scene metadata as plain arrays."""

    roi: str
    s2_date: np.ndarray      # [N] int32
    s2_cloud: np.ndarray     # [N] float32, percent
    s2_shadow: np.ndarray
    s2_nan: np.ndarray
    s1_date: np.ndarray      # [M] int32, may be empty


class AllClearIndex:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        pq = self.root / "index.parquet"
        if not pq.is_file():
            raise FileNotFoundError(f"missing {pq}; run geocr.data.transcode first")
        df = pd.read_parquet(pq)
        missing = set(SCENE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{pq} lacks columns {sorted(missing)}")
        self.tables: dict[str, RoiTable] = {}
        for roi, g in df.groupby("roi", sort=True):
            s2 = g[g.sensor == "s2"].sort_values("i")
            s1 = g[g.sensor == "s1"].sort_values("i")
            if not (s2.i.values == np.arange(len(s2))).all():
                raise ValueError(f"roi {roi}: s2 'i' column is not 0..N-1 date-sorted")
            if len(s1) and not (s1.i.values == np.arange(len(s1))).all():
                raise ValueError(f"roi {roi}: s1 'i' column is not 0..M-1 date-sorted")
            self.tables[str(roi)] = RoiTable(
                roi=str(roi),
                s2_date=s2.date.values.astype(np.int32),
                s2_cloud=s2.cloud30.values.astype(np.float32),
                s2_shadow=s2.shadow30.values.astype(np.float32),
                s2_nan=s2.nan_pct.values.astype(np.float32),
                s1_date=s1.date.values.astype(np.int32),
            )

    def split_rois(self, split: str, split_dir: str = "splits") -> list[str]:
        path = self.root / split_dir / f"{split}.txt"
        rois = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        unknown = [r for r in rois if r not in self.tables]
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} rois not in index, e.g. {unknown[:3]}")
        return rois

    def roi_dir(self, roi: str) -> Path:
        return self.root / "rois" / f"roi{roi}"

    def eligible_targets(self, roi: str, cloud_max: float, shadow_max: float,
                         nan_max: float) -> np.ndarray:
        t = self.tables[roi]
        ok = (t.s2_cloud <= cloud_max) & (t.s2_shadow <= shadow_max) & (t.s2_nan <= nan_max)
        return np.nonzero(ok)[0]
