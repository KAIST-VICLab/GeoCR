"""AllClear observation-set sampling. One batch = one K (frame count), so tensors stack without
ragged padding; SAR is the only per-item-variable slot and is padded. Every batch is a pure
function of (seed, step), which makes resume exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geocr.config import DataCfg
from geocr.data.index import AllClearIndex


@dataclass(frozen=True)
class SampleSpec:
    roi: str
    target_i: int
    frame_is: tuple[int, ...]      # S2 condition frames; empty for SAR-only samples
    s1_is: tuple[int, ...]         # S1 scenes; may be empty
    uncond: bool                   # all conditions -> null token
    #: Stable per-sample id; evaluation noise is keyed on it, not on batch position.
    sid: str = ""


@dataclass
class BatchSpec:
    k: int
    specs: list[SampleSpec] = field(default_factory=list)


class ObservationBatchSampler:
    """``batch(step)`` -> BatchSpec, deterministic in (seed, step)."""

    def __init__(self, index: AllClearIndex, rois: list[str], cfg: DataCfg,
                 batch_size: int, seed: int, draws_per_opt_step: int = 1):
        self.index = index
        self.cfg = cfg
        self.batch_size = batch_size
        self.seed = seed
        #: Sampler draws per optimizer step (grad_accum * world); the K ramp counts
        #: optimizer steps.
        self.draws_per_opt_step = max(1, int(draws_per_opt_step))
        # Keep only rois that can serve at least one (target, K_max frames) sample.
        kmax = max(cfg.k_choices)
        self.rois: list[str] = []
        self._eligible: dict[str, np.ndarray] = {}
        for r in rois:
            elig = index.eligible_targets(r, cfg.target_cloud_max,
                                          cfg.target_shadow_max, cfg.target_nan_max)
            t = index.tables[r]
            if len(elig) == 0 or len(t.s2_date) < kmax + 1:
                continue
            self.rois.append(r)
            self._eligible[r] = elig
        if not self.rois:
            raise ValueError("no roi can serve a sample under the configured filters")

    def _rng(self, step: int) -> np.random.Generator:
        return np.random.default_rng(np.random.PCG64(((self.seed & 0xFFFFFFFF) << 32)
                                                     ^ (step & 0xFFFFFFFF)))

    def batch(self, step: int) -> BatchSpec:
        cfg = self.cfg
        rng = self._rng(step)
        choices, probs = cfg.k_choices, cfg.k_probs
        if cfg.k_ramp_steps:
            opt_step = step // self.draws_per_opt_step
            hi = 1 + int((len(choices) - 1) * min(1.0, opt_step / cfg.k_ramp_steps))
            choices = choices[:hi]
            total = sum(probs[:hi])
            probs = [p / total for p in probs[:hi]]
        k = int(rng.choice(choices, p=probs))
        sar_only = bool(rng.random() < cfg.sar_only_frac)
        out = BatchSpec(k=0 if sar_only else k)
        while len(out.specs) < self.batch_size:
            spec = self._draw(rng, k, sar_only)
            if spec is not None:
                out.specs.append(spec)
        return out

    def _draw(self, rng: np.random.Generator, k: int, sar_only: bool) -> SampleSpec | None:
        cfg = self.cfg
        roi = self.rois[int(rng.integers(len(self.rois)))]
        t = self.index.tables[roi]
        target = int(rng.choice(self._eligible[roi]))
        td = int(t.s2_date[target])

        # Condition frames: uniform draw within the window, excluding the target's date.
        cand = np.nonzero((np.abs(t.s2_date - td) <= cfg.window_days)
                          & (t.s2_date != td)
                          & (t.s2_nan <= cfg.target_nan_max))[0]
        frames: tuple[int, ...] = ()
        if not sar_only:
            if len(cand) < k:
                return None                      # redrawn by the caller
            frames = tuple(int(i) for i in rng.choice(cand, size=k, replace=False))

        # SAR: the nearest s1_max passes inside the window; s1_drop drops them at random.
        s1: tuple[int, ...] = ()
        if len(t.s1_date) and rng.random() >= cfg.s1_drop:
            near = np.nonzero(np.abs(t.s1_date - td) <= cfg.window_days)[0]
            if len(near):
                order = np.argsort(np.abs(t.s1_date[near] - td), kind="stable")
                s1 = tuple(int(i) for i in near[order][: cfg.s1_max])
        if sar_only and not s1:
            return None                          # a SAR-only sample needs SAR
        return SampleSpec(
            roi=roi, target_i=target, frame_is=frames, s1_is=s1,
            uncond=bool(rng.random() < cfg.p_uncond),
        )
