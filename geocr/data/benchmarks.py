"""Benchmark providers: frozen split list -> the training batch dict.

Each provider reads ``<root>/<data.split_dir>/<split>.txt`` (one sample id per line, ``#``
lines skipped) and returns batches that are a pure function of (seed, step). Sentinel-2 pixels
become reflectance (DN / 1e4); 8-bit pixels become pseudo-reflectance png/255 (``_png13``).
``PROCESSING_LEVEL`` names the physical quantity a dataset's pixels carry (``l1c_toa``,
``boa`` or ``display8``); results at different levels are reported separately. Benchmarks
carry no cloud mask, so ``eval_mask`` is all ones.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from geocr.config import DataCfg
from geocr.normalize import Norms


def _read_tif(path: Path) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(path)
    return arr if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] \
        else np.moveaxis(arr, -1, 0)


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path))


def _resize_u8(arr: np.ndarray, resolution: int) -> np.ndarray:
    if arr.shape[0] == resolution:
        return arr
    from PIL import Image
    return np.asarray(Image.fromarray(arr).resize((resolution, resolution),
                                                  Image.BICUBIC))


def _png13(rgb_u8: np.ndarray, nir_u8: np.ndarray | None, resolution: int,
           rgb_gain: tuple[float, float, float]) -> torch.Tensor:
    """HWC uint8 -> 13-band pseudo-reflectance: R,G,B png/255 divided by the live
    ``S2Norm.rgb_gain`` at B4,B3,B2 (so the model input is exactly 2u/255 - 1), NIR png/255
    at B8. Resized to ``resolution`` with PIL bicubic."""
    rgb = _resize_u8(rgb_u8, resolution)
    h, w = rgb.shape[:2]
    out = torch.zeros(13, h, w)
    for c, dst in enumerate((3, 2, 1)):                 # R,G,B -> B4,B3,B2
        out[dst] = torch.from_numpy(
            rgb[..., c].astype(np.float32)) / 255.0 / rgb_gain[c]
    if nir_u8 is not None:
        out[7] = torch.from_numpy(
            _resize_u8(nir_u8, resolution).astype(np.float32)) / 255.0
    return out


class _Provider:
    """Deterministic ``load(step) -> batch dict`` over a list of sample ids."""

    PROCESSING_LEVEL: str = ""

    def __init__(self, root: str | Path, split: str, batch_size: int, seed: int,
                 cfg: DataCfg):
        self.root = Path(root)
        self.batch_size = batch_size
        self.seed = seed
        self.cfg = cfg
        self.norms = Norms()
        split_file = self.root / cfg.split_dir / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"missing {split_file} (data.split_dir={cfg.split_dir!r}); "
                "place the frozen split lists there (see DATA.md)")
        self.ids = [ln.strip() for ln in split_file.read_text().splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
        if not self.ids:
            raise ValueError(f"{split_file} is empty")

    def _rng(self, step: int) -> np.random.Generator:
        return np.random.default_rng(np.random.PCG64(
            ((self.seed & 0xFFFFFFFF) << 32) ^ (step & 0xFFFFFFFF)))

    def load(self, step: int) -> dict:
        rng = self._rng(step)
        picks = [self.ids[int(i)] for i in rng.integers(len(self.ids), size=self.batch_size)]
        samples = [self._sample(p, rng) for p in picks]
        out = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        out["k"] = samples[0]["frames_rgb"].shape[0]
        return out

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        raise NotImplementedError

    def _common(self, frames13_refl: torch.Tensor, target13_refl: torch.Tensor,
                s1_db: torch.Tensor | None, rng: np.random.Generator) -> dict:
        n = self.norms
        h, w = frames13_refl.shape[-2:]
        rgb, ms = n.s2.split(frames13_refl)
        t_rgb, t_ms = n.s2.split(target13_refl[None])
        s1 = torch.zeros(1, 2, h, w)
        s1_valid = torch.zeros(1, dtype=torch.bool)
        if s1_db is not None and rng.random() >= self.cfg.s1_drop:
            s1[0] = n.s1.to_model(s1_db)
            s1_valid[0] = True
        return {
            "frames_rgb": rgb, "frames_ms": ms,
            "s1": s1, "s1_valid": s1_valid,
            "target_rgb": t_rgb[0], "target_ms": t_ms[0],
            "target_refl": target13_refl.clamp(0.0, 1.0),
            "eval_mask": torch.ones(h, w),
            "uncond": torch.tensor(bool(rng.random() < self.cfg.p_uncond)),
        }


class Sen12MSCRProvider(_Provider):
    """Ids are ``s2_cloudy`` tifs; the clear target and SAR live in the parallel ``s2`` and
    ``s1`` trees. 13-band DN / 1e4, K = 1, SAR (dB) when present."""

    KIND = "sen12mscr"
    PROCESSING_LEVEL = "l1c_toa"

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        cloudy = self.root / sid
        clear = self.root / sid.replace("s2_cloudy", "s2")
        s1p = self.root / sid.replace("s2_cloudy", "s1")
        f13 = torch.from_numpy(_read_tif(cloudy).astype(np.float32))[None] / 10000.0
        t13 = torch.from_numpy(_read_tif(clear).astype(np.float32)) / 10000.0
        s1 = torch.from_numpy(_read_tif(s1p).astype(np.float32)) if s1p.is_file() else None
        return self._common(f13, t13, s1, rng)


class Sen2MTCNewProvider(_Provider):
    """Ids are ``Sen2_MTC/<tile>/cloudless/<name>.tif``; the three cloudy frames are
    ``<tile>/cloud/<name>_{0,1,2}.tif``. 4-band R,G,B,NIR DN / 1e4, K = 3."""

    KIND = "sen2mtc_new"
    PROCESSING_LEVEL = "boa"
    #: R,G,B,NIR file order -> S2 band-index positions (B4,B3,B2,B8).
    BAND_POS = (3, 2, 1, 7)

    def _to13(self, arr4: np.ndarray) -> torch.Tensor:
        h, w = arr4.shape[-2:]
        out = torch.zeros(13, h, w)
        for src, dst in enumerate(self.BAND_POS):
            out[dst] = torch.from_numpy(arr4[src].astype(np.float32)) / 10000.0
        return out

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        target = self.root / sid
        cloud_dir = target.parent.parent / "cloud"
        inputs = [cloud_dir / f"{target.stem}_{i}.tif" for i in range(3)]
        missing = [p for p in inputs if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{missing[0]}: expected cloud frames {target.stem}_{{0,1,2}}.tif")
        f13 = torch.stack([self._to13(_read_tif(p)) for p in inputs])
        return self._common(f13, self._to13(_read_tif(target)), None, rng)


class CUHKCRProvider(_Provider):
    """root = .../C-CUHK/CUHK-CR1 or CUHK-CR2 (``{train,test}/{cloud,label}/*.png``, 512 px);
    the NIR tree is ``root.parent/nir/root.name`` (channel 0 of a grey PNG)."""

    KIND = "cuhk_cr"
    PROCESSING_LEVEL = "display8"

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        nir_root = self.root.parent / "nir" / self.root.name
        lab = sid.replace("/cloud/", "/label/")
        res = self.cfg.resolution
        f13 = _png13(_read_png(self.root / sid),
                     _read_png(nir_root / sid)[..., 0], res, self.norms.s2.rgb_gain)[None]
        t13 = _png13(_read_png(self.root / lab),
                     _read_png(nir_root / lab)[..., 0], res, self.norms.s2.rgb_gain)
        return self._common(f13, t13, None, rng)


class TCloudProvider(_Provider):
    """root/{train,test}/{cloud,reference}/N.png (256 px)."""

    KIND = "tcloud"
    PROCESSING_LEVEL = "display8"

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        lab = self.root / sid.replace("/cloud/", "/reference/")
        res = self.cfg.resolution
        f13 = _png13(_read_png(self.root / sid), None, res, self.norms.s2.rgb_gain)[None]
        return self._common(f13, _png13(_read_png(lab), None, res, self.norms.s2.rgb_gain),
                            None, rng)


class RICEProvider(_Provider):
    """root = .../RICE1 or RICE2 ({cloud,label}/{i}.png, 512 px). Pretraining only."""

    KIND = "rice"
    PROCESSING_LEVEL = "display8"

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        res = self.cfg.resolution
        f13 = _png13(_read_png(self.root / sid), None, res, self.norms.s2.rgb_gain)[None]
        t13 = _png13(_read_png(self.root / sid.replace("cloud/", "label/")),
                     None, res, self.norms.s2.rgb_gain)
        return self._common(f13, t13, None, rng)


class Sen2MTCOldProvider(_Provider):
    """root = .../Sen2_MTC_Old ({multipleImage,singleImage}/{clear,cloudy}/*.jpg). Pretraining
    only. Every cloudy frame has a grey ``_ir`` NIR sibling, the clear target does not.
    multipleImage ids are K = 3 (``{stem}_{0,1,2}``), singleImage ids K = 1; one batch is
    always drawn from one subset."""

    KIND = "sen2mtc_old"
    PROCESSING_LEVEL = "display8"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._by_k = {1: [i for i in self.ids if i.startswith("singleImage/")],
                      3: [i for i in self.ids if i.startswith("multipleImage/")]}
        self._by_k = {k: v for k, v in self._by_k.items() if v}

    def load(self, step: int) -> dict:
        rng = self._rng(step)
        ks = sorted(self._by_k)
        sizes = np.array([len(self._by_k[k]) for k in ks], dtype=np.float64)
        k = ks[int(rng.choice(len(ks), p=sizes / sizes.sum()))]
        ids = self._by_k[k]
        picks = [ids[int(i)] for i in rng.integers(len(ids), size=self.batch_size)]
        samples = [self._sample(p, rng) for p in picks]
        out = {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
        out["k"] = k
        return out

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        p = self.root / sid
        cloudy = p.parent.parent / "cloudy"
        stems = [f"{p.stem}_{i}" for i in range(3)] \
            if sid.startswith("multipleImage/") else [p.stem]
        res = self.cfg.resolution
        f13 = torch.stack([
            _png13(_read_png(cloudy / f"{s}.jpg"),
                   _read_png(cloudy / f"{s}_ir.jpg"), res, self.norms.s2.rgb_gain)
            for s in stems])
        t13 = _png13(_read_png(p), None, res, self.norms.s2.rgb_gain)
        return self._common(f13, t13, None, rng)


class WHUS2CRvProvider(_Provider):
    """``extracted/<dir>/{clear,cloud}DNclips/{10m,20m,60m}/<scene>/<N>.tif``; ids are the
    clear 10 m clips. The 13 bands arrive as 384x384x4, 192x192x6 and 64x64x3 rasters (band
    layout of the authors' ``fuseimg.py``) and are replicated (nearest neighbour) onto the
    10 m grid. Pixels are BOA except B10 (L1C TOA). K = 1, no SAR."""

    KIND = "whus2crv"
    PROCESSING_LEVEL = "boa"
    #: Per native resolution, the 13-band destination indices (0=B1 ... 8=B8A ... 12=B12).
    BAND_POS = {"10m": (1, 2, 3, 7), "20m": (4, 5, 6, 8, 11, 12), "60m": (0, 9, 10)}
    UPSAMPLE = {"10m": 1, "20m": 2, "60m": 6}

    def _to13(self, sid: str, role: str) -> torch.Tensor:
        out = None
        for res, dst in self.BAND_POS.items():
            p = self.root / sid.replace("clearDNclips", f"{role}DNclips") \
                               .replace("/10m/", f"/{res}/")
            arr = torch.from_numpy(_read_tif(p).astype(np.float32)) / 10000.0
            if arr.shape[0] != len(dst):
                raise ValueError(f"{p}: expected {len(dst)} bands, got {arr.shape[0]}")
            f = self.UPSAMPLE[res]
            if f > 1:
                arr = arr.repeat_interleave(f, -2).repeat_interleave(f, -1)
            if out is None:
                out = torch.zeros(13, *arr.shape[-2:])
            elif arr.shape[-2:] != out.shape[-2:]:
                raise ValueError(f"{p}: upsampled to {tuple(arr.shape[-2:])}, "
                                 f"expected {tuple(out.shape[-2:])}")
            out[list(dst)] = arr
        return out

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        return self._common(self._to13(sid, "cloud")[None],
                            self._to13(sid, "clear"), None, rng)


class _RGBVariantProvider(_Provider):
    """RGB-only rendering of a reflectance dataset: ids from the native root's split list,
    pixels from ``$CR_EXPORTS/<KIND>/<split>/{cloudy,clear}/<sid_flat>.png`` written by
    ``geocr.data.exporters.export_rgbvariant``. Read like any other 8-bit dataset."""

    PROCESSING_LEVEL = "display8"

    def __init__(self, root, split, batch_size, seed, cfg):
        super().__init__(root, split, batch_size, seed, cfg)
        self.export = Path(os.environ.get("CR_EXPORTS", "CR_exports")) / self.KIND / split
        if not self.export.is_dir():
            raise FileNotFoundError(
                f"{self.KIND}: no export at {self.export}; run `python -m "
                f"geocr.data.exporters.export_rgbvariant --out $CR_EXPORTS --corpus "
                f"{self.KIND}` and set CR_EXPORTS")

    def _sample(self, sid: str, rng: np.random.Generator) -> dict:
        key = Path(sid.replace("/", "__")).with_suffix(".png").name
        res = self.cfg.resolution
        g = self.norms.s2.rgb_gain
        cy = _png13(_read_png(self.export / "cloudy" / key), None, res, g)[None]
        cl = _png13(_read_png(self.export / "clear" / key), None, res, g)
        return self._common(cy, cl, None, rng)


class Sen12MSCRRGBProvider(_RGBVariantProvider):
    KIND = "sen12mscr_rgb"


class WHUS2CRvRGBProvider(_RGBVariantProvider):
    KIND = "whus2crv_rgb"


class Sen2MTCNewRGBProvider(_RGBVariantProvider):
    KIND = "sen2mtc_new_rgb"


PROVIDERS = {c.KIND: c for c in (Sen12MSCRProvider, Sen2MTCNewProvider,
                                 Sen2MTCOldProvider, WHUS2CRvProvider,
                                 CUHKCRProvider, TCloudProvider, RICEProvider,
                                 Sen12MSCRRGBProvider, WHUS2CRvRGBProvider,
                                 Sen2MTCNewRGBProvider)}
