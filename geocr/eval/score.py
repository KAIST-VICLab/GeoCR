"""PSNR/SSIM of prediction directories, and the dataset registry shared with the baseline adapters.

A prediction directory holds ``preds/<key>.npy`` for every id of the test split list, where ``key``
is the id with ``/`` replaced by ``__`` and the extension dropped. Each array is CHW float32
(float16 accepted) in the dataset's canonical domain, neither clipped nor quantised: reflectance
(DN / 10000) for the multispectral settings, png / 255 for the 8-bit RGB settings.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch

from geocr.metrics import lineages as LIN
from geocr.metrics.masked import masked_ergas, masked_metrics
from geocr.metrics.sen2mtc import sen2mtc_metrics
from geocr.normalize import S2_BANDS

CR_DATASETS = Path(os.environ.get("CR_DATASETS", "CR_datasets"))
NO_B10 = [i for i, b in enumerate(S2_BANDS) if b != "B10"]

#: Manifest fields on which prediction dumps compared in one table must agree.
IDENTITY_KEYS = ("corpus", "split", "split_list_sha256", "n_items", "resolution",
                 "crop", "bands", "domain", "processing_level",
                 "ckpt_selection", "train_split_sha256")

#: CUHK-CR1 test images excluded from evaluation (train/validation and within-test duplicates).
CUHK_CR1_EXCLUDE = tuple(f"test/cloud/{n}.png" for n in (
    1, 2, 24, 35, 41, 43, 45, 47, 48, 51, 52, 54, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 68, 69, 70, 74, 75, 76, 77, 78, 79, 80, 81, 91, 102))

CORPORA: dict[str, dict] = {
    "tcloud": dict(
        level="display8", bands=["R", "G", "B"], domain="unit_float",
        resolution=[256, 256], lineage="tcloud", root=CR_DATASETS / "T-CLOUD",
        clear_sub=[("/cloud/", "/reference/")]),
    "cuhk_cr1": dict(
        level="display8", bands=["R", "G", "B", "NIR"], domain="unit_float",
        resolution=[256, 256], lineage="cuhk", root=CR_DATASETS / "C-CUHK" / "CUHK-CR1",
        clear_sub=[("/cloud/", "/label/")], exclude=CUHK_CR1_EXCLUDE),
    "cuhk_cr2": dict(
        level="display8", bands=["R", "G", "B"], domain="unit_float",
        resolution=[256, 256], lineage="cuhk_rgb", root=CR_DATASETS / "C-CUHK" / "CUHK-CR2",
        clear_sub=[("/cloud/", "/label/")]),
    "sen2mtc_new": dict(
        level="boa", bands=["R", "G", "B", "NIR"], domain="reflectance",
        resolution=[256, 256], lineage="sen2mtc", root=CR_DATASETS / "Sen2_MTC_New"),
    "whus2crv": dict(
        level="boa", bands=list(S2_BANDS), domain="reflectance",
        resolution=[384, 384], lineage="masked13",
        root=CR_DATASETS / "WHUS2-CRv" / "extracted"),
    "sen12mscr": dict(
        level="l1c_toa", bands=list(S2_BANDS), domain="reflectance",
        resolution=[256, 256], lineage="masked13", root=CR_DATASETS / "SEN12MS-CR"),
    "sen12mscr_rgb": dict(
        level="display8", bands=["R", "G", "B"], domain="unit_float",
        resolution=[256, 256], lineage="rgbvar", root=CR_DATASETS / "SEN12MS-CR",
        requires_export=True),
    "whus2crv_rgb": dict(
        level="display8", bands=["R", "G", "B"], domain="unit_float",
        resolution=[256, 256], lineage="rgbvar", root=CR_DATASETS / "WHUS2-CRv" / "extracted",
        requires_export=True),
    "sen2mtc_new_rgb": dict(
        level="display8", bands=["R", "G", "B"], domain="unit_float",
        resolution=[256, 256], lineage="rgbvar", root=CR_DATASETS / "Sen2_MTC_New",
        requires_export=True),
}

#: Dataset names used on the Hub and the project page -> registry keys.
SLUGS = {"sen12ms-cr": "sen12mscr", "sen12ms-cr-rgb": "sen12mscr_rgb",
         "sen2-mtc-new": "sen2mtc_new", "sen2-mtc-new-rgb": "sen2mtc_new_rgb",
         "whus2-crv": "whus2crv", "whus2-crv-rgb": "whus2crv_rgb", "t-cloud": "tcloud",
         "cuhk-cr1": "cuhk_cr1", "cuhk-cr2": "cuhk_cr2"}
ALIASES = {"cuhk_cr": "cuhk_cr1", **SLUGS}

#: The per-image metric keys reported as PSNR and SSIM for each protocol.
HEADLINE = {"masked13": ("psnr_all", "ssim_all"), "sen2mtc": ("mtc_psnr", "mtc_ssim"),
            "tcloud": ("tcloud_psnr", "tcloud_ssim"), "cuhk": ("cuhk_psnr", "cuhk_ssim"),
            "cuhk_rgb": ("cuhkrgb_psnr", "cuhkrgb_ssim"),
            "rgbvar": ("rgbvar_psnr", "rgbvar_ssim")}


def resolve_corpus(name: str) -> str:
    return ALIASES.get(name, name)


def sid_flat(sid: str) -> str:
    return sid.replace("/", "__")


def read_split(root: Path, split: str, splits_dir: str) -> list[str]:
    """``<root>/<splits_dir>/<split>.txt``: one id per line, ``#`` comments dropped, order kept."""
    p = Path(root) / splits_dir / f"{split}.txt"
    if not p.is_file():
        raise SystemExit(f"missing split file {p}")
    out = [s for s in (ln.split("#", 1)[0].strip() for ln in p.read_text().splitlines()) if s]
    if not out:
        raise SystemExit(f"{p} is empty")
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _u8_to_unit(a: np.ndarray) -> torch.Tensor:
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1) / 255.0


def ground_truth(corpus: str, split: str = "test", root=None, export=None,
                 splits_dir: str = "splits"):
    """``sid -> [C,H,W]`` float32 reference in the dataset's canonical domain.

    The 8-bit settings read ``<export>/<corpus>/<split>/clear/`` when that export exists
    (mandatory for the RGB-only renderings of the multispectral datasets); otherwise the
    native clear image is decoded and resized exactly as the exporter does."""
    from geocr.data import benchmarks as BM
    cfg = CORPORA[corpus]
    root = Path(root) if root else cfg["root"]
    nb = len(cfg["bands"])
    d = Path(export) / corpus / split if export else None
    if cfg["level"] == "display8" and d is not None and d.is_dir():
        def gt(sid):
            png = Path(sid_flat(sid)).with_suffix(".png").name
            x = _u8_to_unit(BM._read_png(d / "clear" / png))
            if nb == 4:
                x = torch.cat([x[:3], _u8_to_unit(BM._read_png(d / "clear_nir" / png))[:1]], 0)
            return x[:nb]
        return gt
    if cfg["level"] == "display8":
        if cfg.get("requires_export"):
            raise SystemExit(f"{corpus}: the reference images are the 8-bit RGB export, not found "
                             f"at {d}; pass --export <root> or set CR_EXPORTS")
        res = cfg["resolution"][0]

        def gt(sid):
            cid = next((sid.replace(a, b) for a, b in cfg["clear_sub"]
                        if a in sid and (root / sid.replace(a, b)).is_file()), None)
            if cid is None:
                raise SystemExit(f"{sid}: no clear image under {root}")
            x = _u8_to_unit(BM._resize_u8(BM._read_png(root / cid), res))
            if nb == 4:
                n = BM._read_png(root.parent / "nir" / root.name / cid)
                n = n[..., 0] if n.ndim == 3 else n
                x = torch.cat([x[:3], _u8_to_unit(BM._resize_u8(n, res))[:1]], 0)
            return x[:nb]
        return gt
    if corpus == "whus2crv":
        from geocr.config import DataCfg
        w = BM.WHUS2CRvProvider(root, split, 1, 0, DataCfg(split_dir=splits_dir))
        return lambda sid: w._to13(sid, "clear")

    def gt(sid):
        p = root / (sid.replace("s2_cloudy", "s2") if corpus == "sen12mscr" else sid)
        return torch.from_numpy(BM._read_tif(p).astype(np.float32)) / 10000.0
    return gt


def metrics_one(pred: torch.Tensor, gt: torch.Tensor, cfg: dict,
                mask: torch.Tensor | None = None) -> dict[str, float]:
    """Metrics of one ``[C,H,W]`` prediction under its dataset's protocol (``cfg["lineage"]``)."""
    lin = cfg["lineage"]
    p, g = pred[None], gt[None]
    if lin == "masked13":
        m = torch.ones(1, 1, *pred.shape[-2:]) if mask is None else mask[None, None]
        out = masked_metrics(p.clamp(0, 1), g.clamp(0, 1), m)
        out["ergas_no_b10"] = masked_ergas(p[:, NO_B10].clamp(0, 1),
                                           g[:, NO_B10].clamp(0, 1), m)
        return out
    if lin == "sen2mtc":
        return sen2mtc_metrics(p[:, :3], g[:, :3])
    if lin == "tcloud":
        return LIN.tcloud_metrics(p[:, :3], g[:, :3])
    if lin == "cuhk":
        return LIN.cuhk_metrics(p[:, :4], g[:, :4])
    if lin == "cuhk_rgb":
        r = LIN._batched(LIN._cuhk_one, p[:, :3], g[:, :3], lpips=False)
        return {k.replace("cuhk_", "cuhkrgb_"): v for k, v in r.items()}
    if lin == "rgbvar":
        r = LIN._batched(LIN._tcloud_one, p[:, :3], g[:, :3])
        return {k.replace("tcloud_", "rgbvar_"): v for k, v in r.items()}
    raise SystemExit(f"unknown lineage {lin!r}")


def load_pred(run: Path, key: str, cfg: dict, allow_unpredicted: bool = False) -> torch.Tensor:
    """``<run>/preds/<key stem>.npy`` as ``[C,H,W]`` float32. A band that is entirely
    non-finite was not predicted; it is refused unless ``allow_unpredicted``."""
    p = Path(run) / "preds" / (Path(key).stem + ".npy")
    a = np.load(p)
    if a.dtype == np.float16:
        a = a.astype(np.float32)
    if a.ndim != 3 or a.shape[0] != len(cfg["bands"]):
        raise SystemExit(f"{p}: shape {a.shape}, expected {len(cfg['bands'])} bands "
                         f"{cfg['bands']} x H x W")
    dead = [cfg["bands"][i] for i in range(a.shape[0]) if not np.isfinite(a[i]).any()]
    if dead and not allow_unpredicted:
        raise SystemExit(f"{p}: band(s) {dead} are entirely non-finite (not predicted)")
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def check_protocol_identity(mans: list[tuple[Path, dict]], strict: bool = True) -> None:
    """Refuse (``strict``) or warn about dump manifests that lack an IDENTITY_KEYS field or
    disagree on one."""
    def label(p):
        return f"{Path(p).parent.name}/{Path(p).name}"
    ref_path, ref = mans[0]
    problems = []
    for path, m in mans[1:]:
        for k in IDENTITY_KEYS:
            a, b = ref.get(k, "<missing>"), m.get(k, "<missing>")
            if k == "corpus" and isinstance(a, str) and isinstance(b, str):
                a, b = resolve_corpus(a), resolve_corpus(b)
            if a != b:
                problems.append(f"  {k}: {label(ref_path)}={a!r} vs {label(path)}={b!r}")
    problems += [f"  {label(p)}: manifest lacks {k!r}"
                 for p, m in mans for k in IDENTITY_KEYS if k not in m]
    if problems:
        msg = "dumps do not follow the same protocol:\n" + "\n".join(problems)
        if strict:
            raise SystemExit(msg)
        print("WARNING: " + msg)


def eval_ids(run: Path, ids: list[str], cfg: dict) -> list[str]:
    """Require exactly one prediction per split id; return the ids that are scored."""
    stems = {Path(sid_flat(s)).stem for s in ids}
    d = Path(run) / "preds"
    have = {p.stem for p in d.glob("*.npy")} if d.is_dir() else set()
    if have != stems:
        raise SystemExit(
            f"{run}: preds/ does not match the split list ({len(have)} files, {len(stems)} ids); "
            f"missing e.g. {sorted(stems - have)[:5]}, unexpected e.g. {sorted(have - stems)[:5]}")
    excl = {Path(sid_flat(e)).stem for e in cfg.get("exclude", ())}
    if not excl <= stems:
        raise SystemExit(f"exclusion ids not in the split list: {sorted(excl - stems)[:5]}")
    return [s for s in ids if Path(sid_flat(s)).stem not in excl]


def score(run: Path, cfg: dict, ids: list[str], gt, allow_unpredicted: bool = False
          ) -> dict[str, float]:
    """Mean over ``ids`` of the per-image ``metrics_one``, computed single-threaded so that
    every reduction runs in a fixed order and the values are bit-reproducible."""
    rows, threads = [], torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for sid in ids:
            pred, ref = load_pred(run, sid_flat(sid), cfg, allow_unpredicted), gt(sid)
            if pred.shape != ref.shape:
                raise SystemExit(f"{sid}: prediction {tuple(pred.shape)} vs reference "
                                 f"{tuple(ref.shape)}")
            rows.append(metrics_one(pred, ref, cfg))
    finally:
        torch.set_num_threads(threads)
    bad = sorted({k for r in rows for k, v in r.items() if not math.isfinite(v)})
    if bad:
        raise SystemExit(f"{run}: non-finite per-image {bad}")
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
