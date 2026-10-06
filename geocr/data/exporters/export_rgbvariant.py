"""Render the RGB-only datasets derived from SEN12MS-CR, WHUS2-CRv and Sen2_MTC_New.

    python -m geocr.data.exporters.export_rgbvariant --out $CR_EXPORTS [--corpus C] [--aliases]

``u = round(255 * clip(2 * reflectance, 0, 1))`` (``geocr.normalize.RGBVariantNorm``) for the
cloudy input and the clear target alike. Writes ``<out>/<corpus>/<split>/{cloudy,clear}/*.png``
(8-bit RGB, 256 px), ``list.txt`` and ``manifest.json``. WHUS2-CRv is fused to 384 px
(``export_whus2.fuse13``), quantised, then resized to 256 px with PIL bicubic; Sen2_MTC_New uses
cloudy frame 0. ``--split all`` writes ``trainval`` and ``test``; whenever ``trainval`` is written,
``val`` is linked to it (the validation ids are a subset of ``trainval``).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from geocr.data.exporters._common import (
    CR_DATASETS, SplitWriter, export_image_name, geocr_import, read_split, sha256_file,
)
from geocr.data.exporters.export_whus2 import fuse13

CORPORA: dict[str, dict] = {
    "sen12mscr_rgb": dict(source="sen12mscr", root=CR_DATASETS / "SEN12MS-CR", res=256,
                          reader="s2_13band"),
    "whus2crv_rgb": dict(source="whus2crv", root=CR_DATASETS / "WHUS2-CRv" / "extracted",
                         res=256, native=384, reader="whus2_fused"),
    "sen2mtc_new_rgb": dict(source="sen2mtc_new", root=CR_DATASETS / "Sen2_MTC_New", res=256,
                            reader="mtc_4band", frame_index=0),
}


def _rgb_u8(refl_rgb: np.ndarray, v, cfg: dict, g: dict) -> np.ndarray:
    """RGB reflectance CHW -> HWC uint8. Quantise first, then resize the 8-bit pixels."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(refl_rgb))[None].float()
    u8 = torch.round(v.to_u01(t) * 255.0).clamp(0, 255).to(torch.uint8)
    arr = u8[0].permute(1, 2, 0).numpy()
    if cfg.get("native", cfg["res"]) != cfg["res"]:
        arr = g["resize_u8"](arr, cfg["res"])
    return arr


def read_pair(cfg: dict, root: Path, sid: str, g: dict) -> tuple[np.ndarray, np.ndarray]:
    """(cloudy, clear) RGB reflectance, float32 CHW."""
    from geocr.normalize import RGB_IDX
    rgb_idx = list(RGB_IDX)
    if cfg["reader"] == "s2_13band":
        cloudy, clear = root / sid, root / sid.replace("s2_cloudy", "s2")
        return (g["read_tif"](cloudy).astype(np.float32)[rgb_idx] / 1e4,
                g["read_tif"](clear).astype(np.float32)[rgb_idx] / 1e4)
    if cfg["reader"] == "whus2_fused":
        return (fuse13(root, sid, "cloud", g).astype(np.float32)[rgb_idx] / 1e4,
                fuse13(root, sid, "clear", g).astype(np.float32)[rgb_idx] / 1e4)
    clear = root / sid                       # file band order is R,G,B,NIR
    cloudy = clear.parent.parent / "cloud" / f"{clear.stem}_{cfg['frame_index']}.tif"
    return (g["read_tif"](cloudy).astype(np.float32)[:3] / 1e4,
            g["read_tif"](clear).astype(np.float32)[:3] / 1e4)


def export_split(corpus: str, split: str, out_root: Path, root: str | None = None,
                 splits_dir: str = "splits", aliases: bool = False) -> None:
    from geocr.normalize import RGBVariantNorm
    g = geocr_import()
    cfg = CORPORA[corpus]
    root = Path(root) if root else cfg["root"]
    v = RGBVariantNorm()
    nat = cfg.get("native", cfg["res"])
    ids = read_split(root, split, splits_dir)
    keys = [export_image_name(s) for s in ids]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{corpus}/{split}: file name collision")
    w = SplitWriter(out_root / corpus / split)
    for sid, key in zip(ids, keys):
        for sub, refl in zip(("cloudy", "clear"), read_pair(cfg, root, sid, g)):
            if refl.shape[-2:] != (nat, nat):
                raise SystemExit(f"{corpus}/{sid}/{sub}: got {refl.shape[-2:]}, "
                                 f"expected {nat}x{nat}")
            w.png(sub, key, _rgb_u8(refl, v, cfg, g))
    man = dict(corpus=corpus, split=split, source_corpus=cfg["source"], level="display8",
               resolution=[cfg["res"], cfg["res"]], bands=["R", "G", "B"], gain=v.gain,
               resample="none" if nat == cfg["res"] else
               "PIL bicubic after quantisation (geocr.data.benchmarks._resize_u8)",
               split_list_sha256=sha256_file(root / splits_dir / f"{split}.txt"))
    if "frame_index" in cfg:
        man["frame_index"] = cfg["frame_index"]
    w.finish(keys, man, aliases)
    val = out_root / corpus / "val"
    if split == "trainval" and not (val.is_symlink() or val.exists()):
        os.symlink("trainval", val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="export root ($CR_EXPORTS)")
    ap.add_argument("--corpus", default="all", choices=["all", *CORPORA])
    ap.add_argument("--split", default="all", help="all (= trainval,test) or a comma list")
    ap.add_argument("--root", default=None, help="source dataset root (one corpus only)")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--aliases", action="store_true", help="add haze/gt/label directory links")
    args = ap.parse_args()

    corpora = list(CORPORA) if args.corpus == "all" else [args.corpus]
    splits = ["trainval", "test"] if args.split == "all" else args.split.split(",")
    if args.root and len(corpora) != 1:
        raise SystemExit("--root needs exactly one --corpus")
    for c in corpora:
        for s in splits:
            export_split(c, s, Path(args.out), args.root, args.splits_dir, args.aliases)


if __name__ == "__main__":
    main()
