"""Export the 8-bit benchmarks (T-CLOUD, CUHK-CR1, CUHK-CR2) as flat PNG pairs at 256 px.

    python -m geocr.data.exporters.export_display8 --out $CR_EXPORTS [--aliases]

Writes ``<out>/<corpus>/<split>/{cloudy,clear}/<sid_flat>`` (8-bit RGB PNG), single-channel
``{cloudy,clear}_nir/`` for CUHK-CR1, ``list.txt`` and ``manifest.json``. 512 px tiles are
resized here once with ``benchmarks._resize_u8`` (PIL bicubic), so the comparison methods read
the same pixels as the GeoCR provider. ``--aliases`` adds ``haze``/``gt``/``label`` directory
links for methods that expect those names. ``--split all`` = ``trainval,test``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from geocr.data.exporters._common import (
    CR_DATASETS, SplitWriter, geocr_import, read_split, sha256_file, sid_flat,
)

#: ``clear_sub``: (needle, replacement) candidates mapping a cloudy id to its clear pair.
CORPORA: dict[str, dict] = {
    "tcloud": dict(root=CR_DATASETS / "T-CLOUD", native=256, target_res=256,
                   clear_sub=[("/cloud/", "/reference/")], nir=False, level="display8"),
    "cuhk_cr1": dict(root=CR_DATASETS / "C-CUHK" / "CUHK-CR1", native=512, target_res=256,
                     clear_sub=[("/cloud/", "/label/")], nir=True, level="display8"),
    "cuhk_cr2": dict(root=CR_DATASETS / "C-CUHK" / "CUHK-CR2", native=512, target_res=256,
                     clear_sub=[("/cloud/", "/label/")], nir=False, level="display8"),
}
#: GeoCR provider spelling of the same corpus.
CORPUS_ALIASES = {"cuhk_cr": "cuhk_cr1"}


def clear_id(sid: str, cfg: dict, root: Path) -> str | None:
    """The clear-pair id of a cloudy id, or None if no candidate file exists."""
    for needle, repl in cfg["clear_sub"]:
        if needle in sid:
            cand = sid.replace(needle, repl)
            if (root / cand).is_file():
                return cand
    return None


def export_split(corpus: str, split: str, out_root: Path, root: str | None = None,
                 splits_dir: str = "splits", aliases: bool = False) -> None:
    g = geocr_import()
    cfg = CORPORA[corpus]
    root = Path(root) if root else cfg["root"]
    res = cfg["target_res"]
    ids = read_split(root, split, splits_dir)
    keys = [sid_flat(s) for s in ids]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{corpus}/{split}: sid_flat collision")
    nir_root = root.parent / "nir" / root.name
    w = SplitWriter(out_root / corpus / split)
    for sid, key in zip(ids, keys):
        cid = clear_id(sid, cfg, root)
        if cid is None:
            raise SystemExit(f"{root / sid}: no clear pair via {cfg['clear_sub']}")
        jobs = [("cloudy", root / sid, False), ("clear", root / cid, False)]
        if cfg["nir"]:
            jobs += [("cloudy_nir", nir_root / sid, True), ("clear_nir", nir_root / cid, True)]
        for sub, src, is_nir in jobs:
            arr = g["read_png"](src)
            arr = arr[..., 0] if (is_nir and arr.ndim == 3) else arr
            w.png(sub, key, g["resize_u8"](arr, res))
    w.finish(keys, dict(corpus=corpus, split=split, level=cfg["level"],
                        resolution=[res, res], bands=["R", "G", "B"],
                        nir=cfg["nir"], resample="none" if cfg["native"] == res
                        else "PIL bicubic (geocr.data.benchmarks._resize_u8)",
                        split_list_sha256=sha256_file(root / splits_dir / f"{split}.txt")),
             aliases)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="export root ($CR_EXPORTS)")
    ap.add_argument("--corpus", default="all", help="all or a comma list of " + ", ".join(CORPORA))
    ap.add_argument("--split", default="all", help="all (= trainval,test) or a comma list")
    ap.add_argument("--root", default=None, help="dataset root (one corpus only)")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--aliases", action="store_true", help="add haze/gt/label directory links")
    args = ap.parse_args()

    corpora = list(CORPORA) if args.corpus == "all" else args.corpus.split(",")
    corpora = [CORPUS_ALIASES.get(c, c) for c in corpora]
    splits = ["trainval", "test"] if args.split == "all" else args.split.split(",")
    unknown = [c for c in corpora if c not in CORPORA]
    if unknown:
        raise SystemExit(f"unknown corpus {unknown}; have {list(CORPORA)}")
    if args.root and len(corpora) != 1:
        raise SystemExit("--root needs exactly one --corpus")
    for c in corpora:
        for s in splits:
            export_split(c, s, Path(args.out), args.root, args.splits_dir, args.aliases)


if __name__ == "__main__":
    main()
