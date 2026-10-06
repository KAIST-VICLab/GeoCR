"""Helpers shared by the exporters and by the comparison methods' data loaders: split lists,
the flat id rule, and the pixel helpers of ``geocr.data.benchmarks``."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

#: Root holding the downloaded datasets (see DATA.md); ``--root`` overrides one dataset.
CR_DATASETS = Path(os.environ.get("CR_DATASETS", "CR_datasets"))
#: Directory links some methods expect inside an export split (link -> target).
ALIASES = (("haze", "cloudy"), ("gt", "clear"), ("label", "clear"))


def geocr_import() -> dict:
    """The PNG/TIFF readers, the resampler and the WHUS2-CRv band map GeoCR itself uses."""
    from geocr.data.benchmarks import WHUS2CRvProvider, _read_png, _read_tif, _resize_u8
    return dict(read_png=_read_png, read_tif=_read_tif, resize_u8=_resize_u8,
                whus2=WHUS2CRvProvider)


def sid_flat(sid: str) -> str:
    """Split-list id -> flat file key (``/`` -> ``__``), shared by exports and predictions."""
    return sid.replace("/", "__")


def export_image_name(sid: str) -> str:
    """Exports always store PNG bytes, so the flat key gets a ``.png`` suffix."""
    return Path(sid_flat(sid)).with_suffix(".png").name


def read_split(root: Path, split: str, splits_dir: str = "splits") -> list[str]:
    """Ids of ``<root>/<splits_dir>/<split>.txt`` in order; ``#`` comments and blanks dropped."""
    p = Path(root) / splits_dir / f"{split}.txt"
    if not p.is_file():
        raise SystemExit(f"missing split file {p}")
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            out.append(ln)
    if not out:
        raise SystemExit(f"{p} is empty")
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SplitWriter:
    """Writes one export split: PNG files, ``list.txt``, ``manifest.json`` and aliases."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.digests: list[str] = []

    def png(self, sub: str, key: str, arr) -> None:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=False)
        blob = buf.getvalue()
        dst = self.out_dir / sub / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)
        self.digests.append(f"{sub}/{key}\t{hashlib.sha256(blob).hexdigest()}\n")

    def finish(self, keys: list[str], manifest: dict, aliases: bool) -> None:
        """``content_digest`` = sha256 over the sorted ``<sub>/<file>\\t<sha256>`` lines."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "list.txt").write_text("".join(f"{k}\n" for k in sorted(keys)))
        manifest = dict(manifest, n_items=len(keys), content_digest=hashlib.sha256(
            "".join(sorted(self.digests)).encode()).hexdigest())
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if aliases:
            for link, target in ALIASES:
                p = self.out_dir / link
                if not (p.is_symlink() or p.exists()):
                    os.symlink(target, p)
        print(f"  {self.out_dir}: {len(keys)} items")
