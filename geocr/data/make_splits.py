"""Convert the official AllClear roi lists into ``<root>/splits/{train,val,test}.txt``
(digits-only roi ids, the format ``AllClearIndex.split_rois`` reads).

    python -m geocr.data.make_splits --rois_dir <allclear>/metadata/rois --out <root>

Official files, one roi per line ("roi219586"): train_rois_19k.txt (19,013), val_rois_1k.txt
(997), test_rois_3k.txt (3,698). Disjointness and the total are checked.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OFFICIAL = {"train": "train_rois_19k.txt", "val": "val_rois_1k.txt",
            "test": "test_rois_3k.txt"}
EXPECTED_TOTAL = 23_708


def make_splits(rois_dir: str | Path, out_root: str | Path) -> dict[str, int]:
    rois_dir, out_root = Path(rois_dir), Path(out_root)
    splits: dict[str, list[str]] = {}
    for split, fname in OFFICIAL.items():
        keys = []
        for ln in (rois_dir / fname).read_text().splitlines():
            if not ln.strip():
                continue
            m = re.fullmatch(r"(?:roi)?(\d+)", ln.strip())
            if not m:
                raise ValueError(f"{rois_dir / fname}: unrecognized roi line {ln!r}")
            keys.append(m.group(1))
        splits[split] = keys
    sets = {s: set(k) for s, k in splits.items()}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"splits {a}/{b} overlap: "
                                 f"e.g. {sorted(sets[a] & sets[b])[:3]}")
    total = sum(len(k) for k in splits.values())
    if total != EXPECTED_TOTAL:
        raise ValueError(f"splits total {total} rois, expected {EXPECTED_TOTAL}")
    out_dir = out_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, keys in splits.items():
        (out_dir / f"{split}.txt").write_text("\n".join(keys) + "\n")
    return {s: len(k) for s, k in splits.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois_dir", required=True, help="official metadata/rois dir")
    ap.add_argument("--out", required=True, help="transcoded root (writes <out>/splits/)")
    args = ap.parse_args()
    counts = make_splits(args.rois_dir, args.out)
    print(f"wrote {counts} -> {Path(args.out) / 'splits'}")


if __name__ == "__main__":
    main()
