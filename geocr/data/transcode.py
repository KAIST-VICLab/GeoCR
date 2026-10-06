"""AllClear transcode: per-ROI tar.gz -> npy memmaps + index.parquet (layout in ``index.py``).

    python -m geocr.data.transcode --archive <dir of roi*.tar.gz> --out <root> [--workers 16]

Encodings:

* s2: uint16 ``round(DN)``; NaN -> 0 and counted into ``nan_pct``.
* s1: int16 ``round(dB * 100)``; NaN no-data -> -50 dB.
* cld band 1 (cloud probability 0-100): uint8 ``round(prob * 2)``, decode by * 0.5.
  Band 2 is the evaluation mask ``(clouds_30 > 0) | (shadows_thres_30 > 0)`` (tif bands 2 and
  5); ``cloud30``/``shadow30`` are their pixel percentages.

S2 scenes without a cloud/shadow mask are dropped. Dates come from file names
``roi{ID}_{sensor}_{Y}_{M}_{D}_median.tif``; two same-day scenes of one sensor are an error.
Each worker writes ``rois/roi{ID}/rows.json`` last, so ``--skip_existing`` resumes an
interrupted run; the parent assembles ``index.parquet`` from every sidecar.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

_EPOCH = dt.date(2022, 1, 1)
_FNAME = re.compile(r"roi(?P<roi>\d+)_(?P<sensor>s2_toa|s1|cld_shdw)_"
                    r"(?P<y>\d{4})_(?P<m>\d{1,2})_(?P<d>\d{1,2})_median\.tif$")


def _read_tif(data: bytes) -> np.ndarray:
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(data) as mf, mf.open() as src:
            return src.read()
    except ImportError:
        import tifffile
        arr = tifffile.imread(io.BytesIO(data))
        if arr.ndim == 2:
            return arr[None]
        return arr if arr.shape[0] < arr.shape[-1] \
            else np.moveaxis(arr, -1, 0)


def _date_key(name: str) -> tuple | None:
    m = _FNAME.search(name)
    if not m:
        return None
    d = dt.date(int(m["y"]), int(m["m"]), int(m["d"]))
    return m["sensor"], (d - _EPOCH).days, d.timetuple().tm_yday


def _tar_roi(tar_path: Path) -> str:
    return re.search(r"roi(\d+)", tar_path.name).group(1)


def transcode_roi(tar_path: Path, out_root: Path) -> list[dict]:
    roi = _tar_roi(tar_path)
    scenes: dict[str, list[tuple[int, int, bytes]]] = {
        "s2_toa": [], "s1": [], "cld_shdw": []}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            key = _date_key(member.name)
            if key is None:
                continue
            sensor, date, doy = key
            scenes[sensor].append((date, doy, tf.extractfile(member).read()))
    for sensor, v in scenes.items():
        dates = [d for d, _, _ in v]
        if len(dates) != len(set(dates)):
            dup = sorted(d for d in set(dates) if dates.count(d) > 1)
            raise ValueError(f"roi {roi}: same-day duplicate {sensor} scenes at dates {dup[:3]}")
        v.sort(key=lambda x: x[0])

    cld_by_date = {d: b for d, _, b in scenes["cld_shdw"]}
    n_no_cld = sum(1 for d, _, _ in scenes["s2_toa"] if d not in cld_by_date)
    if n_no_cld:
        scenes["s2_toa"] = [s for s in scenes["s2_toa"] if s[0] in cld_by_date]
        print(f"roi {roi}: skipped {n_no_cld} s2 scenes lacking cld_shdw", flush=True)
    if not scenes["s2_toa"]:
        raise ValueError(f"roi {roi}: no usable s2 scenes ({n_no_cld} lacked cld_shdw)")

    roi_dir = out_root / "rois" / f"roi{roi}"
    roi_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    s2_stack, cld_stack = [], []
    for i, (date, doy, blob) in enumerate(scenes["s2_toa"]):
        a = _read_tif(blob).astype(np.float64)                      # [13,256,256]
        nan_pct = 100.0 * float(np.isnan(a).any(axis=0).mean())
        a = np.nan_to_num(a, nan=0.0)
        r = np.round(a)
        round_err = float(np.abs(a - r).max())
        if round_err > 0.5:
            raise ValueError(f"roi {roi} s2 i={i}: DN {round_err} off integer -- corrupt input")
        if r.min() < 0 or r.max() > 65535:
            raise ValueError(f"roi {roi} s2 i={i}: DN outside uint16 range")
        s2_stack.append(r.astype(np.uint16))
        c = np.nan_to_num(_read_tif(cld_by_date[date]).astype(np.float64), nan=0.0)
        prob = np.round(np.clip(c[0], 0, 100) * 2.0).astype(np.uint8)
        cld_frac = int((np.mod(c[1:5], 1.0) != 0.0).sum())
        evalm = ((c[1] > 0) | (c[4] > 0)).astype(np.uint8)
        cld_stack.append(np.stack([prob, evalm]))
        cloud30 = 100.0 * float((c[1] > 0).mean())
        shadow30 = 100.0 * float((c[4] > 0).mean())
        rows.append(dict(roi=roi, sensor="s2", i=i, date=date, doy=doy,
                         cloud30=cloud30, shadow30=shadow30, nan_pct=nan_pct,
                         s2_max_round_err_dn=round_err, cld_frac_count=cld_frac))
    np.save(roi_dir / "s2.npy", np.stack(s2_stack))
    np.save(roi_dir / "cld.npy", np.stack(cld_stack))

    if scenes["s1"]:
        s1_stack = []
        for i, (date, doy, blob) in enumerate(scenes["s1"]):
            a = np.nan_to_num(_read_tif(blob).astype(np.float64), nan=-50.0)
            q = np.round(a * 100.0)
            if q.min() < -32768 or q.max() > 32767:
                raise ValueError(f"roi {roi} s1 i={i}: dB*100 outside int16 range")
            q = q.astype(np.int16)
            rt_err = float(np.abs(q.astype(np.float64) / 100.0 - a).max())
            if rt_err > 0.005 + 1e-9:
                raise ValueError(f"roi {roi} s1 i={i}: round-trip {rt_err} dB > 0.005")
            s1_stack.append(q)
            rows.append(dict(roi=roi, sensor="s1", i=i, date=date, doy=doy,
                             cloud30=np.nan, shadow30=np.nan, nan_pct=0.0))
        np.save(roi_dir / "s1.npy", np.stack(s1_stack))

    # Commit marker, written last and atomically: its presence means the roi is complete.
    tmp = roi_dir / "rows.json.tmp"
    tmp.write_text(json.dumps({"roi": roi, "skipped_no_cld": n_no_cld, "rows": rows}))
    tmp.replace(roi_dir / "rows.json")
    return rows


def _roi_digits(r: str) -> str:
    """Accept both roi forms: '219586' and 'roi219586' -> '219586'."""
    m = re.fullmatch(r"(?:roi)?(\d+)", str(r))
    if not m:
        raise SystemExit(f"--rois entry {r!r} is not a roi id ('219586' or 'roi219586')")
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="dir of per-ROI tar.gz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rois", nargs="*", default=None,
                    help="roi ids, '219586' or 'roi219586' form")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_existing", action="store_true",
                    help="resume: skip rois that already have rois/roi{ID}/rows.json")
    args = ap.parse_args()

    archive, out = Path(args.archive), Path(args.out)
    tars = sorted(archive.glob("*.tar.gz"))
    if args.rois:
        keep = {_roi_digits(r) for r in args.rois}
        tars = [t for t in tars if _tar_roi(t) in keep]
    if not tars:
        raise SystemExit(f"no tar.gz under {archive}" + (" matching --rois" if args.rois else ""))
    out.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        todo = [t for t in tars
                if not (out / "rois" / f"roi{_tar_roi(t)}" / "rows.json").is_file()]
        print(f"--skip_existing: {len(tars) - len(todo)} rois already done, "
              f"{len(todo)} to go", flush=True)
        tars = todo

    failures: list[dict] = []
    with cf.ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(transcode_roi, t, out): t for t in tars}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:               # one bad roi must not stop the corpus
                roi = _tar_roi(futs[fut])
                failures.append(dict(roi=roi, error=f"{type(e).__name__}: {e}"))
                print(f"roi {roi} FAILED: {type(e).__name__}: {e}", flush=True)
            if n % 100 == 0 or n == len(tars):
                print(f"[{n}/{len(tars)}] rois transcoded", flush=True)

    rows: list[dict] = []
    n_skipped = 0
    for sidecar in sorted((out / "rois").glob("roi*/rows.json")):
        try:
            d = json.loads(sidecar.read_text())
        except ValueError as e:
            failures.append(dict(roi=_tar_roi(sidecar.parent), error=f"unreadable sidecar: {e}"))
            continue
        rows.extend(d["rows"])
        n_skipped += d["skipped_no_cld"]
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(out / "index.parquet")
        print(f"wrote {len(df)} scene rows, {df.roi.nunique()} rois -> {out/'index.parquet'}")
    else:
        print("no roi sidecars found: index.parquet not written")
    print(f"{n_skipped} s2 scenes skipped for missing cld_shdw")

    (out / "transcode_failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} rois FAILED -> {out/'transcode_failures.json'}")
    print("0 rois failed")


if __name__ == "__main__":
    main()
