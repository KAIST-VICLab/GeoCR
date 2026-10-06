"""python -m geocr.eval --dataset <dataset> --pred <prediction dir> [--metrics ...] [--out results.json]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from geocr.eval.score import (CORPORA, HEADLINE, SLUGS, eval_ids, ground_truth, load_pred,
                              read_split, resolve_corpus, score, sha256_file, sid_flat)

METRICS = ("fid", "dists", "kid", "dino", "lpips", "ssim", "psnr")
FEATURE_METRICS = ("fid", "dists", "kid", "dino", "lpips")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m geocr.eval",
                                 description="Score cloud-removal predictions on a GeoCR test split.")
    ap.add_argument("--dataset", required=True,
                    help="one of " + ", ".join(SLUGS) + " (keys such as sen12mscr also work)")
    ap.add_argument("--pred", required=True, type=Path,
                    help="prediction directory holding preds/<id>.npy")
    ap.add_argument("--metrics", default=",".join(METRICS), help="comma-separated (default: all)")
    ap.add_argument("--root", type=Path, help="dataset root (default: $CR_DATASETS/<dataset dir>)")
    ap.add_argument("--splits-dir", default="splits",
                    help="directory of test.txt, relative to the root or absolute")
    ap.add_argument("--export", default=os.environ.get("CR_EXPORTS"),
                    help="root of the 8-bit RGB exports (default: $CR_EXPORTS); "
                         "required for the *-rgb datasets")
    ap.add_argument("--dino-weights", type=Path, help="dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    ap.add_argument("--dino-repo", type=Path,
                    help="clone of github.com/facebookresearch/dinov3 "
                         "(not needed if the dinov3 package is installed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, help="also write the results to this JSON file")
    a = ap.parse_args(argv)

    corpus = resolve_corpus(a.dataset)
    if corpus not in CORPORA:
        ap.error(f"unknown dataset {a.dataset!r}")
    metrics = [m.strip().lower() for m in a.metrics.split(",") if m.strip()]
    if not metrics or set(metrics) - set(METRICS):
        ap.error(f"--metrics must be a subset of {','.join(METRICS)}")
    if "dino" in metrics and not a.dino_weights:
        ap.error("dino needs --dino-weights (DINOv3 ViT-L/16 SAT-493M), or drop it from --metrics")

    cfg = CORPORA[corpus]
    root = a.root or cfg["root"]
    ids = eval_ids(a.pred, read_split(root, "test", a.splits_dir), cfg)
    gt = ground_truth(corpus, "test", root, a.export, a.splits_dir)
    # the sen2mtc metrics and RGB views never read the NIR plane
    allow = cfg["lineage"] == "sen2mtc"
    res = dict(dataset=next(k for k, v in SLUGS.items() if v == corpus), corpus=corpus,
               pred=str(a.pred), split_sha256=sha256_file(Path(root) / a.splits_dir / "test.txt"),
               n=len(ids))
    vals = {}
    if {"psnr", "ssim"} & set(metrics):
        detail = score(a.pred, cfg, ids, gt, allow)
        vals["psnr"], vals["ssim"] = (detail[k] for k in HEADLINE[cfg["lineage"]])
        res["detail"] = detail
    if set(FEATURE_METRICS) & set(metrics):
        from geocr.eval.perceptual import feature_ids, perceptual, to_rgb01
        fids = feature_ids(ids)
        lin = cfg["lineage"]
        gv = torch.stack([to_rgb01(gt(s), lin) for s in fids])
        pv = torch.stack([to_rgb01(load_pred(a.pred, sid_flat(s), cfg, allow), lin) for s in fids])
        vals.update(perceptual(pv, gv, metrics, a.device))
        if "dino" in metrics:
            from geocr.eval.dino import dino_cos, load_dinov3_sat
            vals["dino"] = dino_cos(pv, gv, load_dinov3_sat(a.dino_weights, a.dino_repo, a.device))
        res["n_feature"] = len(fids)
    res["metrics"] = {m: vals[m] for m in METRICS if m in metrics}
    if "kid_std" in vals:
        res["kid_std"] = vals["kid_std"]
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
