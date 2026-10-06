#!/usr/bin/env bash
# Predict the test split of one dataset with BBDM.
#   REPO=<patched clone> CR_DATASETS=<datasets> CR_EXPORTS=<exports> [CKPT=<dir>] bash infer.sh <dataset>
# CKPT is the directory holding last_model.pth
# (the downloaded weights, or by default the output of train.sh).
# Optional: OUT (default ./out/bbdm), PY. Predictions: $OUT/pred/<dataset>/{preds/,manifest.json}.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new-rgb) C=sen2mtc_new_rgb ;;
  sen12ms-cr-rgb) C=sen12mscr_rgb ;; whus2-crv-rgb) C=whus2crv_rgb ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|sen2-mtc-new-rgb|sen12ms-cr-rgb|whus2-crv-rgb>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
: "${REPO:?set REPO to the patched BBDM clone}" "${CR_DATASETS:?set CR_DATASETS}" "${CR_EXPORTS:?set CR_EXPORTS}"
CR_DATASETS=$(cd "$CR_DATASETS" && pwd); CR_EXPORTS=$(cd "$CR_EXPORTS" && pwd)
OUT=${OUT:-$PWD/out/bbdm}; mkdir -p "$OUT/pred/$1"; OUT=$(cd "$OUT" && pwd)
CKPT=$(cd "${CKPT:-$OUT/runs/$C/LBBDM-f4/checkpoint}" && pwd)
export CR_DATASETS CR_EXPORTS PYTHONPATH="$KIT/../common:$KIT/../..${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}

cd "$REPO"
"$PY" dump.py --config "$KIT/configs/$C.yaml" --ckpt "$CKPT" --out "$OUT/pred/$1"
echo "score with: PYTHONPATH=$(cd "$KIT/../.." && pwd) python -m geocr.eval --dataset $1 --pred $OUT/pred/$1 --metrics fid,dists,kid,lpips,ssim,psnr"
