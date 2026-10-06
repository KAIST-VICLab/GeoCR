#!/usr/bin/env bash
# Predict the test split of one dataset with HI-Diff.
#   REPO=<patched clone> CR_DATASETS=<datasets> CR_EXPORTS=<exports> [CKPT=<dir>] bash infer.sh <dataset>
# CKPT is the directory holding net_g_latest.pth, net_le_dm_latest.pth and net_d_latest.pth
# (the downloaded weights, or by default the output of train.sh).
# Optional: OUT (default ./out/hidiff), PY. Predictions: $OUT/pred/<dataset>/{preds/,manifest.json}.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new-rgb) C=sen2mtc_new_rgb ;;
  sen12ms-cr-rgb) C=sen12mscr_rgb ;; whus2-crv-rgb) C=whus2crv_rgb ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|sen2-mtc-new-rgb|sen12ms-cr-rgb|whus2-crv-rgb>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
: "${REPO:?set REPO to the patched HI-Diff clone}" "${CR_DATASETS:?set CR_DATASETS}" "${CR_EXPORTS:?set CR_EXPORTS}"
CR_DATASETS=$(cd "$CR_DATASETS" && pwd); CR_EXPORTS=$(cd "$CR_EXPORTS" && pwd)
OUT=${OUT:-$PWD/out/hidiff}; mkdir -p "$OUT/pred/$1"; OUT=$(cd "$OUT" && pwd)
CKPT=$(cd "${CKPT:-$REPO/experiments/geocr_${C}_s2_seed0/models}" && pwd)
export CR_DATASETS CR_EXPORTS PYTHONPATH="$KIT/../common:$KIT/../..${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}

cd "$REPO"
"$PY" dump.py --corpus "$C" --config "$KIT/configs/${C}_s2.yml" --ckpt "$CKPT" --out "$OUT/pred/$1"
echo "score with: PYTHONPATH=$(cd "$KIT/../.." && pwd) python -m geocr.eval --dataset $1 --pred $OUT/pred/$1 --metrics fid,dists,kid,lpips,ssim,psnr"
