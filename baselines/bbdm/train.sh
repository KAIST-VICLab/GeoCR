#!/usr/bin/env bash
# Train BBDM on one dataset (seed 0).
#   REPO=<patched clone> CR_DATASETS=<datasets> CR_EXPORTS=<exports> bash train.sh <dataset>
# <dataset>: t-cloud | cuhk-cr2 | sen2-mtc-new-rgb | sen12ms-cr-rgb | whus2-crv-rgb
# Optional: OUT (default ./out/bbdm), PY.
# Also needs VQF4_CKPT, the vq-f4 model.ckpt (README.md).
# Writes $OUT/runs/<corpus>/LBBDM-f4/checkpoint/last_model.pth.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new-rgb) C=sen2mtc_new_rgb ;;
  sen12ms-cr-rgb) C=sen12mscr_rgb ;; whus2-crv-rgb) C=whus2crv_rgb ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|sen2-mtc-new-rgb|sen12ms-cr-rgb|whus2-crv-rgb>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
: "${REPO:?set REPO to the patched BBDM clone}" "${CR_DATASETS:?set CR_DATASETS}" "${CR_EXPORTS:?set CR_EXPORTS}"
CR_DATASETS=$(cd "$CR_DATASETS" && pwd); CR_EXPORTS=$(cd "$CR_EXPORTS" && pwd)
OUT=${OUT:-$PWD/out/bbdm}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export CR_DATASETS CR_EXPORTS PYTHONPATH="$KIT/../common:$KIT/../..${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}

: "${VQF4_CKPT:?set VQF4_CKPT to the vq-f4 model.ckpt}"
VQ=$(cd "$(dirname "$VQF4_CKPT")" && pwd)/$(basename "$VQF4_CKPT")
mkdir -p "$OUT/runs"
sed "s#\${VQF4_CKPT}#$VQ#" "$KIT/configs/$C.yaml" > "$OUT/runs/$C.yaml"
cd "$REPO"
"$PY" main.py --config "$OUT/runs/$C.yaml" --train --seed 0 --gpu_ids 0 --result_path "$OUT/runs"
