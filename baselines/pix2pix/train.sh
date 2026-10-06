#!/usr/bin/env bash
# Train pix2pix on one dataset (seed 0).
#   REPO=<patched clone> CR_DATASETS=<datasets> CR_EXPORTS=<exports> bash train.sh <dataset>
# <dataset>: t-cloud | cuhk-cr2 | sen2-mtc-new-rgb | sen12ms-cr-rgb | whus2-crv-rgb
# Optional: OUT (default ./out/pix2pix), PY.
# Writes $OUT/runs/<corpus>_pix2pix_seed0/latest_net_G.pth.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new-rgb) C=sen2mtc_new_rgb ;;
  sen12ms-cr-rgb) C=sen12mscr_rgb ;; whus2-crv-rgb) C=whus2crv_rgb ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|sen2-mtc-new-rgb|sen12ms-cr-rgb|whus2-crv-rgb>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
: "${REPO:?set REPO to the patched pix2pix clone}" "${CR_DATASETS:?set CR_DATASETS}" "${CR_EXPORTS:?set CR_EXPORTS}"
CR_DATASETS=$(cd "$CR_DATASETS" && pwd); CR_EXPORTS=$(cd "$CR_EXPORTS" && pwd)
OUT=${OUT:-$PWD/out/pix2pix}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export CR_DATASETS CR_EXPORTS PYTHONPATH="$KIT/../common:$KIT/../..${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}

cd "$REPO"
"$PY" train_geocr.py --config "$KIT/configs/$C.sh" --seed 0 --out "$OUT/runs"
