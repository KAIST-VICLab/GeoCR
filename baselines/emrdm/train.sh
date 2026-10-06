#!/usr/bin/env bash
# Train EMRDM on one dataset (needs natten).
#   REPO=<patched EMRDM clone> CR_DATASETS=<dataset root> bash train.sh <t-cloud|cuhk-cr1|cuhk-cr2|sen2-mtc-new|sen12ms-cr|whus2-crv>
# Optional: GEOCR_ROOT, GEOCR_BASELINES, CR_EXPORTS, OUT, GPU, SEED, PY.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr1) C=cuhk_cr1 ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new) C=sen2mtc_new ;; sen12ms-cr) C=sen12mscr ;; whus2-crv) C=whus2crv ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr1|cuhk-cr2|sen2-mtc-new|sen12ms-cr|whus2-crv>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${REPO:?set REPO to the patched EMRDM clone}" && pwd)
CR_DATASETS=$(cd "${CR_DATASETS:?set CR_DATASETS to the dataset root}" && pwd)
OUT=${OUT:-$PWD/out/emrdm}; mkdir -p "$OUT/configs"; OUT=$(cd "$OUT" && pwd)
export REPO CR_DATASETS OUT
export CR_EXPORTS=${CR_EXPORTS:-$(dirname "$CR_DATASETS")/CR_exports}
export GEOCR_ROOT=${GEOCR_ROOT:-$(cd "$KIT/../.." && pwd)}
export GEOCR_BASELINES=${GEOCR_BASELINES:-$GEOCR_ROOT/baselines/common}
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-0}
export EMRDM_DUMP_ROOT=$OUT/dumps

cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY" main.py -b "$KIT/configs/$C.yaml" -n "emrdm_${C}_seed$SEED" --enable_tf32 \
  --seed "$SEED" --no-test true -l "$OUT/runs"
