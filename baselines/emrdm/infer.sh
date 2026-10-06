#!/usr/bin/env bash
# Predict the test split with an EMRDM checkpoint (last.ckpt; sampling uses the EMA weights).
#   REPO=<patched EMRDM clone> CR_DATASETS=<dataset root> CKPT=<checkpoint> bash infer.sh <t-cloud|cuhk-cr1|cuhk-cr2|sen2-mtc-new|sen12ms-cr|whus2-crv>
# CKPT defaults to the newest checkpoint written by train.sh. Output: $OUT/dumps/<corpus>/test/emrdm/<run_id>/.
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
CKPT=${CKPT:-$(ls -td "$OUT"/runs/*_emrdm_${C}_seed${SEED}/checkpoints/last.ckpt 2>/dev/null | head -1)}
CKPT=$(realpath -m "${CKPT:?set CKPT to last.ckpt}")
case $C in
  tcloud|cuhk_cr1|cuhk_cr2) SAMPLER=(--sampler ResidualEulerEDMSampler --nfe 4) ;;
  sen2mtc_new) SAMPLER=(--sampler TemporalResidualEulerEDMSampler --nfe 5) ;;
  *) SAMPLER=(--sampler ResidualEulerEDMSampler --nfe 5) ;;
esac
DTYPE=float32; [ "$C" = sen12mscr ] && DTYPE=float16
export EMRDM_DUMP_ROOT=$OUT/dumps EMRDM_RUN_ID=$(date +%Y%m%d)_seed$SEED

cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY" main.py -b "$KIT/configs/$C.yaml" -n "emrdm_${C}_seed${SEED}_predict" --enable_tf32 \
  --seed "$SEED" -t false --no-test true --predict true -l "$OUT/runs" \
  model.params.ckpt_path="$CKPT" data.params.batch_size=1
"$PY" -m geocr_bench.dump --dump-dir "$EMRDM_DUMP_ROOT/$C/test/emrdm/$EMRDM_RUN_ID" --corpus "$C" --split test \
  --splits-dir splits --seed "$SEED" --ckpt "$CKPT" --ckpt-selection final --train-split trainval \
  --method-variant "$C.yaml" --dump-dtype "$DTYPE" "${SAMPLER[@]}"
