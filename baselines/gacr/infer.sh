#!/usr/bin/env bash
# Predict the test split with a GACR checkpoint.
#   REPO=<patched GACR clone> CR_DATASETS=<dataset root> CKPT=<checkpoint> bash infer.sh <cuhk-cr2|t-cloud|sen2-mtc-new|sen12ms-cr|cuhk-cr1|whus2-crv>
# CKPT defaults to the final checkpoint written by train.sh. Output: $OUT/dumps/<corpus>/test/gacr/<run_id>/.
# Weights used: 'ema' on cuhk-cr1, cuhk-cr2, sen2-mtc-new; 'model' on t-cloud, sen12ms-cr, whus2-crv (override: STATE_KEY).
# Optional: GEOCR_ROOT, GEOCR_BASELINES, CR_EXPORTS, OUT, GPU, SEED, PY.
set -euo pipefail
case "${1:-}" in
  cuhk-cr2) C=cuhk_cr2 ;; t-cloud) C=tcloud ;; sen2-mtc-new) C=sen2mtc_new ;; sen12ms-cr) C=sen12mscr ;; cuhk-cr1) C=cuhk_cr1 ;; whus2-crv) C=whus2crv ;;
  *) echo "usage: $0 <cuhk-cr2|t-cloud|sen2-mtc-new|sen12ms-cr|cuhk-cr1|whus2-crv>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${REPO:?set REPO to the patched GACR clone}" && pwd)
CR_DATASETS=$(cd "${CR_DATASETS:?set CR_DATASETS to the dataset root}" && pwd)
OUT=${OUT:-$PWD/out/gacr}; mkdir -p "$OUT/configs"; OUT=$(cd "$OUT" && pwd)
export REPO CR_DATASETS OUT
export CR_EXPORTS=${CR_EXPORTS:-$(dirname "$CR_DATASETS")/CR_exports}
export GEOCR_ROOT=${GEOCR_ROOT:-$(cd "$KIT/../.." && pwd)}
export GEOCR_BASELINES=${GEOCR_BASELINES:-$GEOCR_ROOT/baselines/common}
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-0}
case $C in
  cuhk_cr1|cuhk_cr2|sen2mtc_new) STEP=0200000; KEY=ema ;;
  tcloud) STEP=0020000; KEY=model ;;
  *) STEP=0050000; KEY=model ;;
esac
CKPT=$(realpath -m "${CKPT:-$OUT/runs/${C}_seed$SEED/checkpoints/$STEP.pt}")
render() { "$PY" -c 'import os,re,sys; t=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(re.sub(r"\$\{([A-Z0-9_]+)\}",lambda m:os.environ.get(m.group(1),m.group(0)),t))' "$1" "$2"; }

render "$KIT/configs/$C.yaml" "$OUT/configs/$C.yaml"
cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY" dump.py --config "$OUT/configs/$C.yaml" --set misc.seed=$SEED --ckpt "$CKPT" \
  --state-key "${STATE_KEY:-$KEY}" --out "$OUT/dumps/$C/test/gacr/$(date +%Y%m%d)_seed$SEED"
