#!/usr/bin/env bash
# Predict the test split with a DiffCR checkpoint (final_Network.pth).
#   REPO=<patched DiffCR clone> CR_DATASETS=<dataset root> CKPT=<checkpoint> bash infer.sh <t-cloud|sen2-mtc-new|cuhk-cr1|cuhk-cr2>
# CKPT defaults to the newest checkpoint written by train.sh. Output: $OUT/dumps/<corpus>/test/diffcr/<run_id>/.
# Optional: GEOCR_ROOT, GEOCR_BASELINES, CR_EXPORTS, OUT, GPU, SEED, PY.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; sen2-mtc-new) C=sen2mtc_new ;; cuhk-cr1) C=cuhk_cr1 ;; cuhk-cr2) C=cuhk_cr2 ;;
  *) echo "usage: $0 <t-cloud|sen2-mtc-new|cuhk-cr1|cuhk-cr2>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${REPO:?set REPO to the patched DiffCR clone}" && pwd)
CR_DATASETS=$(cd "${CR_DATASETS:?set CR_DATASETS to the dataset root}" && pwd)
OUT=${OUT:-$PWD/out/diffcr}; mkdir -p "$OUT/configs"; OUT=$(cd "$OUT" && pwd)
export REPO CR_DATASETS OUT
export CR_EXPORTS=${CR_EXPORTS:-$(dirname "$CR_DATASETS")/CR_exports}
export GEOCR_ROOT=${GEOCR_ROOT:-$(cd "$KIT/../.." && pwd)}
export GEOCR_BASELINES=${GEOCR_BASELINES:-$GEOCR_ROOT/baselines/common}
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-0}
CKPT=${CKPT:-$(ls -td "$OUT"/runs/$C/train_${C}_seed${SEED}_*/checkpoint/final_Network.pth 2>/dev/null | head -1)}
CKPT=$(realpath -m "${CKPT:?set CKPT to final_Network.pth}")
render() { "$PY" -c 'import os,re,sys; t=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(re.sub(r"\$\{([A-Z0-9_]+)\}",lambda m:os.environ.get(m.group(1),m.group(0)),t))' "$1" "$2"; }

render "$KIT/configs/$C.json" "$OUT/configs/$C.json"
SUMMARY=(); TS=$(dirname "$(dirname "$CKPT")")/train_summary.json; [ -f "$TS" ] && SUMMARY=(--train-summary "$TS")
cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY" dump_geocr.py -c "$OUT/configs/$C.json" -o seed=$SEED --ckpt "$CKPT" --split test \
  --run-id "$(date +%Y%m%d)_seed$SEED" ${SUMMARY[@]+"${SUMMARY[@]}"}
