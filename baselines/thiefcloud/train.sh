#!/usr/bin/env bash
# Train ThiefCloud (two stages) on one dataset.
#   REPO=<patched ThiefCloud clone> CR_DATASETS=<dataset root> bash train.sh <t-cloud|cuhk-cr2|sen2-mtc-new>
# Optional: GEOCR_ROOT, GEOCR_BASELINES, CR_EXPORTS, OUT, GPU, SEED, PY.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new) C=sen2mtc_new ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|sen2-mtc-new>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${REPO:?set REPO to the patched ThiefCloud clone}" && pwd)
CR_DATASETS=$(cd "${CR_DATASETS:?set CR_DATASETS to the dataset root}" && pwd)
OUT=${OUT:-$PWD/out/thiefcloud}; mkdir -p "$OUT/configs"; OUT=$(cd "$OUT" && pwd)
export REPO CR_DATASETS OUT
export CR_EXPORTS=${CR_EXPORTS:-$(dirname "$CR_DATASETS")/CR_exports}
export GEOCR_ROOT=${GEOCR_ROOT:-$(cd "$KIT/../.." && pwd)}
export GEOCR_BASELINES=${GEOCR_BASELINES:-$GEOCR_ROOT/baselines/common}
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-0}
render() { "$PY" -c 'import os,re,sys; t=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(re.sub(r"\$\{([A-Z0-9_]+)\}",lambda m:os.environ.get(m.group(1),m.group(0)),t))' "$1" "$2"; }

render "$KIT/configs/$C.yaml" "$OUT/configs/$C.yaml"
cd "$REPO/ThiefCloud"
CUDA_VISIBLE_DEVICES=$GPU "$PY" geocr_train.py --config "$OUT/configs/$C.yaml" --out "$OUT/runs/$C/seed$SEED" seed=$SEED
