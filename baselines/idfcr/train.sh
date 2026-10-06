#!/usr/bin/env bash
# Train IDF-CR (pixel/) on one dataset.
#   REPO=<patched IDF-CR clone> CR_DATASETS=<dataset root> bash train.sh <t-cloud|cuhk-cr2|whus2-crv>
# Optional: GEOCR_ROOT, GEOCR_BASELINES, CR_EXPORTS, OUT, GPU, SEED, PY.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; whus2-crv) C=whus2crv ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|whus2-crv>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${REPO:?set REPO to the patched IDF-CR clone}" && pwd)
CR_DATASETS=$(cd "${CR_DATASETS:?set CR_DATASETS to the dataset root}" && pwd)
OUT=${OUT:-$PWD/out/idfcr}; mkdir -p "$OUT/configs"; OUT=$(cd "$OUT" && pwd)
export REPO CR_DATASETS OUT
export CR_EXPORTS=${CR_EXPORTS:-$(dirname "$CR_DATASETS")/CR_exports}
export GEOCR_ROOT=${GEOCR_ROOT:-$(cd "$KIT/../.." && pwd)}
export GEOCR_BASELINES=${GEOCR_BASELINES:-$GEOCR_ROOT/baselines/common}
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-0}
render() { "$PY" -c 'import os,re,sys; t=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(re.sub(r"\$\{([A-Z0-9_]+)\}",lambda m:os.environ.get(m.group(1),m.group(0)),t))' "$1" "$2"; }

render "$KIT/configs/$C.yml" "$OUT/configs/$C.yml"
cd "$REPO/pixel"
"$PY" train_geocr.py --config "$OUT/configs/$C.yml" --gpu "$GPU" --seed "$SEED" --out "$OUT/runs/${C}_s$SEED"
