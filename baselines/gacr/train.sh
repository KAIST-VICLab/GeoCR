#!/usr/bin/env bash
# Train GACR on one dataset (needs natten and the DINOv3 ViT-L/16 LVD-1689M teacher).
#   REPO=<patched GACR clone> CR_DATASETS=<dataset root> bash train.sh <cuhk-cr2|t-cloud|sen2-mtc-new|sen12ms-cr|cuhk-cr1|whus2-crv>
# Also required: DINOV3_ROOT=<facebookresearch/dinov3 clone> DINOV3_WEIGHTS=<dinov3_vitl16_pretrain_lvd1689m-*.pth>.
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
: "${DINOV3_ROOT:?set DINOV3_ROOT to a clone of facebookresearch/dinov3}" "${DINOV3_WEIGHTS:?set DINOV3_WEIGHTS to the ViT-L/16 LVD-1689M checkpoint}"
export DINOV3_ROOT DINOV3_WEIGHTS
render() { "$PY" -c 'import os,re,sys; t=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(re.sub(r"\$\{([A-Z0-9_]+)\}",lambda m:os.environ.get(m.group(1),m.group(0)),t))' "$1" "$2"; }

render "$KIT/configs/$C.yaml" "$OUT/configs/$C.yaml"
cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m accelerate.commands.launch --num_processes 1 train.py --config "$OUT/configs/$C.yaml" \
  --set misc.seed=$SEED --exp-name "${C}_seed$SEED"
