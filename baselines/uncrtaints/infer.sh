#!/usr/bin/env bash
# Predict the test split with an UnCRtainTS checkpoint (model.pth.tar).
#   REPO=<patched UnCRtainTS clone> CR_DATASETS=<dataset root> CKPT=<checkpoint> bash infer.sh <sen12ms-cr|whus2-crv|sen2-mtc-new|t-cloud|cuhk-cr1|cuhk-cr2>
# CKPT defaults to the checkpoint written by train.sh. Output: $OUT/dumps/<corpus>/test/<method>/<run_id>/.
# Optional: GEOCR_ROOT, GEOCR_BASELINES, CR_EXPORTS, OUT, GPU, SEED, PY.
set -euo pipefail
case "${1:-}" in
  sen12ms-cr) C=sen12mscr ;; whus2-crv) C=whus2crv ;; sen2-mtc-new) C=sen2mtc_new ;; t-cloud) C=tcloud ;; cuhk-cr1) C=cuhk_cr1 ;; cuhk-cr2) C=cuhk_cr2 ;;
  *) echo "usage: $0 <sen12ms-cr|whus2-crv|sen2-mtc-new|t-cloud|cuhk-cr1|cuhk-cr2>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${REPO:?set REPO to the patched UnCRtainTS clone}" && pwd)
CR_DATASETS=$(cd "${CR_DATASETS:?set CR_DATASETS to the dataset root}" && pwd)
OUT=${OUT:-$PWD/out/uncrtaints}; mkdir -p "$OUT/configs"; OUT=$(cd "$OUT" && pwd)
export REPO CR_DATASETS OUT
export CR_EXPORTS=${CR_EXPORTS:-$(dirname "$CR_DATASETS")/CR_exports}
export GEOCR_ROOT=${GEOCR_ROOT:-$(cd "$KIT/../.." && pwd)}
export GEOCR_BASELINES=${GEOCR_BASELINES:-$GEOCR_ROOT/baselines/common}
PY=${PY:-python}; GPU=${GPU:-0}; SEED=${SEED:-0}
CKPT=$(realpath -m "${CKPT:-$OUT/runs/${C}_seed$SEED/model.pth.tar}")
render() { "$PY" -c 'import os,re,sys; t=open(sys.argv[1]).read(); open(sys.argv[2],"w").write(re.sub(r"\$\{([A-Z0-9_]+)\}",lambda m:os.environ.get(m.group(1),m.group(0)),t))' "$1" "$2"; }

source "$KIT/configs/$C.sh"
render "$KIT/configs/$C.json" "$OUT/configs/$C.json"
mkdir -p "$OUT/weights/${C}_seed0"
ln -sfn "$CKPT" "$OUT/weights/${C}_seed0/model.pth.tar"
FP16=(); [ "$DUMP_FP16" = 1 ] && FP16=(--dump_fp16)
cd "$REPO/model"
CUDA_VISIBLE_DEVICES=$GPU UNCR_S2_BANDS=$S2_BANDS "$PY" dump_geocr.py --load_config "$OUT/configs/$C.json" \
  --corpus "$CORPUS" --corpus_root "$CR_DATASETS/$CORPUS_DIR" --splits_dir splits --export_root "$CR_EXPORTS" \
  --dump_split test --dump_dir "$OUT/dumps" --method_name "$METHOD_NAME" --run_id "$(date +%Y%m%d)_seed$SEED" \
  --weight_folder "$OUT/weights" --experiment_name "${C}_seed0" --res_dir "$OUT/runs/_dump" \
  --batch_size "$DUMP_BATCH" --num_workers "$NUM_WORKERS" --device cuda ${FP16[@]+"${FP16[@]}"}
