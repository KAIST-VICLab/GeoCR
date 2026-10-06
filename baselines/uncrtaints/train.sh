#!/usr/bin/env bash
# Train UnCRtainTS on one dataset.
#   REPO=<patched UnCRtainTS clone> CR_DATASETS=<dataset root> bash train.sh <sen12ms-cr|whus2-crv|sen2-mtc-new|t-cloud|cuhk-cr1|cuhk-cr2>
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

source "$KIT/configs/$C.sh"
DATA=$CR_DATASETS/$CORPUS_DIR
if [ "$MONO" = 1 ]; then TEMPORAL=(--pretrain); else TEMPORAL=(--input_t "$INPUT_T" --n_head 16 --positional_encoding); fi
SAR=(); [ "$USE_SAR" = 1 ] && SAR=(--use_sar)
cd "$REPO/model"
CUDA_VISIBLE_DEVICES=$GPU UNCR_S2_BANDS=$S2_BANDS "$PY" train_reconstruct.py \
  --model uncrtaints --experiment_name "${C}_seed$SEED" --res_dir "$OUT/runs" \
  --corpus "$CORPUS" --corpus_root "$DATA" --splits_dir splits --split_list "$DATA/splits/trainval.txt" \
  --export_root "$CR_EXPORTS" --loss "$LOSS" --scale_by "$SCALE_BY" --block_type mbconv \
  --encoder_widths "[128]" --decoder_widths "[128,128,128,128,128]" \
  --batch_size "$BATCH_SIZE" --lr "$LR" --gamma "$GAMMA" --epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
  --val_every "$VAL_EVERY" --early_stop_patience -1 --ckpt_selection final --save_every_epoch 0 \
  --rdm_seed "$SEED" --num_workers "$NUM_WORKERS" --display_step 500 --plot_every -1 --export_every -1 \
  --device cuda "${TEMPORAL[@]}" ${SAR[@]+"${SAR[@]}"}
