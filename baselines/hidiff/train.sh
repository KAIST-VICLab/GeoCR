#!/usr/bin/env bash
# Train HI-Diff on one dataset (stage 1, then stage 2; seed 0).
#   REPO=<patched clone> CR_DATASETS=<datasets> CR_EXPORTS=<exports> bash train.sh <dataset>
# <dataset>: t-cloud | cuhk-cr2 | sen2-mtc-new-rgb | sen12ms-cr-rgb | whus2-crv-rgb
# Optional: PY. basicsr writes $REPO/experiments/geocr_<corpus>_s{1,2}_seed0/.
set -euo pipefail
case "${1:-}" in
  t-cloud) C=tcloud ;; cuhk-cr2) C=cuhk_cr2 ;; sen2-mtc-new-rgb) C=sen2mtc_new_rgb ;;
  sen12ms-cr-rgb) C=sen12mscr_rgb ;; whus2-crv-rgb) C=whus2crv_rgb ;;
  *) echo "usage: $0 <t-cloud|cuhk-cr2|sen2-mtc-new-rgb|sen12ms-cr-rgb|whus2-crv-rgb>" >&2; exit 2 ;;
esac
KIT=$(cd "$(dirname "$0")" && pwd)
: "${REPO:?set REPO to the patched HI-Diff clone}" "${CR_DATASETS:?set CR_DATASETS}" "${CR_EXPORTS:?set CR_EXPORTS}"
CR_DATASETS=$(cd "$CR_DATASETS" && pwd); CR_EXPORTS=$(cd "$CR_EXPORTS" && pwd)
export CR_DATASETS CR_EXPORTS PYTHONPATH="$KIT/../common:$KIT/../..${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}

cd "$REPO"
mkdir -p options/geocr
for ST in s1 s2; do
  sed "s#\${CR_EXPORTS}#$CR_EXPORTS#g" "$KIT/configs/${C}_$ST.yml" > "options/geocr/${C}_$ST.yml"
  "$PY" train.py -opt "options/geocr/${C}_$ST.yml" --auto_resume
done
