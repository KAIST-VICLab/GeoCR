# UnCRtainTS

**UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series**
(Ebel et al., CVPR Workshops 2023; "CVPRW'23" in the GeoCR tables).

| | |
|---|---|
| Upstream | <https://github.com/PatrickTUM/UnCRtainTS> |
| Pinned commit | `5e1f1b58e993645e765b64e10b6e9c7ff828b36f` |
| Upstream licence | the upstream repository does not include a licence |
| Datasets | SEN12MS-CR, WHUS2-CRv, Sen2_MTC_New, T-CLOUD, CUHK-CR1, CUHK-CR2 |

## Setup

```bash
git clone https://github.com/PatrickTUM/UnCRtainTS && cd UnCRtainTS
git checkout 5e1f1b58e993645e765b64e10b6e9c7ff828b36f
git apply /path/to/GeoCR/baselines/uncrtaints/uncrtaints.patch
```

Install the upstream requirements and this repository's `requirements.txt` (the data loaders import
`geocr` from this repository). Prepare the datasets and copy the split lists as described in
[DATA.md](../../DATA.md). T-CLOUD and CUHK-CR1/CR2 are read from the 8-bit copies in `$CR_EXPORTS`
(DATA.md, "8-bit copies for the comparison methods").

## What the patch changes

| File | Change | Why |
|---|---|---|
| `data/geocr_corpora.py` (new) | split-list datasets: SEN12MS-CR (with SAR), WHUS2-CRv (13 bands on the 10 m grid), Sen2_MTC_New (three cloudy frames), T-CLOUD, CUHK-CR1 (RGB+NIR), CUHK-CR2 | train and test on the split lists |
| `model/dump_geocr.py` (new) | predicts the test list and writes CHW `.npy` files plus `manifest.json` | input of the common evaluation |
| `model/src/geocr_bands.py` (new); `S2_BANDS` in `parse_args.py`, `train_reconstruct.py`, `src/model_utils.py`, `src/losses.py`, `src/backbones/{uncrtaints,base_model}.py` | the output band count is read from `UNCR_S2_BANDS` (default 13) | 3-band (T-CLOUD, CUHK-CR2) and 4-band (Sen2_MTC_New, CUHK-CR1) data |
| `model/train_reconstruct.py`, `model/parse_args.py` | `--corpus`, `--corpus_root`, `--splits_dir`, `--split_list`, `--export_root`, `--max_steps`, `--ckpt_selection final` (keep the last epoch as `model.pth.tar`), `--save_every_epoch`, `--early_stop_patience`; plot band indices for any band count | training on the split lists for a fixed number of steps |
| `data/dataLoader.py` | exact `<season>/<ROI>` match when assigning SEN12MS-CR patches to a split; optional `s2cloudless` import | split assignment by exact ROI identifiers |
| `model/src/backbones/base_model.py` | optional `fvcore` import | only used for profiling |
| `model/test_reconstruct.py` | keep `--batch_size` given on the command line | it was overwritten |

## Train and predict

```bash
export REPO=/path/to/UnCRtainTS CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
GPU=0 bash train.sh sen12ms-cr              # sen12ms-cr | whus2-crv | sen2-mtc-new | t-cloud | cuhk-cr1 | cuhk-cr2
GPU=0 bash infer.sh sen12ms-cr              # uses the checkpoint written by train.sh
CKPT=weights/baselines/sen12ms-cr/uncrtaints/model.pth.tar GPU=0 bash infer.sh sen12ms-cr   # released weights
```

`configs/<dataset>.sh` holds the training settings and `configs/<dataset>.json` the model
configuration that `infer.sh` passes to `dump_geocr.py --load_config`. Outputs go to `$OUT` (default
`./out/uncrtaints`): runs in `runs/<corpus>_seed<seed>/`, predictions in
`dumps/<corpus>/test/<method>/<run_id>/`. Score the predictions as described in
[../README.md](../README.md).

## Configurations and weights

| Dataset | Config | Weights in [`JeonghyeokDo/GeoCR`](https://huggingface.co/JeonghyeokDo/GeoCR) |
|---|---|---|
| SEN12MS-CR | [`configs/sen12mscr.sh`](configs/sen12mscr.sh), [`.json`](configs/sen12mscr.json) | `baselines/sen12ms-cr/uncrtaints/model.pth.tar` |
| WHUS2-CRv | [`configs/whus2crv.sh`](configs/whus2crv.sh), [`.json`](configs/whus2crv.json) | `baselines/whus2-crv/uncrtaints/model.pth.tar` |
| Sen2_MTC_New | [`configs/sen2mtc_new.sh`](configs/sen2mtc_new.sh), [`.json`](configs/sen2mtc_new.json) | `baselines/sen2-mtc-new/uncrtaints/model.pth.tar` |
| T-CLOUD | [`configs/tcloud.sh`](configs/tcloud.sh), [`.json`](configs/tcloud.json) | `baselines/t-cloud/uncrtaints/model.pth.tar` |
| CUHK-CR1 | [`configs/cuhk_cr1.sh`](configs/cuhk_cr1.sh), [`.json`](configs/cuhk_cr1.json) | `baselines/cuhk-cr1/uncrtaints/model.pth.tar` |
| CUHK-CR2 | [`configs/cuhk_cr2.sh`](configs/cuhk_cr2.sh), [`.json`](configs/cuhk_cr2.json) | `baselines/cuhk-cr2/uncrtaints/model.pth.tar` |

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/*/uncrtaints/*" --local-dir weights
```

The released checkpoints are seed 0 (`SEED=0`, the default of the scripts). `model.pth.tar` is the
upstream checkpoint format (`state_dict`, optimizer and scheduler states).
