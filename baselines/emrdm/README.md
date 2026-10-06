# EMRDM

**Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with
Elucidated Design Space** (Liu et al., CVPR 2025; "CVPR'25" in the GeoCR tables).

| | |
|---|---|
| Upstream | <https://github.com/Ly403/EMRDM> |
| Pinned commit | `257d2a15e3572551ce40ab9d62cdded0346455e6` |
| Upstream licence | AGPL-3.0; `emrdm.patch` modifies this code and is distributed under the same licence |
| Datasets | T-CLOUD, CUHK-CR1, CUHK-CR2, Sen2_MTC_New, SEN12MS-CR, WHUS2-CRv |

## Setup

```bash
git clone https://github.com/Ly403/EMRDM && cd EMRDM
git checkout 257d2a15e3572551ce40ab9d62cdded0346455e6
git apply /path/to/GeoCR/baselines/emrdm/emrdm.patch
```

Install the upstream requirements, [`natten`](https://github.com/SHI-Labs/NATTEN) built for your GPU
(we used natten 0.21.7), and this repository's `requirements.txt` (the data loaders import `geocr`
from this repository). Prepare the datasets and copy the split lists as described in
[DATA.md](../../DATA.md).

## What the patch changes

| File | Change | Why |
|---|---|---|
| `geocr_bench/datasets.py`, `datasets_mtc.py`, `datasets_sen12.py` (new) | split-list datasets: T-CLOUD, CUHK-CR1 (RGB+NIR), CUHK-CR2, WHUS2-CRv (13 bands on the 10 m grid), Sen2_MTC_New (three cloudy frames, upstream loader), SEN12MS-CR (with SAR, upstream loader) | train and test on the split lists |
| `geocr_bench/dump.py` (new) | `write_pred` saves each prediction as a CHW `.npy`; the command-line tool checks the predictions and writes `manifest.json` | input of the common evaluation |
| `sgm/models/diffusion.py` | `dump_dir`, `dump_dtype`, `dump_channels` options; with `dump_dir` set, `predict_step` saves the sample mapped from [-1, 1] to [0, 1] (before `scale_01`); `image_path_key` for the single-image engine | prediction output |
| `sgm/data/sentinel/sentinel.py` | exact `<season>/<ROI>` match when assigning SEN12MS-CR patches to a split; path cache named `*.exactroi.pkl`, location `EMRDM_PATHS_CACHE_DIR` | split assignment by exact ROI identifiers |
| `sgm/modules/diffusionmodules/k_diffusion/image_transformer.py` | works when `natten.has_fused_na` is not available | newer natten releases |

## Train and predict

```bash
export REPO=/path/to/EMRDM CR_DATASETS=/path/to/datasets
GPU=0 bash train.sh t-cloud                 # t-cloud | cuhk-cr1 | cuhk-cr2 | sen2-mtc-new | sen12ms-cr | whus2-crv
GPU=0 bash infer.sh t-cloud                 # uses the newest checkpoint written by train.sh
CKPT=weights/baselines/t-cloud/emrdm/last.ckpt GPU=0 bash infer.sh t-cloud   # released weights
```

`infer.sh` runs `main.py --predict true` with the checkpoint and then `python -m geocr_bench.dump`.
Sampling uses the EMA weights stored in `last.ckpt`. Outputs go to `$OUT` (default `./out/emrdm`):
runs in `runs/<time>_emrdm_<corpus>_seed<seed>/`, predictions in `dumps/<corpus>/test/emrdm/<run_id>/`.
Score the predictions as described in [../README.md](../README.md).

## Configurations and weights

| Dataset | Config | Weights in [`JeonghyeokDo/GeoCR`](https://huggingface.co/JeonghyeokDo/GeoCR) |
|---|---|---|
| T-CLOUD | [`configs/tcloud.yaml`](configs/tcloud.yaml) | `baselines/t-cloud/emrdm/last.ckpt` |
| CUHK-CR1 | [`configs/cuhk_cr1.yaml`](configs/cuhk_cr1.yaml) | `baselines/cuhk-cr1/emrdm/last.ckpt` |
| CUHK-CR2 | [`configs/cuhk_cr2.yaml`](configs/cuhk_cr2.yaml) | `baselines/cuhk-cr2/emrdm/last.ckpt` |
| Sen2_MTC_New | [`configs/sen2mtc_new.yaml`](configs/sen2mtc_new.yaml) | `baselines/sen2-mtc-new/emrdm/last.ckpt` |
| SEN12MS-CR | [`configs/sen12mscr.yaml`](configs/sen12mscr.yaml) | `baselines/sen12ms-cr/emrdm/last.ckpt` |
| WHUS2-CRv | [`configs/whus2crv.yaml`](configs/whus2crv.yaml) | `baselines/whus2-crv/emrdm/last.ckpt` |

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/*/emrdm/*" --local-dir weights
```

The released checkpoints are seed 0 (`SEED=0`, the default of the scripts). `last.ckpt` is a
PyTorch Lightning checkpoint (model, EMA and optimizer states). The configs read `CR_DATASETS` and
`EMRDM_DUMP_ROOT` from the environment (`${oc.env:...}`); the scripts set both.
