# IDF-CR

**IDF-CR: Iterative Diffusion Process for Divide-and-Conquer Cloud Removal in Remote-Sensing Images**
(Wang et al., IEEE TGRS 2024; "TGRS'24" in the GeoCR tables).

| | |
|---|---|
| Upstream | <https://github.com/SongYxing/IDF-CR> |
| Pinned commit | `144df841c18fcf8d19e65bd77fa2104b3337a7ee` |
| Upstream licence | no repository-level licence; `pixel/LICENSE` is MIT (Copyright (c) 2019 Penn000), `latent/LICENSE` is Apache-2.0 |
| Datasets | T-CLOUD, CUHK-CR2, WHUS2-CRv |

The kit trains and runs the network in `pixel/` (SPANet); `latent/` is not used.

## Setup

```bash
git clone https://github.com/SongYxing/IDF-CR && cd IDF-CR
git checkout 144df841c18fcf8d19e65bd77fa2104b3337a7ee
git apply /path/to/GeoCR/baselines/idfcr/idfcr.patch
```

Install PyTorch and this repository's `requirements.txt` (the data loaders import `geocr` from this
repository). Prepare the datasets and copy the split lists as described in [DATA.md](../../DATA.md);
the scripts read `$CR_DATASETS/<dataset>/splits/{trainval,test}.txt`.

## What the patch changes

| File | Change | Why |
|---|---|---|
| `pixel/train_geocr.py` (new) | trains the pixel network on `trainval.txt` for a fixed number of optimizer steps and saves `last.pth`; optimizer, losses and attention target as in `train_woGAN.py` / `train_woGAN_whus2.py` | train on the split lists |
| `pixel/dump_geocr.py` (new) | predicts the test list and writes float32 CHW `.npy` files plus `manifest.json` | input of the common evaluation |
| `pixel/geocr_data.py`, `pixel/geocr_eval.py` (new) | split-list datasets; images are read with the readers of `geocr.data.benchmarks`; the six WHUS2-CRv rasters of a sample are derived from one id | data loading |
| `pixel/attrdict.py` (new) | minimal `AttrMap` | the `attrdict` package does not import on Python 3.10+ |
| `pixel/WHUS2/allnet.py` | `sorted()` file listings | the six WHUS2 listings are indexed together |
| `pixel/WHUS2/gdaldiy.py` | GeoTIFF read/write with `tifffile` | GDAL is not needed |
| `pixel/eval.py`, `eval_2.py`, `eval_3.py` | `skimage.metrics.structural_similarity` | `compare_ssim` was removed from scikit-image |
| `pixel/predict_2.py` | save the prediction | it saved the attention map |
| `pixel/models/gen/SPANet.py`, `SPANet_whus2.py` | CPU fallback for the device | runs without a GPU |
| `pixel/train_woGAN.py`, `train_woGAN_whus2.py` | `DataParallel` over the visible GPUs | device ids were fixed to four GPUs |

## Train and predict

```bash
export REPO=/path/to/IDF-CR CR_DATASETS=/path/to/datasets
GPU=0 bash train.sh t-cloud                 # t-cloud | cuhk-cr2 | whus2-crv
GPU=0 bash infer.sh t-cloud                 # uses the checkpoint written by train.sh
CKPT=weights/baselines/t-cloud/idfcr/last.pth GPU=0 bash infer.sh t-cloud   # released weights
```

`GPU` is the physical device index. Outputs go to `$OUT` (default `./out/idfcr`): runs in
`runs/<corpus>_s<seed>/`, predictions in `dumps/<corpus>/test/idfcr-pixel/<run_id>/`. Score the
predictions as described in [../README.md](../README.md).

## Configurations and weights

| Dataset | Config | Weights in [`JeonghyeokDo/GeoCR`](https://huggingface.co/JeonghyeokDo/GeoCR) |
|---|---|---|
| T-CLOUD | [`configs/tcloud.yml`](configs/tcloud.yml) | `baselines/t-cloud/idfcr/last.pth` |
| CUHK-CR2 | [`configs/cuhk_cr2.yml`](configs/cuhk_cr2.yml) | `baselines/cuhk-cr2/idfcr/last.pth` |
| WHUS2-CRv | [`configs/whus2crv.yml`](configs/whus2crv.yml) | `baselines/whus2-crv/idfcr/last.pth` |

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/*/idfcr/*" --local-dir weights
```

The released checkpoints are seed 0 (`SEED=0`, the default of the scripts). `${...}` placeholders in
the configs are filled in by the scripts.
