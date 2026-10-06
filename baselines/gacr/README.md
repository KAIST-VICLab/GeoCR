# GACR

**Interpretation-Oriented Cloud Removal via Observation-Anchored Residual Flow with Geo-Contextual
Alignment** (Wang et al., ECCV 2026; "ECCV'26" in the GeoCR tables).

| | |
|---|---|
| Upstream | <https://github.com/wzy6055/GACR> |
| Pinned commit | `9690a3f50f326532bb7e71715900b2f8d1229c6b` |
| Upstream licence | MIT (`LICENSE` in the upstream repository) |
| Datasets | CUHK-CR2, T-CLOUD, Sen2_MTC_New, SEN12MS-CR, CUHK-CR1, WHUS2-CRv |

## Setup

```bash
git clone https://github.com/wzy6055/GACR && cd GACR
git checkout 9690a3f50f326532bb7e71715900b2f8d1229c6b
git apply /path/to/GeoCR/baselines/gacr/gacr.patch
```

Install the upstream requirements, [`natten`](https://github.com/SHI-Labs/NATTEN) built for your GPU
(we used natten 0.21.7), and this repository's `requirements.txt` (the data loader imports `geocr`
from this repository). Prepare the datasets and copy the split lists as described in
[DATA.md](../../DATA.md).

Training also needs the DINOv3 ViT-L/16 (LVD-1689M) backbone used for geo-contextual alignment: clone
[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), request the weights from Meta
(DINOv3 License), and set `DINOV3_ROOT` (the clone) and `DINOV3_WEIGHTS`
(`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`). Inference does not use DINOv3.

## What the patch changes

| File | Change | Why |
|---|---|---|
| `geocr_data.py` (new) | split-list dataset (`dataset.kind: geocr`): T-CLOUD, CUHK-CR1 (RGB+NIR), CUHK-CR2, Sen2_MTC_New (three cloudy frames), SEN12MS-CR (13 optical bands), WHUS2-CRv (13 bands on the 10 m grid); RGB view for the DINOv3 teacher; per-sample noise seeds | train and test on the split lists |
| `dump.py` (new) | Euler sampling (`misc.eval_num_steps`, 4) with per-sample noise; `--state-key {model,ema}` selects the weights; writes CHW `.npy` files plus `manifest.json` | input of the common evaluation |
| `vfm_hub/hubconf.py` (new) | `torch.hub` entry points for the DINOv3 backbones only; set `gcpa.repo_path` to this directory | loads the teacher without DINOv3's segmentation and depth dependencies |
| `train.py` | `dataset.kind` dispatch; teacher on the input resolution (`gcpa.native_resolution`, patch 16) instead of a fixed resize; global batch = `dataset.batch_size` with gradient accumulation; `--set a.b=c` overrides; optional dev-split evaluation (`logging.dev_select`, off in the configs); writes `train_report.json` | training on the split lists |
| `test.py` | whole test list for `dataset.kind: geocr`; `--set` | `val_size` was applied to the test list |
| `engine.py` | token-count check between teacher and model; optional sampling noise; `loss.proj_loss_impl` | |
| `dataset.py` | `val` split | |
| `models/sgm/evaluator.py` | LPIPS created on first use; 4-band branch for any resolution | |
| `models/k_diffusion/layers.py` | optional `dctorch` import | not used by GACR |
| `models/k_diffusion/image_transformer.py` | works when `natten.has_fused_na` is not available | newer natten releases |

## Train and predict

```bash
export REPO=/path/to/GACR CR_DATASETS=/path/to/datasets
DINOV3_ROOT=/path/to/dinov3 DINOV3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
GPU=0 bash train.sh t-cloud                 # cuhk-cr2 | t-cloud | sen2-mtc-new | sen12ms-cr | cuhk-cr1 | whus2-crv
GPU=0 bash infer.sh t-cloud                 # uses the final checkpoint written by train.sh
CKPT=weights/baselines/t-cloud/gacr/0020000.pt GPU=0 bash infer.sh t-cloud   # released weights
```

A checkpoint holds two weight sets, `model` and `ema`. `infer.sh` uses `ema` for CUHK-CR1, CUHK-CR2
and Sen2_MTC_New, and `model` for T-CLOUD, SEN12MS-CR and WHUS2-CRv (set `STATE_KEY` to override).
Outputs go to `$OUT` (default `./out/gacr`): runs in `runs/<corpus>_seed<seed>/`, predictions in
`dumps/<corpus>/test/gacr/<run_id>/`. Score the predictions as described in
[../README.md](../README.md).

## Configurations and weights

| Dataset | Config | Weights in [`JeonghyeokDo/GeoCR`](https://huggingface.co/JeonghyeokDo/GeoCR) | Weights used |
|---|---|---|---|
| CUHK-CR2 | [`configs/cuhk_cr2.yaml`](configs/cuhk_cr2.yaml) | `baselines/cuhk-cr2/gacr/0200000.pt` | `ema` |
| T-CLOUD | [`configs/tcloud.yaml`](configs/tcloud.yaml) | `baselines/t-cloud/gacr/0020000.pt` | `model` |
| Sen2_MTC_New | [`configs/sen2mtc_new.yaml`](configs/sen2mtc_new.yaml) | `baselines/sen2-mtc-new/gacr/0200000.pt` | `ema` |
| SEN12MS-CR | [`configs/sen12mscr.yaml`](configs/sen12mscr.yaml) | `baselines/sen12ms-cr/gacr/0050000.pt` | `model` |
| CUHK-CR1 | [`configs/cuhk_cr1.yaml`](configs/cuhk_cr1.yaml) | `baselines/cuhk-cr1/gacr/0200000.pt` | `ema` |
| WHUS2-CRv | [`configs/whus2crv.yaml`](configs/whus2crv.yaml) | `baselines/whus2-crv/gacr/0050000.pt` | `model` |

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/*/gacr/*" --local-dir weights
```

The released checkpoints are seed 0 (`SEED=0`, the default of the scripts); the file name is the
training step. Each file holds `model`, `ema`, `opt`, `config` and `steps`. `${...}` placeholders in
the configs are filled in by the scripts.
