# ThiefCloud

**ThiefCloud: A Thickness Fused Thin Cloud Removal Network for Optical Remote Sensing Image With
Self-Supervised Learnable Cloud Prior** (Zhao et al., IEEE TCSVT 2025; "TCSVT'25" in the GeoCR tables).

| | |
|---|---|
| Upstream | <https://github.com/lixinghua5540/ThiefCloud> |
| Pinned commit | `17962e27dff59dacdf5459a0b355d53134b1ae5d` |
| Upstream licence | no licence file; the upstream README states "The code can be used for academic purposes only." |
| Datasets | T-CLOUD, CUHK-CR2, Sen2_MTC_New |

## Setup

```bash
git clone https://github.com/lixinghua5540/ThiefCloud && cd ThiefCloud
git checkout 17962e27dff59dacdf5459a0b355d53134b1ae5d
git apply /path/to/GeoCR/baselines/thiefcloud/thiefcloud.patch
```

Install PyTorch, `timm`, `matplotlib`, `pyyaml` and this repository's `requirements.txt` (the data loaders
import `geocr` from this repository). Training downloads the torchvision VGG-16 weights used by the
perceptual loss. Prepare the datasets and copy the split lists as described in [DATA.md](../../DATA.md).

## What the patch changes

| File | Change | Why |
|---|---|---|
| `ThiefCloud/geocr_data.py` (new) | split-list data for T-CLOUD, CUHK-CR2 and Sen2_MTC_New (cloudy frame 0, RGB bands, DN clipped to [0, 2000] / 2000) | train and test on the split lists |
| `ThiefCloud/geocr_train.py` (new) | the two training stages for a fixed number of optimizer steps (stage 1 with the cloud prior frozen, stage 2 with it trainable), cosine learning rate per stage; writes `stage1_last.pth` and `final.pth` | fixed-step training |
| `ThiefCloud/geocr_dump.py` (new) | predicts the test list and writes CHW `.npy` files plus `manifest.json` | input of the common evaluation |
| `ThiefCloud/model.py` | the cloud-prior (SCPM) checkpoint is `SCPM/checkpointsCDNet_V1/best_CDNet_V1_SateHaze1k.pth` from the repository (override: `THIEFCLOUD_SCPM_CKPT`); `torch.load(..., map_location="cpu", weights_only=False)` | the hard-coded file is not part of the repository; current PyTorch |
| `ThiefCloud/train.py`, `SCPM/src/train_cd.py`, `SCPM/src/test_cd.py` | no `CUDA_VISIBLE_DEVICES` set inside the scripts; `--val_data` and `--pre_state` defaults | GPU selection from the environment |
| `ThiefCloud/test.py`, `ThiefCloud/demo.py` | import the `ThiefCloud` model class | the imported class names do not exist |

## Train and predict

```bash
export REPO=/path/to/ThiefCloud CR_DATASETS=/path/to/datasets
GPU=0 bash train.sh t-cloud                 # t-cloud | cuhk-cr2 | sen2-mtc-new
GPU=0 bash infer.sh t-cloud                 # uses the checkpoint written by train.sh
CKPT=weights/baselines/t-cloud/thiefcloud/final.pth GPU=0 bash infer.sh t-cloud   # released weights
```

Outputs go to `$OUT` (default `./out/thiefcloud`): runs in `runs/<corpus>/seed<seed>/`, predictions in
`dumps/<corpus>/test/thiefcloud/<run_id>/`. Score the predictions as described in
[../README.md](../README.md).

## Configurations and weights

| Dataset | Config | Weights in [`JeonghyeokDo/GeoCR`](https://huggingface.co/JeonghyeokDo/GeoCR) |
|---|---|---|
| T-CLOUD | [`configs/tcloud.yaml`](configs/tcloud.yaml) | `baselines/t-cloud/thiefcloud/final.pth` |
| CUHK-CR2 | [`configs/cuhk_cr2.yaml`](configs/cuhk_cr2.yaml) | `baselines/cuhk-cr2/thiefcloud/final.pth` |
| Sen2_MTC_New | [`configs/sen2mtc_new.yaml`](configs/sen2mtc_new.yaml) | `baselines/sen2-mtc-new/thiefcloud/final.pth` |

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/*/thiefcloud/*" --local-dir weights
```

The released checkpoints are seed 0 (`SEED=0`, the default of the scripts). `final.pth` holds all
weights of the network, including the cloud prior. `${...}` placeholders in the configs are filled in
by the scripts.
