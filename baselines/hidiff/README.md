# HI-Diff

Z. Chen, Y. Zhang, D. Liu, B. Xia, J. Gu, L. Kong and X. Yuan, *Hierarchical Integration
Diffusion Model for Realistic Image Deblurring*, NeurIPS 2023.

- Upstream: <https://github.com/zhengchen1999/HI-Diff> at commit `b3bfd167997e27f8edd57681cf70e5031a0e35f2`
- Licence: Apache-2.0 (the upstream `LICENSE`); HI-Diff builds on BasicSR, Restormer and DiffIR
- Datasets: `t-cloud`, `cuhk-cr2`, `sen2-mtc-new-rgb`, `sen12ms-cr-rgb`, `whus2-crv-rgb`

## Setup

Run from the parent directory of this repository (`GeoCR/`), with `CR_DATASETS` and
`CR_EXPORTS` prepared as in [`DATA.md`](../../DATA.md) and [`../README.md`](../README.md#data).
HI-Diff needs `basicsr==1.4.2`; with torchvision 0.17 or later, change one import in it first
(`torchvision.transforms.functional_tensor` no longer exists):

```bash
git clone https://github.com/zhengchen1999/HI-Diff
git -C HI-Diff checkout b3bfd167997e27f8edd57681cf70e5031a0e35f2
git -C HI-Diff apply "$PWD/GeoCR/baselines/hidiff/hidiff.patch"
pip install basicsr==1.4.2
sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/' \
  "$(python -c 'import importlib.util as u; print(u.find_spec("basicsr").submodule_search_locations[0])')/data/degradations.py"
export REPO=$PWD/HI-Diff CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
```

| File | Purpose |
|---|---|
| `hi_diff/models/HI_Diff_S1_model.py`, `HI_Diff_S2_model.py` (modified) | `train.micro_batch`: forward/backward in equal chunks with one optimizer step per batch, so batch 32 fits one GPU; unset, the upstream step is unchanged |
| `hi_diff/utils/base_model.py`, `train.py` (modified) | `torch.load(..., weights_only=False)` |
| `geocr_data.py` | test-split reader (split lists and 8-bit exports, paired by id) |
| `dump.py` | writes test predictions for `python -m geocr.eval` |

## Train

```bash
bash GeoCR/baselines/hidiff/train.sh t-cloud      # -> HI-Diff/experiments/geocr_tcloud_s{1,2}_seed0/
```

`configs/<corpus>_s1.yml` and `_s2.yml` hold the two stages, which train in turn as in the
repository: stage 1 learns `net_g` and the latent encoder `net_le` (which also sees the clear
image), stage 2 loads them and learns `net_g`, `net_le_dm` and the latent denoiser `net_d`.
Both stages read the `trainval` export through `PairedImageIRDataset` (256 px, random flips and
rotations), with batch 32 in chunks of 8, AdamW (learning rate 2.3784142e-4, weight decay 1e-4),
`CosineAnnealingRestartCyclicLR` and gradient clipping; the network definitions are the
repository's (Transformer with dim 48, 8-step latent diffusion).

| Dataset | Updates per stage |
|---|---:|
| `t-cloud` | 10,000 |
| `cuhk-cr2` | 5,000 |
| `sen2-mtc-new-rgb` | 10,000 |
| `sen12ms-cr-rgb` | 25,000 |
| `whus2-crv-rgb` | 25,000 |

## Predict and score

```bash
CKPT=HI-Diff/experiments/geocr_tcloud_s2_seed0/models bash GeoCR/baselines/hidiff/infer.sh t-cloud
PYTHONPATH=GeoCR python -m geocr.eval --dataset t-cloud --pred out/hidiff/pred/t-cloud \
  --metrics fid,dists,kid,lpips,ssim,psnr     # DINO: see ../README.md#score
```

`CKPT` is a directory holding the stage-2 `net_g_latest.pth`, `net_le_dm_latest.pth` and
`net_d_latest.pth`, either a training run or the downloaded weights. The stage-2 model runs in
test mode: `net_le_dm` encodes the cloudy image, the prior is denoised over the 8 timesteps from
noise seeded per image from the image id, and `net_g` restores the image. The stage-1 encoder
`net_le` is not used for prediction.

## Weights

One checkpoint per dataset (seed 0) on the Hugging Face Hub: `baselines/<dataset>/hidiff/` in
[JeonghyeokDo/GeoCR](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines) holds the three
stage-2 networks in BasicSR format (`{"params": state_dict}`).

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/t-cloud/hidiff/*" --local-dir weights
CKPT=weights/baselines/t-cloud/hidiff bash GeoCR/baselines/hidiff/infer.sh t-cloud
```
