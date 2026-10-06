# pix2pix

P. Isola, J.-Y. Zhu, T. Zhou and A. A. Efros, *Image-to-Image Translation with Conditional
Adversarial Networks*, CVPR 2017.

- Upstream: <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix> at commit
  `2a7afba2895d52556dd5dfe07e8555ef657ced6f`
- Licence: BSD (the upstream `LICENSE`)
- Datasets: `t-cloud`, `cuhk-cr2`, `sen2-mtc-new-rgb`, `sen12ms-cr-rgb`, `whus2-crv-rgb`

## Setup

Run from the parent directory of this repository (`GeoCR/`), with `CR_DATASETS` and
`CR_EXPORTS` prepared as in [`DATA.md`](../../DATA.md) and [`../README.md`](../README.md#data).

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
git -C pytorch-CycleGAN-and-pix2pix checkout 2a7afba2895d52556dd5dfe07e8555ef657ced6f
git -C pytorch-CycleGAN-and-pix2pix apply "$PWD/GeoCR/baselines/pix2pix/pix2pix.patch"
export REPO=$PWD/pytorch-CycleGAN-and-pix2pix CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
```

The patch adds four files and modifies none:

| File | Purpose |
|---|---|
| `geocr_data.py` | reads ids from the split lists and pixels from the 8-bit exports, paired by id |
| `data/geocr_dataset.py` | `--dataset_mode geocr`: training pairs with the repository's aligned transform |
| `train_geocr.py` | trains for a fixed number of updates and writes `latest_net_{G,D}.pth` |
| `dump.py` | writes test predictions for `python -m geocr.eval` |

## Train

```bash
bash GeoCR/baselines/pix2pix/train.sh t-cloud        # -> out/pix2pix/runs/tcloud_pix2pix_seed0/
```

`configs/<corpus>.sh` holds the settings. The model, losses and optimisers are the repository's
pix2pix defaults (`unet_256` generator with batch normalisation and dropout, PatchGAN
discriminator, vanilla GAN loss + 100 × L1, Adam with β1 = 0.5); the training transform is its
aligned recipe (resize to 286, random 256 crop, random flip). Training uses the `trainval` list,
batch 32, learning rate 4.75683e-4 for both networks, and the repository's `linear` policy:
constant for the first half of the epochs, then linear decay, with the number of epochs derived
from the number of updates.

| Dataset | Updates |
|---|---:|
| `t-cloud` | 20,000 |
| `cuhk-cr2` | 10,000 |
| `sen2-mtc-new-rgb` | 20,000 |
| `sen12ms-cr-rgb` | 50,000 |
| `whus2-crv-rgb` | 50,000 |

## Predict and score

```bash
CKPT=out/pix2pix/runs/tcloud_pix2pix_seed0 bash GeoCR/baselines/pix2pix/infer.sh t-cloud
PYTHONPATH=GeoCR python -m geocr.eval --dataset t-cloud --pred out/pix2pix/pred/t-cloud \
  --metrics fid,dists,kid,lpips,ssim,psnr     # DINO: see ../README.md#score
```

`CKPT` is a directory holding `latest_net_G.pth`, either a training run or the downloaded
weights. As in the repository's `test.py`, the generator stays in training mode at test time
(dropout active, batch normalisation over a single image); the dropout noise is seeded per image
from the image id, so the predictions do not depend on the order of the test set.

## Weights

One checkpoint per dataset (seed 0) on the Hugging Face Hub:
`baselines/<dataset>/pix2pix/latest_net_G.pth` in
[JeonghyeokDo/GeoCR](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines), the generator
state dict (54.4 M parameters).

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/t-cloud/pix2pix/*" --local-dir weights
CKPT=weights/baselines/t-cloud/pix2pix bash GeoCR/baselines/pix2pix/infer.sh t-cloud
```
