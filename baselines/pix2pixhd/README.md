# pix2pixHD

T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz and B. Catanzaro, *High-Resolution Image
Synthesis and Semantic Manipulation with Conditional GANs*, CVPR 2018.

- Upstream: <https://github.com/NVIDIA/pix2pixHD> at commit `14b3b3c7fff413086e3b58df52096f16b6891172`
- Licence: BSD (the upstream `LICENSE.txt`)
- Datasets: `t-cloud`, `cuhk-cr2`, `sen2-mtc-new-rgb`, `sen12ms-cr-rgb`, `whus2-crv-rgb`

## Setup

Run from the parent directory of this repository (`GeoCR/`), with `CR_DATASETS` and
`CR_EXPORTS` prepared as in [`DATA.md`](../../DATA.md) and [`../README.md`](../README.md#data).

```bash
git clone https://github.com/NVIDIA/pix2pixHD
git -C pix2pixHD checkout 14b3b3c7fff413086e3b58df52096f16b6891172
git -C pix2pixHD apply "$PWD/GeoCR/baselines/pix2pixhd/pix2pixhd.patch"
export REPO=$PWD/pix2pixHD CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
```

| File | Purpose |
|---|---|
| `data/custom_dataset_data_loader.py` (modified) | `--dataroot geocr:<corpus>` selects the dataset below; other values keep the upstream behaviour |
| `geocr_data.py` | reads ids from the split lists and pixels from the 8-bit exports, paired by id |
| `data/geocr_dataset.py` | cloudy (`label`) / clear (`image`) pairs with the repository's transform |
| `train_geocr.py` | trains for a fixed number of updates and writes `final_net_{G,D}.pth` |
| `dump.py` | writes test predictions for `python -m geocr.eval` |

## Train

```bash
bash GeoCR/baselines/pix2pixhd/train.sh t-cloud      # -> out/pix2pixhd/runs/tcloud_pix2pixhd_seed0/
```

`configs/<corpus>.sh` holds the arguments. The model, losses and optimisers are the repository's
defaults for `--netG global` (9 residual blocks, 4 downsamplings, instance normalisation;
two-scale discriminator; LSGAN, feature-matching and VGG19 perceptual losses; Adam with
β1 = 0.5), run in the documented image-to-image mode (`--label_nc 0 --no_instance`) at 256 px
without resizing, with random horizontal flips. Training uses the `trainval` list, batch 32 and
learning rate 4.75683e-4 for both networks, constant for the first half of the epochs and then
decayed linearly per epoch, with the number of epochs derived from the number of updates.
Training needs a GPU, and torchvision downloads the ImageNet VGG19 weights on the first run.

| Dataset | Updates |
|---|---:|
| `t-cloud` | 20,000 |
| `cuhk-cr2` | 10,000 |
| `sen2-mtc-new-rgb` | 20,000 |
| `sen12ms-cr-rgb` | 50,000 |
| `whus2-crv-rgb` | 50,000 |

## Predict and score

```bash
CKPT=out/pix2pixhd/runs/tcloud_pix2pixhd_seed0 bash GeoCR/baselines/pix2pixhd/infer.sh t-cloud
PYTHONPATH=GeoCR python -m geocr.eval --dataset t-cloud --pred out/pix2pixhd/pred/t-cloud \
  --metrics fid,dists,kid,lpips,ssim,psnr     # DINO: see ../README.md#score
```

`CKPT` is a directory holding `final_net_G.pth`, either a training run or the downloaded
weights. The generator runs in evaluation mode, 16 images per batch.

## Weights

One checkpoint per dataset (seed 0) on the Hugging Face Hub:
`baselines/<dataset>/pix2pixhd/final_net_G.pth` in
[JeonghyeokDo/GeoCR](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines), the generator
state dict (182.4 M parameters).

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/t-cloud/pix2pixhd/*" --local-dir weights
CKPT=weights/baselines/t-cloud/pix2pixhd bash GeoCR/baselines/pix2pixhd/infer.sh t-cloud
```
