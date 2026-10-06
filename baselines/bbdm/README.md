# BBDM

B. Li, K. Xue, B. Liu and Y.-K. Lai, *BBDM: Image-to-image Translation with Brownian Bridge
Diffusion Models*, CVPR 2023. We train the latent model `LBBDM-f4`.

- Upstream: <https://github.com/xuekt98/BBDM> at commit `02c3b13c9f9dfab0853e32123100680a0640c4ed`
- Licence: MIT (the upstream `LICENSE`)
- Datasets: `t-cloud`, `cuhk-cr2`, `sen2-mtc-new-rgb`, `sen12ms-cr-rgb`, `whus2-crv-rgb`

## Setup

Run from the parent directory of this repository (`GeoCR/`), with `CR_DATASETS` and
`CR_EXPORTS` prepared as in [`DATA.md`](../../DATA.md) and [`../README.md`](../README.md#data).
Training also needs the frozen first stage, the latent-diffusion `vq-f4` VQGAN linked from the
BBDM README (the released weights were trained with a `model.ckpt` of 756,175,527 bytes, sha256
`aacf13951f4b18f5af9b47febdc696cf9559305d6de0821084abeaf342439251`).

```bash
git clone https://github.com/xuekt98/BBDM
git -C BBDM checkout 02c3b13c9f9dfab0853e32123100680a0640c4ed
git -C BBDM apply "$PWD/GeoCR/baselines/bbdm/bbdm.patch"
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip && unzip vq-f4.zip -d vq-f4
export REPO=$PWD/BBDM VQF4_CKPT=$PWD/vq-f4/model.ckpt CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
```

| File | Purpose |
|---|---|
| `runners/BaseRunner.py` (modified) | `training.fixed_budget: true` stops training exactly at `n_steps` micro-batches and saves `last_model.pth` there, without validation or sampling during training; `torch.load(..., weights_only=False)` for checkpoint loading |
| `runners/utils.py` (modified) | registers the `geocr_split` dataset |
| `runners/DiffusionBasedModelRunners/BBDMRunner.py`, `model/VQGAN/vqgan.py` (modified) | `weights_only=False`; `ReduceLROnPlateau` without the removed `verbose` argument |
| `geocr_data.py` | reads ids from the split lists and pixels from the 8-bit exports, paired by id |
| `datasets/geocr_split.py` | dataset type `geocr_split` |
| `datasets/__init__.py` | keeps `datasets` the local package when Hugging Face `datasets` is installed |
| `dump.py` | writes test predictions for `python -m geocr.eval` |

## Train

```bash
bash GeoCR/baselines/bbdm/train.sh t-cloud      # -> out/bbdm/runs/tcloud/LBBDM-f4/checkpoint/last_model.pth
```

`configs/<corpus>.yaml` holds the settings, which follow the repository's `LBBDM-f4` template
(UNet with 128 channels, channel multipliers 1/4/8, attention at 32/16/8; linear bridge, `grad`
objective, L1 loss, 1,000 timesteps; Adam with learning rate 1e-4 and `ReduceLROnPlateau` on the
training loss; EMA with decay 0.995 from micro-batch 30,000, updated every 8 optimizer steps; no
flips). Training uses the `trainval` list with micro-batches of 8 and 4 accumulated micro-batches
per update (batch 32), so `n_steps` counts micro-batches.

| Dataset | Updates | `n_steps` |
|---|---:|---:|
| `t-cloud` | 20,000 | 80,000 |
| `cuhk-cr2` | 10,000 | 40,000 |
| `sen2-mtc-new-rgb` | 20,000 | 80,000 |
| `sen12ms-cr-rgb` | 50,000 | 200,000 |
| `whus2-crv-rgb` | 50,000 | 200,000 |

## Predict and score

```bash
CKPT=out/bbdm/runs/tcloud/LBBDM-f4/checkpoint bash GeoCR/baselines/bbdm/infer.sh t-cloud
PYTHONPATH=GeoCR python -m geocr.eval --dataset t-cloud --pred out/bbdm/pred/t-cloud \
  --metrics fid,dists,kid,lpips,ssim,psnr     # DINO: see ../README.md#score
```

`CKPT` is a directory holding `last_model.pth`, either a training run or the downloaded weights.
Sampling uses the EMA weights (the repository's test-time default) and the repository's sampler
(200 of the 1,000 steps, `eta` 1.0, no clipping), one image at a time, with the noise seeded per
image from the image id. The frozen VQGAN is restored from `last_model.pth`, so prediction does
not need `VQF4_CKPT`.

## Weights

One checkpoint per dataset (seed 0) on the Hugging Face Hub:
`baselines/<dataset>/bbdm/last_model.pth` in
[JeonghyeokDo/GeoCR](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines). The file is
the repository's checkpoint format: `model` (the full `LatentBrownianBridgeModel` state dict,
including the frozen `vq-f4` VQGAN), `ema` (EMA weights of the UNet), `step` and `epoch`.

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/t-cloud/bbdm/*" --local-dir weights
CKPT=weights/baselines/t-cloud/bbdm bash GeoCR/baselines/bbdm/infer.sh t-cloud
```
