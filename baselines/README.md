# Comparison methods

Code to retrain the ten comparison methods of the paper on the GeoCR benchmarks, to predict their
test splits, and to score the predictions with `python -m geocr.eval`. As stated in Section 4.2 of
the paper, the baselines are retrained separately on each dataset's training split using official
implementations when available and assessed under a common evaluation pipeline; the reported
baseline set follows the input compatibility of each setting.

Each method directory holds a patch against a pinned upstream commit, one configuration per
dataset, `train.sh`, `infer.sh` and a README with the exact settings. One trained checkpoint per
method and dataset (seed 0) is on the Hugging Face Hub under
[`baselines/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines).

## Methods

| Directory | Method | Venue | Upstream repository and commit | Upstream licence |
|---|---|---|---|---|
| [`pix2pix`](pix2pix) | pix2pix | CVPR'17 | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) `2a7afba2895d52556dd5dfe07e8555ef657ced6f` | BSD |
| [`pix2pixhd`](pix2pixhd) | pix2pixHD | CVPR'18 | [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD) `14b3b3c7fff413086e3b58df52096f16b6891172` | BSD |
| [`bbdm`](bbdm) | BBDM | CVPR'23 | [xuekt98/BBDM](https://github.com/xuekt98/BBDM) `02c3b13c9f9dfab0853e32123100680a0640c4ed` | MIT |
| [`hidiff`](hidiff) | HI-Diff | NeurIPS'23 | [zhengchen1999/HI-Diff](https://github.com/zhengchen1999/HI-Diff) `b3bfd167997e27f8edd57681cf70e5031a0e35f2` | Apache-2.0 |
| [`uncrtaints`](uncrtaints) | UnCRtainTS | CVPRW'23 | [PatrickTUM/UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS) `5e1f1b58e993645e765b64e10b6e9c7ff828b36f` | none |
| [`diffcr`](diffcr) | DiffCR | TGRS'24 | [XavierJiezou/DiffCR](https://github.com/XavierJiezou/DiffCR) `6e8c2ae25fc2b81d07a29d1a2f0a0067c6d45d68` | none |
| [`idfcr`](idfcr) | IDF-CR | TGRS'24 | [SongYxing/IDF-CR](https://github.com/SongYxing/IDF-CR) `144df841c18fcf8d19e65bd77fa2104b3337a7ee` | none at the root; `pixel/` MIT, `latent/` Apache-2.0 |
| [`thiefcloud`](thiefcloud) | ThiefCloud | TCSVT'25 | [lixinghua5540/ThiefCloud](https://github.com/lixinghua5540/ThiefCloud) `17962e27dff59dacdf5459a0b355d53134b1ae5d` | none; the README allows academic use only |
| [`emrdm`](emrdm) | EMRDM | CVPR'25 | [Ly403/EMRDM](https://github.com/Ly403/EMRDM) `257d2a15e3572551ce40ab9d62cdded0346455e6` | AGPL-3.0 |
| [`gacr`](gacr) | GACR | ECCV'26 | [wzy6055/GACR](https://github.com/wzy6055/GACR) `9690a3f50f326532bb7e71715900b2f8d1229c6b` | MIT |

The datasets of each method are the cells of Tables 2, 3, 4 and 10 of the paper:

| | t-cloud | cuhk-cr1 | cuhk-cr2 | sen2-mtc-new | sen12ms-cr | whus2-crv | sen2-mtc-new-rgb | sen12ms-cr-rgb | whus2-crv-rgb |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| pix2pix, pix2pixHD, BBDM, HI-Diff | ✓ | | ✓ | | | | ✓ | ✓ | ✓ |
| UnCRtainTS, EMRDM, GACR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| DiffCR | ✓ | ✓ | ✓ | ✓ | | | | | |
| IDF-CR | ✓ | | ✓ | | | ✓ | | | |
| ThiefCloud | ✓ | | ✓ | ✓ | | | | | |

The upstream licence texts, and a note on the repositories that publish none, are in
[`baselines/licenses`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/licenses) on the
Hub. Cite the original paper of each method you use.

## Environment

We ran every method with Python 3.12, PyTorch 2.11 and torchvision 0.26 on NVIDIA B200 GPUs.
Beyond the GeoCR requirements:

- **BBDM**: PyTorch Lightning, OmegaConf, TensorBoard and einops (its own requirements).
- **HI-Diff**: `basicsr==1.4.2` with a one-line fix for recent torchvision (see
  [`hidiff/README.md`](hidiff/README.md)), OpenCV and einops.
- **EMRDM and GACR**: NATTEN, built from source for the GPU in use; see their READMEs.

## Data

Install the datasets and the frozen split lists under `$CR_DATASETS` as described in
[`DATA.md`](../DATA.md). The methods read 8-bit RGB PNG copies of the benchmarks from
`$CR_EXPORTS`, written once by the exporters in `geocr.data.exporters`, which use the same
readers and resampling as GeoCR:

```bash
export CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
PYTHONPATH=GeoCR python -m geocr.data.exporters.export_display8 --out "$CR_EXPORTS" --aliases  # T-CLOUD, CUHK-CR1/CR2
PYTHONPATH=GeoCR python -m geocr.data.exporters.export_rgbvariant --out "$CR_EXPORTS"          # RGB-only datasets
```

Each export split holds `cloudy/` and `clear/` PNGs named after the split-list ids, `list.txt`
and `manifest.json`; `--aliases` adds the `haze`, `gt` and `label` folder names some methods
expect. The multispectral datasets are read from `$CR_DATASETS` directly (WHUS2-CRv bands through
`geocr.data.exporters.export_whus2.fuse13`).

## Train, predict, score

All commands run from the directory that contains `GeoCR/`. For example, pix2pix on T-CLOUD:

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
git -C pytorch-CycleGAN-and-pix2pix checkout 2a7afba2895d52556dd5dfe07e8555ef657ced6f
git -C pytorch-CycleGAN-and-pix2pix apply "$PWD/GeoCR/baselines/pix2pix/pix2pix.patch"
export REPO=$PWD/pytorch-CycleGAN-and-pix2pix

bash GeoCR/baselines/pix2pix/train.sh t-cloud
bash GeoCR/baselines/pix2pix/infer.sh t-cloud          # CKPT=<dir> to use downloaded weights
```

The scripts put `GeoCR/` and `baselines/common/` on `PYTHONPATH`: the adapters import the dataset
registry as `rescore` and the exporters under their file names, both thin aliases of
`geocr.eval.score` and `geocr.data.exporters` in [`common/`](common). The method READMEs give the
arguments, outputs and settings of each script.

### Score

`infer.sh` writes `preds/<id>.npy` (one float32 CHW array per test image, unclipped) and a
`manifest.json` to its output directory, which `geocr.eval` scores against the test split:

```bash
PYTHONPATH=GeoCR python -m geocr.eval --dataset t-cloud --pred out/pix2pix/pred/t-cloud \
  --metrics fid,dists,kid,lpips,ssim,psnr
```

DINO similarity needs the DINOv3 ViT-L/16 SAT-493M weights, which Meta distributes under the
DINOv3 License: add `dino` to `--metrics` and pass `--dino-weights <dinov3_vitl16_pretrain_sat493m.pth>`.
For the 8-bit and RGB-only datasets the reference images are read from `$CR_EXPORTS`.

## Layout

```text
baselines/
├── README.md
├── common/          rescore.py and exporters/*.py: aliases used by the adapters
└── <method>/
    ├── README.md
    ├── <method>.patch
    ├── configs/     one file per dataset (two per dataset for HI-Diff)
    ├── train.sh
    └── infer.sh
```
