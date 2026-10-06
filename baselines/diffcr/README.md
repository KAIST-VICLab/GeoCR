# DiffCR

**DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal From Optical Satellite Images**
(Zou et al., IEEE TGRS 2024; "TGRS'24" in the GeoCR tables).

| | |
|---|---|
| Upstream | <https://github.com/XavierJiezou/DiffCR> |
| Pinned commit | `6e8c2ae25fc2b81d07a29d1a2f0a0067c6d45d68` |
| Upstream licence | the upstream repository does not include a licence |
| Datasets | T-CLOUD, Sen2_MTC_New, CUHK-CR1, CUHK-CR2 |

## Setup

```bash
git clone https://github.com/XavierJiezou/DiffCR && cd DiffCR
git checkout 6e8c2ae25fc2b81d07a29d1a2f0a0067c6d45d68
git apply /path/to/GeoCR/baselines/diffcr/diffcr.patch
```

Install the upstream requirements and this repository's `requirements.txt` (the data loader imports
`geocr` from this repository). Prepare the datasets and copy the split lists as described in
[DATA.md](../../DATA.md).

## What the patch changes

| File | Change | Why |
|---|---|---|
| `data/geocr_dataset.py` (new) | split-list dataset for T-CLOUD, Sen2_MTC_New (three cloudy frames), CUHK-CR1 (RGB+NIR) and CUHK-CR2; for single-image datasets the cloudy image fills the three condition inputs | train and test on the split lists |
| `models/geocr_model.py` (new) | `PaletteGeoCR`: trains for `budget.steps` optimizer steps and writes `checkpoint/final_Network.pth` (the `denoise_fn.*` weights) and `train_summary.json` | fixed-step training |
| `dump_geocr.py` (new) | DPM-Solver++ sampling (20 steps) with a per-sample noise seed; writes CHW `.npy` files plus `manifest.json` | input of the common evaluation |
| `core/praser.py`, `run.py` | `-o key.path=value` config overrides; `DIFFCR_NO_CODE_BACKUP=1` skips the code copy | seeds and run names from the command line |
| `models/ours/nafnet_double_encoder_splitcaCond_splitcaUnet.py` | the output convolution has `img_channel` channels | 4-band CUHK-CR1 |
| `models/network_x0_dpm_solver.py` | import the module file that exists | broken import |
| `core/dpm_solver_pytorch.py` | broadcast `alpha_t`/`sigma_t` over the image dimensions | shape error in `x_start` mode |
| `core/logger.py` | `LogTracker` indexing | current pandas |

## Train and predict

```bash
export REPO=/path/to/DiffCR CR_DATASETS=/path/to/datasets
GPU=0 bash train.sh t-cloud                 # t-cloud | sen2-mtc-new | cuhk-cr1 | cuhk-cr2
GPU=0 bash infer.sh t-cloud                 # uses the newest checkpoint written by train.sh
CKPT=weights/baselines/t-cloud/diffcr/final_Network.pth GPU=0 bash infer.sh t-cloud   # released weights
```

Outputs go to `$OUT` (default `./out/diffcr`): runs in `runs/<corpus>/train_<corpus>_seed<seed>_<time>/`,
predictions in `dumps/<corpus>/test/diffcr/<run_id>/`. Score the predictions as described in
[../README.md](../README.md).

## Configurations and weights

| Dataset | Config | Weights in [`JeonghyeokDo/GeoCR`](https://huggingface.co/JeonghyeokDo/GeoCR) |
|---|---|---|
| T-CLOUD | [`configs/tcloud.json`](configs/tcloud.json) | `baselines/t-cloud/diffcr/final_Network.pth` |
| Sen2_MTC_New | [`configs/sen2mtc_new.json`](configs/sen2mtc_new.json) | `baselines/sen2-mtc-new/diffcr/final_Network.pth` |
| CUHK-CR1 | [`configs/cuhk_cr1.json`](configs/cuhk_cr1.json) | `baselines/cuhk-cr1/diffcr/final_Network.pth` |
| CUHK-CR2 | [`configs/cuhk_cr2.json`](configs/cuhk_cr2.json) | `baselines/cuhk-cr2/diffcr/final_Network.pth` |

```bash
hf download JeonghyeokDo/GeoCR --include "baselines/*/diffcr/*" --local-dir weights
```

The released checkpoints are seed 0 (`SEED=0`, the default of the scripts). `${...}` placeholders in
the configs are filled in by the scripts.
