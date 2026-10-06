# GeoCR Model Zoo

Results of the paper (arXiv:2609.32510, Tables 2, 3, 4 and 10) with the weights that correspond to each row. All weights are in the Hugging Face repository [JeonghyeokDo/GeoCR](https://huggingface.co/JeonghyeokDo/GeoCR): GeoCR under `transformer/` and `lora/`, the retrained comparison methods under `baselines/<dataset>/<method>/`, one checkpoint each. The code that trains and runs each comparison method is in [`baselines/`](baselines/).

FID, DISTS, KID, DINO similarity and LPIPS are computed on RGB views; PSNR and SSIM use the bands listed for each setting ([DATA.md](DATA.md#evaluation-settings)). Lower is better for FID, DISTS, KID and LPIPS; higher is better for DINO similarity, SSIM and PSNR.

## CUHK-CR2

Table 2 · `--dataset cuhk-cr2` · PSNR/SSIM bands: RGB · N<sub>test</sub> = 111

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 194.8 | 0.241 | 0.1201 | 0.538 | 0.319 | 0.452 | 19.46 | [`latest_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/pix2pix) |
| pix2pixHD | CVPR'18 | 257.1 | 0.262 | 0.2130 | 0.379 | 0.304 | 0.401 | 19.34 | [`final_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/pix2pixhd) |
| BBDM | CVPR'23 | 514.6 | 0.542 | 0.5827 | 0.131 | 0.737 | 0.326 | 17.68 | [`last_model.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/bbdm) |
| HI-Diff | NeurIPS'23 | 165.7 | 0.192 | 0.0769 | 0.664 | 0.253 | 0.638 | 23.54 | [`net_g_latest.pth`, `net_le_dm_latest.pth`, `net_d_latest.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/hidiff) |
| UnCRtainTS | CVPRW'23 | 165.6 | 0.265 | 0.0761 | 0.529 | 0.345 | 0.586 | 22.12 | [`model.pth.tar`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/uncrtaints) |
| DiffCR | TGRS'24 | 245.2 | 0.285 | 0.1874 | 0.615 | 0.353 | 0.582 | 22.85 | [`final_Network.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/diffcr) |
| IDF-CR | TGRS'24 | 167.0 | 0.214 | 0.0873 | 0.645 | 0.270 | 0.641 | 23.18 | [`last.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/idfcr) |
| ThiefCloud | TCSVT'25 | 136.3 | 0.226 | 0.0593 | 0.668 | 0.232 | 0.638 | 23.88 | [`final.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/thiefcloud) |
| EMRDM | CVPR'25 | 104.4 | 0.167 | 0.0259 | 0.727 | 0.208 | 0.654 | 23.61 | [`last.ckpt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/emrdm) |
| GACR | ECCV'26 | 125.0 | 0.178 | 0.0496 | 0.739 | 0.217 | 0.620 | 23.45 | [`0200000.pt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr2/gacr) |
| **GeoCR (w/o FT)** | – | 93.6 | 0.153 | 0.0200 | 0.784 | 0.194 | 0.607 | 23.21 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 94.7 | 0.152 | 0.0194 | 0.779 | 0.196 | 0.601 | 23.11 | [`lora/cuhk-cr2/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/cuhk-cr2) |

## Sen2_MTC_New, full-band (RGB + NIR)

Table 3a · `--dataset sen2-mtc-new` · PSNR/SSIM bands: RGB+B8 · N<sub>test</sub> = 687

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| UnCRtainTS | CVPRW'23 | 113.4 | 0.258 | 0.0597 | 0.471 | 0.428 | 0.588 | 17.15 | [`model.pth.tar`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new/uncrtaints) |
| DiffCR | TGRS'24 | 95.9 | 0.297 | 0.0388 | 0.499 | 0.312 | 0.599 | 19.26 | [`final_Network.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new/diffcr) |
| ThiefCloud | TCSVT'25 | 143.6 | 0.333 | 0.0767 | 0.310 | 0.508 | 0.481 | 14.92 | [`final.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new/thiefcloud) |
| EMRDM | CVPR'25 | 96.6 | 0.218 | 0.0459 | 0.533 | 0.314 | 0.640 | 18.20 | [`last.ckpt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new/emrdm) |
| GACR | ECCV'26 | 87.4 | 0.223 | 0.0358 | 0.535 | 0.320 | 0.626 | 18.90 | [`0200000.pt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new/gacr) |
| **GeoCR (w/o FT)** | – | 52.9 | 0.182 | 0.0032 | 0.642 | 0.221 | 0.661 | 20.77 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 52.7 | 0.181 | 0.0031 | 0.643 | 0.220 | 0.660 | 20.76 | [`lora/sen2-mtc-new/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/sen2-mtc-new) |

## Sen2_MTC_New, RGB-only

Table 3b · `--dataset sen2-mtc-new-rgb` · PSNR/SSIM bands: RGB · N<sub>test</sub> = 687

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 161.6 | 0.305 | 0.1188 | 0.224 | 0.480 | 0.541 | 20.16 | [`latest_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new-rgb/pix2pix) |
| pix2pixHD | CVPR'18 | 153.9 | 0.292 | 0.1121 | 0.274 | 0.365 | 0.646 | 24.00 | [`final_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new-rgb/pix2pixhd) |
| BBDM | CVPR'23 | 179.4 | 0.392 | 0.1243 | 0.243 | 0.504 | 0.652 | 24.48 | [`last_model.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new-rgb/bbdm) |
| HI-Diff | NeurIPS'23 | 125.8 | 0.302 | 0.0695 | 0.390 | 0.398 | 0.742 | 25.66 | [`net_g_latest.pth`, `net_le_dm_latest.pth`, `net_d_latest.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen2-mtc-new-rgb/hidiff) |
| **GeoCR (w/o FT)** | – | 79.1 | 0.252 | 0.0223 | 0.489 | 0.313 | 0.706 | 25.41 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 78.4 | 0.265 | 0.0313 | 0.520 | 0.333 | 0.712 | 25.07 | [`lora/sen2-mtc-new-rgb/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/sen2-mtc-new-rgb) |

## SEN12MS-CR, full-band (13 spectral bands)

Table 4a · `--dataset sen12ms-cr` · PSNR/SSIM bands: 13 bands · N<sub>test</sub> = 7,899

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| UnCRtainTS | CVPRW'23 | 80.5 | 0.276 | 0.0497 | 0.440 | 0.313 | 0.884 | 28.77 | [`model.pth.tar`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr/uncrtaints) |
| EMRDM | CVPR'25 | 75.7 | 0.274 | 0.0475 | 0.467 | 0.304 | 0.873 | 28.70 | [`last.ckpt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr/emrdm) |
| GACR | ECCV'26 | 96.9 | 0.331 | 0.0679 | 0.358 | 0.333 | 0.828 | 27.76 | [`0050000.pt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr/gacr) |
| **GeoCR (w/o FT)** | – | 28.4 | 0.202 | 0.0091 | 0.571 | 0.215 | 0.866 | 29.29 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 29.5 | 0.205 | 0.0098 | 0.569 | 0.216 | 0.866 | 29.19 | [`lora/sen12ms-cr/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/sen12ms-cr) |

## SEN12MS-CR, RGB-only

Table 4b · `--dataset sen12ms-cr-rgb` · PSNR/SSIM bands: RGB · N<sub>test</sub> = 7,899

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 145.9 | 0.317 | 0.1000 | 0.281 | 0.465 | 0.632 | 21.64 | [`latest_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr-rgb/pix2pix) |
| pix2pixHD | CVPR'18 | 70.8 | 0.239 | 0.0443 | 0.420 | 0.277 | 0.772 | 27.23 | [`final_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr-rgb/pix2pixhd) |
| BBDM | CVPR'23 | 96.1 | 0.284 | 0.0536 | 0.331 | 0.367 | 0.710 | 25.44 | [`last_model.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr-rgb/bbdm) |
| HI-Diff | NeurIPS'23 | 62.7 | 0.284 | 0.0342 | 0.438 | 0.320 | 0.837 | 28.28 | [`net_g_latest.pth`, `net_le_dm_latest.pth`, `net_d_latest.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/sen12ms-cr-rgb/hidiff) |
| **GeoCR (w/o FT)** | – | 43.9 | 0.244 | 0.0246 | 0.532 | 0.326 | 0.712 | 22.15 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 52.1 | 0.263 | 0.0375 | 0.529 | 0.308 | 0.781 | 24.59 | [`lora/sen12ms-cr-rgb/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/sen12ms-cr-rgb) |

## T-CLOUD

Table 10a · `--dataset t-cloud` · PSNR/SSIM bands: RGB · N<sub>test</sub> = 588

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 107.6 | 0.254 | 0.0509 | 0.450 | 0.330 | 0.635 | 20.13 | [`latest_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/pix2pix) |
| pix2pixHD | CVPR'18 | 60.8 | 0.169 | 0.0173 | 0.616 | 0.177 | 0.741 | 24.75 | [`final_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/pix2pixhd) |
| BBDM | CVPR'23 | 186.6 | 0.422 | 0.1179 | 0.259 | 0.546 | 0.541 | 22.69 | [`last_model.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/bbdm) |
| HI-Diff | NeurIPS'23 | 40.2 | 0.124 | 0.0070 | 0.726 | 0.114 | 0.874 | 30.54 | [`net_g_latest.pth`, `net_le_dm_latest.pth`, `net_d_latest.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/hidiff) |
| UnCRtainTS | CVPRW'23 | 63.3 | 0.180 | 0.0188 | 0.640 | 0.178 | 0.809 | 26.28 | [`model.pth.tar`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/uncrtaints) |
| DiffCR | TGRS'24 | 139.6 | 0.358 | 0.0732 | 0.376 | 0.482 | 0.402 | 20.67 | [`final_Network.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/diffcr) |
| IDF-CR | TGRS'24 | 84.4 | 0.220 | 0.0304 | 0.467 | 0.236 | 0.787 | 25.99 | [`last.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/idfcr) |
| ThiefCloud | TCSVT'25 | 40.4 | 0.123 | 0.0058 | 0.726 | 0.109 | 0.861 | 29.36 | [`final.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/thiefcloud) |
| EMRDM | CVPR'25 | 36.5 | 0.121 | 0.0045 | 0.759 | 0.110 | 0.867 | 28.25 | [`last.ckpt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/emrdm) |
| GACR | ECCV'26 | 39.2 | 0.120 | 0.0058 | 0.747 | 0.113 | 0.858 | 29.56 | [`0020000.pt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/t-cloud/gacr) |
| **GeoCR (w/o FT)** | – | 36.5 | 0.126 | 0.0024 | 0.763 | 0.122 | 0.785 | 27.15 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 36.3 | 0.125 | 0.0022 | 0.751 | 0.122 | 0.786 | 27.23 | [`lora/t-cloud/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/t-cloud) |

## CUHK-CR1, RGB+NIR

Table 10b · `--dataset cuhk-cr1` · PSNR/SSIM bands: RGB+NIR · N<sub>test</sub> = 98

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| UnCRtainTS | CVPRW'23 | 134.8 | 0.204 | 0.0283 | 0.712 | 0.311 | 0.679 | 23.82 | [`model.pth.tar`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr1/uncrtaints) |
| DiffCR | TGRS'24 | 237.6 | 0.291 | 0.1360 | 0.649 | 0.318 | 0.573 | 22.75 | [`final_Network.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr1/diffcr) |
| EMRDM | CVPR'25 | 77.8 | 0.115 | -0.0006 | 0.845 | 0.146 | 0.760 | 25.64 | [`last.ckpt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr1/emrdm) |
| GACR | ECCV'26 | 97.1 | 0.133 | 0.0099 | 0.825 | 0.160 | 0.726 | 25.02 | [`0200000.pt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/cuhk-cr1/gacr) |
| **GeoCR (w/o FT)** | – | 80.4 | 0.125 | -0.0025 | 0.825 | 0.165 | 0.680 | 23.88 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 81.5 | 0.125 | -0.0027 | 0.821 | 0.167 | 0.677 | 23.83 | [`lora/cuhk-cr1/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/cuhk-cr1) |

## WHUS2-CRv, native multispectral

Table 10c · `--dataset whus2-crv` · PSNR/SSIM bands: 13 bands · N<sub>test</sub> = 3,746

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| UnCRtainTS | CVPRW'23 | 26.9 | 0.142 | 0.0062 | 0.808 | 0.139 | 0.926 | 31.14 | [`model.pth.tar`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv/uncrtaints) |
| IDF-CR | TGRS'24 | 40.0 | 0.179 | 0.0115 | 0.691 | 0.188 | 0.858 | 29.00 | [`last.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv/idfcr) |
| EMRDM | CVPR'25 | 16.9 | 0.101 | 0.0009 | 0.875 | 0.097 | 0.937 | 32.55 | [`last.ckpt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv/emrdm) |
| GACR | ECCV'26 | 22.6 | 0.230 | 0.0033 | 0.789 | 0.138 | 0.882 | 31.09 | [`0050000.pt`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv/gacr) |
| **GeoCR (w/o FT)** | – | 18.9 | 0.111 | 0.0027 | 0.850 | 0.106 | 0.905 | 32.29 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 18.5 | 0.107 | 0.0025 | 0.858 | 0.103 | 0.903 | 32.15 | [`lora/whus2-crv/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/whus2-crv) |

## WHUS2-CRv, RGB-only

Table 10d · `--dataset whus2-crv-rgb` · PSNR/SSIM bands: RGB · N<sub>test</sub> = 3,746

| Method | Venue | FID↓ | DISTS↓ | KID↓ | DINO↑ | LPIPS↓ | SSIM↑ | PSNR↑ | Weights |
|---|---|---|---|---|---|---|---|---|---|
| pix2pix | CVPR'17 | 95.1 | 0.238 | 0.0523 | 0.432 | 0.274 | 0.746 | 23.75 | [`latest_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv-rgb/pix2pix) |
| pix2pixHD | CVPR'18 | 27.9 | 0.148 | 0.0057 | 0.751 | 0.145 | 0.828 | 26.77 | [`final_net_G.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv-rgb/pix2pixhd) |
| BBDM | CVPR'23 | 50.4 | 0.213 | 0.0171 | 0.413 | 0.291 | 0.639 | 25.19 | [`last_model.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv-rgb/bbdm) |
| HI-Diff | NeurIPS'23 | 17.8 | 0.107 | 0.0018 | 0.853 | 0.096 | 0.894 | 29.91 | [`net_g_latest.pth`, `net_le_dm_latest.pth`, `net_d_latest.pth`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/baselines/whus2-crv-rgb/hidiff) |
| **GeoCR (w/o FT)** | – | 21.6 | 0.137 | 0.0032 | 0.786 | 0.133 | 0.833 | 27.16 | [`transformer/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/transformer) |
| **GeoCR (LoRA)** | – | 20.7 | 0.138 | 0.0039 | 0.811 | 0.134 | 0.804 | 25.35 | [`lora/whus2-crv-rgb/`](https://huggingface.co/JeonghyeokDo/GeoCR/tree/main/lora/whus2-crv-rgb) |

## Licences

GeoCR weights: see [LICENSE-WEIGHTS.md](LICENSE-WEIGHTS.md). Each comparison-method checkpoint follows the terms of its upstream code, listed in [`baselines/README.md`](baselines/README.md) and, with the licence texts, in [`baselines/README.md` on the Hub](https://huggingface.co/JeonghyeokDo/GeoCR/blob/main/baselines/README.md).
