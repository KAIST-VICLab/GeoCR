# Data

GeoCR is pretrained on the training splits of ten cloud removal datasets (883,331 cloud-free targets)
and evaluated on six of them, plus RGB-only versions of SEN12MS-CR, Sen2_MTC_New and WHUS2-CRv
(paper Section 4.1 and Appendix F). This repository does not redistribute any imagery: download each
dataset from its distributor, arrange it as below, and install the frozen split lists from
[`splits/`](splits/). Use of each dataset is subject to its distributor's terms.

| Dataset | Imagery | Used for | Download | Terms stated by the distributor |
|---|---|---|---|---|
| AllClear | Sentinel-2 + Sentinel-1 | pretraining | [project page](https://allclear.cs.cornell.edu/) | CC BY-NC 4.0 |
| SEN12MS-CR | Sentinel-2 + Sentinel-1 | pretraining, evaluation | [mediaTUM](https://mediatum.ub.tum.de/1554803) | CC BY 4.0 |
| WHUS2-CRv | Sentinel-2 | pretraining, evaluation | Zenodo [train](https://doi.org/10.5281/zenodo.8035347), [val/test](https://doi.org/10.5281/zenodo.8035349) | CC BY 4.0 |
| Sen2_MTC_Old | Sentinel-2 | pretraining | [Harvard Dataverse](https://doi.org/10.7910/DVN/BSETKZ) | CC0 1.0 |
| Sen2_MTC_New | Sentinel-2 | pretraining, evaluation | [Google Drive](https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link) | none stated |
| T-CLOUD | Landsat-8 | pretraining, evaluation | [Baidu Netdisk](https://pan.baidu.com/s/1LtkcdxMbJQTgEr-JvTM1Ug) (code `t63d`) | none stated |
| RICE1, RICE2 | Google Earth, Landsat-8 | pretraining | [Google Drive](https://drive.google.com/file/d/1CricZtIj28BGFvkD_x-W8fSexPiDtgHk/view?usp=share_link), [Baidu Netdisk](https://pan.baidu.com/s/1h6SFWSnzH7GQJoM2UxO_ng) | none stated |
| CUHK-CR1, CUHK-CR2 | Jilin-1 KF01B | pretraining, evaluation | [Hugging Face](https://huggingface.co/littlebeen/DE), [Baidu Netdisk](https://pan.baidu.com/s/12yNH5eowjGA1fFM5sXUzBw) (code `bean`) | none stated |

## Directory layout

All dataset paths are relative to `$CR_DATASETS`; rendered PNG copies go to `$CR_EXPORTS`.

```text
$CR_DATASETS/
├── AllClear/              index.parquet, rois/roi<ID>/*.npy, splits/      (transcoded, see below)
├── SEN12MS-CR/            ROIs{1158_spring,1868_summer,1970_fall,2017_winter}_{s1,s2,s2_cloudy}/, splits/
├── WHUS2-CRv/extracted/   {train,val,test}/{clearDNclips,cloudDNclips}/{10m,20m,60m}/<scene>/<N>.tif, splits/
├── Sen2_MTC_Old/          {multipleImage,singleImage}/{clear,cloudy}/*.jpg, splits/
├── Sen2_MTC_New/          Sen2_MTC/<tile>/{cloud,cloudless}/*.tif, splits/
├── T-CLOUD/               {train,test}/{cloud,reference}/<n>.png, splits/
├── RICE/RICE1/            {cloud,label}/<n>.png, splits/
├── RICE/RICE2/            {cloud,label,mask}/<n>.png, splits/
└── C-CUHK/                CUHK-CR1/, CUHK-CR2/ ({train,test}/{cloud,label}/<n>.png, splits/),
                           nir/{CUHK-CR1,CUHK-CR2}/
$CR_EXPORTS/
├── sen12mscr_rgb/, whus2crv_rgb/, sen2mtc_new_rgb/   RGB-only datasets: <split>/{cloudy,clear}/*.png
└── tcloud/, cuhk_cr1/, cuhk_cr2/                     8-bit copies read by the comparison methods
```

Set both roots before running the commands below:

```bash
export CR_DATASETS=/path/to/datasets CR_EXPORTS=/path/to/exports
```

## Datasets

### AllClear

Hangyu Zhou, Chia-Hsiang Kao, Cheng Perng Phoo, Utkarsh Mall, Bharath Hariharan, and Kavita Bala.
AllClear: A comprehensive dataset and benchmark for cloud removal in satellite imagery. *Advances in
Neural Information Processing Systems*, 37:53571–53597, 2024.

- **Download:** <https://allclear.cs.cornell.edu/> (official downloader: `download.py` in
  <https://github.com/Zhou-Hangyu/allclear>): 23,708 per-ROI archives (about 3.3 TB) and `metadata.tar.gz`.
- **Terms:** CC BY-NC 4.0.
- **Preparation:** GeoCR reads a transcoded copy (per-ROI NumPy arrays plus an index, over 4 TB), made
  once from the archives. `make_splits` writes the official ROI split, identical to
  `splits/AllClear/splits/`.

```bash
BASE=https://allclear.cs.cornell.edu/dataset/allclear
mkdir -p allclear/data
wget -c -P allclear "$BASE/metadata.tar.gz" && tar -xzf allclear/metadata.tar.gz -C allclear
awk 1 allclear/metadata/rois/{train_rois_19k,val_rois_1k,test_rois_3k}.txt \
  | xargs -P 8 -I{} wget -c -q -P allclear/data "$BASE/data/{}.tar.gz"
python -m geocr.data.transcode --archive allclear/data --out "$CR_DATASETS/AllClear" --workers 16
python -m geocr.data.make_splits --rois_dir allclear/metadata/rois --out "$CR_DATASETS/AllClear"
```

### SEN12MS-CR

Patrick Ebel, Andrea Meraner, Michael Schmitt, and Xiao Xiang Zhu. Multisensor data fusion for cloud
removal in global and all-season Sentinel-2 imagery. *IEEE Transactions on Geoscience and Remote
Sensing*, 59(7):5866–5878, 2020.

- **Download:** <https://mediatum.ub.tum.de/1554803> (DOI 10.14459/2020mp1554803): twelve archives
  `ROIs{1158_spring,1868_summer,1970_fall,2017_winter}_{s1,s2,s2_cloudy}.tar.gz`.
- **Terms:** CC BY 4.0; the distributor notes that the data contain Copernicus Sentinel data, whose
  terms also apply.
- **Preparation:** extract the twelve archives into `$CR_DATASETS/SEN12MS-CR/`.

```bash
mkdir -p "$CR_DATASETS/SEN12MS-CR"
for f in ROIs*.tar.gz; do tar -xzf "$f" -C "$CR_DATASETS/SEN12MS-CR"; done
```

### WHUS2-CRv

Jun Li, Yuejie Zhang, Qinghong Sheng, Zhaocong Wu, Bo Wang, Zhongwen Hu, Guanting Shen, Michael
Schmitt, and Matthieu Molinier. Thin cloud removal fusing full spectral and spatial features for
Sentinel-2 imagery. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*,
15:8759–8775, 2022.

- **Download:** Zenodo [10.5281/zenodo.8035347](https://doi.org/10.5281/zenodo.8035347) (`train.zip`,
  `train.z01`–`train.z08`) and [10.5281/zenodo.8035349](https://doi.org/10.5281/zenodo.8035349)
  (`val.zip`, `val.z01`–`val.z02`, `test.zip`, `test.z01`–`test.z02`). The scene split is Table A1 of
  `WHUS2-CRv.pdf` in <https://github.com/Neooolee/WHUS2-CRv>.
- **Terms:** CC BY 4.0.
- **Preparation:** put the 15 parts in `$CR_DATASETS/WHUS2-CRv/` and extract them (e.g. with 7-Zip).
  The released `train/`, `val/` and `test/` folders do not follow the Table A1 scene split; the shipped
  lists regroup the clips by scene (`splits/WHUS2-CRv/extracted/splits/scene_split.csv`).

```bash
cd "$CR_DATASETS/WHUS2-CRv"
for s in train val test; do 7z x "$s.zip" -oextracted; done
```

### Sen2_MTC_Old

Vishnu Sarukkai, Anirudh Jain, Burak Uzkent, and Stefano Ermon. Cloud removal in satellite images
using spatiotemporal generative networks. In *2020 IEEE Winter Conference on Applications of Computer
Vision (WACV)*, pp. 1785–1794. IEEE, 2020.

- **Download:** Harvard Dataverse [doi:10.7910/DVN/BSETKZ](https://doi.org/10.7910/DVN/BSETKZ):
  `multipleImage.tar.gz` and `singleImage.tar.gz`.
- **Terms:** CC0 1.0.
- **Preparation:** the dataset ships no split; `splits/Sen2_MTC_Old/splits/` holds the one we use.

```bash
mkdir -p "$CR_DATASETS/Sen2_MTC_Old"
tar -xzf multipleImage.tar.gz -C "$CR_DATASETS/Sen2_MTC_Old"
tar -xzf singleImage.tar.gz -C "$CR_DATASETS/Sen2_MTC_Old"
```

### Sen2_MTC_New

Gi-Luen Huang and Pei-Yuan Wu. CTGAN: Cloud transformer generative adversarial network. In *2022 IEEE
International Conference on Image Processing (ICIP)*, pp. 511–515. IEEE, 2022.

- **Download:** `CTGAN.zip` from the Google Drive link in the README of
  <https://github.com/come880412/CTGAN>.
- **Terms:** none stated for the data (the CTGAN code repository is AGPL-3.0).
- **Preparation:** `CTGAN.zip` contains the dataset as `Sen2_MTC.zip`. The split is the official tile
  split.

```bash
unzip CTGAN.zip CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC.zip
unzip CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC.zip -d "$CR_DATASETS/Sen2_MTC_New"
```

### T-CLOUD

Haidong Ding, Yue Zi, and Fengying Xie. Uncertainty-based thin cloud removal network via conditional
variational autoencoders. In *Asian Conference on Computer Vision*, pp. 52–68. Springer, 2022.

- **Download:** Baidu Netdisk <https://pan.baidu.com/s/1LtkcdxMbJQTgEr-JvTM1Ug> (extraction code
  `t63d`; a Baidu account is required), linked from <https://github.com/haidong-Ding/Cloud-Removal>.
- **Terms:** none stated.
- **Preparation:** arrange the download as `$CR_DATASETS/T-CLOUD/{train,test}/{cloud,reference}/<n>.png`.
  The test split is the official test set.

### RICE1 and RICE2

Daoyu Lin, Guangluan Xu, Xiaoke Wang, Yang Wang, Xian Sun, and Kun Fu. A remote sensing image dataset
for cloud removal. *arXiv preprint arXiv:1901.00600*, 2019.

- **Download:** `RICE_DATASET.zip` from Google Drive or Baidu Netdisk, linked from
  <https://github.com/BUPTLdy/RICE_DATASET>.
- **Terms:** none stated.
- **Preparation:** RICE1 and RICE2 are used for pretraining only.

```bash
unzip RICE_DATASET.zip -d "$CR_DATASETS/RICE"
```

### CUHK-CR1 and CUHK-CR2

Jialu Sui, Yiyang Ma, Wenhan Yang, Xiaokang Zhang, Man-On Pun, and Jiaying Liu. Diffusion enhancement
for cloud removal in ultra-resolution remote sensing imagery. *IEEE Transactions on Geoscience and
Remote Sensing*, 62:1–14, 2024.

- **Download:** `C-CUHK.rar` from <https://huggingface.co/littlebeen/DE>, or Baidu Netdisk
  <https://pan.baidu.com/s/12yNH5eowjGA1fFM5sXUzBw> (code `bean`); both are linked from
  <https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal>.
- **Terms:** none stated.
- **Preparation:** the archive is RAR5 and its top folder is `C-CUHK - 副本`; the RGB and NIR images sit
  in separate trees, which the loaders combine.

```bash
wget https://huggingface.co/littlebeen/DE/resolve/main/C-CUHK.rar
LANG=C.UTF-8 bsdtar -xf C-CUHK.rar
mv "C-CUHK - 副本" "$CR_DATASETS/C-CUHK"
```

## Split lists

When the datasets are in place, copy the frozen lists next to them (from the repository root):

```bash
cp -r splits/*/ "$CR_DATASETS"/
```

## Rendered copies

### RGB-only datasets

The RGB-only versions of SEN12MS-CR, Sen2_MTC_New and WHUS2-CRv are rendered once from the native
data with u = round[255 · clip(2ρ, 0, 1)] and use the split lists of their source dataset. GeoCR and
the comparison methods read the same rendered images.

```bash
python -m geocr.data.exporters.export_rgbvariant --out "$CR_EXPORTS"
```

### 8-bit copies for the comparison methods

The comparison methods read T-CLOUD and CUHK-CR1/CR2 as flat folders of 8-bit PNG pairs (see
[`baselines/README.md`](baselines/README.md)); `--aliases` adds the folder names some of them expect.

```bash
python -m geocr.data.exporters.export_display8 --out "$CR_EXPORTS" --aliases
```

## Preprocessing

The data loaders apply the preprocessing of Appendix F; no manual step is needed.

- Sentinel-2 digital numbers are converted to reflectance ρ = DN/10⁴ and normalized as
  2 clip(ρ, 0, 1) − 1. Eight-bit imagery is normalized as 2u/255 − 1. SAR in dB is clipped to
  [−30, 0] and mapped to [−1, 1] as 2[clip(s, −30, 0) + 30]/30 − 1.
- WHUS2-CRv bands are aligned to the 10 m grid by nearest-neighbor replication of the 20 m and 60 m
  bands.
- AllClear cloudy observations are sampled without replacement within ±40 days of the target,
  excluding its acquisition date; SAR is the nearest available acquisition within this window.
  Eligible targets contain at most 10% cloud, 10% shadow, and 1% no-data coverage.

## Split checks

As stated in Appendix F:

> We use exact region-of-interest (ROI) identifiers for SEN12MS-CR, correct scene assignments in
> WHUS2-CRv, and exclude 104 Sen2_MTC_Old samples that overlap Sen2_MTC_New test tiles. CUHK-CR1
> evaluation excludes 32 train/validation duplicates and four within-test duplicates. RICE1/2
> contribute only pretraining data because their supplied splits contain duplicate images.

| Check | Shipped record |
|---|---|
| SEN12MS-CR ROI identifiers | [`splits/SEN12MS-CR/splits/roi_split.csv`](splits/SEN12MS-CR/splits/roi_split.csv) |
| WHUS2-CRv scene assignments | [`splits/WHUS2-CRv/extracted/splits/scene_split.csv`](splits/WHUS2-CRv/extracted/splits/scene_split.csv) |
| Sen2_MTC_Old samples on Sen2_MTC_New test tiles | [`splits/Sen2_MTC_Old/splits/excluded_sen2mtc_new_test_tiles.txt`](splits/Sen2_MTC_Old/splits/excluded_sen2mtc_new_test_tiles.txt) |
| CUHK-CR1 duplicates excluded from evaluation | [`splits/C-CUHK/CUHK-CR1/splits/test_exclude.txt`](splits/C-CUHK/CUHK-CR1/splits/test_exclude.txt) |
| RICE1/2 lists used in pretraining | [`splits/RICE/`](splits/RICE/) |

All lists and their checksums are described in [`splits/README.md`](splits/README.md).

## Pretraining mixture

Table 11 of the paper. Bands describe optical targets; K is the number of cloudy observations. Mix
denotes the source sampling probability.

| Source | Optical representation | SAR | K | Mix (%) | Training targets |
|---|---|---|---|---:|---:|
| AllClear | 13 bands, L1C TOA | VV/VH | 1–3 | 53.14 | 662,022 |
| SEN12MS-CR | 13 bands, L1C TOA | VV/VH | 1 | 32.93 | 107,143 |
| WHUS2-CRv | 13 bands, BOA; B10 TOA | – | 1 | 8.81 | 18,816 |
| Sen2_MTC_Old | RGB; NIR in cloudy inputs only | – | 3 | 2.00 | 88,874 |
| Sen2_MTC_New | RGB+B8, BOA | – | 3 | 1.10 | 2,380 |
| T-CLOUD | RGB, 8-bit | – | 1 | 1.10 | 2,234 |
| RICE2 | RGB, 8-bit | – | 1 | 0.27 | 553 |
| CUHK-CR1 | RGB+NIR, 8-bit | – | 1 | 0.25 | 508 |
| CUHK-CR2 | RGB+NIR, 8-bit | – | 1 | 0.21 | 426 |
| RICE1 | RGB, 8-bit | – | 1 | 0.19 | 375 |
| **Total** | | | | **100.00** | **883,331** |

## Evaluation settings

Table 12 of the paper. N<sub>test</sub> is the size of each evaluated test split; CUHK-CR1 counts
reflect the duplicate exclusions above.

| Setting | PSNR/SSIM bands | N<sub>test</sub> |
|---|---|---:|
| SEN12MS-CR | 13 bands | 7,899 |
| Sen2_MTC_New | RGB+B8 | 687 |
| WHUS2-CRv | 13 bands | 3,746 |
| T-CLOUD | RGB | 588 |
| CUHK-CR1 | RGB+NIR | 98 |
| CUHK-CR2 | RGB | 111 |
| SEN12MS-CR, RGB-only | RGB | 7,899 |
| Sen2_MTC_New, RGB-only | RGB | 687 |
| WHUS2-CRv, RGB-only | RGB | 3,746 |
