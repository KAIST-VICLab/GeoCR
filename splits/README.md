# Frozen splits

The exact sample lists used for pretraining, LoRA adaptation and evaluation, plus the split-check
results described in Appendix F of the paper. No imagery is included. In the `.txt` lists, each line
names one sample by an image path relative to the dataset directory (an ROI number for AllClear).

The folders mirror the dataset directories under `$CR_DATASETS` (see [DATA.md](../DATA.md)), so
one command, run from the repository root, puts every list where the data loaders read it:

```bash
cp -r splits/*/ "$CR_DATASETS"/
```

| File (under `splits/`) | Contents | Entries |
|---|---|---|
| `AllClear/splits/{train,val,test}.txt` | Official AllClear ROI split (`train_rois_19k.txt`, `val_rois_1k.txt`, `test_rois_3k.txt`) with the `roi` prefix removed | 19,013 / 997 / 3,698 ROIs |
| `SEN12MS-CR/splits/{train,val,test}.txt` | ROI-disjoint SEN12MS-CR split built from exact ROI identifiers | 107,143 / 7,176 / 7,899 |
| `SEN12MS-CR/splits/roi_split.csv` | Split and patch count of each ROI (155 train, 10 val, 10 test) | 175 ROIs |
| `WHUS2-CRv/extracted/splits/{train,val,test}.txt` | WHUS2-CRv clips grouped by the scene split of Table A1 in the dataset's `WHUS2-CRv.pdf` | 18,816 / 1,888 / 3,746 |
| `WHUS2-CRv/extracted/splits/scene_split.csv` | Per scene: its split (`official_split`), the Zenodo directory that holds it (`zenodo_dir`), and its clip count | 123 scenes |
| `Sen2_MTC_Old/splits/{train,val,test}.txt` | Tile-disjoint split (the dataset ships none) | 88,874 / 4,445 / 4,890 |
| `Sen2_MTC_Old/splits/excluded_sen2mtc_new_test_tiles.txt` | Sen2_MTC_Old samples on tiles of the Sen2_MTC_New test split; excluded from all three Sen2_MTC_Old lists | 104 |
| `Sen2_MTC_New/splits/{train,val,test}.txt` | Official tile split | 2,380 / 350 / 687 |
| `T-CLOUD/splits/{train,val,test}.txt` | Official test set; `train` and `val` partition the official training set | 2,234 / 117 / 588 |
| `RICE/RICE1/splits/{train,val}.txt` | RICE1 lists read in pretraining (RICE is used for pretraining only) | 375 / 25 |
| `RICE/RICE2/splits/{train,val}.txt` | RICE2 lists read in pretraining | 553 / 36 |
| `C-CUHK/CUHK-CR1/splits/{train,val,test}.txt` | Official test set; `train` and `val` partition the official training set | 508 / 26 / 134 |
| `C-CUHK/CUHK-CR1/splits/test_exclude.txt` | CUHK-CR1 test images excluded from evaluation: 32 train/validation duplicates and 4 within-test duplicates, each with the image it duplicates | 36 |
| `C-CUHK/CUHK-CR2/splits/{train,val,test}.txt` | Official test set; `train` and `val` partition the official training set | 426 / 22 / 111 |
| `trainval.txt` in the SEN12MS-CR, WHUS2-CRv, Sen2_MTC_New, T-CLOUD, CUHK-CR1 and CUHK-CR2 folders | `train.txt` followed by `val.txt`: the training split for LoRA adaptation | 114,319 / 20,704 / 2,730 / 2,351 / 534 / 448 |
| `SHA256SUMS` | Checksums of every file above (`cd splits && sha256sum -c SHA256SUMS`) | |

Pretraining reads each dataset's `train.txt`. The RGB-only versions of SEN12MS-CR, Sen2_MTC_New and
WHUS2-CRv use the lists of their source dataset. CUHK-CR1 is evaluated on `test.txt` minus
`test_exclude.txt` (98 images); every other test set is evaluated in full.
