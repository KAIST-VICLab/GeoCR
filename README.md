<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
    <img src="assets/logo.svg" alt="GeoCR" width="440">
  </picture>
</p>

<div align="center">
<h2>GeoCR: Learning a Generalist Cloud Removal Prior from Heterogeneous Observations</h2>
<div>
    <a href='https://jeonghyeokdo.github.io/' target='_blank'>Jeonghyeok Do</a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://www.viclab.kaist.ac.kr/' target='_blank'>Munchurl Kim</a><sup>†</sup>
</div>
<div>
    Korea Advanced Institute of Science and Technology (KAIST), South Korea
</div>
<div>
    <sup>†</sup>Corresponding author
</div>
<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/GeoCR_site/" target='_blank'>
        <img alt="Project Page" src="https://img.shields.io/badge/🏠-Project%20Page-blue">
        </a>
        <!-- ARXIV_BADGE_START -->
        <img alt="arXiv (coming soon)" src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg">
        <!-- ARXIV_BADGE_END -->
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/GeoCR">
    </h4>
</div>
</div>

---

<div align="center">
    <h4>
        This repository is the official implementation of "GeoCR: Learning a Generalist Cloud Removal Prior from Heterogeneous Observations".
    </h4>
</div>

## 📧 News
- **Sep 2026:** This repository is created. The code will be released in this repository.

## 📖 Abstract
Cloud removal methods are typically specialized to individual datasets and input configurations, limiting reuse across sensors, spectral bands, and observation settings. We introduce **GeoCR**, a generalist model that unifies RGB-only-based CR and multispectral-based CR from single- or multi-temporal cloudy observations, with optional SAR guidance, within a single network. To accommodate different spectral and sensing domains, compact input and output stems extend a pretrained RGB autoencoder while keeping its encoder and decoder trunks frozen. This shared latent interface enables a single flow transformer to jointly model clean RGB and non-RGB latents, conditioned on separate cloudy-observation streams and optional SAR tokens. Through joint pretraining on the training splits of ten datasets comprising 883,331 cloud-free target images, GeoCR learns a *shared cloud removal prior* across these heterogeneous configurations. The same pretrained checkpoint supports direct inference without dataset-specific fine-tuning and efficient adaptation through low-rank adaptation (LoRA). We evaluate GeoCR against general image restoration and cloud removal methods on test splits of the contributing datasets under full-band and RGB-only settings. GeoCR achieves the best FID and DISTS on full-band SEN12MS-CR and Sen2_MTC_New and RGB-only CUHK-CR2, outperforming existing models and demonstrating the effectiveness of a reusable generative model across diverse settings.

## 📊 Results

### Qualitative Comparison
<div align="center">
    <img src="assets/qualitative_fig1.jpg" alt="Qualitative comparison on five cloud removal benchmarks" width="100%">
</div>
<p align="center"><em><b>Qualitative comparison on cloud-removal benchmarks.</b> (a) Input cloudy, (b) UnCRtainTS, (c) EMRDM, (d) GACR, (e) GeoCR without fine-tuning, (f) GeoCR with LoRA adaptation, and (g) ground-truth cloud-free image.</em></p>

### Cross-Dataset Comparison
GeoCR achieves the best FID and DISTS on full-band Sen2_MTC_New and SEN12MS-CR and RGB-only CUHK-CR2.

<div align="center">
    <img src="assets/radar.png" alt="FID and DISTS of cloud removal methods on five benchmarks" width="56%">
</div>
<p align="center"><em>FID and DISTS on the five primary benchmarks, normalized for each dataset–metric pair as 100 × best/value (larger is better).</em></p>

### Quantitative Comparison
**Bold** and <ins>underlined</ins> values indicate the best and second-best results.

<p align="center"><em><b>Table 2: Quantitative comparison on CUHK-CR2.</b> Its RGB evaluation setting supports both RGB-only image translation/restoration methods and specialized cloud removal methods, enabling comparison across all ten baselines.</em></p>
<div align="center">
    <img src="assets/table2_cuhk_cr2.png" alt="Table 2: Quantitative comparison on CUHK-CR2" width="100%">
</div>
<br>
<p align="center"><em><b>Table 3: Quantitative comparison on Sen2_MTC_New.</b> (a) Full-band setting. (b) RGB setting.</em></p>
<div align="center">
    <img src="assets/table3_sen2_mtc_new.png" alt="Table 3: Quantitative comparison on Sen2_MTC_New, (a) full-band and (b) RGB-only" width="62%">
</div>
<br>
<p align="center"><em><b>Table 4: Quantitative comparison on SEN12MS-CR.</b> (a) Full-band setting. (b) RGB setting.</em></p>
<div align="center">
    <img src="assets/table4_sen12ms_cr.png" alt="Table 4: Quantitative comparison on SEN12MS-CR, (a) full-band and (b) RGB-only" width="62%">
</div>
<br>

**Please visit our [project page](https://kaist-viclab.github.io/GeoCR_site/) for the interactive comparison and more results.**

## 🖼️ Method Overview

<div align="center">
    <img src="assets/framework.jpg" alt="Overview of the GeoCR framework" width="100%">
</div>
<p align="center"><em><b>GeoCR framework.</b> A shared latent interface enables joint cloud removal across heterogeneous spectral, temporal, and SAR configurations.</em></p>

- **Shared latent interface:** compact stems map non-RGB and SAR inputs into a frozen pretrained RGB autoencoder.
- **One generalist prior:** a single flow transformer is jointly pretrained on 10 datasets (883,331 cloud-free targets).
- **Two modes:** the same checkpoint is used directly (**GeoCR (w/o FT)**) or adapted with LoRA (**GeoCR (LoRA)**).

## 🚀 Code Release Plan
**The following components are planned for release in this repository:**

- [ ] Inference code
- [ ] Training scripts
- [ ] Evaluation scripts

## 📑 Citation
If you find GeoCR useful, please consider citing:
```BibTeX
@article{do2026geocr,
  title={GeoCR: Learning a Generalist Cloud Removal Prior from Heterogeneous Observations},
  author={Do, Jeonghyeok and Kim, Munchurl},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
