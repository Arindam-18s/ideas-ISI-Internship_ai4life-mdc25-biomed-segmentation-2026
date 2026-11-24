# Biomedical Image Segmentation: CNN & Vision Transformer Benchmarks (AI4Life MDC25)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Affiliation](https://img.shields.io/badge/IDEAS--TIH-ISI%20Kolkata-8A2BE2.svg)](https://www.isical.ac.in/)

Official codebase for biomedical image segmentation research conducted during the **Autumn 2026 Research Internship** at **IDEAS-TIH (Institute of Data Engineering, Analytics and Science Foundation), Indian Statistical Institute (ISI), Kolkata**. 

This repository implements, benchmarks, and analyzes pure convolutional, attention-gated, and hierarchical Vision Transformer architectures on the **[AI4Life MDC25 Grand Challenge](https://ai4life-mdc25.grand-challenge.org/)** dataset.

---

## 📌 Overview

Biomedical microscopy and volumetric imaging present distinct challenges including low signal-to-noise ratios, subtle boundary transitions, and high structural variability. In this work, we systematically evaluate the transition from local receptive fields (CNNs) to global self-attention mechanisms (Vision Transformers) across three canonical encoder-decoder paradigms:

1. **U-Net** *(Ronneberger et al., MICCAI 2015)* — Standard symmetric encoder-decoder baseline with direct skip connections for spatial feature preservation.
2. **Attention U-Net** *(Oktay et al., MIDL 2018)* — Incorporates soft Attention Gates (AGs) in skip connections to suppress irrelevant background regions and highlight salient foreground structures.
3. **Swin-UNet** *(Cao et al., ECCVW 2022)* — Pure Transformer-based U-shaped architecture leveraging shifted-window self-attention (Swin Transformer blocks) and patch-expanding layers for multi-scale semantic representation.

---

## 📂 Repository Structure

```text
isi-ideas-ai4life-mdc25-segmentation/
├── configs/                  # YAML/JSON hyperparameter & experiment configurations
│   ├── unet.yaml
│   ├── attention_unet.yaml
│   └── swin_unet.yaml
├── data/                     # Dataset loaders, preprocessing & augmentation pipelines
│   ├── dataset.py
│   └── transforms.py
├── models/                   # PyTorch model definitions
│   ├── unet.py
│   ├── attention_unet.py
│   └── swin_unet.py
├── utils/                    # Loss functions (Dice, BCE, Focal), metrics & visualization
│   ├── losses.py
│   └── metrics.py
├── train.py                  # Unified training & validation loop
├── evaluate.py               # Inference, test-time augmentation & leaderboard submission generation
├── requirements.txt          # Python dependencies
└── README.md
```


## 🙏 Acknowledgments

This work was carried out during the **Autumn 2026 Research Internship** at **IDEAS-TIH, Indian Statistical Institute (ISI), Kolkata**.

* Grateful to **Mr. Aditya Shankar Pal** *(PhD Scholar, ISI Kolkata)* for guidance, mentorship, and compute support.
* We thank the organizers of the **AI4Life MDC25 Grand Challenge** for providing the benchmark dataset and evaluation platform.

---

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
2. Oktay, O., et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL.
3. Cao, H., et al. (2022). *Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation*. ECCV Workshops.
