# Noise-Aware Deep Learning for Microscopy Image Denoising
### AI4Life MDC25 · 2D Nuclei Denoising · Autumn 2026 Research Internship

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Task-Microscopy%20Denoising-2E8B57.svg)](https://ai4life-mdc25.grand-challenge.org/)
[![Affiliation](https://img.shields.io/badge/IDEAS--ISI%20Kolkata-Research%20Internship-8A2BE2.svg)](https://www.isical.ac.in/)

Research code, experiments, and analysis for a supervised fluorescence microscopy image-denoising project conducted during the **Autumn 2026 Research Internship** at **IDEAS – Institute of Data Engineering, Analytics and Science Foundation, Indian Statistical Institute (ISI), Kolkata**.

The project investigates how convolutional and transformer-based denoising models reconstruct clean microscopy images from noisy inputs. The work is based on the **AI4Life Microscopy Supervised Denoising Challenge (MDC25), 2D Nuclei task**.

This is an **image denoising research project**, not a biomedical image segmentation benchmark.

---

## Overview

Microscopy images often contain noise that reduces contrast and hides important cellular structures, including nuclear boundaries and internal texture.

In supervised denoising, a model receives a noisy microscopy image and learns to reconstruct its corresponding ground-truth (GT) image.

The primary objective is to reduce noise while preserving meaningful biological structures.

This project compares traditional image-processing methods with multiple deep-learning architectures:

1. **Gaussian and Median Filters** – Classical non-learning denoising baselines.
2. **U-Net** – Convolutional encoder-decoder architecture with skip connections.
3. **Edge-Aware U-Net** – Incorporates an additional edge-preservation loss.
4. **Noise-Aware U-Net** – Uses an estimated noise level to condition the network.
5. **Noise-Aware U-Net + Edge Loss** – Combines noise conditioning and boundary preservation.
6. **DnCNN** – Residual convolutional denoising network evaluated in the supplementary analysis.
7. **Swin-Based Denoiser** – Transformer-based image restoration model.
8. **Noise-Aware Swin + Edge Loss** – Additional experiment defined in the main notebook; the final saved run was not completed.

### Research Questions

- How much improvement does a trained U-Net provide over noisy images and classical filters?
- Does edge-aware training help preserve nuclear boundaries?
- Does noise-level conditioning improve reconstruction under different noise conditions?
- How well do models trained on 256 × 256 images generalize to 1024 × 1024 images?
- Do improved numerical metrics correspond to visually preserved biological details?

---

## Dataset

The project uses paired noisy and ground-truth microscopy images from the AI4Life 2D Nuclei denoising dataset.

| Dataset Group | Image Pairs | Image Dimensions | Purpose |
|---|---:|---:|---|
| Small images | 2,457 | 256 × 256 | Training, validation and cross-validation |
| Large images | 20 | 1024 × 1024 | Separate hold-out evaluation |

The main notebook uses a **3-fold cross-validation** workflow for the small-image experiments.

One illustrated fold contains:

- Training images: 1,475
- Validation images: 163
- Test images: 819

The large-image group is evaluated separately to investigate **domain shift**—the difficulty of applying a model to images with different dimensions, intensity distributions and noise characteristics.

### Preprocessing

The notebooks implement the following general workflow:

- Pair noisy images with their corresponding ground-truth images.
- Inspect image dimensions, intensity ranges and data types.
- Normalize images into a model-friendly range.
- Convert grayscale images into PyTorch tensors.
- Train models using paired noisy and GT images.
- Select checkpoints using validation performance.
- Evaluate on held-out test images.

For large images, the main notebook uses overlapping 256 × 256 tiles with a stride of 128 pixels. Predictions from overlapping regions are combined to reconstruct the complete image.

**Note:** The three notebooks use different normalization methods, test splits, model sizes and training durations. Their absolute scores should therefore be compared within their respective experimental setups.

---

## Methodology

### Classical Baselines

Gaussian and median filters are used to establish traditional denoising baselines.

These methods are computationally simple but may remove fine details or blur nuclear boundaries.

### U-Net

U-Net uses an encoder-decoder architecture:

- The encoder extracts image features.
- The decoder reconstructs the denoised image.
- Skip connections preserve spatial information between corresponding encoder and decoder stages.

The standard U-Net serves as the primary deep-learning baseline.

### Edge-Aware Learning

An additional edge loss compares image gradients between the predicted image and the ground truth.

The objective is to encourage the model to preserve nuclear boundaries instead of producing excessively smooth outputs.

### Noise-Aware Learning

The noise-aware architecture receives an estimated noise descriptor in addition to the noisy image.

In the main experiment, noise information is incorporated using feature-conditioning mechanisms, including FiLM-style conditioning.

The intention is to help the network adapt its denoising behaviour to different noise levels.

### Swin-Based Denoising

The Swin-based model investigates transformer-style image restoration.

Unlike conventional convolutional models, Swin Transformer blocks use shifted-window self-attention to model relationships between image regions.

The model is evaluated as an alternative to U-Net, particularly under the large-image hold-out condition.

---

## Main Results

The following results are taken from the main challenge notebook's final cross-validation summary.

Values are reported as **mean ± standard deviation across three folds**.

### In-Domain Evaluation — 256 × 256 Images

| Model | PSNR (dB) | SSIM | Edge Error | Dice |
|---|---:|---:|---:|---:|
| Standard U-Net (E3) | 28.789 ± 0.232 | 0.882 ± 0.006 | 0.054 ± 0.002 | 0.920 ± 0.003 |
| Noise-Aware U-Net + Edge (E6) | **28.889 ± 0.180** | **0.883 ± 0.005** | **0.053 ± 0.002** | **0.921 ± 0.003** |
| Swin Denoiser (E7) | 26.711 ± 0.227 | 0.840 ± 0.005 | 0.064 ± 0.002 | 0.889 ± 0.008 |

On images similar to the training data, the standard U-Net performs strongly.

The combined noise-aware and edge-aware model improves PSNR by only **0.10 dB** and SSIM by **0.002** compared with the standard U-Net.

These differences are small and should not be interpreted as a decisive improvement.

### Large-Image Hold-Out — 1024 × 1024 Images

| Model | PSNR (dB) | SSIM | Edge Error | Dice |
|---|---:|---:|---:|---:|
| Standard U-Net (E3) | 13.041 ± 0.611 | 0.352 ± 0.011 | 0.112 ± 0.007 | 0.558 ± 0.156 |
| Noise-Aware U-Net + Edge (E6) | 14.828 ± 0.313 | 0.459 ± 0.006 | 0.103 ± 0.005 | 0.466 ± 0.069 |
| Swin Denoiser (E7) | **15.010 ± 0.160** | **0.508 ± 0.026** | **0.080 ± 0.004** | **0.744 ± 0.004** |

The large-image evaluation reveals a substantial **domain-shift problem**.

Compared with the standard U-Net:

- E6 improves hold-out PSNR by approximately **1.79 dB**.
- E7 improves hold-out PSNR by approximately **1.97 dB**.
- E7 achieves the highest reported hold-out SSIM and Dice.
- E7 also achieves the lowest reported hold-out edge error.

However, the large-image outputs still show tile seams, brightness inconsistencies and some loss of nuclear texture.

These results represent partial improvement rather than a complete solution to generalization.

---

## Supplementary Results

A separate saved-checkpoint analysis notebook evaluates 369 test images.

| Model | PSNR (dB) | SSIM | Mean MAE |
|---|---:|---:|---:|
| Standard U-Net | **33.565** | **0.9488** | **0.01189** |
| U-Net + Edge Loss | 33.181 | 0.9449 | 0.01260 |
| DnCNN | 32.498 | 0.9242 | 0.01446 |

In this separate experiment, the standard U-Net performs best among the three evaluated models.

A separate instructional experiment reports:

| Model | PSNR (dB) |
|---|---:|
| Standard U-Net | 29.92 |
| Noise-Aware U-Net | 30.84 |

These supplementary results are not combined with the main challenge results because the experimental setups differ.

---

## Evaluation Metrics

### PSNR — Peak Signal-to-Noise Ratio

PSNR measures pixel-level reconstruction error using a logarithmic scale.

A higher PSNR generally indicates that the predicted image is closer to the ground truth.

### SSIM — Structural Similarity Index

SSIM compares local brightness, contrast and structure.

A value closer to 1 indicates stronger structural similarity between the prediction and GT image.

### Edge Error

Edge error measures differences between image gradients.

A lower value indicates that the predicted boundaries are closer to the ground-truth boundaries.

### Dice Coefficient

Dice measures overlap between thresholded foreground masks.

- 1.0 indicates perfect overlap.
- 0.0 indicates no overlap.

Dice is used as an additional structural measure; it does not mean that the project is a segmentation benchmark.

### MAE — Mean Absolute Error

MAE measures the average absolute difference between predicted and target pixel values.

A lower MAE indicates smaller average pixel-level errors.

**Important:** No single metric fully captures biological image quality. Numerical evaluation is therefore supported by visual comparisons of noisy inputs, GT images, model predictions and zoomed nuclei regions.

---

## Visual Results

The research notebooks include visual comparisons of:

- Noisy input images
- Ground-truth microscopy images
- Standard U-Net predictions
- Edge-aware and noise-aware predictions
- Swin-based denoising outputs
- Zoomed nuclei crops
- Foreground and structural comparisons

The large-image examples are particularly important because they reveal tile boundaries and brightness errors that may not be fully represented by aggregate metrics.

Visual inspection also shows that high PSNR does not necessarily guarantee preservation of fine nuclear texture.

---

## Repository Structure

The repository is organized around the research notebooks and supporting documentation.

```text
.
├── Associate Research Notebooks/
│   ├── 06_results_analysis-Abinash.ipynb
│   ├── notebook64feb9036d - Suyash...ipynb
│   └── Noise_Aware_DeepLearning_U_NET_for_MicroScopyImageDenoising-Arindam-S.ipynb
├── Noise_Aware_Denoising_Final_Presentation...
├── Noise_Aware_UNET-ATTNTN-SWIN...
└── README.md
```
The notebooks are the primary record of:

- Dataset exploration and preprocessing
- Model architecture experiments
- Training and validation
- Checkpoint evaluation
- Metric calculation
- Visual result generation

Notebook filenames may contain additional suffixes depending on how they were downloaded.

---

## Running the Notebooks

The notebooks were developed using Python and PyTorch, with GPU acceleration where available.

1. Clone or download the repository.
2. Create a Python environment. Python 3.10+ is recommended.
3. Install the dependencies imported by the notebooks.
4. Download the AI4Life 2D Nuclei dataset.
5. Update the dataset path in the notebook if required.
6. Open the notebooks in Jupyter or Google Colab.
7. Run the preprocessing, training, checkpoint-loading and evaluation cells in order.

Before reproducing the results, check:

- Dataset location and version
- PyTorch and CUDA compatibility
- GPU availability
- Random seeds
- Saved checkpoint paths
- Normalization settings

Exact reproduction requires the same dataset version, preprocessing procedure and model checkpoints.

---

## Limitations & Future Work

* **Small Large-Image Hold-Out:** The large-image evaluation set contains only 20 image pairs.
* **Configuration Inconsistency:** Different notebooks use varying experimental configurations.
* **Tiled Inference Artifacts:** Tiled inference can introduce visible seams and brightness inconsistencies.
* **Texture Loss:** Fine nuclear textures may be lost during denoising.

### Future Work

* **Dataset Expansion:** Conduct further evaluation on larger and independent microscopy datasets.
* **Tile Blending:** Improve tile blending to reduce seams and brightness shifts.
* **Generalization:** Investigate model generalization across diverse imaging conditions.
* **Additional Experiments:** Complete remaining model experiments.
* **Biological Impact:** Evaluate performance on downstream biological applications.

---

## Research Team

**Autumn 2026 Research Internship**  
**IDEAS – Indian Statistical Institute, Kolkata**

This research project was conducted collaboratively by three research interns under the guidance of a project mentor.

### Research Interns

| Name | Academic Background | LinkedIn | GitHub |
|---|---|---|---|
| **Abinash Mallick** | M.Sc. in Data Science, Symbiosis Institute of Geoinformatics | [LinkedIn](https://www.linkedin.com/in/abinash-mallick-3a481a27b) | [GitHub](https://github.com/abinash128648/Noise-Aware-Microscopy-Denoising) |
| **Suyash Jaiswal** | B.Tech. Computer Science (Data Science), Haldia Institute of Technology | [LinkedIn](https://www.linkedin.com/in/suyashjaiswal7) | [GitHub](https://github.com/suyash-jaiswal-7/Microscopy-Denoising) |
| **Arindam Samadder** | Master of Computer Applications (MCA), Techno India University | [LinkedIn](https://www.linkedin.com/in/arindam-samadder) | [GitHub](https://github.com/Arindam-18s/ideas-ISI-Internship_ai4life-mdc25-biomed-segmentation-2026) |

### Project Mentor

**Mr. Aditya Shankar Pal**  
*PhD Scholar, Indian Statistical Institute, Kolkata* - **LinkedIn:** [View Profile](https://in.linkedin.com/in/aditya-shankar-pal-133a74161)

---

## Acknowledgments

We sincerely thank **IDEAS – Institute of Data Engineering, Analytics and Science Foundation, Indian Statistical Institute (ISI), Kolkata**, for providing the research internship opportunity, mentorship and research support.

We are grateful to our project mentor, **Mr. Aditya Shankar Pal**, for his valuable guidance, technical insights and continuous support throughout the project.

We also thank the organizers of the **AI4Life MDC25 Grand Challenge** for providing the microscopy denoising benchmark and dataset that made this research possible.

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
2. Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*. IEEE TIP.
3. Liang, J., et al. (2021). *SwinIR: Image Restoration Using Swin Transformer*. ICCVW.
4. Perez, E., et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI.
5. Immerkær, J. (1996). *Fast Noise Variance Estimation*. Computer Vision and Image Understanding.
6. AI4Life Microscopy Supervised Denoising Challenge 2025, 2D Nuclei task: https://ai4life-mdc25.grand-challenge.org/
7. Jacquemet, G. (2021). *Noisy Nuclei Dataset for Testing Deep Learning-Based Denoising Tools*. Zenodo.
