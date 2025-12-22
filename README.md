<div align="center">

<h2>⚒️ Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights</h2>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.12%2B-orange)](https://pytorch.org/)
<img src="https://img.shields.io/github/stars/Gandolfczjh/APDE?style=social" alt="GitHub stars" />

[**English**](./README.md) | [**中文**](./README_zh.md)

</div>

> **Note**: This project focuses on the research of adversarial patch attacks and defenses for Object Detectors, providing a complete pipeline for attacks, defenses, and evaluation.
> 
> This is the official repository for the paper [Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](https://arxiv.org/abs/2508.00649) accepted by ICCV2025.

## 📅 Roadmap & To-Do List

The current development plan is as follows, and we will continue to update this list:

- [ ] **1. Detector Evaluation Framework**
    - [ ] Integrate mainstream detector interfaces (YOLOv2/3/4/5/7, Faster R-CNN, SSD, CenterNet, ...).
    - [ ] Provide standardized robustness evaluation metrics (mAP under Attack, Attack Success Rate).

- [ ] **2. APDE Dataset**
    - [ ] Release the **A**dversarial **P**atch **D**efense **E**valuation dataset download link.
    - [ ] Provide data preprocessing and DataLoader scripts.

- [x] **3. Retrained Defense**
    - [x] Release retrained defense weights (SAC, Adyolo, NAPGuard).
    - [x] Provide Model Zoo table comparing Adv mAP before and after retraining.

- [ ] **4. Continuous Updates**
    - [ ] **Attack Codes**: Integrate the latest attack algorithms (T-SEA, ...).
    - [ ] **Patches**: Update adversarial patches to provide references for new defense works.

---

## 🚀 Introduction

With the application of deep learning in autonomous driving and security surveillance, the safety of object detectors has attracted significant attention. This repository aims to provide a unified platform for:
1. Generating high-quality adversarial samples (Adversarial Patch Attacks).
2. Evaluating the vulnerability of existing detectors when facing attacks.
3. Improving model defense capabilities through the APDE dataset and adversarial training.

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/Gandolfczjh/APDE](https://github.com/Gandolfczjh/APDE)
cd APDE

# Create a virtual environment
conda create -n APDE python=3.10
conda activate APDE

# Install dependencies
pip install -r requirements.txt
```

## 🧠 Retrained Defense

To verify the effectiveness of the APDE Dataset, we performed retraining on three mainstream defense methods (SAC, Adyolo, NAPGuard) using this dataset.

The table below shows the detection performance (mAP) before and after retraining under various adversarial patch attacks. It is worth noting that the AdvCloak and AdvTshirt attacks in the last two rows were NOT included in our retraining set (Out-of-Domain / Unseen). Experimental results show that retraining with APDE not only improves defense against known attacks but also significantly enhances generalization capabilities against unknown (out-of-domain) patches.

| Attack Method | SAC <br> *(Original / Retrained)* | Adyolo <br> *(Original / Retrained)* | NAPGuard <br> *(Original / Retrained)* |
| :--- | :---: | :---: | :---: |
| **T-SEA** | 51.82 / **71.61** | 66.61 / **72.47** | 83.61 / **86.31** |
| **TC-EGA** | 58.16 / **71.36** | 63.49 / **70.91** | 68.51 / **85.30** |
| **Advpatch** | 56.53 / **73.29** | 65.54 / **72.07** | 78.45 / **85.10** |
| **GNAP** | 70.03 / **76.86** | 72.94 / **78.52** | 78.96 / **85.42** |
| **DM-NAP** | 68.50 / **76.48** | 69.26 / **76.83** | 71.37 / **85.71** |
| **_Out-of-Domain (Unseen)_** | | | |
| **AdvCloak** | 4.17 / **71.29** | 18.29 / **22.36** | 52.21 / **73.16** |
| **AdvTshirt** | 34.27 / **64.47** | 8.19 / **37.53** | 50.21 / **70.89** |

**Weights Download:**
> **[Google Drive](https://drive.google.com/drive/folders/13y1xrvXo-p1JI7yGucAFHSy_wuqO_sFT?us)**
> *Google Drive Includes weights for SAC-Retrained, Adyolo-Retrained, NAPGuard-Retrained, etc.*

This repository integrates the following three typical defense methods and provides the corresponding retrained weights:

* **SAC (Segment and Complete)**: Locates adversarial patches via a segmentation network, removes them, and uses image inpainting technology to restore the background, thereby recovering detector performance.

* **Adyolo (Adversarial YOLO)**: Introduces a new "adversarial patch" class during the training phase, enabling the detector to actively identify and ignore adversarial patches in the scene, preventing them from interfering with normal object detection.

* **NAPGuard**: Specifically designed for naturalistic adversarial patches, it distinguishes generated adversarial textures from natural objects by analyzing texture and pixel distribution features.
