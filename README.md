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
    - [x] Provide a reconstruction dataset reader and Hugging Face export tools.

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
git clone https://github.com/Gandolfczjh/APDE.git
cd APDE

# Create a virtual environment
conda create -n APDE python=3.10
conda activate APDE

# Install dependencies
pip install -r requirements.txt
```

## APDE dataset

The reconstructed dataset is complete and validated: **94 patch types, 94,000 patched images**, with one mask and annotation file per image. The Hugging Face download link is **pending publication**; no dataset is hosted in this Git repository yet. See the [publishing guide](docs/HUGGINGFACE_RELEASE.md) and [dataset card](docs/HF_DATASET_CARD.md).

| Split | Patch types | Patched images |
|---|---:|---:|
| Train | 57 | 57,000 |
| Test | 37 | 37,000 |
| Total | 94 | 94,000 |

This is a reconstruction, **not the exact original dataset used for the paper**. The paper reports 56,400/37,600 images; this release keeps each 1,000-image patch type entirely in one split. Clean sources are shared across patch types, including across train/test. Split isolation applies to patch types, not source photographs.

The source pool contains 1,000 positives (81 INRIA Test + 919 COCO val2017) sampled without replacement from 288 + 2,693 positive candidates, and 1,000 annotation-negative backgrounds (162 INRIA + 838 COCO). Images are padded/resized to 416×416. Masks are lossless binary PNG (0/255); annotation rows are `person x1 y1 x2 y2` and `patch x1 y1 x2 y2` in pixel coordinates. Patch right/bottom coordinates are exclusive.

### Directory layout

```text
APDE/
├── clean/
│   ├── positive/{images,labels}/        # 1,000 clean positive samples
│   └── negative/{images,labels}/        # 1,000 clean negative samples
├── groups/<method>/<detector>/
│   ├── images/                         # 1,000 RGB PNGs per patch type
│   ├── masks/                          # 1,000 binary PNG masks
│   ├── labels/                         # 1,000 TXT annotations
│   ├── patch.png                       # Frozen patch used for this group
│   ├── samples.jsonl                   # Paths, split, and hashes
│   └── complete.json
├── images/<method>/<detector>/          # Compatibility symlink to groups
├── annotations/<method>/
│   ├── <detector>_label/                # Compatibility symlink
│   └── <detector>_mask/                 # Compatibility symlink
├── train.jsonl                         # 57,000 sample records
├── test.jsonl                          # 37,000 sample records
├── split_plan.json
├── positive_sources.json
├── negative_sources.json
├── source_report.json
├── build_report.json
├── audit_report.json
├── pipeline_status.json
├── code_snapshot/
└── preview.jpg
```

The eight matrix families (`advpatch`, `TCEGA`, `tsea-pgd`, `tsea-mim`, `TCA`, `tsea`, `GNAP`, `DM-NAP`) each cover 11 detectors: `yolov2`, `yolov3`, `yolov4`, `yolov5`, `yolov7`, `ssd`, `centernet`, `retinanet`, `mask_rcnn`, `faster_rcnn`, `ddetr`.

The remaining six types are **test-only**: `AdvCloak/yolov2`, `AdvCloak/yolov3`, `AdvTshirt/yolov2`, `AA/yolov2`, `AdvSticker/yolov3`, and `UPC/yolov3`. The train/test JSONL manifests define the split; there are no duplicated train/test image directories.

### Read the local dataset

```bash
python -m pip install Pillow
```

```python
from apde_data import APDEDataset

data = APDEDataset("APDE", split="test")
sample = data[0]
image, mask = sample["image"], sample["mask"]  # RGB / L PIL images
person_boxes, patch_boxes = sample["person_boxes"], sample["patch_boxes"]
```

The reader works with PyTorch `DataLoader`; supply a transform and a custom collate function for variable-length boxes. For HF export and `load_dataset()` usage, follow [HUGGINGFACE_RELEASE.md](docs/HUGGINGFACE_RELEASE.md).

### Provenance and limits

Four missing GNAP combinations were retrained with corrected settings. AdvCloak/YOLOv3 was recovered from its original paper's embedded patch figure. AA, AdvSticker, and UPC are newly trained detector adaptations; UPC is weak under the small-patch placement used here. These are not recovered original-author checkpoints. Detailed provenance and diagnostic results are in the [dataset card](docs/HF_DATASET_CARD.md).

The defense scores below belong to the published paper and have not been rerun on this reconstruction. Dataset source-image and patch-artifact terms must be documented separately from the code repository's MIT badge before public data redistribution.

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
