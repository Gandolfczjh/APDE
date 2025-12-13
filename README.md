# Overview
This is the official repository for the paper [Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](https://arxiv.org/abs/2508.00649) accepted by ICCV2025.


# ⚒️ Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.12%2B-orange)](https://pytorch.org/)
<img src="https://img.shields.io/github/stars/Gandolfczjh/APDE?style=social" alt="GitHub stars" />

> **注意**: 本项目致力于目标检测器（Object Detectors）的对抗补丁攻击和防御研究，提供攻击、防御及测评流程。

## 📅 Roadmap & To-Do List

目前项目的开发计划如下，我们将持续更新此列表：

- [ ]  **1. 检测器测评框架 (Detector Evaluation Framework)**
    - [ ] 集成主流检测器接口 (YOLOv2/3/4/5/7, Faster R-CNN, SSD, CenterNet, ...)。
    - [ ] 提供标准化的鲁棒性评估指标 (mAP under Attack, Attack Success Rate)。

- [ ] **2. APDE 数据集 (APDE Dataset)**
    - [ ] 发布 **A**dversarial **P**atch **D**efense **E**valuation 数据集下载链接。
    - [ ] 提供数据预处理与 DataLoader 脚本。

- [x] **3. 重训练防御 (Retrained Defense)**
    - [x] 发布重训练防御权重（SAC, Adyolo, NAPGuard）。
    - [x] 提供 Model Zoo 表格，对比各种防御重训练前后的 Adv mAP。

- [ ] **4. 持续更新的攻击代码与补丁 (Continuous Updates)**
    - [ ] **Attack Codes**: 集成最新的攻击算法（T-SEA, ...）。
    - [ ] **Patches**: 更新对抗补丁，为新的防御工作提供参考

---

## 🚀 Introduction

随着深度学习在自动驾驶和安防监控中的应用，目标检测器的安全性备受关注。本仓库旨在提供一个统一的平台，用于：
1.  生成高质量的对抗样本（对抗补丁攻击）。
2.  评估现有检测器在面对攻击时的脆弱性。
3.  通过 APDE 数据集和对抗训练提升模型的防御能力。

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

为了验证 **APDE 数据集** 的有效性，我们使用该数据集对三种主流的防御方法（SAC, Adyolo, NAPGuard）进行了重训练（Retraining）。

下表展示了重训练前后模型在面对不同对抗补丁攻击下的检测性能（mAP）。**需要特别注意的是，表格最后两行的 AdvCloak 和 AdvTshirt 攻击并未包含在我们的重训练集中（Out-of-Domain / Unseen）。** 实验结果表明，使用 APDE 重训练不仅提升了对已知攻击的防御力，更显著提高了模型对**未知（域外）补丁**的泛化防御能力。

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
> *包含 SAC-Retrained, Adyolo-Retrained, NAPGuard-Retrained 等权重文件。*

本仓库集成了以下三种典型的防御方法，并提供了相应的重训练权重：

* **SAC (Segment and Complete)**: 通过分割网络定位对抗补丁，将其剔除后利用图像修复技术（Inpainting）还原背景，从而恢复检测器的性能。
* **Adyolo (Adversarial YOLO)**: 在训练阶段引入一个新的“对抗补丁”类别，使检测器能够主动识别并忽略画面中的对抗补丁，防止其干扰正常物体检测。
* **NAPGuard**: 专门针对自然伪装风格的对抗补丁设计，通过分析纹理和像素分布特征来区分生成的对抗纹理与自然物体。
