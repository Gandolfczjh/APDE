<div align="center">

<h2>⚒️ Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights</h2>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.12%2B-orange)](https://pytorch.org/)
<img src="https://img.shields.io/github/stars/Gandolfczjh/APDE?style=social" alt="GitHub stars" />

[**English**](./README.md) | [**中文**](./README_zh.md)

</div>



> **注意**: 本项目致力于目标检测器（Object Detectors）的对抗补丁攻击和防御研究，提供攻击、防御及测评流程。
> 
> This is the official repository for the paper [Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](https://arxiv.org/abs/2508.00649) accepted by ICCV2025.



## 📅 Roadmap & To-Do List

目前项目的开发计划如下，我们将持续更新此列表：

- [ ]  **1. 检测器测评框架 (Detector Evaluation Framework)**
    - [ ] 集成主流检测器接口 (YOLOv2/3/4/5/7, Faster R-CNN, SSD, CenterNet, ...)。
    - [ ] 提供标准化的鲁棒性评估指标 (mAP under Attack, Attack Success Rate)。

- [ ] **2. APDE 数据集 (APDE Dataset)**
    - [ ] 发布 **A**dversarial **P**atch **D**efense **E**valuation 数据集下载链接。
    - [x] 提供重建版数据读取与 Hugging Face 导出工具。

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
git clone https://github.com/Gandolfczjh/APDE.git
cd APDE

# Create a virtual environment
conda create -n APDE python=3.10
conda activate APDE

# Install dependencies
pip install -r requirements.txt

```

## APDE 数据集

重建已完成并通过完整性校验：**94 类补丁、94,000 张合成图像**，每张图像均有对应掩码和标注。Hugging Face 下载链接**待发布**，本 Git 仓库暂不存放数据文件。发布步骤见 [HF 上传指南](docs/HUGGINGFACE_RELEASE.md)，版本与来源说明见 [Dataset Card](docs/HF_DATASET_CARD.md)。

| 划分 | 补丁类型 | 合成图像 |
|---|---:|---:|
| 训练集 | 57 | 57,000 |
| 测试集 | 37 | 37,000 |
| 总计 | 94 | 94,000 |

这是重新制作的数据集，**并非论文原始数据的逐字节恢复**。论文原文为 56,400/37,600 张；本版为保持每类 1,000 张且补丁类型完全隔离，采用 57,000/37,000 划分。不同补丁类型共用 clean 原图，因此训练/测试集保证的是**补丁类型隔离，原始照片并不隔离**。

从 INRIA Test 的 288 张和 COCO val2017 的 2,693 张正样本中，无放回随机抽取 1,000 张（INRIA 81 + COCO 919）；另准备 1,000 张标注中无人的负样本（INRIA 162 + COCO 838）。图像补边缩放至 416×416，掩码为无损二值 PNG（0/255）。TXT 标注格式为 `person x1 y1 x2 y2` 或 `patch x1 y1 x2 y2`，使用像素坐标；补丁框右、下边界不包含在框内。

### 文件夹结构

```text
APDE/
├── clean/
│   ├── positive/{images,labels}/        # 1,000 张 clean 正样本
│   └── negative/{images,labels}/        # 1,000 张 clean 负样本
├── groups/<攻击方法>/<检测器>/
│   ├── images/                         # 每类 1,000 张 RGB PNG
│   ├── masks/                          # 每类 1,000 张二值掩码
│   ├── labels/                         # 每类 1,000 个 TXT 标注
│   ├── patch.png                       # 生成该组数据时固定的补丁
│   ├── samples.jsonl                   # 路径、划分与哈希
│   └── complete.json
├── images/<攻击方法>/<检测器>/           # 指向 groups 的兼容软链接
├── annotations/<攻击方法>/
│   ├── <检测器>_label/                  # 标注软链接
│   └── <检测器>_mask/                   # 掩码软链接
├── train.jsonl                         # 57,000 条训练样本
├── test.jsonl                          # 37,000 条测试样本
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

八种方法 `advpatch`、`TCEGA`、`tsea-pgd`、`tsea-mim`、`TCA`、`tsea`、`GNAP`、`DM-NAP` 各覆盖 11 个检测器：`yolov2`、`yolov3`、`yolov4`、`yolov5`、`yolov7`、`ssd`、`centernet`、`retinanet`、`mask_rcnn`、`faster_rcnn`、`ddetr`。

其余六类**全部进入测试集**：`AdvCloak/yolov2`、`AdvCloak/yolov3`、`AdvTshirt/yolov2`、`AA/yolov2`、`AdvSticker/yolov3`、`UPC/yolov3`。训练和测试由 JSONL 清单划分，不额外复制图片到 train/test 文件夹。

### 读取本地数据

```bash
python -m pip install Pillow
```

```python
from apde_data import APDEDataset

data = APDEDataset("APDE", split="test")
sample = data[0]
image, mask = sample["image"], sample["mask"]  # RGB / L 模式 PIL 图像
person_boxes, patch_boxes = sample["person_boxes"], sample["patch_boxes"]
```

配合 PyTorch `DataLoader` 使用时，可通过 transform 转换张量，并使用自定义 collate 函数处理不同数量的框。HF 导出与 `load_dataset()` 用法见 [上传指南](docs/HUGGINGFACE_RELEASE.md)。

### 来源与复现范围

四类缺失 GNAP 已按修正后的参数重训；AdvCloak/YOLOv3 从原论文内嵌补丁图恢复；AA、AdvSticker、UPC 为新训练的检测器适配实现，并非找回的原作者训练权重。UPC 在本版小补丁粘贴规则下效果较弱。来源和实测结果详见 [Dataset Card](docs/HF_DATASET_CARD.md)。

下面的防御性能表来自原论文，尚未在本重建版上重新测量。数据公开发布前，需要单独明确原图与第三方补丁的许可和署名要求，不能直接以代码仓库的 MIT 徽章作为全部数据的许可。

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
