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

- [x] **2. APDE 数据集 (APDE Dataset)**
    - [x] 发布 **A**dversarial **P**atch **D**efense **E**valuation 数据集下载链接。
    - [x] 提供数据读取、完整性校验与 Hugging Face 下载和处理示例。

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

重建已完成并通过完整性校验：**94 类补丁、94,000 张合成图像**，每张图像均有对应掩码和标注。完整数据集已在 **[Hugging Face](https://huggingface.co/datasets/Gandolfczjh/APDE)** 发布，本 Git 仓库存放补丁原图与读取工具。下载与处理方法见下方[使用示例](#从-hugging-face-下载与使用-apde)，数据格式、来源与评估说明见 [Dataset Card](docs/HF_DATASET_CARD.md)。

| 划分 | 补丁类型 | 合成图像 |
|---|---:|---:|
| 训练集 | 57 | 57,000 |
| 测试集 | 37 | 37,000 |
| 总计 | 94 | 94,000 |

这是重新制作的数据集，**并非论文原始数据的逐字节恢复**。论文原文为 56,400/37,600 张；本版为保持每类 1,000 张且补丁类型完全隔离，采用 57,000/37,000 划分。不同补丁类型共用 clean 原图，因此训练/测试集保证的是**补丁类型隔离，原始照片并不隔离**。

从 INRIA Test 的 288 张和 COCO val2017 的 2,693 张正样本中，无放回随机抽取 1,000 张（INRIA 81 + COCO 919）；另准备 1,000 张标注中无人的负样本（INRIA 162 + COCO 838）。图像补边缩放至 416×416，掩码为无损二值 PNG（0/255）。TXT 标注格式为 `person x1 y1 x2 y2` 或 `patch x1 y1 x2 y2`，使用像素坐标；补丁框右、下边界不包含在框内。

### 从 Hugging Face 下载与使用 APDE

**数据集：[Gandolfczjh/APDE](https://huggingface.co/datasets/Gandolfczjh/APDE)**

在本代码仓库中安装数据读取依赖：

```bash
python -m pip install -r requirements-data.txt
```

| 配置 | 划分 | 内容 |
|---|---|---|
| `patched`（默认） | `train`：57,000；`test`：37,000 | 合成图像、二值掩码、TXT 标注与元信息 |
| `clean` | `positive`：1,000；`negative`：1,000 | clean 原图及人物框 |
| `patches` | `all`：94 | 补丁原图及训练/测试归属 |

#### 在线加载

```python
from datasets import load_dataset

data = load_dataset("Gandolfczjh/APDE", "patched")
assert len(data["train"]) == 57000
assert len(data["test"]) == 37000
sample = data["test"][0]

clean = load_dataset("Gandolfczjh/APDE", "clean")
patches = load_dataset("Gandolfczjh/APDE", "patches", split="all")
```

首次调用会下载并处理所选配置，后续调用复用 Hugging Face 缓存。若只想先查看样本，可以流式读取，无需等待整个配置下载完成：

```python
stream = load_dataset("Gandolfczjh/APDE", "patched", split="test", streaming=True)
sample = next(iter(stream))
```

#### 完整下载与离线加载

完整发布包包含 97 个 Parquet 文件，约 20 GB；处理缓存或导出图片还需要额外磁盘空间。以下命令在克隆的代码仓库中执行，`APDE` 表示存放数据的子目录：

```bash
hf download Gandolfczjh/APDE --repo-type dataset --local-dir APDE
python tools/verify_apde_hf.py APDE
```

下载中断后，重复执行下载命令即可继续。下载完成后，从本地 Parquet 文件加载：

```python
from datasets import load_dataset

data = load_dataset("parquet", data_files={
    "train": "APDE/data/train/*.parquet",
    "test": "APDE/data/test/*.parquet",
})
sample = data["test"][0]
```

HF 发布包将图片、掩码和标注内嵌在 Parquet 中，包含 `data/train`、`data/test`、`clean` 和 `patches` 目录，与下方原始 PNG/JSONL 目录结构不同。HF 下载文件请用 `load_dataset()` 读取；`APDEDataset("APDE")` 用于包含 `train.jsonl` 和 `test.jsonl` 的原始目录。

#### 处理图片、掩码与标注

```python
from pathlib import Path
import numpy as np
from apde_data.dataset import parse_labels

image = sample["image"].convert("RGB")  # PIL 图像，416 x 416
mask = sample["mask"].convert("L")      # 0 为背景，255 为补丁
boxes = parse_labels(sample["labels"])
person_boxes = boxes["person_boxes"]   # 像素坐标 [x1, y1, x2, y2]
patch_boxes = boxes["patch_boxes"]

image_array = np.asarray(image, dtype=np.float32) / 255.0  # H x W x 3
mask_array = (np.asarray(mask) > 0).astype(np.uint8)        # H x W，取值 0/1

# 将一个样本导出为普通 PNG/TXT 文件。
image.save("sample.png")
mask.save("sample_mask.png")
Path("sample.txt").write_text(sample["labels"], encoding="utf-8")
```

每条合成样本还包含 `id`、`source_id`、`patch_id`、`method`、`detector`、`attack_goal` 和 `split`。clean 样本的人物框保存在 `person_boxes_json`，可用 `json.loads()` 解析。几何变换时须同步处理图片、掩码和框；掩码缩放使用最近邻插值。训练与评估时保留提供的补丁类型划分。

更多用法见 HF 官方[加载文档](https://huggingface.co/docs/datasets/loading)和[下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。

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

<!-- APDE patch gallery -->

### 94 张补丁一览

展示高度统一为 96 像素，保留原始宽高比；点击图片可查看原图。每张图标注检测器及训练/测试归属。

#### AdvPatch

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/advpatch/yolov2.png"><img src="assets/patches/advpatch/yolov2.png" alt="advpatch/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/yolov3.png"><img src="assets/patches/advpatch/yolov3.png" alt="advpatch/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/yolov4.png"><img src="assets/patches/advpatch/yolov4.png" alt="advpatch/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/yolov5.png"><img src="assets/patches/advpatch/yolov5.png" alt="advpatch/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/yolov7.png"><img src="assets/patches/advpatch/yolov7.png" alt="advpatch/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/ssd.png"><img src="assets/patches/advpatch/ssd.png" alt="advpatch/ssd patch" width="96" height="96"></a><br><sub>SSD<br>训练</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/advpatch/centernet.png"><img src="assets/patches/advpatch/centernet.png" alt="advpatch/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/retinanet.png"><img src="assets/patches/advpatch/retinanet.png" alt="advpatch/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/mask_rcnn.png"><img src="assets/patches/advpatch/mask_rcnn.png" alt="advpatch/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/faster_rcnn.png"><img src="assets/patches/advpatch/faster_rcnn.png" alt="advpatch/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/advpatch/ddetr.png"><img src="assets/patches/advpatch/ddetr.png" alt="advpatch/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>测试</sub></td>
  </tr>
</table>

#### TC-EGA

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/TCEGA/yolov2.png"><img src="assets/patches/TCEGA/yolov2.png" alt="TCEGA/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/yolov3.png"><img src="assets/patches/TCEGA/yolov3.png" alt="TCEGA/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/yolov4.png"><img src="assets/patches/TCEGA/yolov4.png" alt="TCEGA/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/yolov5.png"><img src="assets/patches/TCEGA/yolov5.png" alt="TCEGA/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/yolov7.png"><img src="assets/patches/TCEGA/yolov7.png" alt="TCEGA/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/ssd.png"><img src="assets/patches/TCEGA/ssd.png" alt="TCEGA/ssd patch" width="96" height="96"></a><br><sub>SSD<br>训练</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/TCEGA/centernet.png"><img src="assets/patches/TCEGA/centernet.png" alt="TCEGA/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/retinanet.png"><img src="assets/patches/TCEGA/retinanet.png" alt="TCEGA/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/mask_rcnn.png"><img src="assets/patches/TCEGA/mask_rcnn.png" alt="TCEGA/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/faster_rcnn.png"><img src="assets/patches/TCEGA/faster_rcnn.png" alt="TCEGA/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCEGA/ddetr.png"><img src="assets/patches/TCEGA/ddetr.png" alt="TCEGA/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>测试</sub></td>
  </tr>
</table>

#### T-SEA-PGD

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/yolov2.png"><img src="assets/patches/tsea-pgd/yolov2.png" alt="tsea-pgd/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/yolov3.png"><img src="assets/patches/tsea-pgd/yolov3.png" alt="tsea-pgd/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/yolov4.png"><img src="assets/patches/tsea-pgd/yolov4.png" alt="tsea-pgd/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/yolov5.png"><img src="assets/patches/tsea-pgd/yolov5.png" alt="tsea-pgd/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/yolov7.png"><img src="assets/patches/tsea-pgd/yolov7.png" alt="tsea-pgd/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/ssd.png"><img src="assets/patches/tsea-pgd/ssd.png" alt="tsea-pgd/ssd patch" width="96" height="96"></a><br><sub>SSD<br>训练</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/centernet.png"><img src="assets/patches/tsea-pgd/centernet.png" alt="tsea-pgd/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/retinanet.png"><img src="assets/patches/tsea-pgd/retinanet.png" alt="tsea-pgd/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/mask_rcnn.png"><img src="assets/patches/tsea-pgd/mask_rcnn.png" alt="tsea-pgd/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/faster_rcnn.png"><img src="assets/patches/tsea-pgd/faster_rcnn.png" alt="tsea-pgd/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-pgd/ddetr.png"><img src="assets/patches/tsea-pgd/ddetr.png" alt="tsea-pgd/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>训练</sub></td>
  </tr>
</table>

#### T-SEA-MIM

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/yolov2.png"><img src="assets/patches/tsea-mim/yolov2.png" alt="tsea-mim/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/yolov3.png"><img src="assets/patches/tsea-mim/yolov3.png" alt="tsea-mim/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/yolov4.png"><img src="assets/patches/tsea-mim/yolov4.png" alt="tsea-mim/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/yolov5.png"><img src="assets/patches/tsea-mim/yolov5.png" alt="tsea-mim/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/yolov7.png"><img src="assets/patches/tsea-mim/yolov7.png" alt="tsea-mim/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/ssd.png"><img src="assets/patches/tsea-mim/ssd.png" alt="tsea-mim/ssd patch" width="96" height="96"></a><br><sub>SSD<br>训练</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/centernet.png"><img src="assets/patches/tsea-mim/centernet.png" alt="tsea-mim/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/retinanet.png"><img src="assets/patches/tsea-mim/retinanet.png" alt="tsea-mim/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/mask_rcnn.png"><img src="assets/patches/tsea-mim/mask_rcnn.png" alt="tsea-mim/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/faster_rcnn.png"><img src="assets/patches/tsea-mim/faster_rcnn.png" alt="tsea-mim/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea-mim/ddetr.png"><img src="assets/patches/tsea-mim/ddetr.png" alt="tsea-mim/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>测试</sub></td>
  </tr>
</table>

#### TCA

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/TCA/yolov2.png"><img src="assets/patches/TCA/yolov2.png" alt="TCA/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/yolov3.png"><img src="assets/patches/TCA/yolov3.png" alt="TCA/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/yolov4.png"><img src="assets/patches/TCA/yolov4.png" alt="TCA/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/yolov5.png"><img src="assets/patches/TCA/yolov5.png" alt="TCA/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/yolov7.png"><img src="assets/patches/TCA/yolov7.png" alt="TCA/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/ssd.png"><img src="assets/patches/TCA/ssd.png" alt="TCA/ssd patch" width="96" height="96"></a><br><sub>SSD<br>测试</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/TCA/centernet.png"><img src="assets/patches/TCA/centernet.png" alt="TCA/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/retinanet.png"><img src="assets/patches/TCA/retinanet.png" alt="TCA/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/mask_rcnn.png"><img src="assets/patches/TCA/mask_rcnn.png" alt="TCA/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/faster_rcnn.png"><img src="assets/patches/TCA/faster_rcnn.png" alt="TCA/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/TCA/ddetr.png"><img src="assets/patches/TCA/ddetr.png" alt="TCA/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>测试</sub></td>
  </tr>
</table>

#### T-SEA

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/tsea/yolov2.png"><img src="assets/patches/tsea/yolov2.png" alt="tsea/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/yolov3.png"><img src="assets/patches/tsea/yolov3.png" alt="tsea/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/yolov4.png"><img src="assets/patches/tsea/yolov4.png" alt="tsea/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/yolov5.png"><img src="assets/patches/tsea/yolov5.png" alt="tsea/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/yolov7.png"><img src="assets/patches/tsea/yolov7.png" alt="tsea/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/ssd.png"><img src="assets/patches/tsea/ssd.png" alt="tsea/ssd patch" width="96" height="96"></a><br><sub>SSD<br>训练</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/tsea/centernet.png"><img src="assets/patches/tsea/centernet.png" alt="tsea/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/retinanet.png"><img src="assets/patches/tsea/retinanet.png" alt="tsea/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/mask_rcnn.png"><img src="assets/patches/tsea/mask_rcnn.png" alt="tsea/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/faster_rcnn.png"><img src="assets/patches/tsea/faster_rcnn.png" alt="tsea/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/tsea/ddetr.png"><img src="assets/patches/tsea/ddetr.png" alt="tsea/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>训练</sub></td>
  </tr>
</table>

#### GNAP

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/GNAP/yolov2.png"><img src="assets/patches/GNAP/yolov2.png" alt="GNAP/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/yolov3.png"><img src="assets/patches/GNAP/yolov3.png" alt="GNAP/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/yolov4.png"><img src="assets/patches/GNAP/yolov4.png" alt="GNAP/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/yolov5.png"><img src="assets/patches/GNAP/yolov5.png" alt="GNAP/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/yolov7.png"><img src="assets/patches/GNAP/yolov7.png" alt="GNAP/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/ssd.png"><img src="assets/patches/GNAP/ssd.png" alt="GNAP/ssd patch" width="96" height="96"></a><br><sub>SSD<br>测试</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/GNAP/centernet.png"><img src="assets/patches/GNAP/centernet.png" alt="GNAP/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/retinanet.png"><img src="assets/patches/GNAP/retinanet.png" alt="GNAP/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/mask_rcnn.png"><img src="assets/patches/GNAP/mask_rcnn.png" alt="GNAP/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/faster_rcnn.png"><img src="assets/patches/GNAP/faster_rcnn.png" alt="GNAP/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/GNAP/ddetr.png"><img src="assets/patches/GNAP/ddetr.png" alt="GNAP/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>测试</sub></td>
  </tr>
</table>

#### DM-NAP

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/yolov2.png"><img src="assets/patches/DM-NAP/yolov2.png" alt="DM-NAP/yolov2 patch" width="96" height="96"></a><br><sub>YOLOv2<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/yolov3.png"><img src="assets/patches/DM-NAP/yolov3.png" alt="DM-NAP/yolov3 patch" width="96" height="96"></a><br><sub>YOLOv3<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/yolov4.png"><img src="assets/patches/DM-NAP/yolov4.png" alt="DM-NAP/yolov4 patch" width="96" height="96"></a><br><sub>YOLOv4<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/yolov5.png"><img src="assets/patches/DM-NAP/yolov5.png" alt="DM-NAP/yolov5 patch" width="96" height="96"></a><br><sub>YOLOv5<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/yolov7.png"><img src="assets/patches/DM-NAP/yolov7.png" alt="DM-NAP/yolov7 patch" width="96" height="96"></a><br><sub>YOLOv7<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/ssd.png"><img src="assets/patches/DM-NAP/ssd.png" alt="DM-NAP/ssd patch" width="96" height="96"></a><br><sub>SSD<br>测试</sub></td>
  </tr>
  <tr>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/centernet.png"><img src="assets/patches/DM-NAP/centernet.png" alt="DM-NAP/centernet patch" width="96" height="96"></a><br><sub>CenterNet<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/retinanet.png"><img src="assets/patches/DM-NAP/retinanet.png" alt="DM-NAP/retinanet patch" width="96" height="96"></a><br><sub>RetinaNet<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/mask_rcnn.png"><img src="assets/patches/DM-NAP/mask_rcnn.png" alt="DM-NAP/mask_rcnn patch" width="96" height="96"></a><br><sub>Mask R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/faster_rcnn.png"><img src="assets/patches/DM-NAP/faster_rcnn.png" alt="DM-NAP/faster_rcnn patch" width="96" height="96"></a><br><sub>Faster R-CNN<br>训练</sub></td>
    <td align="center" width="120"><a href="assets/patches/DM-NAP/ddetr.png"><img src="assets/patches/DM-NAP/ddetr.png" alt="DM-NAP/ddetr patch" width="96" height="96"></a><br><sub>D-DETR<br>训练</sub></td>
  </tr>
</table>

#### 六类仅用于测试的补丁

<table>
  <tr>
    <td align="center" width="120"><a href="assets/patches/AdvCloak/yolov2.png"><img src="assets/patches/AdvCloak/yolov2.png" alt="AdvCloak/yolov2 patch" width="59" height="96"></a><br><sub>AdvCloak / YOLOv2<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/AdvCloak/yolov3.png"><img src="assets/patches/AdvCloak/yolov3.png" alt="AdvCloak/yolov3 patch" width="58" height="96"></a><br><sub>AdvCloak / YOLOv3<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/AdvTshirt/yolov2.png"><img src="assets/patches/AdvTshirt/yolov2.png" alt="AdvTshirt/yolov2 patch" width="59" height="96"></a><br><sub>AdvTshirt / YOLOv2<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/AA/yolov2.png"><img src="assets/patches/AA/yolov2.png" alt="AA/yolov2 patch" width="96" height="96"></a><br><sub>AA / YOLOv2<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/AdvSticker/yolov3.png"><img src="assets/patches/AdvSticker/yolov3.png" alt="AdvSticker/yolov3 patch" width="96" height="96"></a><br><sub>AdvSticker / YOLOv3<br>测试</sub></td>
    <td align="center" width="120"><a href="assets/patches/UPC/yolov3.png"><img src="assets/patches/UPC/yolov3.png" alt="UPC/yolov3 patch" width="96" height="96"></a><br><sub>UPC / YOLOv3<br>测试</sub></td>
  </tr>
</table>

<!-- /APDE patch gallery -->

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

配合 PyTorch `DataLoader` 使用时，可通过 transform 转换张量，并使用自定义 collate 函数处理不同数量的框。HF 下载文件的读取与处理方法见上方[使用示例](#从-hugging-face-下载与使用-apde)。

### 来源与复现范围

四类缺失 GNAP 已按修正后的参数重训；AdvCloak/YOLOv3 从原论文内嵌补丁图恢复；AA、AdvSticker、UPC 为新训练的检测器适配实现，并非找回的原作者训练权重。UPC 在本版小补丁粘贴规则下效果较弱。来源和实测结果详见 [Dataset Card](docs/HF_DATASET_CARD.md)。

下面的防御性能表来自原论文，尚未在本重建版上重新测量。

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
