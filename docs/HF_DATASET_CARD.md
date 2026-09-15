---
language:
- en
task_categories:
- object-detection
- image-segmentation
size_categories:
- 10K<n<100K
tags:
- adversarial-patches
- robustness
- apde
configs:
- config_name: patched
  default: true
  data_files:
  - split: train
    path: data/train/*.parquet
  - split: test
    path: data/test/*.parquet
- config_name: clean
  data_files:
  - split: positive
    path: clean/positive.parquet
  - split: negative
    path: clean/negative.parquet
- config_name: patches
  data_files:
  - split: all
    path: patches/patches.parquet
---

# APDE

Dataset for evaluating adversarial patch defenses on object detectors. Related paper:
[Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](https://arxiv.org/abs/2508.00649), ICCV 2025.
[Code and documentation](https://github.com/Gandolfczjh/APDE).

This release is a **reconstruction**, not a byte-for-byte recovery of the dataset used in the paper. The paper's 56,400/37,600 split and reported defense scores must not be treated as measurements on this release.

## Contents

| Configuration | Split | Rows |
|---|---|---:|
| patched | train | 57,000 |
| patched | test | 37,000 |
| clean | positive | 1,000 |
| clean | negative | 1,000 |
| patches | all | 94 |

The 94,000 patched RGB PNG images are 416x416, with corresponding lossless binary masks and pixel-coordinate TXT annotations embedded in Parquet. Each patch type has 1,000 samples. `image` and `mask` are Hugging Face Image features. `labels` contains one `person x1 y1 x2 y2` or `patch x1 y1 x2 y2` record per line. Patch bounding boxes have exclusive right/bottom limits. Mask values are 0 (background) and 255 (patch support).

Metadata columns include globally unique `id`, `source_id`, `source_dataset`, `patch_id`, `method`, `detector`, `attack_goal`, `split`, and image/mask/patch SHA-256 hashes. Clean samples include original filenames and original file hashes. Original filesystem paths are excluded.

```python
from datasets import load_dataset

dataset = load_dataset("Gandolfczjh/APDE", "patched")
sample = dataset["test"][0]
image, mask, labels = sample["image"], sample["mask"], sample["labels"]
clean = load_dataset("Gandolfczjh/APDE", "clean")
patches = load_dataset("Gandolfczjh/APDE", "patches", split="all")
```

## Sources and split

Uniform sampling without replacement from INRIA-Person Test positives (288) and COCO val2017 person-positive images (2,693), seed 15089, selected 81 INRIA and 919 COCO images. Negative sampling selected 162 INRIA and 838 COCO images, seed 15090. Negative means no person in the source annotations; it does not imply manual verification that every image is person-free. Person crowd annotations are retained.

Train/test are disjoint by patch type: 57 training types and 37 test types. **Clean source photographs are shared across train/test patch groups.** This is not a source-image-disjoint split. The six special types are test-only: AdvCloak/YOLOv2, AdvCloak/YOLOv3, AdvTshirt/YOLOv2, AA/YOLOv2, AdvSticker/YOLOv3, UPC/YOLOv3.

The eight 11-detector families are AdvPatch, TC-EGA, T-SEA-PGD, T-SEA-MIM, TCA, T-SEA, GNAP, and DM-NAP. Exact storage names and assignments are in `split_plan.json`.

## Reconstruction and limitations

Images are padded with RGB 127 and resized to 416x416. Hiding patches are placed on each annotated person box: 20% of the box's long side (minimum 10 pixels), centered horizontally, vertical position sampled at 40%-60% of box height, and rotation within ±15 degrees. Appearing attacks use negative backgrounds and patch long sides of 72-128 pixels. Placement randomness is deterministic per source and patch type.

- AdvCloak/YOLOv3 was recovered from the embedded 150x250 patch image in Fig.3(j), page 7 of the [ECCV 2020 paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490001.pdf).
- AdvCloak/YOLOv2 and AdvTshirt/YOLOv2 are retained workspace artifacts without recovered optimizer histories.
- AA/YOLOv2 and AdvSticker/YOLOv3 are newly trained detector adaptations, not recovered original-author checkpoints.
- UPC/YOLOv3 is a newly trained dog-shaped adaptation of the [Faster-RCNN-based UPC implementation](https://github.com/mesunhlf/UPC-tf). Its YOLO objectness/class/box losses replace RPN/ROI terms. The reference dog comes from COCO train2017 image 69383, annotation 1504. Its effect is weak under this release's small-patch placement protocol.
- The three new adaptations use 2,000 optimizer steps on INRIA Train. Four missing GNAP combinations were retrained for 1,000 epochs with corrected latent quantization and Adam settings.
- Physical robustness and equality with the original paper's attack strength are not established by file-integrity checks.

On the first 100 fixed evaluation sources at confidence 0.5, AA and AdvSticker caused person detections at the patch on 84 and 99 negative images, respectively (clean: zero). UPC reduced GT matches at IoU 0.5 from 201/303 to 193/303, while raw person detections increased from 224 to 226. These are diagnostic counts, not mAP or physical-world ASR.

## Intended use

Reproducible study of patch localization, image restoration, and detector robustness in offline research. Evaluate on the documented split and report its limitations.

## Licensing and attribution — publisher action required

Before making this dataset repository public, complete the license metadata and document applicable redistribution terms for INRIA-Person, COCO source images, and third-party patch artifacts. The code repository's MIT badge does **not** establish a blanket license for all dataset images. No new dataset license is asserted by this card template. Retain source attribution when redistributing.

## Integrity

The source reconstruction passed image/mask/annotation consistency checks and patch-disjoint split checks. Export verifies original sample hashes, manifest membership, counts, and byte-preserving Parquet round-trips. `SHA256SUMS` covers Parquet shards; `export_report.json` records export completion. Use `sha256sum -c SHA256SUMS` after downloading the release files.
