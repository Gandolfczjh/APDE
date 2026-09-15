# Publishing APDE on Hugging Face

This guide publishes **APDE**. Its 57,000/37,000 patch-disjoint split differs from the paper's 56,400/37,600 split.

## 1. Prepare an isolated publishing environment

On the server holding the dataset, clone/update this repository and use an isolated environment so the old detector environment's `huggingface_hub` and `transformers` dependencies remain usable:

```bash
git clone https://github.com/Gandolfczjh/APDE.git
cd APDE
python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-data.txt
```

## 2. Export a self-contained release

```bash
python tools/export_apde_hf.py \
  --source APDE \
  --output hf/APDE
```

The source must have a passing 94-type `audit_report.json`. Choose an empty output folder with sufficient disk space (roughly another dataset-sized allocation). This command performs no upload. For interrupted exports, repeat the command with `--resume`: each existing shard is compared against the source before reuse; differing shards are rejected. The exported dataset card is preserved so publisher edits are not lost. Four shards are processed concurrently by default; use `--workers 1` on memory-constrained hosts. Only an export with `export_report.json` containing `complete: true` is ready for release.

The export contains 97 Parquet files: 57 train shards, 37 test shards, two clean-source shards, and one 94-patch shard. Each image/mask keeps its original PNG bytes. This avoids uploading the raw tree's approximately 282,000 small sample files or duplicating its compatibility symlinks. Source server paths, process logs, environment paths, and checkpoints are excluded.

```text
APDE/
├── README.md                  # HF dataset card; complete publisher fields
├── data/train/*.parquet       # 57 shards, 57,000 samples
├── data/test/*.parquet        # 37 shards, 37,000 samples
├── clean/positive.parquet
├── clean/negative.parquet
├── patches/patches.parquet
├── split_plan.json
├── export_report.json
└── SHA256SUMS
```

Local smoke check:

```bash
python tools/verify_apde_hf.py hf/APDE
```

This verifies every shard checksum/count and checks Hugging Face image decoding. To load all rows locally:

```python
from datasets import load_dataset

data = load_dataset("parquet", data_files={
    "train": "hf/APDE/data/train/*.parquet",
    "test": "hf/APDE/data/test/*.parquet",
})
assert len(data["train"]) == 57000
assert len(data["test"]) == 37000
sample = data["test"][0]
assert sample["image"].size == sample["mask"].size == (416, 416)
```

## 3. Finish the dataset card

Edit the exported `README.md`:

- Replace `YOUR_HF_NAMESPACE` with your account or organization name.
- Set the appropriate dataset license metadata and source redistribution/attribution terms. Do not inherit the GitHub code's MIT badge for source photographs automatically.
- Retain the split differences, shared source-image caveat, and patch provenance/limitations.

See the official [dataset card guide](https://huggingface.co/docs/hub/datasets-cards).

## 4. Log in, create a dataset repository, and upload

Create an account if needed. Generate a token with write permission for your dataset repository, then enter it interactively; do not put the token into the scripts or Git repository.

```bash
hf auth login
hf auth whoami
hf repos create YOUR_HF_NAMESPACE/APDE --repo-type dataset --private
hf upload YOUR_HF_NAMESPACE/APDE hf/APDE --repo-type dataset
```

Alternatively create the repository at <https://huggingface.co/new-dataset>. Private staging is recommended while checking the card, license, and viewer. Once ready, switch visibility to public in the dataset repository's Settings. You may create it public initially if all publisher fields are already settled.

The current official CLI recommends `hf upload`, which supports large folders and resuming by repeating the same command. `hf upload-large-folder` is deprecated. Consult the [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload) and [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) for your installed version.

## 5. Verify and link the published release

```python
from datasets import load_dataset

data = load_dataset("YOUR_HF_NAMESPACE/APDE", "patched")
assert len(data["train"]) == 57000
assert len(data["test"]) == 37000
assert data["test"][0]["image"].size == (416, 416)
```

Verify the `patched`, `clean`, and `patches` configurations in the HF viewer, then replace the pending-download notice in both GitHub READMEs with the actual dataset link. The upload tool does not invent a namespace or claim publication before it occurs.
