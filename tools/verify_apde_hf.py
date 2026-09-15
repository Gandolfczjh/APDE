"""Validate exported shard hashes/counts and decode each HF image configuration."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import DatasetCard

    report = json.loads((root/'export_report.json').read_text())
    if not report.get('complete'):
        raise ValueError('Export is incomplete')
    expected_files = set()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        checksum, relative = line.split('  ', 1)
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Checksum path outside export root')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f'Hash mismatch: {relative}')
        expected_files.add(path)
    if {p.resolve() for p in root.rglob('*.parquet')} != expected_files:
        raise ValueError('Unexpected or missing Parquet files')
    expected = {'data/train':57000,'data/test':37000,'clean':2000,'patches':94}
    for folder, count in expected.items():
        paths=sorted((root/folder).glob('*.parquet'))
        actual=sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
        if actual!=count:
            raise ValueError(f'{folder}: expected {count}, found {actual}')
        data=load_dataset('parquet',data_files=[str(p) for p in paths],split='train',streaming=True)
        row=next(iter(data))
        if row['image'].mode!='RGB':
            raise ValueError(f'Image feature did not decode as RGB: {folder}')
        if folder.startswith('data/'):
            if row['image'].size!=(416,416) or row['mask'].size!=(416,416):
                raise ValueError('Invalid decoded image dimensions')
            if set(row['mask'].getdata())!={0,255}:
                raise ValueError('Invalid binary mask')
        print(f'{folder}: {actual} rows; Hugging Face image decoding passed')
    card=DatasetCard.load(root/'README.md')
    if {config['config_name'] for config in card.data.configs}!={'patched','clean','patches'}:
        raise ValueError('Dataset card config mismatch')
    if len(expected_files)!=97:
        raise ValueError('Expected 97 shards')
    print('All 97 shard hashes, counts, card configurations and decoding checks passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    verify(parser.parse_args().root)
