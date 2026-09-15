"""Export the completed reconstruction to self-contained Hugging Face Parquet.

No network calls, repository creation, or uploads. Original PNG bytes are kept.
One shard per patch type; clean sources and patch images are separate configs.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apde_data.dataset import parse_labels, resolve_file


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_value(data, filename):
    return {'bytes': data, 'path': filename}


def export(source, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = source.resolve()
    output = output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output and source directories must be separate')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; interrupted exports are not publication-ready')
    audit = json.loads((source / 'audit_report.json').read_text())
    if not audit.get('complete') or audit.get('valid_types') != 94:
        raise ValueError('Source has not passed the complete 94-type audit')
    plan = json.loads((source / 'split_plan.json').read_text())['groups']
    if len(plan) != 94 or Counter(g['split'] for g in plan) != {'train': 57, 'test': 37}:
        raise ValueError('Expected 94 types with the 57/37 reconstruction split')
    output.mkdir(parents=True, exist_ok=True)
    checksums = {}

    def write_table(rows, relative, images):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        # HF Image feature metadata enables image and mask rendering in the viewer.
        features = {}
        for field in table.schema:
            if field.name in images:
                features[field.name] = {'_type': 'Image'}
            elif pa.types.is_string(field.type):
                features[field.name] = {'_type': 'Value', 'dtype': 'string'}
            else:
                raise ValueError(f'Unexpected export column: {field}')
        metadata = {b'huggingface': json.dumps({'info': {'features': features}}).encode()}
        table = table.replace_schema_metadata(metadata)
        temp = path.with_suffix('.parquet.tmp')
        pq.write_table(table, temp, compression='zstd', row_group_size=100)
        # Validate the schema, embedded bytes and row counts after serialization.
        restored = pq.read_table(temp)
        if not restored.equals(table, check_metadata=True):
            raise RuntimeError(f'Parquet round-trip mismatch: {relative}')
        os.replace(temp, path)
        checksums[relative] = sha(path.read_bytes())

    source_records = {}
    for kind in ['positive', 'negative']:
        records = json.loads((source / f'{kind}_sources.json').read_text())
        if len(records) != 1000:
            raise ValueError(f'Expected 1000 {kind} sources')
        clean = []
        for row in records:
            source_records[row['id']] = row
            # Use dataset-relative paths; never serialize source workstation paths.
            filename = row['id'] + '.png'
            raw = (source / 'clean' / kind / 'images' / filename).read_bytes()
            clean.append(dict(source_id=row['id'], source_dataset=row['dataset'],
                original_filename=Path(row['source']).name,
                original_sha256=row['source_sha256'], person_boxes_json=json.dumps(row['boxes']),
                image=image_value(raw, filename)))
        write_table(clean, f'clean/{kind}.parquet', {'image'})

    patch_rows = []
    public_plan = []
    totals = Counter()
    split_rows = {}
    for split in ['train', 'test']:
        entries = [json.loads(line) for line in (source / f'{split}.jsonl').read_text().splitlines()]
        split_rows[split] = Counter(json.dumps(r, sort_keys=True) for r in entries)
    consumed = {'train': Counter(), 'test': Counter()}
    for group in plan:
        dest = resolve_file(source, 'groups/' + group['key'])
        info = json.loads((dest / 'complete.json').read_text())
        manifest = (dest / 'samples.jsonl').read_bytes()
        if sha(manifest) != info['manifest_sha256']:
            raise ValueError(f'Manifest hash mismatch: {group["key"]}')
        patch = (dest / 'patch.png').read_bytes()
        if sha(patch) != info['patch_sha256']:
            raise ValueError(f'Patch hash mismatch: {group["key"]}')
        records = [json.loads(line) for line in manifest.splitlines()]
        if len(records) != 1000 or len({r['id'] for r in records}) != 1000:
            raise ValueError(f'Invalid sample count: {group["key"]}')
        output_rows = []
        for row in records:
            if row['split'] != group['split'] or row['group'] != group['key']:
                raise ValueError('Group/split mismatch')
            data = {k: resolve_file(source, row[k]).read_bytes() for k in ['image', 'mask', 'label']}
            for key, raw in data.items():
                if sha(raw) != row[key + '_sha256']:
                    raise ValueError(f'Content hash mismatch: {row[key]}')
            parse_labels(data['label'].decode())
            record = source_records[row['source_id']]
            output_rows.append(dict(id=f'{group["key"]}/{row["id"]}', source_id=row['source_id'],
                source_dataset=record['dataset'], patch_id=group['key'], method=group['method'],
                detector=group['detector'], attack_goal=group['task'], split=group['split'],
                patch_sha256=info['patch_sha256'], image_sha256=row['image_sha256'],
                mask_sha256=row['mask_sha256'], image=image_value(data['image'], row['id']+'.png'),
                mask=image_value(data['mask'], row['id']+'.mask.png'), labels=data['label'].decode()))
            consumed[group['split']][json.dumps(row, sort_keys=True)] += 1
        relative=f'data/{group["split"]}/{group["method"]}__{group["detector"]}.parquet'
        write_table(output_rows, relative, {'image', 'mask'})
        patch_rows.append(dict(patch_id=group['key'], method=group['method'], detector=group['detector'],
            split=group['split'], attack_goal=group['task'], sha256=info['patch_sha256'],
            image=image_value(patch, group['key'].replace('/', '__')+'.png')))
        public_plan.append({k: group[k] for k in ['key','method','detector','task','shape','split']})
        totals[group['split']] += len(output_rows)
        print(f'{group["key"]}: {len(output_rows)} rows exported', flush=True)
    if consumed != split_rows or totals != {'train': 57000, 'test': 37000}:
        raise ValueError('Export does not exactly match root train/test manifests')
    write_table(patch_rows, 'patches/patches.parquet', {'image'})
    (output/'split_plan.json').write_text(json.dumps(public_plan, indent=2))
    card=Path(__file__).resolve().parents[1]/'docs/HF_DATASET_CARD.md'
    (output/'README.md').write_text(card.read_text(encoding='utf-8'), encoding='utf-8')
    (output/'SHA256SUMS').write_text(''.join(f'{value}  {key}\n' for key,value in sorted(checksums.items())))
    (output/'export_report.json').write_text(json.dumps(dict(complete=True,version='2026-09-reconstruction',
        counts=dict(totals),clean_positive=1000,clean_negative=1000,patches=94,
        parquet_files=len(checksums),source_audit=audit),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.source,args.output)
