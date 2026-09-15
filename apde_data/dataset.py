import json
from pathlib import Path

from PIL import Image


def resolve_file(root, relative):
    """Only accept dataset-relative files contained in root."""
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f'Expected a relative dataset path: {relative}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves dataset root: {relative}')
    return path


def parse_labels(text):
    result = {'person_boxes': [], 'patch_boxes': []}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] not in {'person', 'patch'}:
            raise ValueError(f'Invalid annotation: {line}')
        box = list(map(float, fields[1:]))
        if not (0 <= box[0] < box[2] <= 416 and 0 <= box[1] < box[3] <= 416):
            raise ValueError(f'Invalid box: {line}')
        result[fields[0] + '_boxes'].append(box)
    return result


class APDEDataset:
    """Map-style reader returning RGB/L PIL images, boxes, and metadata.

    Pass a callable transform(sample) for tensor conversion/augmentation.
    With torch DataLoader, use a custom collate_fn for variable-length boxes.
    """
    def __init__(self, root, split='train', transform=None):
        if split not in {'train', 'test'}:
            raise ValueError('split must be train or test')
        self.root = Path(root).resolve()
        self.transform = transform
        with (self.root / f'{split}.jsonl').open(encoding='utf-8') as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if any(row['split'] != split for row in self.records):
            raise ValueError('Manifest contains a different split')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(resolve_file(self.root, row['image'])) as image:
            rgb = image.convert('RGB')
        with Image.open(resolve_file(self.root, row['mask'])) as image:
            mask = image.convert('L')
        labels = parse_labels(resolve_file(self.root, row['label']).read_text())
        sample = dict(id=f'{row["group"]}/{row["id"]}', source_id=row['source_id'],
                      patch_id=row['group'], split=row['split'], attack_goal=row['task'],
                      image=rgb, mask=mask, **labels)
        return self.transform(sample) if self.transform else sample
