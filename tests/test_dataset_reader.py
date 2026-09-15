import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from apde_data import APDEDataset
from apde_data.dataset import parse_labels, resolve_file


class DatasetReaderTest(unittest.TestCase):
    def test_read_actual_image_modes_and_variable_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (416, 416), 'red').save(root/'image.png')
            Image.new('L', (416, 416), 255).save(root/'mask.png')
            (root/'label.txt').write_text('person 1 2 300 400\npatch 50 60 90 100\n')
            row = dict(id='one', source_id='coco_one', group='AA/yolov2', split='test',
                       task='appearing', image='image.png', mask='mask.png', label='label.txt')
            (root/'test.jsonl').write_text(json.dumps(row)+'\n')
            reader = APDEDataset(root, 'test')
            self.assertEqual(len(reader), 1)
            item = reader[0]
            self.assertEqual((item['image'].mode,item['mask'].mode), ('RGB','L'))
            self.assertEqual(item['patch_boxes'], [[50.,60.,90.,100.]])
            row['split']='train'
            (root/'test.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError): APDEDataset(root,'test')

    def test_path_and_annotation_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in ['../outside.png',str(root/'absolute.png')]:
                with self.assertRaises(ValueError): resolve_file(root,path)
        for label in ['patch 5 5 1 1','person 0 0 417 416','patch nan 0 1 1','unknown 0 0 1 1']:
            with self.assertRaises(ValueError): parse_labels(label)
        self.assertEqual(parse_labels(''), {'person_boxes':[],'patch_boxes':[]})


if __name__=='__main__': unittest.main()
