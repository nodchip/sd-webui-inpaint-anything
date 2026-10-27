import ast
import importlib.util
from pathlib import Path
import types
import unittest

import cv2
import numpy as np


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mask_editor', root / 'inpalib/mask_editor.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source = ast.parse((root / 'scripts/inpaint_anything.py').read_text(encoding='utf-8'))
functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in ('editor_epoch_is_obsolete', 'input_image_upload', 'run_padding', 'run_sam')]
for function in functions:
    function.decorator_list = []


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.previous = np.full((4, 5, 3), 20, np.uint8)
        self.regions = [{'segmentation': np.ones((4, 5), bool)}]
        self.state = {'orig_image': self.previous, 'sam_masks': self.regions,
                      'sam_color_image': np.zeros_like(self.previous), 'sam_region_labels': object(),
                      'pad_mask': None, 'mask_image': np.full_like(self.previous, 255), 'mask_editor': None,
                      'editor_epoch': 'client:1'}
        self.env = {'sam_dict': self.state, 'np': np, 'cv2': cv2, 'MaskEditor': module.MaskEditor,
                    'USE_FORGE_CANVAS': True,
                    'get_sketch_image': lambda value: value,
                    'create_sam_preview': lambda image, color: cv2.addWeighted(image, .5, color, .5, 0),
                    'inpalib': types.SimpleNamespace(create_seg_color_image=lambda image, masks: np.zeros_like(image)),
                    'gr': types.SimpleNamespace(update=lambda **kwargs: kwargs),
                    'ia_logging': types.SimpleNamespace(info=lambda *args: None),
                    'get_webui_setting': lambda *args: 127}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<upload>', 'exec'), self.env)

    def test_same_size_keeps_regions_and_resets_mask_and_history(self):
        image = np.full_like(self.previous, 40)
        result = self.env['input_image_upload'](image, self.previous)
        self.assertIs(result[1], True)
        self.assertIs(self.state['sam_masks'], self.regions)
        self.assertEqual(np.count_nonzero(self.state['mask_image']), 0)
        self.assertEqual(np.count_nonzero(self.state['mask_editor'].mask), 0)
        self.assertFalse(self.state['mask_editor'].undo())

    def test_different_size_discards_regions_and_initializes_editor(self):
        image = np.zeros((3, 6, 3), np.uint8)
        self.env['input_image_upload'](image, self.previous)
        self.assertIsNone(self.state['sam_masks'])
        self.assertIsNone(self.state['sam_region_labels'])
        self.assertEqual(self.state['mask_editor'].mask.shape, (3, 6))

    def test_padding_updates_editor_and_preserves_original_and_padding_mask(self):
        result = self.env['run_padding'](self.previous, 1.2, 1, .5, .5, 'edge', 'padded')
        self.assertEqual(result[0].shape, (4, 6, 3))
        self.assertIs(result[3], True)
        self.assertIs(self.state['orig_image'], self.previous)
        self.assertEqual(self.state['editor_epoch'], 'padded')
        self.assertEqual(self.state['mask_editor'].mask.shape, (4, 6))
        self.assertEqual(self.state['pad_mask']['segmentation'].shape, (4, 6))
        self.assertEqual(np.count_nonzero(self.state['pad_mask']['segmentation']), 4)
        self.assertIsNone(self.state['sam_masks'])

    def test_older_upload_does_not_replace_newer_image(self):
        current = np.full_like(self.previous, 40)
        self.env['input_image_upload'](current, self.previous, 'client:3')
        self.env['input_image_upload'](self.previous, current, 'client:2')
        self.assertIs(self.state['orig_image'], current)
        self.assertEqual(self.state['editor_epoch'], 'client:3')

    def test_stale_padding_and_sam_do_not_replace_current_state(self):
        self.state['editor_epoch'] = 'client:3'
        padding = self.env['run_padding'](self.previous, 1.2, 1, .5, .5, 'edge', 'client:2')
        sam = self.env['run_sam'](self.previous, 'unused', self.previous, False, 'client:2')
        self.assertEqual(len(padding), 5)
        self.assertIn('cancelled', sam[1])
        self.assertIs(self.state['orig_image'], self.previous)
        self.assertIs(self.state['sam_masks'], self.regions)


if __name__ == '__main__':
    unittest.main()
