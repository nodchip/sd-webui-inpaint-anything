import ast
import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import types
import unittest

import numpy as np
from PIL import Image


root = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / 'inpalib' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


masklib, editorlib = load('masklib'), load('mask_editor')
source = ast.parse((root / 'scripts/inpaint_anything.py').read_text(encoding='utf-8'))
names = {'editor_epoch_is_obsolete', 'editor_png', 'editor_state', 'editor_command', 'editor_recover'}
functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
for function in functions:
    function.decorator_list = []


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        region = np.zeros((4, 5), bool)
        region[1:3, 1:4] = True
        self.state = {'editor_epoch': 'first', 'mask_editor': editorlib.MaskEditor(np.zeros((4, 5), np.uint8)),
                      'mask_image': np.zeros((4, 5, 3), np.uint8), 'sam_masks': [{'segmentation': region}],
                      'sam_region_labels': None, 'sam_color_image': np.zeros((4, 5, 3), np.uint8)}
        self.env = {'sam_dict': self.state, 'np': np, 'base64': base64, 'BytesIO': BytesIO, 'Image': Image,
                    'json': json, 'inpalib': masklib, 'ia_logging': types.SimpleNamespace(error=lambda *a: None)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<editor>', 'exec'), self.env)

    def send(self, **kwargs):
        return json.loads(self.env['editor_command'](json.dumps(kwargs)))

    def test_select_updates_output_and_rgb_mask(self):
        result = self.send(epoch='first', seq=1, action='select', points=[[2, 2]], width=1, ignore_black=True)
        self.assertEqual(result['seq'], 1)
        mask = np.array(Image.open(BytesIO(base64.b64decode(result['mask']))))
        self.assertEqual(np.count_nonzero(mask), 6)
        np.testing.assert_array_equal(mask, self.state['mask_image'][:, :, 0])
        self.assertNotIn('colors', result)

    def test_stale_commands_do_not_modify_mask_or_history(self):
        result = self.send(epoch='old', seq=1, action='invert')
        self.assertTrue(result['stale'])
        self.assertEqual(np.count_nonzero(self.state['mask_image']), 0)
        self.assertEqual(len(self.state['mask_editor'].history), 0)

    def test_recovery_invalidates_previous_generation_preserving_mask_and_undo(self):
        self.send(epoch='first', seq=1, action='invert')
        result = json.loads(self.env['editor_recover']('second'))
        self.assertEqual(result['epoch'], 'second')
        self.assertEqual(np.count_nonzero(self.state['mask_image']), 60)
        self.assertTrue(self.send(epoch='first', seq=2, action='undo')['stale'])
        self.send(epoch='second', seq=3, action='undo')
        self.assertEqual(np.count_nonzero(self.state['mask_image']), 0)

    def test_error_does_not_change_mask(self):
        result = self.send(epoch='first', seq=1, action='unknown')
        self.assertIn('error', result)
        self.assertEqual(np.count_nonzero(self.state['mask_image']), 0)

    def test_old_recovery_cannot_replace_newer_epoch(self):
        self.state['editor_epoch'] = 'client:3'
        result = json.loads(self.env['editor_recover']('client:2'))
        self.assertEqual(result['epoch'], 'client:3')
        self.assertEqual(self.state['editor_epoch'], 'client:3')


if __name__ == '__main__':
    unittest.main()
