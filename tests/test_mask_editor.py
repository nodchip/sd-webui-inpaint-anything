import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load(name):
    path = Path(__file__).resolve().parents[1] / 'inpalib' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


masklib = load('masklib')
MaskEditor = load('mask_editor').MaskEditor


class MaskEditorTests(unittest.TestCase):
    def setUp(self):
        first = np.zeros((8, 12), dtype=bool)
        second = np.zeros_like(first)
        first[1:7, 1:6] = True
        second[1:7, 4:10] = True
        self.labels = masklib.create_region_labels(
            [{'segmentation': first}, {'segmentation': second}], first.shape)
        self.editor = MaskEditor(np.zeros(first.shape, np.uint8))

    def test_overlap_and_repeated_drag_toggle_each_visible_region_once(self):
        self.editor.select(self.labels, [(4, 3)])
        self.assertTrue(np.all(self.editor.mask[self.labels == 1] == 255))
        self.assertTrue(np.all(self.editor.mask[self.labels == 2] == 0))
        before = self.editor.mask.copy()
        self.editor.select(self.labels, [(2, 3), (8, 3), (2, 3), (8, 3)])
        self.assertTrue(np.all(self.editor.mask[self.labels == 1] == 0))
        self.assertTrue(np.all(self.editor.mask[self.labels == 2] == 255))
        self.editor.undo()
        np.testing.assert_array_equal(self.editor.mask, before)

    def test_brush_partial_region_fill_clear_and_undo(self):
        self.editor.brush([(2, 2)], 1, True)
        self.editor.select(self.labels, [(2, 2)])
        self.assertTrue(np.all(self.editor.mask[self.labels == 1] == 255))
        self.editor.select(self.labels, [(2, 2)])
        self.assertTrue(np.all(self.editor.mask[self.labels == 1] == 0))
        self.editor.undo()
        self.assertTrue(np.all(self.editor.mask[self.labels == 1] == 255))
        self.editor.brush([(2, 2)], 1, False)
        self.assertEqual(self.editor.mask[2, 2], 0)
        self.assertEqual(self.editor.mask[3, 2], 255)

    def test_black_region_and_brush_width(self):
        self.assertFalse(self.editor.select(self.labels, [(0, 0)]))
        self.assertEqual(len(self.editor.history), 0)
        self.editor.select(self.labels, [(0, 0)], ignore_black=False)
        self.assertTrue(np.all(self.editor.mask[self.labels == 0] == 255))
        self.editor.reset(np.zeros_like(self.labels))
        self.editor.select(self.labels, [(6, 3)], width=5)
        self.assertTrue(np.all(self.editor.mask[self.labels == 1] == 255))
        self.assertTrue(np.all(self.editor.mask[self.labels == 2] == 255))

    def test_expand_invert_undo_and_reset_history(self):
        self.editor.select(self.labels, [(2, 3)])
        before = self.editor.mask.copy()
        self.editor.expand(1)
        self.assertTrue(np.any(self.editor.mask[self.labels == 2]))
        self.editor.invert()
        self.editor.undo()
        self.editor.undo()
        np.testing.assert_array_equal(self.editor.mask, before)
        self.editor.clear()
        self.assertEqual(np.count_nonzero(self.editor.mask), 0)
        self.editor.undo()
        np.testing.assert_array_equal(self.editor.mask, before)
        self.editor.clear_history()
        self.assertFalse(self.editor.undo())
        np.testing.assert_array_equal(self.editor.mask, before)
        self.editor.reset(np.zeros_like(before))
        self.assertEqual(np.count_nonzero(self.editor.mask), 0)

    def test_history_limit_and_rgb_output(self):
        editor = MaskEditor(np.zeros((2, 3), np.uint8), history_limit=2)
        for _ in range(3):
            editor.invert()
        self.assertTrue(editor.undo())
        self.assertTrue(editor.undo())
        self.assertFalse(editor.undo())
        self.assertEqual(editor.rgb_mask().shape, (2, 3, 3))
        self.assertTrue(np.all(editor.mask == 255))


if __name__ == '__main__':
    unittest.main()
