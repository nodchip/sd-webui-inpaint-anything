import importlib.util
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "inpalib" / "masklib.py"
spec = importlib.util.spec_from_file_location("inpaint_anything_masklib", path)
masklib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masklib)


class CreateMaskTests(unittest.TestCase):
    def assert_mask(self, brush, masks, ignore_black, expected):
        labels = masklib.create_region_labels(masks, brush.shape[:2])
        expected_rgb = np.repeat(expected[:, :, np.newaxis], 3, axis=2)
        for cached in (None, labels):
            actual = masklib.create_mask_image(brush, masks, ignore_black, cached)
            np.testing.assert_array_equal(actual, expected_rgb)

    def test_overlap_priority_and_black_area(self):
        first = np.zeros((3, 4), dtype=bool)
        first[:2, :2] = True
        second = np.zeros((3, 4), dtype=bool)
        second[:2, 1:3] = True
        masks = [{"segmentation": first}, {"segmentation": second}]

        brush = np.zeros((3, 4), dtype=np.uint8)
        brush[0, 1] = 255
        self.assert_mask(brush, masks, True, first.astype(np.uint8) * 255)

        brush[:] = 0
        brush[0, 2] = 255
        self.assert_mask(brush, masks, True, (second & ~first).astype(np.uint8) * 255)

        brush[:] = 0
        brush[2, 3] = 255
        self.assert_mask(brush, masks, True, np.zeros((3, 4), dtype=np.uint8))
        self.assert_mask(brush, masks, False, (~(first | second)).astype(np.uint8) * 255)

    def test_more_than_255_regions(self):
        shape = (23, 23)
        masks = []
        for index in range(520):
            segmentation = np.zeros(shape, dtype=bool)
            segmentation.flat[index] = True
            masks.append({"segmentation": segmentation})
        brush = np.zeros(shape, dtype=np.uint8)
        brush.flat[255] = 255
        brush.flat[519] = 255
        expected = np.zeros(shape, dtype=np.uint8)
        expected.flat[255] = 255
        expected.flat[519] = 255
        self.assert_mask(brush, masks, True, expected)

    def test_empty_regions_and_empty_brush(self):
        brush = np.zeros((3, 4), dtype=np.uint8)
        self.assert_mask(brush, [], True, np.zeros_like(brush))
        self.assert_mask(brush, [], False, np.zeros_like(brush))
        brush[0, 0] = 255
        self.assert_mask(brush, [], True, np.zeros_like(brush))
        self.assert_mask(brush, [], False, np.full_like(brush, 255))

    def test_nonbinary_regions_use_compatible_fallback(self):
        segmentation = np.array([[2, 0], [0, 0]], dtype=np.uint8)
        masks = [{"segmentation": segmentation}]
        brush = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        self.assertIsNone(masklib.create_region_labels(masks, brush.shape))
        expected = np.array([[254, 0], [0, 0]], dtype=np.uint8)
        self.assert_mask(brush, masks, True, expected)

    def test_float_brush_uses_compatible_fallback(self):
        segmentation = np.array([[False, False], [False, True]])
        masks = [{"segmentation": segmentation}]
        labels = masklib.create_region_labels(masks, segmentation.shape)
        brush = np.array([[np.nan, 0], [0, 0]], dtype=np.float32)
        expected = np.array([[0, 0], [0, 255]], dtype=np.uint8)
        actual = masklib.create_mask_image(brush, masks, True, labels)
        np.testing.assert_array_equal(actual, np.repeat(expected[:, :, None], 3, axis=2))


if __name__ == "__main__":
    unittest.main()
