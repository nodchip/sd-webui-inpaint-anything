"""Operations on the unified Inpaint Anything selection mask."""

import cv2
import numpy as np


class MaskEditor:
    """Keep the binary mask and a bounded history at image resolution."""

    def __init__(self, mask, history_limit=16):
        self.history_limit = history_limit
        self.reset(mask)

    def reset(self, mask):
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional image")
        self.mask = np.where(mask != 0, 255, 0).astype(np.uint8)
        self.history = []

    def clear_history(self):
        self.history.clear()

    def _commit(self, new_mask):
        if np.array_equal(self.mask, new_mask):
            return False
        self.history.append(self.mask.copy())
        if len(self.history) > self.history_limit:
            self.history.pop(0)
        self.mask = new_mask
        return True

    def _stroke(self, points, width):
        points = np.asarray(points, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("stroke must contain image coordinates")
        width = max(1, int(width))
        stroke = np.zeros(self.mask.shape, np.uint8)
        if len(points) > 1:
            cv2.polylines(stroke, [points], False, 1, width, lineType=cv2.LINE_8)
        radius = width // 2
        for point in (points[0], points[-1]):
            cv2.circle(stroke, tuple(point), radius, 1, -1, lineType=cv2.LINE_8)
        return stroke.astype(bool)

    def select(self, labels, points, width=1, ignore_black=True):
        if labels is None or labels.shape != self.mask.shape:
            raise ValueError("SAM regions do not match the mask")
        stroke = self._stroke(points, width)
        indices = np.unique(labels[stroke])
        if ignore_black:
            indices = indices[indices != 0]
        updated = self.mask.copy()
        for index in indices:
            region = labels == index
            updated[region] = 0 if np.all(self.mask[region] != 0) else 255
        return self._commit(updated)

    def brush(self, points, width, add):
        stroke = self._stroke(points, width)
        updated = self.mask.copy()
        updated[stroke] = 255 if add else 0
        return self._commit(updated)

    def expand(self, iterations):
        iterations = max(1, min(100, int(iterations)))
        return self._commit(cv2.dilate(self.mask, np.ones((3, 3), np.uint8), iterations=iterations))

    def invert(self):
        return self._commit(cv2.bitwise_not(self.mask))

    def clear(self):
        return self._commit(np.zeros_like(self.mask))

    def clear(self):
        return self._commit(np.zeros_like(self.mask))

    def undo(self):
        if not self.history:
            return False
        self.mask = self.history.pop()
        return True

    def rgb_mask(self):
        return np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)
