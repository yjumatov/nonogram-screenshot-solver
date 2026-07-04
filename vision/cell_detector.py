"""Classify a single board cell as filled (green), crossed (red X), or
unknown/empty — via HSV color segmentation, no machine learning.

In this game, both states are solid-colored rounded tiles covering most of
the cell: filled cells are a lime-green tile, crossed cells are a brick-red
tile with a white "X" glyph cut out of it (not a thin red line on a white
background). Cells that are neither — plain white, or covered by the
decorative "not yet revealed" texture used for untouched cells — are unknown.
"""

import cv2
import numpy as np

from models.puzzle import CellState

# OpenCV hue range is 0-179.
_GREEN_HSV_LOW = np.array([35, 60, 60])
_GREEN_HSV_HIGH = np.array([85, 255, 255])

# Red wraps around hue 0, so it needs two ranges. Lower saturation bound is
# forgiving since the game's cross tiles are a muted brick red, not pure red.
_RED_HSV_LOW_1 = np.array([0, 50, 60])
_RED_HSV_HIGH_1 = np.array([10, 255, 255])
_RED_HSV_LOW_2 = np.array([170, 50, 60])
_RED_HSV_HIGH_2 = np.array([179, 255, 255])

# Fraction of a cell's pixels that must match a color for it to count. Both
# states are solid tiles (minus rounded corners and, for cross cells, the
# white X glyph cut out of the middle), so they share one generous-but-safe
# threshold well below full coverage.
_COLOR_FRACTION_THRESHOLD = 0.3


def classify_cell(cell_image: np.ndarray) -> CellState:
    """Return the CellState best matching a cropped cell image."""
    if cell_image.size == 0:
        return CellState.UNKNOWN

    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    green_mask = cv2.inRange(hsv, _GREEN_HSV_LOW, _GREEN_HSV_HIGH)
    red_mask = cv2.inRange(hsv, _RED_HSV_LOW_1, _RED_HSV_HIGH_1) | cv2.inRange(
        hsv, _RED_HSV_LOW_2, _RED_HSV_HIGH_2
    )

    green_fraction = cv2.countNonZero(green_mask) / total_pixels
    red_fraction = cv2.countNonZero(red_mask) / total_pixels

    if green_fraction >= _COLOR_FRACTION_THRESHOLD and green_fraction >= red_fraction:
        return CellState.FILLED
    if red_fraction >= _COLOR_FRACTION_THRESHOLD:
        return CellState.CROSS
    return CellState.UNKNOWN
