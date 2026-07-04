"""Classify a single board cell as filled (green), crossed (red X), or
unknown/empty — via HSV color segmentation, no machine learning.

NOTE: these HSV bounds are reasonable starting points for a typical
"green fill / red X" mobile nonogram theme, not values calibrated against a
real screenshot of the target game. They will likely need retuning once a
sample screenshot is available.
"""

import cv2
import numpy as np

from models.puzzle import CellState

# OpenCV hue range is 0-179.
_GREEN_HSV_LOW = np.array([35, 60, 60])
_GREEN_HSV_HIGH = np.array([85, 255, 255])

# Red wraps around hue 0, so it needs two ranges.
_RED_HSV_LOW_1 = np.array([0, 60, 60])
_RED_HSV_HIGH_1 = np.array([10, 255, 255])
_RED_HSV_LOW_2 = np.array([170, 60, 60])
_RED_HSV_HIGH_2 = np.array([179, 255, 255])

# Fraction of a cell's pixels that must match a color for it to count. Green
# cells are typically solid fills (high coverage); red cells are typically a
# thin X stroke (much lower coverage), so they get separate thresholds.
_GREEN_FRACTION_THRESHOLD = 0.15
_RED_FRACTION_THRESHOLD = 0.06


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

    if green_fraction >= _GREEN_FRACTION_THRESHOLD and green_fraction >= red_fraction:
        return CellState.FILLED
    if red_fraction >= _RED_FRACTION_THRESHOLD:
        return CellState.CROSS
    return CellState.UNKNOWN
