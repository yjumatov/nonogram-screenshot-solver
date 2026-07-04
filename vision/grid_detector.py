"""Locate the puzzle grid within a screenshot and produce a straightened image
plus the pixel geometry of its cells and clue regions.

Tuned for screenshots from a single mobile nonogram game where the board
renders as a large, roughly axis-aligned grid with thin per-cell gridlines,
row clues to the left of the board, and column clues above it. Because these
are digital screenshots rather than photos, "perspective correction" here is
mostly a straightening/crop step, but it still routes through a homography so
it degrades gracefully for screenshots with minor skew (e.g. a photo of a
screen instead of a native screenshot).

NOTE: the thresholds below are reasonable starting points, not values
calibrated against a real screenshot of the target game — they will likely
need tuning once a sample image is available.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class GridGeometry:
    """Pixel geometry of the detected board, in the coordinate space of the
    straightened grid image returned by `load_and_straighten`.
    """

    board_left: int
    board_top: int
    cell_size: float
    num_rows: int
    num_cols: int
    row_clue_region: Tuple[int, int, int, int]  # (x0, y0, x1, y1), left of the board
    col_clue_region: Tuple[int, int, int, int]  # (x0, y0, x1, y1), above the board

    def cell_bbox(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Pixel bounding box (x0, y0, x1, y1) of one board cell."""
        x0 = self.board_left + round(col * self.cell_size)
        x1 = self.board_left + round((col + 1) * self.cell_size)
        y0 = self.board_top + round(row * self.cell_size)
        y1 = self.board_top + round((row + 1) * self.cell_size)
        return x0, y0, x1, y1


def load_and_straighten(image_path: str) -> np.ndarray:
    """Load the screenshot and correct any perspective skew.

    Finds the largest quadrilateral contour (assumed to be the puzzle
    card/background) and warps it to a straight, axis-aligned rectangle. If
    no confident quadrilateral is found, the screenshot is assumed to already
    be axis-aligned and is returned unchanged.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * perimeter, True)

    image_area = image.shape[0] * image.shape[1]
    if len(approx) != 4 or cv2.contourArea(approx) < 0.2 * image_area:
        return image

    corners = _order_corners(approx.reshape(4, 2).astype("float32"))
    top_left, top_right, bottom_right, bottom_left = corners

    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Sort 4 arbitrary corner points into (top-left, top-right, bottom-right, bottom-left)."""
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    top_left = points[np.argmin(total)]
    bottom_right = points[np.argmax(total)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def detect_grid(image: np.ndarray) -> GridGeometry:
    """Find the board's cell grid lines and infer board size plus the
    clue-text regions to its left and above it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    row_lines = _find_grid_lines(binary.sum(axis=1))
    col_lines = _find_grid_lines(binary.sum(axis=0))
    if len(row_lines) < 2 or len(col_lines) < 2:
        raise ValueError("Could not detect a puzzle grid in this image.")

    board_top, board_bottom = row_lines[0], row_lines[-1]
    board_left, board_right = col_lines[0], col_lines[-1]

    num_rows = len(row_lines) - 1
    num_cols = len(col_lines) - 1
    cell_size = (
        (board_right - board_left) / num_cols + (board_bottom - board_top) / num_rows
    ) / 2

    return GridGeometry(
        board_left=board_left,
        board_top=board_top,
        cell_size=cell_size,
        num_rows=num_rows,
        num_cols=num_cols,
        row_clue_region=(0, board_top, board_left, board_bottom),
        col_clue_region=(board_left, 0, board_right, board_top),
    )


def _find_grid_lines(profile: np.ndarray, min_gap_fraction: float = 0.02) -> List[int]:
    """Collapse a row/column ink-density profile into grid-line pixel positions,
    merging peaks closer together than a minimum expected cell size.
    """
    if profile.max() == 0:
        return []

    threshold = profile.max() * 0.5
    candidates = np.where(profile > threshold)[0]
    if len(candidates) == 0:
        return []

    min_gap = max(int(len(profile) * min_gap_fraction), 3)
    lines = [int(candidates[0])]
    for position in candidates[1:]:
        if position - lines[-1] > min_gap:
            lines.append(int(position))
    return lines
