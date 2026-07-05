"""Locate the puzzle grid within a screenshot and produce a straightened image
plus the pixel geometry of its cells and clue badges.

Tuned for screenshots from a single mobile nonogram game where the board
renders as a large, roughly axis-aligned grid, with a dark-navy pill-shaped
clue badge to the left of every row and above every column. Because these
are digital screenshots rather than photos, "perspective correction" here is
mostly a straightening/crop step, but it still routes through a homography so
it degrades gracefully for screenshots with minor skew (e.g. a photo of a
screen instead of a native screenshot).

Board size and cell geometry are derived from the clue badges rather than
from counting interior gridlines. That's deliberate: cells that haven't been
solved yet can be covered in a decorative texture (a preview of the hidden
picture), and this game only draws the thick every-5-cells separator lines
over that texture, not the thin per-cell ones — so counting interior lines
undercounts the board whenever a puzzle has unsolved cells in the middle of
it. A clue badge, in contrast, is always fully rendered for every row and
column regardless of solved state, so counting *those* is robust.
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

    board_left: float
    board_top: float
    cell_width: float
    cell_height: float
    num_rows: int
    num_cols: int
    row_clue_boxes: List[Tuple[int, int, int, int]]  # one per row, (x0, y0, x1, y1)
    col_clue_boxes: List[Tuple[int, int, int, int]]  # one per column, (x0, y0, x1, y1)

    def cell_bbox(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Pixel bounding box (x0, y0, x1, y1) of one board cell."""
        x0 = round(self.board_left + col * self.cell_width)
        x1 = round(self.board_left + (col + 1) * self.cell_width)
        y0 = round(self.board_top + row * self.cell_height)
        y1 = round(self.board_top + (row + 1) * self.cell_height)
        return x0, y0, x1, y1


# A found quadrilateral only gets warped if it's already nearly
# axis-aligned (each corner within this fraction of the image's diagonal
# from where an axis-aligned rectangle's corner would be). Native app
# screenshots are never genuinely skewed, so a quad that's *far* from
# axis-aligned is a false-positive contour match (e.g. the grid's own
# irregular silhouette from its rounded clue badges) rather than real
# perspective distortion — warping to it does active damage.
_MAX_CORNER_SKEW_FRACTION = 0.03


def load_and_straighten(image_path: str) -> np.ndarray:
    """Load the screenshot and correct any perspective skew.

    Finds the largest quadrilateral contour (assumed to be the puzzle
    card/background) and warps it to a straight, axis-aligned rectangle —
    but only if that quadrilateral is already close to axis-aligned (see
    _MAX_CORNER_SKEW_FRACTION). If no confident, near-rectangular
    quadrilateral is found, the screenshot is assumed to already be
    axis-aligned and is returned unchanged.
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
    if not _is_nearly_axis_aligned(corners, image.shape):
        return image

    top_left, top_right, bottom_right, bottom_left = corners

    width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def _is_nearly_axis_aligned(corners: np.ndarray, image_shape) -> bool:
    """Whether an ordered (TL, TR, BR, BL) quad is close enough to its own
    axis-aligned bounding box to be a real (near-zero) perspective
    correction rather than a false-positive contour match."""
    x_coords, y_coords = corners[:, 0], corners[:, 1]
    bounding_box_corners = np.array([
        [x_coords.min(), y_coords.min()],
        [x_coords.max(), y_coords.min()],
        [x_coords.max(), y_coords.max()],
        [x_coords.min(), y_coords.max()],
    ])
    diagonal = np.hypot(image_shape[0], image_shape[1])
    max_offset = np.linalg.norm(corners - bounding_box_corners, axis=1).max()
    return max_offset <= _MAX_CORNER_SKEW_FRACTION * diagonal


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Sort 4 arbitrary corner points into (top-left, top-right, bottom-right, bottom-left)."""
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    top_left = points[np.argmin(total)]
    bottom_right = points[np.argmax(total)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


# Clue badges are a dark navy blue: darker than the white/colored cells and
# the page background, but distinctly lighter than the grid's lines in most
# theme variants. That brightness gap isn't reliable on its own though — some
# theme variants render grid lines as a dark *gray* (not pure black) whose
# brightness overlaps the badge's, and gray isn't distinguishable from navy
# by value alone. Requiring some actual color (R/G/B channels spread apart,
# not equal) is what keeps navy badges separate from any dark neutral-gray
# grid line, regardless of that line's exact brightness.
_NAVY_MIN_VALUE = 25
_NAVY_MAX_VALUE = 150
_NAVY_MIN_SATURATION = 15

# A badge must be at least this big (relative to image area) to count as a
# real clue badge rather than compression/anti-aliasing noise.
_MIN_BADGE_AREA_FRACTION = 0.001

# Badges are identified by which edge of the image they hug: row badges sit
# flush against the left edge, column badges flush against the top edge.
# The offset margins are deliberately generous (not a hairline cutoff): a
# real badge's detected bounding box can shift by a few pixels depending on
# exact rendering/masking, and a too-tight cutoff has been observed to miss
# a real badge sitting just barely past it.
_ROW_BADGE_MAX_LEFT_OFFSET_FRACTION = 0.03
_COL_BADGE_MAX_TOP_OFFSET_FRACTION = 0.08

# A badge is a solid pill shape, so its pixel count should fill most of its
# own bounding box. This rejects a different artifact than the area/position
# filters: an anti-aliased ring where the board's outer border blends into a
# colored (non-white) page background can incidentally match the navy color
# range all the way around the board, but as a thin outline it fills only a
# small fraction of its (board-sized) bounding box.
_MIN_BADGE_FILL_RATIO = 0.5


def _navy_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of pixels in the clue badges' navy color range."""
    blue, green, red = image[:, :, 0].astype(np.int16), image[:, :, 1].astype(np.int16), image[:, :, 2].astype(np.int16)
    channel_max = image.max(axis=2)
    channel_min = image.min(axis=2)
    is_in_value_range = (channel_max > _NAVY_MIN_VALUE) & (channel_max < _NAVY_MAX_VALUE)
    is_colored = (channel_max - channel_min) >= _NAVY_MIN_SATURATION
    # "Navy" is fundamentally a shade of blue, so blue must be the strongest
    # channel. Without this, an anti-aliased pixel where the board's outer
    # border blends into a colored (e.g. green) page background can land in
    # the same dark-and-colored bucket as a real badge, forming a thin ring
    # around the whole board that bridges unrelated badges into one blob.
    is_blue_dominant = (blue >= green) & (blue >= red)
    mask = (is_in_value_range & is_colored & is_blue_dominant).astype(np.uint8) * 255
    # An opening (erode then dilate) clears out thin noise and breaks any
    # accidental single-pixel bridges between a badge and something else.
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _find_clue_badges(image: np.ndarray):
    """Find every row-clue and column-clue badge via connected components on
    a navy-color mask, classifying each blob by shape and position. Returns
    (row_boxes, col_boxes) sorted in reading order, each box (x0, y0, x1, y1).
    """
    height, width = image.shape[:2]
    mask = _navy_mask(image)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = _MIN_BADGE_AREA_FRACTION * width * height
    row_badges, col_badges = [], []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if area < _MIN_BADGE_FILL_RATIO * w * h:
            continue
        cx, cy = centroids[i]
        # Row badges are wide-and-short and hug the left edge; column badges
        # are tall-and-narrow and hug the top edge.
        if x < _ROW_BADGE_MAX_LEFT_OFFSET_FRACTION * width and w > h:
            row_badges.append((cy, x, y, x + w, y + h))
        elif y < _COL_BADGE_MAX_TOP_OFFSET_FRACTION * height and h > w:
            col_badges.append((cx, x, y, x + w, y + h))

    row_badges.sort(key=lambda b: b[0])
    col_badges.sort(key=lambda b: b[0])
    row_boxes = [box for _, *box in row_badges]
    col_boxes = [box for _, *box in col_badges]
    return row_boxes, col_boxes


def detect_grid(image: np.ndarray) -> GridGeometry:
    """Find the board's size and cell geometry from its row/column clue badges."""
    row_boxes, col_boxes = _find_clue_badges(image)
    if len(row_boxes) < 2 or len(col_boxes) < 2:
        raise ValueError("Could not detect a puzzle grid in this image.")

    row_centers = [(y0 + y1) / 2 for _, y0, _, y1 in row_boxes]
    col_centers = [(x0 + x1) / 2 for x0, _, x1, _ in col_boxes]
    cell_height = float(np.median(np.diff(row_centers)))
    cell_width = float(np.median(np.diff(col_centers)))

    # Badge centers line up with cell centers, so the board's top-left
    # corner is half a cell above/left of the first row/column's center.
    board_top = row_centers[0] - cell_height / 2
    board_left = col_centers[0] - cell_width / 2

    return GridGeometry(
        board_left=board_left,
        board_top=board_top,
        cell_width=cell_width,
        cell_height=cell_height,
        num_rows=len(row_boxes),
        num_cols=len(col_boxes),
        row_clue_boxes=row_boxes,
        col_clue_boxes=col_boxes,
    )
