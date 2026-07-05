"""Read clue numbers out of a cropped clue-badge image, e.g. "344" -> [3, 4, 4].

This game renders row clues as a horizontal run of digit glyphs with no
separator inside a pill to the left of the row, and column clues as a
vertical stack of digit glyphs, one clue value per line, inside a pill above
the column. Most clue values are a single digit (0-9), but a line with only
one block can have a value of 10 or more, rendered as an ordinary multi-digit
number ("11") — visually indistinguishable from several single-digit blocks
concatenated ("1", "1") since neither case uses a separator. Adjacent digits
are grouped into one clue value based on spacing (row) or shared line
(column); see _group_row_digit_boxes and _group_column_lines.

Rather than asking Tesseract to segment and read the whole badge in one
pass — which proved unreliable here (it would occasionally drop a digit,
misread one, or fold the badge's rounded-cap corner into a bogus extra
character) — each digit glyph is located ourselves via connected components
and OCR'd individually with Tesseract in single-character mode. This also
naturally discards the badge's rounded cap: its corner (left edge for row
badges, top edge for column badges) is page background peeking around the
curve, which shows up as its own connected component touching that edge and
is filtered out accordingly.
"""

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

_OCR_CORRECTIONS = {
    "I": "1", "l": "1", "|": "1", "i": "1",
    "S": "5", "s": "5",
    "O": "0", "o": "0",
    "B": "8",
    # This game's "1" glyph has a top serif/flag that Tesseract's
    # digit-only mode can't place at all (empty result) but its general
    # language model reads as one of these letters instead.
    "T": "1", "n": "1",
}

_CHAR_WHITELIST = "0123456789IlSOoBi|"
# No single Tesseract page-segmentation mode reads every digit glyph in this
# font reliably — psm 13 ("raw line") is needed for some glyphs (a flagged
# "1", a "5" that psm 10 drops entirely) but blank on others (a "0" that only
# psm 6 reads). Rather than chase one "correct" mode, try a short list, each
# with and without the digit whitelist, and take the first one that yields a
# digit — cheap, since each attempt is a single isolated glyph.
_DIGIT_PSM_MODES = (13, 6)

_MIN_DIGIT_AREA = 15  # pixels; discards antialiasing specks
_DIGIT_CROP_PADDING = 6
_OCR_BORDER_PADDING = 10  # Tesseract reads isolated glyphs more reliably with a quiet margin

# A row badge can hold either several single-digit blocks concatenated with
# no separator (e.g. "344" -> blocks 3, 4, 4) or, when a line has only one
# block and it's 10+, a single multi-digit number (e.g. "11" -> one block of
# 11) — visually ambiguous from digit shapes alone since neither uses a
# separator. The two cases differ in spacing: digits of one number are
# kerned tight, separate blocks sit further apart. Calibrated against real
# screenshots: gaps within one multi-digit number measured <=0.11x digit
# width; gaps between genuinely separate blocks measured >=0.27x. 0.2 sits
# with margin in between.
_SAME_NUMBER_MAX_GAP_RATIO = 0.2


def _correct_ocr_text(text: str) -> str:
    """Map commonly-confused OCR glyphs back to the digits they represent."""
    return "".join(_OCR_CORRECTIONS.get(char, char) for char in text)


def _find_digit_boxes(thresholded: np.ndarray, orientation: str) -> List[Tuple[int, int, int, int]]:
    """Locate each digit glyph in a black-text-on-white badge image, sorted
    in reading order, discarding artifacts from the badge crop's own edges.
    """
    height, width = thresholded.shape[:2]
    text_mask = 255 - thresholded
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)

    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < _MIN_DIGIT_AREA:
            continue
        # The rounded cap's corner (page background showing around the
        # curve) forms its own component flush against the capped edge:
        # left for row badges, top for column badges. Real digit glyphs sit
        # inset from that edge, so this never discards actual text.
        if orientation == "row" and x == 0:
            continue
        if orientation == "column" and y == 0:
            continue
        # The badge's opposite (flat, grid-facing) edge can likewise
        # produce a hairline artifact if the crop overshoots the badge by a
        # pixel or two, picking up a sliver of the (bright) board behind it.
        # A real glyph is always inset from every edge, so a component
        # flush against the crop boundary and needle-thin in one dimension
        # is never real text.
        touches_far_edge = (orientation == "row" and x + w >= width) or (
            orientation == "column" and y + h >= height
        )
        if touches_far_edge and min(w, h) <= 2:
            continue
        boxes.append((x, y, x + w, y + h))

    if orientation == "row":
        boxes.sort(key=lambda box: box[0])
        return boxes
    return _order_column_digits(boxes)


def _group_column_lines(boxes: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
    """Group column digit boxes into lines top to bottom, sorted
    left-to-right within a line. Column clues are normally one digit per
    line, but two can render side-by-side on the same line if that's more
    compact (e.g. "11") — those always belong to the same multi-digit clue
    value, since a column badge never puts two *separate* clue values on
    one line. Boxes whose y-ranges substantially overlap are treated as the
    same line; a plain sort by y alone would order same-line digits by
    whichever has the smaller y0, not by reading order.
    """
    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))  # (y0, x0)

    lines: List[List[Tuple[int, int, int, int]]] = []
    for box in boxes:
        _, y0, _, y1 = box
        if lines:
            _, line_y0, _, line_y1 = lines[-1][0]
            overlap = min(y1, line_y1) - max(y0, line_y0)
            shorter_height = min(y1 - y0, line_y1 - line_y0)
            if overlap > 0.5 * shorter_height:
                lines[-1].append(box)
                continue
        lines.append([box])

    return [sorted(line, key=lambda box: box[0]) for line in lines]


def _order_column_digits(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Flatten grouped column lines into overall reading order: top to
    bottom, left-to-right within a line."""
    ordered = []
    for line in _group_column_lines(boxes):
        ordered.extend(line)
    return ordered


def _padded_crop_bounds(
    box: Tuple[int, int, int, int],
    all_boxes: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Padding around a digit box, clamped so it never crosses into another
    digit — whichever direction that neighbor happens to be in.

    Digits are normally stacked vertically one per line, but two can render
    side-by-side on the same line instead (e.g. "10"), just a couple of
    pixels apart — closer than the padding alone would respect. Rather than
    assume a fixed layout, this checks every other box directly: one only
    constrains this box's left/right padding if it overlaps vertically
    (i.e. it's a horizontal neighbor), and only constrains top/bottom if it
    overlaps horizontally (i.e. it's a vertical neighbor).
    """
    x0, y0, x1, y1 = box
    min_x0, max_x1, min_y0, max_y1 = 0, width, 0, height
    for other in all_boxes:
        if other is box:
            continue
        ox0, oy0, ox1, oy1 = other
        if min(y1, oy1) - max(y0, oy0) > 0:  # vertical overlap -> horizontal neighbor
            if ox1 <= x0:
                min_x0 = max(min_x0, (ox1 + x0) // 2)
            if ox0 >= x1:
                max_x1 = min(max_x1, (x1 + ox0) // 2)
        if min(x1, ox1) - max(x0, ox0) > 0:  # horizontal overlap -> vertical neighbor
            if oy1 <= y0:
                min_y0 = max(min_y0, (oy1 + y0) // 2)
            if oy0 >= y1:
                max_y1 = min(max_y1, (y1 + oy0) // 2)

    cx0 = max(x0 - _DIGIT_CROP_PADDING, min_x0)
    cy0 = max(y0 - _DIGIT_CROP_PADDING, min_y0)
    cx1 = min(x1 + _DIGIT_CROP_PADDING, max_x1)
    cy1 = min(y1 + _DIGIT_CROP_PADDING, max_y1)
    return cx0, cy0, cx1, cy1


def _group_row_digit_boxes(
    boxes: List[Tuple[int, int, int, int]]
) -> List[List[Tuple[int, int, int, int]]]:
    """Group left-to-right row digit boxes into one sublist per clue value:
    tightly-spaced adjacent digits are the same multi-digit number, a wider
    gap starts a new (separate) block clue. See _SAME_NUMBER_MAX_GAP_RATIO.
    """
    groups: List[List[Tuple[int, int, int, int]]] = []
    for box in boxes:
        if groups:
            prev = groups[-1][-1]
            gap = box[0] - prev[2]
            width = min(prev[2] - prev[0], box[2] - box[0])
            if width > 0 and gap / width <= _SAME_NUMBER_MAX_GAP_RATIO:
                groups[-1].append(box)
                continue
        groups.append([box])
    return groups


def read_clue_numbers(clue_image: np.ndarray, orientation: str = "row") -> List[int]:
    """OCR a single clue badge and return its clue values as a list of ints.

    orientation: "row" for a horizontal digit run, or "column" for digits
    stacked vertically. Determines reading order, which edge of the badge is
    the rounded cap to discard, and how digits are grouped into clue values
    (see _group_row_digit_boxes / _group_column_lines).
    """
    if clue_image.size == 0:
        return []

    gray = cv2.cvtColor(clue_image, cv2.COLOR_BGR2GRAY) if clue_image.ndim == 3 else clue_image
    if gray.size == 0:
        return []

    # White text on a dark badge: THRESH_BINARY_INV maps the (bright) text
    # above the OTSU-picked threshold to black and the (dark) badge
    # background below it to white, producing the black-on-white image
    # Tesseract expects.
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = thresholded.shape[:2]
    digit_boxes = _find_digit_boxes(thresholded, orientation)
    groups = _group_row_digit_boxes(digit_boxes) if orientation == "row" else _group_column_lines(digit_boxes)

    clue_values = []
    for group in groups:
        digit_chars = []
        for box in group:
            cx0, cy0, cx1, cy1 = _padded_crop_bounds(box, digit_boxes, width, height)
            digit_crop = thresholded[cy0:cy1, cx0:cx1]
            digit_crop = cv2.copyMakeBorder(
                digit_crop,
                _OCR_BORDER_PADDING, _OCR_BORDER_PADDING, _OCR_BORDER_PADDING, _OCR_BORDER_PADDING,
                cv2.BORDER_CONSTANT, value=255,
            )
            digit = _ocr_single_digit(digit_crop)
            if digit is not None:
                digit_chars.append(str(digit))
        if digit_chars:
            clue_values.append(int("".join(digit_chars)))

    return clue_values


def _ocr_single_digit(digit_crop: np.ndarray) -> Optional[int]:
    """OCR one isolated digit glyph, or return None if no attempt reads a digit."""
    for psm in _DIGIT_PSM_MODES:
        for whitelist in (_CHAR_WHITELIST, None):
            config = f"--psm {psm}"
            if whitelist:
                config += f" -c tessedit_char_whitelist={whitelist}"
            raw_text = pytesseract.image_to_string(digit_crop, config=config)
            match = re.search(r"\d", _correct_ocr_text(raw_text))
            if match:
                return int(match.group())

    return None
