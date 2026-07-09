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
# font reliably. psm 6 is tried first since it's the more generally
# accurate of the two — psm 13 ("raw line") has been observed to
# confidently misread an otherwise-unambiguous "5" as "7", a wrong answer
# that "prefer any single-character result" (see _ocr_single_digit) can't
# catch on its own. psm 13 stays as a fallback since some glyphs need it (a
# flagged "1", a "0" that psm 6 alone leaves blank). Each mode is tried with
# and without the digit whitelist — cheap, since each attempt is a single
# isolated glyph.
_DIGIT_PSM_MODES = (6, 13)

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

# Used when checking for a dimmed-text cluster (see _clue_text_threshold):
# how far above the background's own gray level to start looking, so its
# own anti-aliasing tail isn't mistaken for a second cluster.
_BACKGROUND_TAIL_MARGIN = 20
# How many pixels a candidate cluster needs at its single brightest gray
# level to count as a genuine dimmed digit rather than anti-aliasing noise
# around a curved glyph's edge (e.g. "0", "8", "9" all shed a fair number of
# stray mid-brightness pixels along their curve, which can otherwise look
# like a small second cluster). Calibrated against real screenshots: a
# curved glyph's anti-aliasing noise peaked at ~114 pixels; a genuine dimmed
# digit peaked at ~437.
_MIN_DIM_CLUSTER_PEAK = 150

# See _looks_like_five_not_seven.
_FIVE_VS_SEVEN_BOTTOM_LEFT_INK_RATIO = 0.5


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

    boxes = _drop_undersized_fragments(boxes)

    if orientation == "row":
        boxes.sort(key=lambda box: box[0])
        return boxes
    return _order_column_digits(boxes)


# Relative to the median digit height in a badge (see _drop_undersized_fragments).
_MIN_DIGIT_HEIGHT_RATIO = 0.5


def _drop_undersized_fragments(
    boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """Discard a component much shorter than its badge-mates.

    Every digit glyph in this font renders at the same height within one
    badge, so a component well short of that height is a stray artifact,
    not a genuine glyph. This catches cases the edge-touching filters above
    miss: a real screenshot had a cap-corner sliver (background peeking
    through the pill's rounded corner) that touched the badge's top edge
    without touching its left edge — the specific combination those filters
    check for — and got read as a bogus extra "1", inflating that row's
    clue by one. Comparing against the *other digits actually found in this
    badge* catches any such fragment regardless of which edge or corner it
    happens to touch.
    """
    if len(boxes) < 2:
        return boxes
    heights = sorted(y1 - y0 for _, y0, _, y1 in boxes)
    median_height = heights[len(heights) // 2]
    return [box for box in boxes if (box[3] - box[1]) >= _MIN_DIGIT_HEIGHT_RATIO * median_height]


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


def _clue_text_threshold(gray: np.ndarray) -> int:
    """Pick the black/white cutoff separating a badge's solid background
    from its text.

    This game dims a clue value's text (a distinct, noticeably darker gray
    rather than pure white) once that block is already satisfied on the
    board — so a badge can have two different "text" brightness levels at
    once, not one. Plain Otsu assumes only two clusters (text vs
    background) and, with three actually present, has been observed to
    lump the dimmed digit in with the background, dropping it entirely.

    This starts from Otsu's own threshold, then checks the gap between the
    background's level and that threshold for a genuine second cluster —
    distinguished from ordinary anti-aliasing noise (which curved glyphs
    like "0"/"8"/"9" shed a fair amount of) by requiring a real peak, not
    just scattered low counts (_MIN_DIM_CLUSTER_PEAK). If one is found, the
    cutoff moves to the valley between the background and it, so both text
    tiers are captured; otherwise Otsu's own threshold is used unchanged,
    since lowering it needlessly has been observed to measurably hurt
    Tesseract's accuracy even on already-clean glyphs.
    """
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    background_level = int(np.argmax(histogram))
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_threshold = int(otsu_threshold)

    search_start = background_level + _BACKGROUND_TAIL_MARGIN
    if otsu_threshold <= search_start:
        return otsu_threshold

    # Inclusive of otsu_threshold itself: Otsu's own cutoff groups gray
    # values <= it with the background, so a cleanly-separated dim cluster
    # sitting exactly at that boundary value would otherwise be excluded
    # from this search by one.
    window = histogram[search_start : otsu_threshold + 1]
    if window.max() < _MIN_DIM_CLUSTER_PEAK:
        return otsu_threshold

    dim_cluster_peak = search_start + int(np.argmax(window))
    if dim_cluster_peak <= search_start:
        return otsu_threshold

    valley = histogram[search_start:dim_cluster_peak]
    return search_start + int(np.argmin(valley))


def _looks_like_five_not_seven(tight_digit_crop: np.ndarray) -> bool:
    """Whether a digit Tesseract read as "7" is actually a "5".

    This font's "5" has been observed to be confidently misread as "7" by
    psm 13 with no other mode corroborating either reading to fall back on
    (unlike other 5-vs-7 mix-ups psm-mode-ordering alone resolves). The two
    shapes are reliably distinguishable by how much ink sits in the
    bottom-left quadrant: a "5"'s bottom curve wraps back to the left,
    while a "7"'s diagonal stroke leaves that corner almost empty.
    Calibrated across every "5" and "7" in the real screenshot corpus
    (resized to a fixed size first, to remove any bias from crops of
    different pixel scale): genuine "7"s measured a ratio of 0.36-0.468,
    genuine "5"s measured 0.46-0.59 — comfortably separated by 0.5.
    """
    if tight_digit_crop.size == 0:
        return False
    resized = cv2.resize(tight_digit_crop, (40, 40), interpolation=cv2.INTER_NEAREST)
    bottom_left = resized[20:, :20]
    ink_ratio = (bottom_left == 0).sum() / bottom_left.size
    return ink_ratio > _FIVE_VS_SEVEN_BOTTOM_LEFT_INK_RATIO


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

    # White text on a dark badge: THRESH_BINARY_INV maps text brighter than
    # the threshold to black and the (dark) badge background below it to
    # white, producing the black-on-white image Tesseract expects.
    threshold_value = _clue_text_threshold(gray)
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

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
            if digit == 7 and _looks_like_five_not_seven(thresholded[box[1] : box[3], box[0] : box[2]]):
                digit = 5
            if digit is not None:
                digit_chars.append(str(digit))
        if digit_chars:
            clue_values.append(int("".join(digit_chars)))

    return clue_values


def _ocr_single_digit(digit_crop: np.ndarray) -> Optional[int]:
    """OCR one isolated digit glyph, or return None if no attempt reads a digit.

    Tries every psm/whitelist combination and prefers any single-character
    result over a multi-character one. A cleanly segmented isolated glyph
    should always read as exactly one character; a psm mode occasionally
    hallucinates a bogus extra stroke off this font's flagged "1" glyph as
    its own leading digit (e.g. reading "41" for a lone "1"), and picking
    "the first digit found" in that string grabs the spurious one. A
    multi-character result is itself a sign of that split, so it's only
    used as a last resort if no attempt yields a clean single digit.
    """
    fallback = None
    for psm in _DIGIT_PSM_MODES:
        for whitelist in (_CHAR_WHITELIST, None):
            config = f"--psm {psm}"
            if whitelist:
                config += f" -c tessedit_char_whitelist={whitelist}"
            raw_text = pytesseract.image_to_string(digit_crop, config=config)
            corrected = _correct_ocr_text(raw_text).strip()
            if len(corrected) == 1 and corrected.isdigit():
                return int(corrected)
            if fallback is None:
                match = re.search(r"\d", corrected)
                if match:
                    fallback = int(match.group())

    return fallback
