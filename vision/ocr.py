"""Read clue numbers out of a cropped clue-badge image, e.g. "344" -> [3, 4, 4].

This game renders row clues as a horizontal run of digits with no separators
inside a pill to the left of the row (e.g. "344"), and column clues as a
vertical stack of single digits, one per line, inside a pill above the
column (e.g. "1215" read top-to-bottom as four separate digits). Either way,
every clue value is a single digit (0-9).

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
# psm 13 ("raw line", bypassing Tesseract's usual segmentation heuristics)
# proved the most reliable for a single isolated digit glyph; psm 10
# ("single character") looks like the more obviously "correct" mode but
# empirically dropped some digits (e.g. "5") that psm 13 reads fine.
_DIGIT_CONFIG = f"--psm 13 -c tessedit_char_whitelist={_CHAR_WHITELIST}"
# Fallback when the whitelisted pass reads nothing at all: some glyphs in
# this font (a flagged "1") aren't recognized as any digit even loosely, so
# retry with no character restriction and lean on _OCR_CORRECTIONS instead.
_FALLBACK_DIGIT_CONFIG = "--psm 13"

_MIN_DIGIT_AREA = 15  # pixels; discards antialiasing specks
_DIGIT_CROP_PADDING = 6
_OCR_BORDER_PADDING = 10  # Tesseract reads isolated glyphs more reliably with a quiet margin


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

    sort_key = (lambda box: box[1]) if orientation == "column" else (lambda box: box[0])
    boxes.sort(key=sort_key)
    return boxes


def read_clue_numbers(clue_image: np.ndarray, orientation: str = "row") -> List[int]:
    """OCR a single clue badge and return its clue values as a list of ints.

    orientation: "row" for a horizontal digit run, or "column" for digits
    stacked vertically. Determines reading order and which edge of the
    badge is the rounded cap to discard.
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

    digits = []
    height, width = thresholded.shape[:2]
    for x0, y0, x1, y1 in _find_digit_boxes(thresholded, orientation):
        cx0 = max(x0 - _DIGIT_CROP_PADDING, 0)
        cy0 = max(y0 - _DIGIT_CROP_PADDING, 0)
        cx1 = min(x1 + _DIGIT_CROP_PADDING, width)
        cy1 = min(y1 + _DIGIT_CROP_PADDING, height)
        digit_crop = thresholded[cy0:cy1, cx0:cx1]
        digit_crop = cv2.copyMakeBorder(
            digit_crop,
            _OCR_BORDER_PADDING, _OCR_BORDER_PADDING, _OCR_BORDER_PADDING, _OCR_BORDER_PADDING,
            cv2.BORDER_CONSTANT, value=255,
        )

        digit = _ocr_single_digit(digit_crop)
        if digit is not None:
            digits.append(digit)

    return digits


def _ocr_single_digit(digit_crop: np.ndarray) -> Optional[int]:
    raw_text = pytesseract.image_to_string(digit_crop, config=_DIGIT_CONFIG)
    match = re.search(r"\d", _correct_ocr_text(raw_text))
    if match:
        return int(match.group())

    fallback_text = pytesseract.image_to_string(digit_crop, config=_FALLBACK_DIGIT_CONFIG)
    match = re.search(r"\d", _correct_ocr_text(fallback_text))
    if match:
        return int(match.group())

    return None
