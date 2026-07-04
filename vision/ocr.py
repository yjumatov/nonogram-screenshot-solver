"""Read clue numbers out of a cropped clue-strip image, e.g. "344" -> [3, 4, 4].

This game renders row clues as a horizontal run of digits with no separators
inside a pill to the left of the row (e.g. "344"), and column clues as a
vertical stack of single digits, one per line, inside a pill above the
column (e.g. "1215" read top-to-bottom as four separate lines). Either way,
the underlying clue values are single digits (0-9), so a bare run with no
separators is split one character at a time.

Clue text is white on a dark badge, which is the reverse of the usual
"dark text on light background" OCR assumption.
"""

import re
from typing import List

import cv2
import numpy as np
import pytesseract

_OCR_CORRECTIONS = {
    "I": "1", "l": "1", "|": "1", "i": "1",
    "S": "5", "s": "5",
    "O": "0", "o": "0",
    "B": "8",
}

_CHAR_WHITELIST = "0123456789IlSOoBi|"
# Row clues are one horizontal line of digits; column clues are several
# digits stacked as separate lines. Tesseract needs to be told which shape
# to expect.
_ROW_CONFIG = f"--psm 7 -c tessedit_char_whitelist={_CHAR_WHITELIST}"
_COLUMN_CONFIG = f"--psm 6 -c tessedit_char_whitelist={_CHAR_WHITELIST}"


def _correct_ocr_text(text: str) -> str:
    """Map commonly-confused OCR glyphs back to the digits they represent."""
    return "".join(_OCR_CORRECTIONS.get(char, char) for char in text)


def read_clue_numbers(clue_image: np.ndarray, orientation: str = "row") -> List[int]:
    """OCR a single clue strip/cell and return its clue values as a list of ints.

    orientation: "row" for a horizontal digit run (single text line), or
    "column" for digits stacked vertically (multiple text lines).
    """
    gray = cv2.cvtColor(clue_image, cv2.COLOR_BGR2GRAY) if clue_image.ndim == 3 else clue_image
    if gray.size == 0:
        return []

    # White text on a dark badge: THRESH_BINARY_INV maps the (bright) text
    # above the OTSU-picked threshold to black and the (dark) badge
    # background below it to white, producing the black-on-white image
    # Tesseract expects.
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    config = _ROW_CONFIG if orientation == "row" else _COLUMN_CONFIG
    raw_text = pytesseract.image_to_string(thresholded, config=config)
    corrected = _correct_ocr_text(raw_text)
    tokens = re.findall(r"\d+", corrected)
    if not tokens:
        return []

    if len(tokens) > 1:
        return [int(token) for token in tokens]

    return [int(digit) for digit in tokens[0]]
