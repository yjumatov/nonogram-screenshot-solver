"""Read clue numbers out of a cropped clue-strip image, e.g. "344" -> [3, 4, 4].

This game renders clue lists as either space-separated numbers ("2 2 1 5")
or, when every clue is a single digit, a run of digits with no separators
("2215" -> [2, 2, 1, 5]). We assume clues never exceed 9 (true for the board
sizes this game uses), so a bare digit run is split one character at a time.
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

_TESSERACT_CONFIG = "--psm 6 -c tessedit_char_whitelist=0123456789IlSOoBi|"


def _correct_ocr_text(text: str) -> str:
    """Map commonly-confused OCR glyphs back to the digits they represent."""
    return "".join(_OCR_CORRECTIONS.get(char, char) for char in text)


def read_clue_numbers(clue_image: np.ndarray) -> List[int]:
    """OCR a single clue strip/cell and return its clue values as a list of ints."""
    gray = cv2.cvtColor(clue_image, cv2.COLOR_BGR2GRAY) if clue_image.ndim == 3 else clue_image
    if gray.size == 0:
        return []

    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    raw_text = pytesseract.image_to_string(thresholded, config=_TESSERACT_CONFIG)
    corrected = _correct_ocr_text(raw_text)
    tokens = re.findall(r"\d+", corrected)
    if not tokens:
        return []

    if len(tokens) > 1:
        return [int(token) for token in tokens]

    return [int(digit) for digit in tokens[0]]
