"""Regression tests for clue OCR against synthetic pill-badge images.

These mimic the real game's badge shape (rounded-cap pill, white-on-navy
text) closely enough to catch a real bug found during development: the
rounded cap's corner registering as a bogus extra digit. They intentionally
stick to short digit runs — Tesseract's accuracy is sensitive to the exact
font, and cv2.putText's font is only an approximation of the real game's, so
longer synthetic runs are prone to font-mismatch misreads that don't
reflect real behavior (verified separately against actual screenshots).
"""

import unittest

import cv2
import numpy as np

from vision.ocr import read_clue_numbers

_NAVY = (110, 60, 20)  # BGR
_ORANGE_BG = (140, 200, 245)  # BGR


def _make_row_badge(digits, cell=77, pill_width=59):
    """A horizontal pill: rounded cap on the left, flat on the right (where
    it meets the board), matching a row-clue badge."""
    height, width = cell, pill_width * 2 + 30
    img = np.full((height, width, 3), _ORANGE_BG, dtype=np.uint8)
    cv2.rectangle(img, (pill_width // 2, 4), (width - 4, height - 4), _NAVY, -1)
    cv2.circle(img, (pill_width // 2, height // 2), pill_width // 2 - 4, _NAVY, -1)
    text = "".join(str(d) for d in digits)
    cv2.putText(img, text, (pill_width, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def _make_column_badge(digits, cell=77, pill_width=59):
    """A vertical pill: rounded cap on top, flat on the bottom (where it
    meets the board), matching a column-clue badge with one digit per line."""
    height, width = cell * len(digits) + 40, pill_width
    img = np.full((height, width, 3), _ORANGE_BG, dtype=np.uint8)
    cv2.rectangle(img, (4, pill_width // 2), (width - 4, height - 4), _NAVY, -1)
    cv2.circle(img, (width // 2, pill_width // 2), pill_width // 2 - 4, _NAVY, -1)
    for i, digit in enumerate(digits):
        y = pill_width + i * cell + 45
        cv2.putText(img, str(digit), (width // 2 - 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return img


class TestOcr(unittest.TestCase):
    def test_reads_row_clue_digit_run(self):
        badge = _make_row_badge([2, 2])
        self.assertEqual(read_clue_numbers(badge, orientation="row"), [2, 2])

    def test_reads_column_clue_stacked_digits(self):
        badge = _make_column_badge([1, 2, 4, 5])
        self.assertEqual(read_clue_numbers(badge, orientation="column"), [1, 2, 4, 5])

    def test_reads_column_clue_single_digit(self):
        badge = _make_column_badge([8])
        self.assertEqual(read_clue_numbers(badge, orientation="column"), [8])

    def test_reads_column_clue_repeated_digit(self):
        badge = _make_column_badge([1, 1, 1, 1])
        self.assertEqual(read_clue_numbers(badge, orientation="column"), [1, 1, 1, 1])

    def test_empty_crop_returns_empty_list(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(read_clue_numbers(empty, orientation="row"), [])


if __name__ == "__main__":
    unittest.main()
