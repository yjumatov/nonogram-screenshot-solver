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

from vision.ocr import _clue_text_threshold, _group_row_digit_boxes, _order_column_digits, read_clue_numbers

_NAVY = (110, 60, 20)  # BGR
_ORANGE_BG = (140, 200, 245)  # BGR


def _make_row_badge(digits, cell=77, pill_width=59, block_gap=30):
    """A horizontal pill: rounded cap on the left, flat on the right (where
    it meets the board), matching a row-clue badge. Each digit is drawn
    separately with a visible gap between them, matching the real game's
    spacing between separate single-digit blocks (a single multi-digit clue
    value would be drawn with normal tight kerning instead — see
    read_clue_numbers's digit-grouping logic)."""
    height, width = cell, pill_width * len(digits) + block_gap * len(digits) + 30
    img = np.full((height, width, 3), _ORANGE_BG, dtype=np.uint8)
    cv2.rectangle(img, (pill_width // 2, 4), (width - 4, height - 4), _NAVY, -1)
    cv2.circle(img, (pill_width // 2, height // 2), pill_width // 2 - 4, _NAVY, -1)
    x = pill_width
    for digit in digits:
        cv2.putText(img, str(digit), (x, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        x += block_gap
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

    def test_reads_row_clue_with_a_tightly_kerned_multi_digit_number(self):
        # digits drawn as one cv2.putText string, with real font kerning,
        # unlike _make_row_badge's separate-block spacing above — matches a
        # lone block of size 11 rather than blocks 1 and 1.
        height, width = 77, 150
        img = np.full((height, width, 3), _ORANGE_BG, dtype=np.uint8)
        cv2.rectangle(img, (30, 4), (width - 4, height - 4), _NAVY, -1)
        cv2.circle(img, (30, height // 2), 26, _NAVY, -1)
        cv2.putText(img, "11", (55, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        self.assertEqual(read_clue_numbers(img, orientation="row"), [11])

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


class TestGroupRowDigitBoxes(unittest.TestCase):
    """Adjacent row digits are one multi-digit clue value (e.g. a lone block
    of size 11) when tightly kerned, or separate single-digit blocks (e.g.
    blocks 1 and 1) when spaced further apart — distinguished by gap-to-width
    ratio, calibrated against real screenshots."""

    def test_tightly_spaced_digits_are_one_group(self):
        # "11" as a single two-digit number: gap 2px vs digit width 18px.
        boxes = [(31, 12, 49, 48), (51, 12, 69, 48)]
        self.assertEqual(_group_row_digit_boxes(boxes), [boxes])

    def test_widely_spaced_digits_are_separate_groups(self):
        # "4 2" as two separate blocks: gap 13px vs digit width 39px.
        boxes = [(80, 16, 119, 67), (132, 16, 164, 68)]
        self.assertEqual(_group_row_digit_boxes(boxes), [[boxes[0]], [boxes[1]]])

    def test_single_digit_is_its_own_group(self):
        boxes = [(10, 10, 30, 50)]
        self.assertEqual(_group_row_digit_boxes(boxes), [boxes])


class TestOrderColumnDigits(unittest.TestCase):
    """Column digits are normally stacked one per line, but two can render
    side-by-side on the same line instead (e.g. "10") — reading order must
    still come out top-to-bottom, then left-to-right within a line."""

    def test_stacked_digits_read_top_to_bottom(self):
        boxes = [(10, 100, 30, 140), (10, 10, 30, 50), (10, 55, 30, 95)]
        self.assertEqual(
            _order_column_digits(boxes),
            [(10, 10, 30, 50), (10, 55, 30, 95), (10, 100, 30, 140)],
        )

    def test_side_by_side_digits_read_left_to_right(self):
        # A left-to-right pair sharing a line ("10"), positioned so that a
        # naive sort-by-y0 would (wrongly) put the right-hand box first.
        left = (6, 148, 23, 182)
        right = (25, 147, 53, 184)
        self.assertEqual(_order_column_digits([right, left]), [left, right])


class TestClueTextThreshold(unittest.TestCase):
    """This game dims a clue digit's text once its block is already
    satisfied on the board, so a badge can have two distinct text
    brightness levels instead of one — a real screenshot exposed a bug
    where plain Otsu thresholding lumped the dimmed digit's cluster in with
    the background and dropped it entirely."""

    def test_includes_a_dimmed_digit_cluster(self):
        image = np.full((100, 100), 60, dtype=np.uint8)  # navy background
        image[10:20, 10:50] = 130  # dimmed ("already satisfied") digit
        image[60:90, 10:90] = 250  # full-brightness digit
        threshold = _clue_text_threshold(image)
        self.assertGreater(threshold, 60)
        self.assertLessEqual(threshold, 130)

    def test_ignores_curved_glyph_antialiasing_noise(self):
        # A curved glyph (e.g. "0") sheds a modest amount of scattered
        # mid-brightness edge noise that must not be mistaken for a real
        # second text tier.
        rng = np.random.default_rng(0)
        image = np.full((100, 100), 60, dtype=np.uint8)
        noise_coords = rng.integers(10, 90, size=(40, 2))
        for x, y in noise_coords:
            image[y, x] = 100
        image[60:90, 10:90] = 250
        otsu_expected, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.assertEqual(_clue_text_threshold(image), int(otsu_expected))

    def test_single_text_tier_matches_plain_otsu(self):
        image = np.full((100, 100), 60, dtype=np.uint8)
        image[10:90, 10:90] = 250
        otsu_expected, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.assertEqual(_clue_text_threshold(image), int(otsu_expected))


if __name__ == "__main__":
    unittest.main()
