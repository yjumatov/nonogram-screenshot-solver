"""Tests for the backtracking search layer on top of the line-solving engine."""

import unittest

from solver.engine import solve_puzzle
from solver.search import solve_with_backtracking

# Real clues from an actual game screenshot where pure line-solving stalls
# with 92 cells still undetermined (verified separately) — a genuine case
# for backtracking, not a hand-crafted one.
ROW_CLUES = [
    [3, 2], [2, 1], [4, 1], [6, 1, 3], [5, 1, 2], [2, 3, 4], [7, 1, 1], [7, 4],
    [1, 1, 2, 2], [4, 1], [3, 2, 1, 2], [4, 1, 3], [2, 3, 1, 2], [1, 4, 3],
    [3, 2, 1, 1, 1],
]
COLUMN_CLUES = [
    [1, 9], [5, 4, 1], [9, 2], [3, 2, 1, 3], [7, 1, 3], [1, 2, 4, 5], [8], [2],
    [1], [1, 1, 1, 1], [2, 1], [2, 3], [3, 2, 5], [5, 4], [1, 1, 1, 1, 2],
]


class TestSearch(unittest.TestCase):
    def test_line_solving_alone_leaves_this_puzzle_unsolved(self):
        # Establishes the premise: this specific puzzle is a real case where
        # the unmodified engine needs help, so the backtracking test below
        # is actually exercising the search and not just line-solving.
        board = solve_puzzle(ROW_CLUES, COLUMN_CLUES)
        unknown_count = sum(1 for row in board for cell in row if cell == " ")
        self.assertGreater(unknown_count, 0)

    def test_backtracking_fully_solves_it(self):
        board = solve_with_backtracking(ROW_CLUES, COLUMN_CLUES)
        self.assertIsNotNone(board)
        unknown_count = sum(1 for row in board for cell in row if cell == " ")
        self.assertEqual(unknown_count, 0)

        # The solved board must actually satisfy every row/column clue.
        for row, clue in zip(board, ROW_CLUES):
            self.assertEqual(_runs(row), clue)
        for col_index, clue in enumerate(COLUMN_CLUES):
            column = [row[col_index] for row in board]
            self.assertEqual(_runs(column), clue)

    def test_contradictory_board_returns_none(self):
        # Row clue [1] needs one filled cell, but both cells are pre-seeded
        # crossed out — no valid placement exists.
        result = solve_with_backtracking([[1]], [[1], [1]], [["X", "X"]])
        self.assertIsNone(result)


def _runs(line):
    """Convert a solved line (list of 'O'/'X') into its clue-style run lengths."""
    runs = []
    count = 0
    for status in line:
        if status == "O":
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs or [0]


if __name__ == "__main__":
    unittest.main()
