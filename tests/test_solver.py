"""Smoke tests for the solver API, including partial-board seeding."""

import unittest

from models.puzzle import CellState
from solver.api import solve

U, F, X = CellState.UNKNOWN, CellState.FILLED, CellState.CROSS


class TestSolver(unittest.TestCase):
    def test_solves_simple_puzzle_from_blank_board(self):
        # A 3x3 "plus sign":
        #  . O .
        #  O O O
        #  . O .
        row_clues = [[1], [3], [1]]
        column_clues = [[1], [3], [1]]

        result = solve(row_clues, column_clues)

        expected = [
            [X, F, X],
            [F, F, F],
            [X, F, X],
        ]
        self.assertEqual(result, expected)

    def test_solves_when_seeded_with_partial_board(self):
        row_clues = [[1], [3], [1]]
        column_clues = [[1], [3], [1]]
        current_board = [
            [U, U, U],
            [U, F, U],  # one cell already known filled, as if read from a screenshot
            [U, U, U],
        ]

        result = solve(row_clues, column_clues, current_board)

        expected = [
            [X, F, X],
            [F, F, F],
            [X, F, X],
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
