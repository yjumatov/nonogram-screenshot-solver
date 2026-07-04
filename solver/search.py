"""Backtracking search layered on top of the unchanged line-solving engine.

Pure line-solving (solver/engine.py) proves everything it can without ever
guessing, which is enough for most puzzles but not all — some genuinely
require a guess-and-check step to fully resolve. This adds exactly that
without touching a single deduction rule in engine.py: guess a cell, re-run
the (unmodified) line solver with that guess seeded in, and keep the guess
if it doesn't raise ContradictionError; if it does, the opposite value is
forced. Recurses if line-solving still leaves cells undetermined after a
guess.
"""

from typing import List, Optional

from solver.engine import ContradictionError, solve_puzzle

Board = List[List[str]]


def _find_unknown_cell(board: Board):
    """Return the (row, col) of the first still-unknown cell, or None if fully solved."""
    for row_index, row in enumerate(board):
        for col_index, status in enumerate(row):
            if status == " ":
                return row_index, col_index
    return None


def _with_guess(board: Board, row: int, col: int, status: str) -> Board:
    """Return a copy of board with one cell set to status, for a backtracking guess."""
    guessed = [list(existing_row) for existing_row in board]
    guessed[row][col] = status
    return guessed


def solve_with_backtracking(
    row_clues: List[List[int]],
    col_clues: List[List[int]],
    initial_board: Optional[Board] = None,
) -> Optional[Board]:
    """Solve a nonogram, falling back to guess-and-check where pure
    line-solving can't fully determine the board on its own.

    Returns the solved board (2D list of "O"/"X" status characters), or
    None if the clues/board are contradictory — which a valid puzzle read
    correctly should never be, but guards against a bad vision-pipeline read
    rather than guessing forever.
    """
    try:
        board = solve_puzzle(row_clues, col_clues, initial_board)
    except ContradictionError:
        return None

    unknown = _find_unknown_cell(board)
    if unknown is None:
        return board

    row, col = unknown
    for guess in ("O", "X"):
        result = solve_with_backtracking(row_clues, col_clues, _with_guess(board, row, col, guess))
        if result is not None:
            return result

    return None
