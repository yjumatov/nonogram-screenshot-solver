"""Public, reusable API around the nonogram solving engine.

The UI and vision pipeline should only ever call `solve()` from this module —
never reach into `solver.engine` directly. This keeps the mature solving
algorithm in engine.py untouched and isolated behind a stable interface.
"""

from typing import List, Optional

from models.puzzle import CellState
from solver.search import solve_with_backtracking

# CellState <-> the single-character status codes the engine works with internally.
_STATE_TO_CHAR = {
    CellState.UNKNOWN: " ",
    CellState.FILLED: "O",
    CellState.CROSS: "X",
}
_CHAR_TO_STATE = {char: state for state, char in _STATE_TO_CHAR.items()}


def solve(
    row_clues: List[List[int]],
    column_clues: List[List[int]],
    current_board: Optional[List[List[CellState]]] = None,
) -> List[List[CellState]]:
    """Solve a nonogram, optionally starting from a partially-known board.

    row_clues:      e.g. [[3], [1, 2], [5]]
    column_clues:   e.g. [[2], [4], [1, 1]]
    current_board:  2D list (height x width) of CellState, e.g. the board as
                     read off a screenshot. Cells that are CellState.UNKNOWN
                     are left for the solver to determine. Pass None to solve
                     from a completely blank board.

    Line-solving alone doesn't fully determine every puzzle; where it stalls,
    this falls back to guess-and-check (solver.search) until the board is
    fully solved or a guess is proven wrong.

    Returns a 2D list (height x width) of CellState, fully solved unless the
    given clues/board are contradictory.
    """
    row_total = sum(sum(clue) for clue in row_clues)
    col_total = sum(sum(clue) for clue in column_clues)
    if row_total != col_total:
        # Every valid nonogram satisfies this identity (both sides count the
        # same filled cells, just grouped by row vs by column), so a mismatch
        # means the read clues themselves are inconsistent — most likely a
        # screenshot taken mid-load/mid-animation before all clue badges had
        # rendered, rather than a misread digit (which this can't localize).
        raise ValueError(
            f"The row clues add up to {row_total} filled cells but the column clues add up to "
            f"{col_total} — these can't both be right for the same board. This usually means the "
            "screenshot was captured before the puzzle finished loading/rendering; try again with "
            "a fresh screenshot of the fully-loaded puzzle."
        )

    initial_board = None
    if current_board is not None:
        initial_board = [
            [_STATE_TO_CHAR[cell] for cell in row] for row in current_board
        ]

    result_chars = solve_with_backtracking(row_clues, column_clues, initial_board)
    if result_chars is None:
        raise ValueError("These clues and current board are contradictory — no valid solution exists.")

    return [[_CHAR_TO_STATE[char] for char in row] for row in result_chars]
