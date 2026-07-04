"""Public, reusable API around the nonogram solving engine.

The UI and vision pipeline should only ever call `solve()` from this module —
never reach into `solver.engine` directly. This keeps the mature solving
algorithm in engine.py untouched and isolated behind a stable interface.
"""

from typing import List, Optional

from models.puzzle import CellState
from solver import engine

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

    Returns a 2D list (height x width) of CellState. Cells the solver
    couldn't fully determine remain CellState.UNKNOWN.
    """
    initial_board = None
    if current_board is not None:
        initial_board = [
            [_STATE_TO_CHAR[cell] for cell in row] for row in current_board
        ]

    result_chars = engine.solve_puzzle(row_clues, column_clues, initial_board)
    return [[_CHAR_TO_STATE[char] for char in row] for row in result_chars]
