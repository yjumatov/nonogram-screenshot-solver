"""Shared data types describing a nonogram puzzle as read from a screenshot."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class CellState(Enum):
    """State of a single board cell, as detected from a screenshot (or produced by the solver)."""

    UNKNOWN = " "
    FILLED = "O"
    CROSS = "X"


@dataclass
class Puzzle:
    """A nonogram puzzle extracted from a screenshot.

    row_clues[i] / column_clues[j] are lists of ints, e.g. [3] or [1, 2].
    board[i][j] is the CellState currently visible in row i, column j.
    """

    row_clues: List[List[int]]
    column_clues: List[List[int]]
    board: List[List[CellState]]

    @property
    def height(self) -> int:
        return len(self.row_clues)

    @property
    def width(self) -> int:
        return len(self.column_clues)
