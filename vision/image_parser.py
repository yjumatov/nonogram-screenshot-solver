"""Top-level vision pipeline orchestration.

parse_puzzle(): screenshot path -> Puzzle (clues + current board state).
render_solution(): the solved board -> the original screenshot with newly
solved cells overlaid on top.
"""

from PIL import Image, ImageDraw

from models.puzzle import CellState, Puzzle
from vision import grid_detector
from vision.cell_detector import classify_cell
from vision.grid_detector import GridGeometry
from vision.ocr import read_clue_numbers

_FILLED_OVERLAY_RGBA = (34, 197, 94, 140)  # semi-transparent green
_CROSS_OVERLAY_RGBA = (220, 38, 38, 170)  # semi-transparent red
_CROSS_STROKE_WIDTH_FRACTION = 0.12


def parse_puzzle(image_path: str) -> Puzzle:
    """Run the full vision pipeline: load, straighten, detect the grid,
    read clues, and read the current board state.
    """
    image = grid_detector.load_and_straighten(image_path)
    geometry = grid_detector.detect_grid(image)

    row_clues = [_read_clue_strip(image, geometry, row=row) for row in range(geometry.num_rows)]
    column_clues = [_read_clue_strip(image, geometry, col=col) for col in range(geometry.num_cols)]
    board = _read_board(image, geometry)

    return Puzzle(row_clues=row_clues, column_clues=column_clues, board=board)


def _read_clue_strip(image, geometry: GridGeometry, row: int = None, col: int = None):
    if row is not None:
        x0, y0, x1, y1 = geometry.row_clue_region
        y0 = y0 + round(row * geometry.cell_size)
        y1 = y0 + round(geometry.cell_size)
    else:
        x0, y0, x1, y1 = geometry.col_clue_region
        x0 = x0 + round(col * geometry.cell_size)
        x1 = x0 + round(geometry.cell_size)

    strip = image[y0:y1, x0:x1]
    return read_clue_numbers(strip) or [0]


def _read_board(image, geometry: GridGeometry):
    board = []
    for row in range(geometry.num_rows):
        board_row = []
        for col in range(geometry.num_cols):
            x0, y0, x1, y1 = geometry.cell_bbox(row, col)
            board_row.append(classify_cell(image[y0:y1, x0:x1]))
        board.append(board_row)
    return board


def render_solution(image_path: str, puzzle: Puzzle, solved_board) -> Image.Image:
    """Overlay newly-solved cells on the original screenshot.

    Cells already Filled/Cross in the detected puzzle are left untouched
    (they're already visible in the source screenshot). Cells the solver
    newly determined get a semi-transparent green fill or red X.
    """
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Re-run grid detection to get pixel geometry; cheap and keeps Puzzle a
    # plain data object with no vision-specific state attached.
    image = grid_detector.load_and_straighten(image_path)
    geometry = grid_detector.detect_grid(image)
    scale_x = base.width / image.shape[1]
    scale_y = base.height / image.shape[0]

    for row in range(geometry.num_rows):
        for col in range(geometry.num_cols):
            was_known = puzzle.board[row][col] != CellState.UNKNOWN
            new_state = solved_board[row][col]
            if was_known or new_state == CellState.UNKNOWN:
                continue

            x0, y0, x1, y1 = geometry.cell_bbox(row, col)
            box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)

            if new_state == CellState.FILLED:
                draw.rectangle(box, fill=_FILLED_OVERLAY_RGBA)
            else:
                _draw_cross(draw, box)

    return Image.alpha_composite(base, overlay).convert("RGB")


def _draw_cross(draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    width = max(int((x1 - x0) * _CROSS_STROKE_WIDTH_FRACTION), 2)
    draw.line([(x0, y0), (x1, y1)], fill=_CROSS_OVERLAY_RGBA, width=width)
    draw.line([(x0, y1), (x1, y0)], fill=_CROSS_OVERLAY_RGBA, width=width)
