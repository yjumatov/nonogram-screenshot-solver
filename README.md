# Nonogram Screenshot Solver

Upload a screenshot of a nonogram (picross) puzzle from a specific mobile game, and get back
the same screenshot with the solution filled in.

```
Open App -> Upload Screenshot -> Solve -> Result Image
```

That's the entire app. There's no puzzle editor, no manual entry, no game mode — just
upload, solve, done.

The solving algorithm is the original constraint-propagation ("line solving") engine from
this project's earlier command-line solver, unchanged, in [`solver/engine.py`](solver/engine.py).
Line-solving alone doesn't fully determine every puzzle, so [`solver/search.py`](solver/search.py)
adds a guess-and-check layer on top — without changing a single deduction rule in
`engine.py` — that guesses a cell, re-runs the unmodified engine, and keeps the guess unless
it proves contradictory. Everything else here — the vision pipeline and the desktop UI — is
new, built around that engine.

## How it works

1. **Upload** a screenshot ([`ui/app_window.py`](ui/app_window.py)).
2. **Detect the grid** — straighten out any perspective skew, then find the board's size and
   cell geometry from its row/column clue badges (robust to unsolved cells rendering a
   decorative texture instead of plain white) ([`vision/grid_detector.py`](vision/grid_detector.py)).
3. **Read the clues** — OCR the row/column clue numbers, correcting common misreads
   (`I`/`l` → `1`, `S` → `5`, `O` → `0`) ([`vision/ocr.py`](vision/ocr.py)).
4. **Read the board** — classify each cell as filled (green), crossed out (red), or unknown,
   using HSV color segmentation ([`vision/cell_detector.py`](vision/cell_detector.py)).
5. **Solve** — hand the clues and current board to the existing solver
   ([`solver/api.py`](solver/api.py)).
6. **Render** — overlay the newly-solved cells (semi-transparent green fills, semi-transparent
   red X's) on top of the original screenshot; cells that were already visible are left alone
   ([`vision/image_parser.py`](vision/image_parser.py)).

The vision pipeline is deliberately narrow: it's tuned for one game's screenshot layout
(clue strips to the left of / above the board, green fill / red X cell styling), not a
generic nonogram parser. That's what keeps it deterministic and simple instead of a full
computer-vision project in its own right.

## Project structure

```
app/       entry point
ui/        the (only) window: upload / preview / solve / result / save
vision/    screenshot -> Puzzle, and solved board -> overlay image
solver/    the original solving engine (unchanged) plus a backtracking layer, behind a reusable API
models/    shared data types (Puzzle, CellState)
tests/     solver smoke tests
```

## Setup (VS Code)

Requires Python 3.9+ **with a working Tk** and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
installed on your system.

**On macOS, don't use the system/Command Line Tools Python for this.** It bundles Tk 8.5
(from 2009), which predates Dark Mode and renders this app's window as entirely blank —
not a bug in this app's code, a broken Tk on a modern OS. Use a Homebrew Python instead,
which links against a modern Tk:

```bash
brew install tesseract python-tk@3.12   # installs a modern Tcl/Tk alongside Python 3.12
python3.12 -m venv .venv                # use the Homebrew python3.12, not /usr/bin/python3
source .venv/bin/activate               # in VS Code, select this interpreter (Cmd/Ctrl+Shift+P -> Python: Select Interpreter)
pip install -r requirements.txt
```

On Linux: `apt install tesseract-ocr python3-tk`, then the usual `python3 -m venv .venv`.

If the window ever renders blank again, it's almost certainly this: run
`python3 -c "import tkinter as tk; print(tk.Tk().call('info','patchlevel'))"` — if it prints
`8.5.x`, you're on the broken system Tk and need to switch interpreters as above.

## Running the app

```bash
source .venv/bin/activate
python -m app.main
```

## Running tests

```bash
source .venv/bin/activate
python -m unittest discover tests
```

## Calibration note

`vision/grid_detector.py`, `vision/cell_detector.py`, and `vision/ocr.py` have been verified
end-to-end against a real screenshot from the target game (15x15 board, clue OCR, cell
classification, solve, and render all correct). If a screenshot from a different puzzle size
or a visual theme variant doesn't detect cleanly, the constants near the top of those three
files are the place to retune — in particular `_NAVY_MIN_VALUE`/`_NAVY_MAX_VALUE` in
`grid_detector.py` (clue badge color) and the HSV ranges in `cell_detector.py` (fill/cross
color).
