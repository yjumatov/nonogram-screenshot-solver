# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-07-05

Initial public release. Transforms the original command-line Nonogram solver into a
screenshot-based desktop application.

### Added
- Desktop UI (Tkinter): upload a screenshot, preview it, solve, view and save the result.
- Vision pipeline (`vision/`): grid detection via clue-badge recognition (robust to
  decorative textures on unsolved cells), per-digit OCR of row/column clues, and HSV-based
  cell state classification (filled / crossed / unknown).
- Backtracking layer (`solver/search.py`) on top of the original line-solving engine, so the
  solver always completes a puzzle when a valid solution exists, instead of stopping wherever
  pure constraint propagation stalls.
- Solution overlay rendering: newly-solved cells are drawn as semi-transparent green fills /
  red X's directly on the original screenshot.
- Reusable `solver.api.solve()` entry point, and `ContradictionError` in the engine as a
  backtracking signal.
- Test suite covering the solver, backtracking search, and OCR.

### Changed
- Restructured the project into `app/`, `ui/`, `vision/`, `solver/`, `models/`, `tests/`.
- The original solving algorithm (`solver/engine.py`) is preserved with its deduction rules
  unchanged; the only additions are optional partial-board seeding and contradiction
  detection for the backtracking layer to use.

### Removed
- The original CLI demo, puzzle-scraping test data, and GIF-rendering script — this project
  has a single workflow (screenshot in, solved screenshot out), so the game/demo tooling
  around the original solver no longer applies.
