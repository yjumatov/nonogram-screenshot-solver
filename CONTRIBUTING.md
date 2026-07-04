# Contributing

Thanks for considering a contribution! This is a small, focused project, so the process is
intentionally lightweight.

## Getting set up

Follow the [Installation](README.md#installation) steps in the README, then confirm
everything works:

```bash
source .venv/bin/activate
python -m unittest discover tests
python -m app.main
```

## Where things live

See [Project Structure](README.md#project-structure) in the README. A few things worth
knowing before you dig in:

- **`solver/engine.py` is the original line-solving algorithm** from
  [bbukaty/nonograms](https://github.com/bbukaty/nonograms), preserved deliberately unchanged
  in its deduction rules. If you're improving solving behavior, prefer adding to
  `solver/search.py` (the backtracking layer) rather than editing `engine.py`'s rules — and if
  you do need to touch `engine.py`, please call that out clearly in your PR description.
- **`vision/` is deliberately narrow**, tuned to one mobile game's specific screenshot style
  (clue badge shape/position, fill/cross tile colors) rather than built as a generic Nonogram
  image parser. If you're adding support for a different game or theme, a configurable
  "profile" of constants (see the top of each `vision/*.py` file) is more in the spirit of this
  project than a rewrite.
- **`ui/app_window.py` is intentionally minimal** — one window, four controls, no settings. New
  features should fit that scope, or belong in a separate discussion first.

## Making changes

1. Open an issue first for anything beyond a small fix, so we can agree on the approach.
2. Keep changes focused — a single PR should do one thing.
3. Add or update tests for any behavior change (`tests/`).
4. Run the full test suite before opening a PR:
   ```bash
   python -m unittest discover tests
   ```
5. If you touch the vision pipeline, please describe what you tested it against (e.g. "verified
   end-to-end against N real screenshots") — this pipeline is calibrated against real examples,
   not synthetic fonts/colors, and regressions here are easy to introduce without noticing.

## Reporting bugs

Please include:
- What you expected vs. what happened
- The screenshot that triggered it, if possible (redact anything sensitive)
- Any error message shown by the app

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you're
expected to uphold it.
