"""The entire UI: upload a screenshot, preview it, solve, view/save the result.

Deliberately minimal — one window, four controls, no editor, no settings.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

from PIL import Image, ImageTk

from solver.api import solve
from vision.image_parser import parse_puzzle
from vision.image_parser import render_solution

PREVIEW_MAX_SIZE = (480, 480)

# Explicit colors throughout, rather than Tk/system defaults: this app was
# found running under a very old Tk (8.5, from the macOS Command Line Tools
# Python), which predates real Dark Mode support — under a dark system
# appearance its default widget colors resolved to text and borders with no
# contrast against the window background, rendering as blank space even
# though the widgets (and their images) were genuinely there.
_BG = "#f0f0f0"
_FG = "#111111"
_IMAGE_SLOT_BG = "#ffffff"


class NonogramApp(tk.Tk):
    """Single-window app: Upload -> Preview -> Solve -> Result -> Save."""

    def __init__(self):
        super().__init__()
        self.title("Nonogram Screenshot Solver")
        self.configure(bg=_BG)

        self._source_image_path: Optional[str] = None
        self._result_image: Optional[Image.Image] = None

        self._build_widgets()

    def _build_widgets(self):
        controls = tk.Frame(self, bg=_BG)
        controls.pack(padx=10, pady=10)

        self.upload_button = tk.Button(controls, text="Upload Image", command=self._on_upload)
        self.upload_button.grid(row=0, column=0, padx=5)

        self.solve_button = tk.Button(controls, text="Solve", command=self._on_solve, state=tk.DISABLED)
        self.solve_button.grid(row=0, column=1, padx=5)

        self.save_button = tk.Button(controls, text="Save Result", command=self._on_save, state=tk.DISABLED)
        self.save_button.grid(row=0, column=2, padx=5)

        images = tk.Frame(self, bg=_BG)
        images.pack(padx=10, pady=(0, 10))

        preview_col = tk.Frame(images, bg=_BG)
        preview_col.grid(row=0, column=0, padx=10)
        tk.Label(preview_col, text="Uploaded Screenshot", bg=_BG, fg=_FG).pack()
        self.preview_label = self._make_image_slot(preview_col)

        result_col = tk.Frame(images, bg=_BG)
        result_col.grid(row=0, column=1, padx=10)
        tk.Label(result_col, text="Solved Result", bg=_BG, fg=_FG).pack()
        self.result_label = self._make_image_slot(result_col)

        self.status_label = tk.Label(self, text="Upload a screenshot to begin.", anchor="w", bg=_BG, fg=_FG)
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _make_image_slot(self, parent: tk.Frame) -> tk.Label:
        """A fixed-size placeholder box that will later hold a photo.

        The box's size is set on an outer Frame (always pixel units in
        Tkinter) rather than on the Label itself, since a Label's width/height
        options are character-cell units until an image is set and pixel
        units after — mixing the two on the same widget is a classic way to
        end up with an image that silently fails to render.
        """
        box = tk.Frame(parent, width=PREVIEW_MAX_SIZE[0], height=PREVIEW_MAX_SIZE[1], bg=_IMAGE_SLOT_BG,
                       relief=tk.SUNKEN, borderwidth=1)
        box.pack()
        box.pack_propagate(False)
        label = tk.Label(box, bg=_IMAGE_SLOT_BG)
        label.pack(expand=True)
        return label

    def _set_status(self, text: str):
        self.status_label.config(text=text)

    def _show_image(self, label: tk.Label, image: Image.Image):
        preview = image.copy()
        preview.thumbnail(PREVIEW_MAX_SIZE)
        photo = ImageTk.PhotoImage(preview)
        label.configure(image=photo)
        label.image = photo  # keep a reference so it isn't garbage collected

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select a nonogram screenshot",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            image = Image.open(path)
            image.load()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return

        self._source_image_path = path
        self._show_image(self.preview_label, image)
        self.result_label.configure(image="")
        self.result_label.image = None
        self._result_image = None
        self.save_button.config(state=tk.DISABLED)
        self.solve_button.config(state=tk.NORMAL)
        self._set_status(f"Loaded {path}. Click Solve to continue.")

    def _on_solve(self):
        if not self._source_image_path:
            return

        self._set_status("Detecting puzzle...")
        self.update_idletasks()
        try:
            puzzle = parse_puzzle(self._source_image_path)
            solved_board = solve(puzzle.row_clues, puzzle.column_clues, puzzle.board)
            self._result_image = render_solution(self._source_image_path, puzzle, solved_board)
        except Exception as exc:
            messagebox.showerror("Solve failed", str(exc))
            self._set_status("Solve failed. See error for details.")
            return

        self._show_image(self.result_label, self._result_image)
        self.save_button.config(state=tk.NORMAL)
        self._set_status("Solved. You can save the result image.")

    def _on_save(self):
        if self._result_image is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save solved puzzle",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return

        self._result_image.save(path)
        self._set_status(f"Saved to {path}.")


def main():
    app = NonogramApp()
    app.mainloop()


if __name__ == "__main__":
    main()
