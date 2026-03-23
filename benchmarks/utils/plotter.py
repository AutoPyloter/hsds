"""
plotter.py
==========
Central convergence-history visualisation for the Harmonix benchmark suite.

The :class:`ConvergencePlotter` renders fitness and constraint-penalty
curves on a shared *y*-axis using a **penalty–fitness alignment** trick:
when the first feasible (zero-penalty) solution is found at iteration *k*,
all penalty values for iterations 0 … *k* are shifted upward by the fitness
at that feasible point.  This keeps both curves visually continuous and
comparable on the same scale.

The :func:`plot_comparison` helper overlays multiple history CSV files on
a single figure for side-by-side method comparison.

Academic styling is applied globally via Seaborn's *whitegrid* palette
with publication-quality defaults (300 DPI, tight bounding box, larger
tick labels).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Global academic style
# ---------------------------------------------------------------------------
_STYLE_APPLIED = False


def _apply_academic_style() -> None:
    """Apply a publication-quality Matplotlib style exactly once."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    plt.rcParams.update(
        {
            # --- figure ---
            "figure.figsize": (12, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # --- font ---
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # --- grid ---
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.55,
            # --- lines ---
            "lines.linewidth": 2.0,
            "lines.antialiased": True,
            # --- borders ---
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    _STYLE_APPLIED = True


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class HistoryRecord:
    """A single row from a convergence-history CSV."""

    iteration: int
    fitness: float
    penalty: float


@dataclass
class TextBox:
    """Configuration for an annotation text-box on the plot."""

    text: str
    x: float = 0.99
    y: float = 0.11
    facecolor: str = "wheat"
    alpha: float = 0.5


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
def read_history_csv(path: Union[str, Path]) -> List[HistoryRecord]:
    """
    Read a harmonix history CSV into a list of :class:`HistoryRecord`.

    Expected columns: ``iteration``, ``best_fitness``, ``best_penalty``.
    Additional columns (e.g. variable values) are silently ignored.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    list[HistoryRecord]
    """
    records: List[HistoryRecord] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(
                HistoryRecord(
                    iteration=int(row["iteration"]),
                    fitness=float(row["best_fitness"]),
                    penalty=float(row["best_penalty"]),
                )
            )
    return records


# ---------------------------------------------------------------------------
# ConvergencePlotter
# ---------------------------------------------------------------------------
class ConvergencePlotter:
    """
    Dual-axis convergence plotter with penalty–fitness alignment.

    The key insight: when the optimizer transitions from infeasible to
    feasible space, the penalty series is **offset** by the first feasible
    fitness value so that both curves share a smooth visual transition on
    the *y*-axis.

    Typical usage::

        plotter = ConvergencePlotter("history_data.csv")
        plotter.set_labels(title="Welded Beam — Dependent Space")
        plotter.add_info_box("Cost: 1.84\\nPenalty: 0.0")
        plotter.plot(save_path="convergence.png")

    Parameters
    ----------
    history_file : str | Path
        Path to a harmonix history CSV.
    """

    def __init__(self, history_file: Union[str, Path]) -> None:
        self.history_file: Path = Path(history_file)
        self.iterations_data: List[HistoryRecord] = read_history_csv(self.history_file)

        # --- configurable labels ---
        self.title: str = "Optimization Convergence History"
        self.x_label: str = "Iterations"
        self.y_fit_label: str = "Objective Function (Cost)"
        self.y_pen_label: str = "Constraint Violation (Penalty)"

        # --- optional axis limits ---
        self.x_lim: Optional[Tuple[float, float]] = None
        self.y_lim: Optional[Tuple[float, float]] = None

        # --- annotation text-boxes ---
        self.text_boxes: List[TextBox] = []

    # ------------------------------------------------------------------ API
    def set_labels(
        self,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_fit_label: Optional[str] = None,
        y_pen_label: Optional[str] = None,
    ) -> "ConvergencePlotter":
        """Configure plot title and axis labels.  Returns *self* for chaining."""
        if title is not None:
            self.title = title
        if x_label is not None:
            self.x_label = x_label
        if y_fit_label is not None:
            self.y_fit_label = y_fit_label
        if y_pen_label is not None:
            self.y_pen_label = y_pen_label
        return self

    def set_axis_limits(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> "ConvergencePlotter":
        """Set explicit axis limits.  Returns *self* for chaining."""
        if x_min is not None and x_max is not None:
            self.x_lim = (x_min, x_max)
        if y_min is not None and y_max is not None:
            self.y_lim = (y_min, y_max)
        return self

    def add_info_box(
        self,
        text: str,
        x_pos: float = 0.99,
        y_pos: float = 0.11,
        facecolor: str = "wheat",
        alpha: float = 0.5,
    ) -> "ConvergencePlotter":
        """
        Add an annotation text-box at axes-relative coordinates.

        Multiple boxes may be added; they are rendered in the order they
        were registered.

        Returns *self* for chaining.
        """
        self.text_boxes.append(TextBox(text=text, x=x_pos, y=y_pos, facecolor=facecolor, alpha=alpha))
        return self

    # --------------------------------------------------------------- render
    def plot(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Render the convergence plot.

        Three scenarios are handled automatically:

        1. **All penalties zero** — only the fitness curve is drawn.
        2. **No feasible solution found** — only the penalty curve is drawn.
        3. **Mixed** — penalty is drawn (shifted) up to the feasibility
           transition point, then fitness takes over.

        Parameters
        ----------
        save_path : str | Path | None
            If given, the figure is saved at 300 DPI.  Otherwise ``plt.show()``
            is called.
        """
        _apply_academic_style()

        if not self.iterations_data:
            print("No data to plot!")
            return

        # --- Phase detection ------------------------------------------------
        first_zero_idx: int = -1
        first_feasible_fitness: float = -1.0

        for i, rec in enumerate(self.iterations_data):
            if rec.penalty == 0.0:
                first_zero_idx = i
                first_feasible_fitness = rec.fitness
                break

        all_zero: bool = all(r.penalty == 0.0 for r in self.iterations_data)

        iterations = [r.iteration for r in self.iterations_data]
        fitness_values = [r.fitness for r in self.iterations_data]
        penalty_values = [r.penalty for r in self.iterations_data]

        # Pre-compute stats
        min_fitness: Optional[float] = min(fitness_values) if fitness_values else None
        last_fitness: Optional[float] = fitness_values[-1] if fitness_values else None
        max_penalty: float = max(penalty_values) if penalty_values else 0.0
        first_fitness: Optional[float] = fitness_values[0] if fitness_values else None

        # Physical y-values for reference lines
        y_pen_zero_phys: Optional[float] = 0.0
        y_pen_max_phys: Optional[float] = max_penalty

        # --- Build draw series (penalty–fitness alignment) ------------------
        penalties_draw: List[Optional[float]] = []
        fitness_draw: List[Optional[float]] = []

        if all_zero:
            # Scenario 1: purely feasible run — no penalty curve
            penalties_draw = [None] * len(self.iterations_data)
            fitness_draw = list(fitness_values)
            y_pen_zero_phys = None
            y_pen_max_phys = None
            first_feasible_fitness_ref: Optional[float] = None

        elif first_zero_idx == -1:
            # Scenario 2: never reached feasibility — no fitness curve
            penalties_draw = list(penalty_values)
            fitness_draw = [None] * len(self.iterations_data)
            min_fitness = None
            last_fitness = None
            y_pen_zero_phys = None
            first_feasible_fitness_ref = None

        else:
            # Scenario 3: transition — shift penalties up by first feasible
            first_feasible_fitness_ref = first_feasible_fitness
            for i, rec in enumerate(self.iterations_data):
                if i <= first_zero_idx:
                    penalties_draw.append(rec.penalty + first_feasible_fitness)
                    fitness_draw.append(None)
                else:
                    penalties_draw.append(None)
                    fitness_draw.append(rec.fitness)

            y_pen_zero_phys = 0.0 + first_feasible_fitness
            y_pen_max_phys = max_penalty + first_feasible_fitness
            min_fitness = last_fitness

        # --- Figure ---------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel(self.x_label)

        valid_fit_iters = [iterations[i] for i, v in enumerate(fitness_draw) if v is not None]
        valid_fit_vals = [v for v in fitness_draw if v is not None]

        if valid_fit_vals:
            ax1.set_ylabel(self.y_fit_label, color="#2ca02c")
            ax1.tick_params(axis="y", labelcolor="#2ca02c")
            ax1.plot(
                valid_fit_iters,
                valid_fit_vals,
                color="#2ca02c",
                linestyle="-",
                label=self.y_fit_label,
                linewidth=2,
            )
        else:
            ax1.set_ylabel("")
            ax1.set_yticks([])

        ax2 = ax1.twinx()
        if not valid_fit_vals:
            ax2.grid(True, linestyle="--", alpha=0.55)

        valid_pen_iters = [iterations[i] for i, v in enumerate(penalties_draw) if v is not None]
        valid_pen_vals = [v for v in penalties_draw if v is not None]

        if valid_pen_vals:
            ax2.set_ylabel(self.y_pen_label, color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax2.plot(
                valid_pen_iters,
                valid_pen_vals,
                color="#d62728",
                linestyle="-",
                label=self.y_pen_label,
                linewidth=2,
            )
        else:
            ax2.set_ylabel("")
            ax2.set_yticks([])

        # --- Axis scaling ---------------------------------------------------
        if self.y_lim is not None:
            ax1.set_ylim(*self.y_lim)
            ax2.set_ylim(*self.y_lim)
        else:
            all_y: List[float] = []
            if valid_fit_vals:
                all_y.extend(valid_fit_vals)
            if valid_pen_vals:
                all_y.extend(valid_pen_vals)
            if min_fitness is not None:
                all_y.append(min_fitness)
            if all_zero and first_fitness is not None:
                all_y.append(first_fitness)
            if y_pen_zero_phys is not None:
                all_y.append(y_pen_zero_phys)
            if y_pen_max_phys is not None:
                all_y.append(y_pen_max_phys)

            if all_y:
                lo, hi = min(all_y), max(all_y)
                pad = 0.5 if hi == lo else (hi - lo) * 0.05
                ax1.set_ylim(lo - pad, hi + pad)
                ax2.set_ylim(lo - pad, hi + pad)

        if self.x_lim is not None:
            ax1.set_xlim(*self.x_lim)
            ax2.set_xlim(*self.x_lim)

        # --- Tick formatters (penalty–fitness alignment) --------------------
        _fzidx = first_zero_idx
        _fffit = first_feasible_fitness if first_zero_idx != -1 else None
        _allz = all_zero

        def _penalty_formatter(x: float, _pos: int) -> str:
            if _fzidx == -1:
                original = x
            else:
                original = x - (_fffit if _fffit is not None else 0.0)
            return f"{original:.2f}" if original >= -0.001 else ""

        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_penalty_formatter))

        def _fitness_formatter(x: float, _pos: int) -> str:
            if _fzidx != -1 and not _allz and _fffit is not None:
                if x > _fffit + 1e-5:
                    return ""
            return f"{x:.2f}"

        if valid_fit_vals:
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_fitness_formatter))

        # --- Reference lines ------------------------------------------------
        if first_zero_idx != -1 and not all_zero and first_feasible_fitness is not None:
            ax1.axhline(
                first_feasible_fitness,
                color="#1f77b4",
                linestyle=":",
                linewidth=1,
                alpha=0.5,
            )

        if all_zero and min_fitness is not None:
            ax1.axhline(
                min_fitness,
                color="cyan",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
            )

        if first_zero_idx != -1 and not all_zero and last_fitness is not None:
            ylims = ax1.get_ylim()
            if ylims[0] <= last_fitness <= ylims[1]:
                ax1.axhline(
                    last_fitness,
                    color="cyan",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                )

        if first_zero_idx != -1 and not all_zero and y_pen_zero_phys is not None:
            ylims = ax2.get_ylim()
            if ylims[0] <= y_pen_zero_phys <= ylims[1]:
                ax2.axhline(
                    y_pen_zero_phys,
                    color="purple",
                    linestyle="-.",
                    linewidth=1,
                    alpha=0.6,
                )

        ax1.set_title(self.title, pad=15)

        # --- Unified legend -------------------------------------------------
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        if all_lines:
            ax2.legend(
                all_lines,
                all_labels,
                loc="upper right",
                framealpha=0.85,
                edgecolor="#cccccc",
            )

        # --- User-defined info boxes ----------------------------------------
        for box in self.text_boxes:
            fig.text(
                box.x,
                box.y,
                box.text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc=box.facecolor,
                    alpha=box.alpha,
                    ec="k",
                    lw=0.7,
                ),
            )

        fig.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Multi-run shaded-area helper
# ---------------------------------------------------------------------------
def plot_multi_run_convergence(
    history_files: Sequence[Union[str, Path]],
    *,
    label: str = "Mean Fitness",
    color: str = "#2ca02c",
    title: str = "Multi-Run Convergence (Mean ± Std Dev)",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Overlay the **mean** fitness curve with a shaded ±1 σ band from
    multiple independent runs.

    Parameters
    ----------
    history_files : sequence of paths
        Each path should point to a harmonix history CSV from an
        independent run.  All runs must use the same ``history_every``
        setting so iteration grids align.
    label : str
        Legend label for this series.
    color : str
        Line and fill colour.
    title : str
        Axes title (only applied when *ax* is ``None``).
    save_path : str | Path | None
        If given, save the figure at 300 DPI.
    ax : Axes | None
        Existing axes to draw on.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for plotting (for further customisation).
    """
    _apply_academic_style()

    all_runs: List[List[HistoryRecord]] = [read_history_csv(p) for p in history_files]

    # Align on the shortest run
    min_len = min(len(r) for r in all_runs)
    iterations = [all_runs[0][i].iteration for i in range(min_len)]

    fitness_matrix = np.array([[run[i].fitness for i in range(min_len)] for run in all_runs])
    mean = fitness_matrix.mean(axis=0)
    std = fitness_matrix.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(title, pad=15)
        owns_figure = True
    else:
        owns_figure = False

    ax.plot(iterations, mean, color=color, linewidth=2, label=label)
    ax.fill_between(
        iterations,
        mean - std,
        mean + std,
        color=color,
        alpha=0.18,
        label=f"{label} ± 1σ",
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective Function (Cost)")
    ax.legend(framealpha=0.85, edgecolor="#cccccc")
    ax.grid(True, linestyle="--", alpha=0.55)

    if save_path and owns_figure:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Multi-run plot saved to {save_path}")

    return ax


# ---------------------------------------------------------------------------
# Comparative overlay helper
# ---------------------------------------------------------------------------
def plot_comparison(
    csv_paths: Dict[str, Union[str, Path]],
    *,
    title: str = "Static Penalty vs Dependent Space",
    metric: str = "fitness",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Overlay convergence curves from multiple methods on a single figure.

    Parameters
    ----------
    csv_paths : dict[str, path]
        Mapping from display label to history CSV path.
        Example::

            {
                "Static Penalty": "static_penalty/history_data.csv",
                "Dependent Space": "dependent_space/history_data.csv",
            }

    title : str
        Figure title.
    metric : ``"fitness"`` | ``"penalty"``
        Which column to plot.
    save_path : str | Path | None
        If given, save the figure at 300 DPI.
    """
    _apply_academic_style()

    palette = [
        "#d62728",  # red
        "#2ca02c",  # green
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#9467bd",  # purple
    ]
    line_styles = ["-", "--", "-.", ":", "-"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (label, csv_path) in enumerate(csv_paths.items()):
        records = read_history_csv(csv_path)
        iters = [r.iteration for r in records]
        values = [r.fitness if metric == "fitness" else r.penalty for r in records]
        ax.plot(
            iters,
            values,
            color=palette[idx % len(palette)],
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2,
            label=label,
        )

    ylabel = "Objective Function (Cost)" if metric == "fitness" else "Constraint Violation (Penalty)"
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=15)
    ax.legend(framealpha=0.85, edgecolor="#cccccc")

    fig.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
