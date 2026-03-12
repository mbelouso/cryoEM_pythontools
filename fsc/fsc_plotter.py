"""
fsc_plotter.py
==============
Publication-quality FSC curve plotting using Matplotlib and Seaborn.

Functions
---------
plot_fsc        – single axes with one or more FSC curves and threshold lines
plot_fsc_grid   – multi-panel grid, one panel per FSC result
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Sequence

# ── default aesthetic constants ───────────────────────────────────────────────
_PALETTE       = "tab10"
_LINEWIDTH     = 2.0
_THRESHOLD_CFG = {
    0.143: dict(color="#e05c5c", linestyle="--", linewidth=1.4, label="FSC = 0.143"),
    0.500: dict(color="#4a90d9", linestyle=":",  linewidth=1.4, label="FSC = 0.500"),
}
_FIG_DPI       = 150


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_seaborn_style() -> None:
    sns.set_theme(style="ticks", context="notebook",
                  rc={"axes.spines.top": False, "axes.spines.right": False})


def _resolution_xlabel() -> str:
    return "Resolution (Å)"


def _format_resolution_axis(ax: matplotlib.axes.Axes, resolution_A: np.ndarray) -> None:
    """Replace numeric x-tick positions with Ångström labels.

    The x-axis stores reciprocal-space values (Å, i.e. 1/freq_cyc_per_px * px_size),
    which decrease from left to right.  We show every other tick to avoid crowding.
    """
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))

    def _fmt(val, _):
        if val <= 0:
            return ""
        return f"{val:.1f}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))


def _annotate_resolution(
    ax: matplotlib.axes.Axes,
    result: dict,
    thresholds: list[float],
    y_offset_step: float = 0.08,
) -> None:
    """Annotate the axes with resolution estimates for each threshold."""
    base_y = 0.97
    for i, thr in enumerate(thresholds):
        res = result["resolutions"].get(thr, np.nan)
        if np.isfinite(res):
            color = _THRESHOLD_CFG.get(thr, {}).get("color", "grey")
            ax.text(
                0.98, base_y - i * y_offset_step,
                f"d{thr:.3f} = {res:.2f} Å",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )


# ── public API ────────────────────────────────────────────────────────────────

def plot_fsc(
    results: dict | Sequence[dict],
    labels: list[str] | None = None,
    thresholds: list[float] | None = None,
    title: str = "Fourier Shell Correlation",
    figsize: tuple[float, float] = (8, 5),
    palette: str | list[str] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show_annotations: bool = True,
    xlim: tuple[float, float] = (8.0, 2.0),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot one or more FSC curves on a single axes.

    Parameters
    ----------
    results : dict or list of dict
        Output(s) from :func:`fsc_calculator.compute_fsc_from_files`.
    labels : list of str, optional
        Override the ``label`` key inside each result dict.
    thresholds : list of float, optional
        Which FSC threshold lines to draw.  Defaults to ``[0.143, 0.5]``.
    title : str
        Axes title.
    figsize : tuple
        Figure size in inches.
    palette : str or list of str, optional
        Seaborn / Matplotlib colour palette for the curves.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot onto (creates a new figure if None).
    show_annotations : bool
        Whether to annotate estimated resolutions in the corner.
    xlim : tuple of float
        Resolution axis limits in Ångströms as (low_res, high_res), e.g. (8.0, 2.0).
        The axis runs left→right from low to high resolution.

    Returns
    -------
    fig, ax
    """
    _apply_seaborn_style()

    if isinstance(results, dict):
        results = [results]

    if thresholds is None:
        thresholds = [0.143, 0.5]

    n = len(results)
    colors = sns.color_palette(palette or _PALETTE, n_colors=n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=_FIG_DPI)
    else:
        fig = ax.get_figure()

    # ── draw FSC curves ───────────────────────────────────────────────────────
    for idx, res in enumerate(results):
        lbl = (labels[idx] if labels and idx < len(labels) else res.get("label", f"FSC {idx+1}"))
        x   = res["resolution_A"]
        y   = res["fsc"]

        # clip infinite values (DC component) to a sensible max
        finite_mask = np.isfinite(x)
        x = x[finite_mask]
        y = y[finite_mask]

        ax.plot(x, y, color=colors[idx], linewidth=_LINEWIDTH, label=lbl, zorder=3)

        if show_annotations and n == 1:
            _annotate_resolution(ax, res, thresholds)

    # ── threshold lines ───────────────────────────────────────────────────────
    # Draw the line but keep it out of the legend; label it directly on the
    # right-hand edge of the axes so it never overlaps the curve legend.
    for thr in thresholds:
        cfg = _THRESHOLD_CFG.get(thr, dict(color="grey", linestyle="--", linewidth=1.2))
        ax.axhline(
            thr,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            zorder=2,
        )
        # Only draw the right-edge label when thr is inside the y-axis range
        ax.annotate(
            f" FSC={thr:.3f}",
            xy=(1.0, thr),
            xycoords=("axes fraction", "data"),
            ha="left", va="center",
            fontsize=8, color=cfg["color"],
            annotation_clip=False,
        )

    # ── axes formatting ───────────────────────────────────────────────────────
    # Resolution axis: large Å (low resolution) on the left, small Å on the right
    ax.set_xlim(xlim[0], xlim[1])   # e.g. 8.0 → 2.0
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(_resolution_xlabel(), fontsize=11)
    ax.set_ylabel("FSC", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    # Legend contains only FSC curve entries; placed lower-left where curves
    # have decayed to ~0, away from the threshold line labels on the right.
    ax.legend(fontsize=9, framealpha=0.8, loc="lower left")
    ax.axhline(0, color="k", linewidth=0.6, zorder=1)

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_fsc_grid(
    results: Sequence[dict],
    thresholds: list[float] | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (6, 4),
    suptitle: str = "Fourier Shell Correlation",
    palette: str | list[str] | None = None,
    xlim: tuple[float, float] = (8.0, 2.0),
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot one FSC curve per panel in a grid layout.

    Parameters
    ----------
    results : list of dict
        Outputs from :func:`fsc_calculator.compute_fsc_from_files`.
    thresholds : list of float, optional
        FSC threshold lines to draw.
    ncols : int
        Number of columns in the panel grid.
    figsize_per_panel : tuple
        Width × height of each individual panel in inches.
    suptitle : str
        Overall figure title.
    palette : str or list, optional
        Colour palette for curves across panels.
    xlim : tuple of float
        Resolution axis limits in Ångströms as (low_res, high_res).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    _apply_seaborn_style()

    if thresholds is None:
        thresholds = [0.143, 0.5]

    n      = len(results)
    ncols  = min(ncols, n)
    nrows  = int(np.ceil(n / ncols))
    colors = sns.color_palette(palette or _PALETTE, n_colors=n)

    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows + 0.6  # extra space for suptitle
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), dpi=_FIG_DPI)
    axes_flat  = np.array(axes).flatten()

    for idx, (res, ax) in enumerate(zip(results, axes_flat)):
        plot_fsc(
            res,
            thresholds=thresholds,
            title=res.get("label", f"FSC {idx+1}"),
            figsize=figsize_per_panel,
            palette=[colors[idx]],
            ax=ax,
            show_annotations=True,
            xlim=xlim,
        )

    # Hide any unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes_flat[:n]
