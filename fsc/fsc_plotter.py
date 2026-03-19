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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
_HEATMAP_CMAP  = "viridis_r"


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


def _heatmap_metric_value(result: dict, threshold: float = 0.143) -> float:
    """Return scalar used for heatmap colouring.

    Priority:
    1) resolution at the requested FSC threshold
    2) post-alignment correlation
    3) NaN
    """
    res_map = result.get("resolutions", {}) or {}
    v = res_map.get(threshold, np.nan)
    if np.isfinite(v):
        return float(v)

    aln = result.get("alignment", {}) or {}
    c = aln.get("overall_correlation_after", np.nan)
    if np.isfinite(c):
        return float(c)

    return np.nan


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
    color_mode: str = "auto",
    heatmap_threshold: float = 0.143,
    heatmap_cmap: str = _HEATMAP_CMAP,
    legend_mode: str = "auto",
    max_legend_entries: int = 10,
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
    color_mode : {"auto", "heatmap", "categorical"}
        Curve colouring strategy for overlays.
    heatmap_threshold : float
        FSC threshold used to derive per-curve heat values from resolution.
    heatmap_cmap : str
        Matplotlib colormap used when ``color_mode`` selects heatmap.
    legend_mode : {"auto", "full", "compact", "off"}
        Strategy for curve legend display.
    max_legend_entries : int
        Maximum legend entries shown in compact mode.

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
    if color_mode not in {"auto", "heatmap", "categorical"}:
        raise ValueError("color_mode must be one of: 'auto', 'heatmap', 'categorical'")
    if legend_mode not in {"auto", "full", "compact", "off"}:
        raise ValueError("legend_mode must be one of: 'auto', 'full', 'compact', 'off'")

    use_heatmap = (color_mode == "heatmap") or (color_mode == "auto" and n > 6)
    if use_heatmap:
        heat_values = np.array(
            [_heatmap_metric_value(r, threshold=heatmap_threshold) for r in results],
            dtype=np.float64,
        )
        finite = np.isfinite(heat_values)
        if np.any(finite):
            vmin = float(np.min(heat_values[finite]))
            vmax = float(np.max(heat_values[finite]))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(heatmap_cmap)
            colors = [cmap(norm(v)) if np.isfinite(v) else (0.7, 0.7, 0.7, 0.8) for v in heat_values]
        else:
            use_heatmap = False
            colors = sns.color_palette(palette or _PALETTE, n_colors=n)
    else:
        colors = sns.color_palette(palette or _PALETTE, n_colors=n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=_FIG_DPI)
    else:
        fig = ax.get_figure()

    # ── draw FSC curves ───────────────────────────────────────────────────────
    curve_handles: list[matplotlib.lines.Line2D] = []
    curve_labels: list[str] = []
    for idx, res in enumerate(results):
        lbl = (labels[idx] if labels and idx < len(labels) else res.get("label", f"FSC {idx+1}"))
        x   = res["resolution_A"]
        y   = res["fsc"]

        # clip infinite values (DC component) to a sensible max
        finite_mask = np.isfinite(x)
        x = x[finite_mask]
        y = y[finite_mask]

        line, = ax.plot(x, y, color=colors[idx], linewidth=_LINEWIDTH, label=lbl, zorder=3, alpha=0.95)
        curve_handles.append(line)
        curve_labels.append(lbl)

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

    if use_heatmap:
        sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(heatmap_cmap))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.05)
        cbar.set_label(f"Resolution at FSC={heatmap_threshold:.3f} (Å)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # ── axes formatting ───────────────────────────────────────────────────────
    # Resolution axis: large Å (low resolution) on the left, small Å on the right
    ax.set_xlim(xlim[0], xlim[1])   # e.g. 8.0 → 2.0
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(_resolution_xlabel(), fontsize=11)
    ax.set_ylabel("FSC", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Adaptive legend behavior to avoid unreadable overlays with many curves.
    if legend_mode == "auto":
        if n <= max_legend_entries and not use_heatmap:
            legend_mode_eff = "full"
        elif n <= max_legend_entries:
            legend_mode_eff = "compact"
        else:
            legend_mode_eff = "off"
    else:
        legend_mode_eff = legend_mode

    if legend_mode_eff == "full":
        ax.legend(fontsize=8, framealpha=0.85, loc="lower left", ncol=1)
    elif legend_mode_eff == "compact":
        keep = min(max_legend_entries, n)
        if use_heatmap:
            # Show only extreme curves in heatmap mode for context.
            metric = np.array([_heatmap_metric_value(r, threshold=heatmap_threshold) for r in results], dtype=np.float64)
            finite_idx = np.flatnonzero(np.isfinite(metric))
            if finite_idx.size > 0:
                order = finite_idx[np.argsort(metric[finite_idx])]
                low = order[: max(1, keep // 2)]
                high = order[-max(1, keep - len(low)) :]
                show_idx = np.unique(np.concatenate([low, high]))
            else:
                show_idx = np.arange(keep)
        else:
            show_idx = np.arange(keep)

        handles = [curve_handles[i] for i in show_idx]
        labels_show = [curve_labels[i] for i in show_idx]
        extra = n - len(show_idx)
        if extra > 0:
            labels_show.append(f"... +{extra} more")
            handles.append(matplotlib.lines.Line2D([], [], color="none"))
        ax.legend(handles, labels_show, fontsize=8, framealpha=0.85, loc="lower left", ncol=1)
    elif legend_mode_eff == "off":
        ax.text(
            0.01,
            0.01,
            f"{n} FSC curves shown",
            transform=ax.transAxes,
            fontsize=8,
            color="#444444",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )

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
