"""
fsc — Fourier Shell Correlation toolkit for cryoEM single-particle analysis.

Modules
-------
fsc_calculator : load MRC volumes and compute FSC curves
fsc_plotter    : publication-quality FSC curve plotting with seaborn
"""

from .fsc_calculator import (
    load_volume,
    compute_fsc,
    fsc_to_resolution,
    compute_fsc_from_files,
    estimate_translation_phase_correlation,
    apply_fourier_shift,
    apply_euler_rotation,
    align_volume_to_reference,
)
from .fsc_plotter import plot_fsc, plot_fsc_grid

__all__ = [
    "load_volume",
    "compute_fsc",
    "fsc_to_resolution",
    "compute_fsc_from_files",
    "estimate_translation_phase_correlation",
    "apply_fourier_shift",
    "apply_euler_rotation",
    "align_volume_to_reference",
    "plot_fsc",
    "plot_fsc_grid",
]
