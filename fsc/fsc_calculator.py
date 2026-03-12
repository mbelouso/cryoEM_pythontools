"""
fsc_calculator.py
=================
Core routines for computing Fourier Shell Correlation (FSC) between
pairs of 3-D cryoEM volumes stored in MRC format.

Typical usage
-------------
>>> from fsc.fsc_calculator import compute_fsc_from_files
>>> result = compute_fsc_from_files("half1.mrc", "half2.mrc")
>>> print(f"Resolution at FSC=0.143: {result['resolution_A']:.2f} Å")
"""

from __future__ import annotations

import numpy as np
import mrcfile
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_volume(mrc_path: str | Path) -> tuple[np.ndarray, float]:
    """Load an MRC file and return its data and voxel size.

    Parameters
    ----------
    mrc_path : str or Path
        Path to the MRC file.

    Returns
    -------
    data : np.ndarray, shape (Z, Y, X), dtype float32
        Volume data.
    voxel_size : float
        Voxel size in Ångströms (taken from the X dimension).
    """
    with mrcfile.open(mrc_path, mode="r") as mrc:
        data = mrc.data.copy().astype(np.float32)
        voxel_size = float(mrc.voxel_size.x)
    return data, voxel_size


# ---------------------------------------------------------------------------
# FSC calculation
# ---------------------------------------------------------------------------

def compute_fsc(
    vol1: np.ndarray,
    vol2: np.ndarray,
    num_shells: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fourier Shell Correlation between two equal-sized 3-D volumes.

    Parameters
    ----------
    vol1, vol2 : np.ndarray
        3-D arrays with identical shape.
    num_shells : int, optional
        Number of radial shells.  Defaults to ``box_size // 2``.

    Returns
    -------
    shell_centers : np.ndarray, shape (num_shells,)
        Spatial frequency at the centre of each shell, in cycles per pixel.
    fsc_values : np.ndarray, shape (num_shells,)
        FSC value for each shell (range −1 to 1).

    Raises
    ------
    ValueError
        If the two volumes do not have the same shape.
    """
    if vol1.shape != vol2.shape:
        raise ValueError(
            f"Volumes must have the same shape; got {vol1.shape} and {vol2.shape}"
        )

    box_size = vol1.shape[0]
    if num_shells is None:
        num_shells = box_size // 2

    # --- 3-D FFTs ----------------------------------------------------------------
    f1 = np.fft.fftn(vol1)
    f2 = np.fft.fftn(vol2)

    # --- Radial frequency grid ---------------------------------------------------
    freq = np.fft.fftfreq(box_size)          # cycles per pixel, range [-0.5, 0.5)
    kz, ky, kx = np.meshgrid(freq, freq, freq, indexing="ij")
    radius = np.sqrt(kx**2 + ky**2 + kz**2)   # shape (Z, Y, X)

    # --- Shell edges and centres -------------------------------------------------
    shell_edges = np.linspace(0.0, 0.5, num_shells + 1)
    shell_centers = 0.5 * (shell_edges[:-1] + shell_edges[1:])

    fsc_values = np.zeros(num_shells, dtype=np.float64)

    for i in range(num_shells):
        mask = (radius >= shell_edges[i]) & (radius < shell_edges[i + 1])
        if not np.any(mask):
            continue
        f1_s = f1[mask]
        f2_s = f2[mask]

        numerator   = np.real(np.sum(f1_s * np.conj(f2_s)))
        denom_sq    = np.sum(np.abs(f1_s) ** 2) * np.sum(np.abs(f2_s) ** 2)
        if denom_sq > 0:
            fsc_values[i] = numerator / np.sqrt(denom_sq)

    return shell_centers, fsc_values


# ---------------------------------------------------------------------------
# Resolution estimation
# ---------------------------------------------------------------------------

def fsc_to_resolution(
    shell_centers: np.ndarray,
    fsc_values: np.ndarray,
    voxel_size: float,
    threshold: float = 0.143,
) -> float:
    """Estimate resolution from an FSC curve by linear interpolation at a threshold.

    Parameters
    ----------
    shell_centers : np.ndarray
        Spatial frequencies in cycles per pixel.
    fsc_values : np.ndarray
        FSC values.
    voxel_size : float
        Pixel size in Ångströms.
    threshold : float
        FSC criterion (default 0.143 for gold-standard half-map FSC).

    Returns
    -------
    float
        Estimated resolution in Ångströms, or ``np.nan`` if the curve never
        drops below the threshold.
    """
    for i in range(len(fsc_values) - 1):
        y0, y1 = fsc_values[i], fsc_values[i + 1]
        if y0 >= threshold >= y1:
            x0, x1 = shell_centers[i], shell_centers[i + 1]
            # Linear interpolation
            if (y1 - y0) != 0:
                freq_cross = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            else:
                freq_cross = x0
            if freq_cross > 0:
                return voxel_size / freq_cross
    return np.nan


# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

def compute_fsc_from_files(
    path1: str | Path,
    path2: str | Path,
    num_shells: int | None = None,
    thresholds: list[float] | None = None,
) -> dict:
    """Load two MRC volumes, compute FSC, and estimate resolution.

    Parameters
    ----------
    path1, path2 : str or Path
        Paths to the two MRC half-map files.
    num_shells : int, optional
        Number of radial frequency shells.
    thresholds : list of float, optional
        FSC threshold values for which to compute resolution estimates.
        Defaults to ``[0.143, 0.5]``.

    Returns
    -------
    dict with keys:
        ``label``        – short label built from both filenames
        ``path1``        – resolved path of first volume
        ``path2``        – resolved path of second volume
        ``shell_centers``– np.ndarray of spatial frequencies (cyc/px)
        ``resolution_A`` – np.ndarray of spatial frequencies in Å
        ``fsc``          – np.ndarray of FSC values
        ``voxel_size``   – voxel size in Å (taken from path1)
        ``resolutions``  – dict mapping threshold → resolution in Å
    """
    if thresholds is None:
        thresholds = [0.143, 0.5]

    vol1, px1 = load_volume(path1)
    vol2, _   = load_volume(path2)

    shell_centers, fsc_values = compute_fsc(vol1, vol2, num_shells=num_shells)

    # Convert spatial frequency (cyc/px) → resolution (Å)
    # Avoid division by zero at shell_centers[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        resolution_A = np.where(
            shell_centers > 0,
            px1 / shell_centers,
            np.inf,
        )

    resolutions = {
        thr: fsc_to_resolution(shell_centers, fsc_values, px1, threshold=thr)
        for thr in thresholds
    }

    label = f"{Path(path1).stem} vs {Path(path2).stem}"

    return {
        "label":         label,
        "path1":         str(Path(path1).resolve()),
        "path2":         str(Path(path2).resolve()),
        "shell_centers": shell_centers,
        "resolution_A":  resolution_A,
        "fsc":           fsc_values,
        "voxel_size":    px1,
        "resolutions":   resolutions,
    }
