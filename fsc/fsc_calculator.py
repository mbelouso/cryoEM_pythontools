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

try:
    from scipy.ndimage import rotate as _scipy_rotate
    from scipy.ndimage import map_coordinates as _scipy_map_coordinates
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback path when scipy is unavailable
    _scipy_rotate = None
    _scipy_map_coordinates = None
    _HAS_SCIPY = False


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
# Fourier-domain rigid translation alignment
# ---------------------------------------------------------------------------

def _quadratic_peak_offset(left: float, center: float, right: float) -> float:
    """Return subpixel offset in [-0.5, 0.5] from three samples around a peak."""
    denom = left - 2.0 * center + right
    if np.isclose(denom, 0.0):
        return 0.0
    delta = 0.5 * (left - right) / denom
    return float(np.clip(delta, -0.5, 0.5))


def _normalized_correlation(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """Compute Pearson-like global correlation coefficient between two volumes."""
    a = np.asarray(vol1, dtype=np.float64)
    b = np.asarray(vol2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Volume shapes differ: {a.shape} vs {b.shape}")

    a -= np.mean(a)
    b -= np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= np.finfo(np.float64).eps:
        return np.nan
    return float(np.sum(a * b) / denom)


def _downsample_for_rotation_search(
    vol: np.ndarray,
    target_max_dim: int = 96,
) -> tuple[np.ndarray, int]:
    """Downsample by integer stride for faster real-space fitting."""
    max_dim = max(vol.shape)
    stride = max(1, int(np.ceil(max_dim / target_max_dim)))
    return vol[::stride, ::stride, ::stride], stride


def apply_euler_rotation(
    volume: np.ndarray,
    angles_deg_zyx: tuple[float, float, float],
    interpolation_order: int = 1,
) -> np.ndarray:
    """Rotate a volume with intrinsic Z→Y→X Euler angles (degrees).

    Parameters
    ----------
    volume : np.ndarray
        3-D array to rotate.
    angles_deg_zyx : tuple[float, float, float]
        Rotation angles in degrees as (rz, ry, rx).
    interpolation_order : int
        Spline interpolation order for scipy.ndimage.rotate.

    Returns
    -------
    np.ndarray
        Rotated volume with dtype float32.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume; got ndim={volume.ndim}")
    if not _HAS_SCIPY:
        raise ImportError(
            "Rotational alignment requires scipy. Install scipy to enable this feature."
        )

    rz, ry, rx = angles_deg_zyx
    rotated = _scipy_rotate(
        volume,
        angle=rz,
        axes=(1, 2),
        reshape=False,
        order=interpolation_order,
        mode="nearest",
        prefilter=(interpolation_order > 1),
    )
    rotated = _scipy_rotate(
        rotated,
        angle=ry,
        axes=(0, 2),
        reshape=False,
        order=interpolation_order,
        mode="nearest",
        prefilter=(interpolation_order > 1),
    )
    rotated = _scipy_rotate(
        rotated,
        angle=rx,
        axes=(0, 1),
        reshape=False,
        order=interpolation_order,
        mode="nearest",
        prefilter=(interpolation_order > 1),
    )
    return rotated.astype(np.float32, copy=False)


def _sample_points_and_weights(
    volume: np.ndarray,
    max_points: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a sparse, intensity-weighted voxel set for real-space fitting.

    ChimeraX uses C-optimized interpolation over map points; for Python, using
    all voxels is prohibitively slow. This function keeps only informative
    points (high absolute intensity) and caps sample size.
    """
    flat = volume.ravel().astype(np.float64, copy=False)
    abs_flat = np.abs(flat)

    if flat.size == 0:
        raise ValueError("Volume has no voxels")

    q = float(np.clip(quantile, 0.0, 1.0))
    threshold = float(np.quantile(abs_flat, q))
    candidate_idx = np.flatnonzero(abs_flat >= threshold)

    if candidate_idx.size == 0:
        candidate_idx = np.arange(flat.size, dtype=np.int64)

    max_pts = max(1024, int(max_points))
    if candidate_idx.size > max_pts:
        # Keep strongest magnitudes only.
        cand_abs = abs_flat[candidate_idx]
        top_local = np.argpartition(cand_abs, -max_pts)[-max_pts:]
        candidate_idx = candidate_idx[top_local]

    nz, ny, nx = volume.shape
    z = candidate_idx // (ny * nx)
    rem = candidate_idx % (ny * nx)
    y = rem // nx
    x = rem % nx

    points = np.column_stack((z, y, x)).astype(np.float64, copy=False)
    weights = flat[candidate_idx]
    return points, weights


def _rodrigues_rotation_matrix(axis_zyx: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from axis-angle using Rodrigues' formula."""
    axis = np.asarray(axis_zyx, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n <= np.finfo(np.float64).eps or abs(angle_rad) <= np.finfo(np.float64).eps:
        return np.eye(3, dtype=np.float64)
    axis /= n

    a0, a1, a2 = axis
    k = np.array(
        [[0.0, -a2, a1], [a2, 0.0, -a0], [-a1, a0, 0.0]],
        dtype=np.float64,
    )
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.eye(3, dtype=np.float64) + s * k + (1.0 - c) * (k @ k)


def _matrix_to_euler_zyx_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to intrinsic Z→Y→X Euler angles in degrees."""
    # Intrinsic ZYX == extrinsic XYZ. These formulas follow the XYZ convention.
    sy = np.clip(r[0, 2], -1.0, 1.0)
    ry = np.arcsin(sy)
    cy = np.cos(ry)

    if abs(cy) > 1e-8:
        rx = np.arctan2(-r[1, 2], r[2, 2])
        rz = np.arctan2(-r[0, 1], r[0, 0])
    else:
        # Gimbal lock fallback.
        rx = np.arctan2(r[2, 1], r[1, 1])
        rz = 0.0

    return tuple(np.rad2deg([rz, ry, rx]).tolist())


def _apply_rigid_to_points(
    points_zyx: np.ndarray,
    rotation_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    center_zyx: np.ndarray,
) -> np.ndarray:
    """Apply rigid transform around center to point coordinates."""
    centered = points_zyx - center_zyx[None, :]
    moved = centered @ rotation_zyx.T
    return moved + center_zyx[None, :] + shift_zyx[None, :]


def _compose_rigid(
    rotation_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    step_rotation_zyx: np.ndarray,
    step_shift_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose an incremental rigid step with current rigid transform."""
    new_rotation = step_rotation_zyx @ rotation_zyx
    new_shift = step_rotation_zyx @ shift_zyx + step_shift_zyx
    return new_rotation, new_shift


def _sample_values_and_gradients(
    reference: np.ndarray,
    gradients_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
    points_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample reference values and gradients at floating point coordinates."""
    coords = [points_zyx[:, 0], points_zyx[:, 1], points_zyx[:, 2]]
    values = _scipy_map_coordinates(reference, coords, order=1, mode="nearest")
    gz = _scipy_map_coordinates(gradients_zyx[0], coords, order=1, mode="nearest")
    gy = _scipy_map_coordinates(gradients_zyx[1], coords, order=1, mode="nearest")
    gx = _scipy_map_coordinates(gradients_zyx[2], coords, order=1, mode="nearest")
    grads = np.column_stack((gz, gy, gx))
    return values, grads


def _resample_rigid(
    moving: np.ndarray,
    rotation_zyx: np.ndarray,
    shift_zyx: np.ndarray,
    center_zyx: np.ndarray,
) -> np.ndarray:
    """Resample moving map onto reference grid using current rigid transform."""
    z = np.arange(moving.shape[0], dtype=np.float64)
    y = np.arange(moving.shape[1], dtype=np.float64)
    x = np.arange(moving.shape[2], dtype=np.float64)
    qz, qy, qx = np.meshgrid(z, y, x, indexing="ij")
    q = np.column_stack((qz.ravel(), qy.ravel(), qx.ravel()))

    inv_rotation = rotation_zyx.T
    src = (q - center_zyx[None, :] - shift_zyx[None, :]) @ inv_rotation.T
    src += center_zyx[None, :]
    coords = [src[:, 0], src[:, 1], src[:, 2]]
    out = _scipy_map_coordinates(moving, coords, order=1, mode="nearest")
    return out.reshape(moving.shape).astype(np.float32, copy=False)


def _fit_rigid_real_space(
    reference: np.ndarray,
    moving: np.ndarray,
    optimize_translation: bool,
    optimize_rotation: bool,
    max_steps: int,
    ijk_step_size_min: float,
    ijk_step_size_max: float,
    rotation_step_min_deg: float,
    rotation_step_max_deg: float,
    max_rotation_deg: float,
    sample_max_points: int,
    sample_quantile: float,
    segment_steps: int = 4,
    init_rotation: np.ndarray | None = None,
    init_shift: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """ChimeraX-like rigid fit using real-space gradients and adaptive step size."""
    if reference.shape != moving.shape:
        raise ValueError(f"Volume shapes differ: {reference.shape} vs {moving.shape}")

    center_zyx = 0.5 * (np.array(reference.shape, dtype=np.float64) - 1.0)
    points_zyx, weights = _sample_points_and_weights(
        moving,
        max_points=sample_max_points,
        quantile=sample_quantile,
    )

    gradients_zyx = tuple(np.gradient(reference.astype(np.float64), edge_order=1))
    if init_rotation is None:
        rotation = np.eye(3, dtype=np.float64)
    else:
        rotation = np.asarray(init_rotation, dtype=np.float64).copy()
    if init_shift is None:
        shift = np.zeros(3, dtype=np.float64)
    else:
        shift = np.asarray(init_shift, dtype=np.float64).copy()

    step_size = float(ijk_step_size_max)
    rot_step = float(max(rotation_step_max_deg, rotation_step_min_deg))

    cut_step_size_threshold = 0.25
    step_cut_factor = 0.5
    step_grow_factor = 1.2

    step_types = []
    if optimize_translation:
        step_types.append("translation")
    if optimize_rotation:
        step_types.append("rotation")

    step_count = 0
    fit_score = np.nan

    while step_count < max_steps and step_size > ijk_step_size_min:
        start_pts = _apply_rigid_to_points(points_zyx, rotation, shift, center_zyx)

        for sub in range(segment_steps):
            if not step_types:
                break

            step_kind = step_types[sub % len(step_types)]
            moved_pts = _apply_rigid_to_points(points_zyx, rotation, shift, center_zyx)
            vals, grads = _sample_values_and_gradients(reference, gradients_zyx, moved_pts)
            fit_score = float(np.dot(weights, vals))

            if step_kind == "translation":
                grad_direction = np.sum(weights[:, None] * grads, axis=0)
                grad_norm = float(np.linalg.norm(grad_direction))
                if grad_norm > np.finfo(np.float64).eps:
                    delta = grad_direction * (step_size / grad_norm)
                    rotation, shift = _compose_rigid(
                        rotation,
                        shift,
                        np.eye(3, dtype=np.float64),
                        delta,
                    )

            else:  # rotation step
                rel = moved_pts - center_zyx[None, :]
                torque = np.sum(weights[:, None] * np.cross(rel, grads), axis=0)
                torque_norm = float(np.linalg.norm(torque))
                if torque_norm <= np.finfo(np.float64).eps:
                    continue

                axis = torque / torque_norm
                motion_scale = np.linalg.norm(np.cross(rel, axis[None, :]), axis=1)
                max_motion_scale = float(np.max(motion_scale))
                if max_motion_scale <= np.finfo(np.float64).eps:
                    continue

                angle_rad = step_size / max_motion_scale
                angle_deg = min(np.rad2deg(angle_rad), rot_step)
                if angle_deg <= np.finfo(np.float64).eps:
                    continue

                step_rot = _rodrigues_rotation_matrix(axis, np.deg2rad(angle_deg))
                test_rot, test_shift = _compose_rigid(
                    rotation,
                    shift,
                    step_rot,
                    np.zeros(3, dtype=np.float64),
                )

                if max_rotation_deg > 0:
                    rz, ry, rx = _matrix_to_euler_zyx_deg(test_rot)
                    if (
                        abs(rz) > max_rotation_deg
                        or abs(ry) > max_rotation_deg
                        or abs(rx) > max_rotation_deg
                    ):
                        continue

                rotation, shift = test_rot, test_shift

        end_pts = _apply_rigid_to_points(points_zyx, rotation, shift, center_zyx)
        displacements = np.linalg.norm(end_pts - start_pts, axis=1)
        max_motion = float(np.max(displacements))

        motion_cutoff = cut_step_size_threshold * segment_steps * step_size
        if max_motion < motion_cutoff:
            step_size *= step_cut_factor
            rot_step *= step_cut_factor
        else:
            step_size = min(step_size * step_grow_factor, ijk_step_size_max)
            rot_step = min(rot_step * 1.15, rotation_step_max_deg)

        rot_step = max(rot_step, rotation_step_min_deg)
        step_count += segment_steps

    return rotation, shift, step_count, fit_score


def apply_fourier_shift(volume: np.ndarray, shift_zyx: tuple[float, float, float]) -> np.ndarray:
    """Shift a 3-D volume by fractional voxels using Fourier phase ramps.

    Parameters
    ----------
    volume : np.ndarray
        3-D array to shift.
    shift_zyx : tuple[float, float, float]
        Shift in voxels to apply as (dz, dy, dx). Positive values move content
        toward increasing indices.

    Returns
    -------
    np.ndarray
        Shifted volume with dtype float32.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume; got ndim={volume.ndim}")

    dz, dy, dx = shift_zyx
    nz, ny, nx = volume.shape

    fz = np.fft.fftfreq(nz)
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    kz, ky, kx = np.meshgrid(fz, fy, fx, indexing="ij")

    phase = np.exp(-2j * np.pi * (kz * dz + ky * dy + kx * dx))
    shifted = np.fft.ifftn(np.fft.fftn(volume) * phase)
    return np.real(shifted).astype(np.float32, copy=False)


def estimate_translation_phase_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    subpixel: bool = True,
) -> tuple[tuple[float, float, float], float]:
    """Estimate translation to align ``moving`` onto ``reference``.

    Parameters
    ----------
    reference, moving : np.ndarray
        Equal-shaped 3-D arrays.
    subpixel : bool
        If True, refine the integer peak with quadratic interpolation.

    Returns
    -------
    shift_zyx : tuple[float, float, float]
        Estimated shift in voxels to apply to ``moving``.
    peak_value : float
        Peak value of the normalized phase-correlation volume.
    """
    if reference.shape != moving.shape:
        raise ValueError(
            f"Volumes must have the same shape; got {reference.shape} and {moving.shape}"
        )
    if reference.ndim != 3:
        raise ValueError(f"Expected 3-D volumes; got ndim={reference.ndim}")

    f_ref = np.fft.fftn(reference)
    f_mov = np.fft.fftn(moving)

    cps = f_ref * np.conj(f_mov)
    mag = np.abs(cps)
    eps = np.finfo(np.float64).eps
    cps /= np.maximum(mag, eps)

    corr = np.fft.ifftn(cps)
    corr_abs = np.abs(corr)

    peak_idx = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    peak_val = float(corr_abs[peak_idx])

    # Convert wrapped FFT index to signed integer shift.
    int_shift = []
    for idx, n in zip(peak_idx, reference.shape):
        s = float(idx)
        if s > n // 2:
            s -= n
        int_shift.append(s)

    shift = np.array(int_shift, dtype=np.float64)

    if subpixel:
        refined = shift.copy()
        for ax, (idx, n) in enumerate(zip(peak_idx, reference.shape)):
            i0 = (idx - 1) % n
            i1 = idx
            i2 = (idx + 1) % n

            if ax == 0:
                left, center, right = corr_abs[i0, peak_idx[1], peak_idx[2]], corr_abs[i1, peak_idx[1], peak_idx[2]], corr_abs[i2, peak_idx[1], peak_idx[2]]
            elif ax == 1:
                left, center, right = corr_abs[peak_idx[0], i0, peak_idx[2]], corr_abs[peak_idx[0], i1, peak_idx[2]], corr_abs[peak_idx[0], i2, peak_idx[2]]
            else:
                left, center, right = corr_abs[peak_idx[0], peak_idx[1], i0], corr_abs[peak_idx[0], peak_idx[1], i1], corr_abs[peak_idx[0], peak_idx[1], i2]

            refined[ax] += _quadratic_peak_offset(float(left), float(center), float(right))
        shift = refined

    # The signed peak location is the translation to apply to moving so it
    # aligns with reference (same convention as skimage phase correlation).
    shift_to_apply = tuple(shift.tolist())
    return shift_to_apply, peak_val


def align_volume_to_reference(
    reference: np.ndarray,
    moving: np.ndarray,
    subpixel: bool = True,
    rotational_alignment: bool = False,
    max_rotation_deg: float = 12.0,
    coarse_angle_step_deg: float = 6.0,
    fine_angle_step_deg: float = 2.0,
    rotation_search_target_dim: int = 96,
    speed_preset: str = "balanced",
) -> tuple[np.ndarray, dict]:
    """Align ``moving`` to ``reference`` using real-space rigid fitting.

    Returns
    -------
    aligned : np.ndarray
        Aligned moving volume.
    meta : dict
        Alignment metadata with keys ``shift_zyx`` and ``peak_correlation``.
    """
    if not _HAS_SCIPY:
        raise ImportError(
            "Real-space map fitting requires scipy (ndimage.map_coordinates)."
        )

    if reference.shape != moving.shape:
        raise ValueError(f"Volume shapes differ: {reference.shape} vs {moving.shape}")

    pre_corr = _normalized_correlation(reference, moving)

    preset = (speed_preset or "balanced").lower()
    if preset not in {"fast", "balanced", "accurate"}:
        raise ValueError("speed_preset must be one of: 'fast', 'balanced', 'accurate'")

    # Presets define how aggressively the iterative real-space fitting runs.
    if preset == "fast":
        max_steps = 32
        ijk_step_size_min = 0.10
        ijk_step_size_max = 0.42
        coarse_angle_step_deg = max(coarse_angle_step_deg, 1.0)
        fine_angle_step_deg = max(fine_angle_step_deg, 0.3)
        rotation_search_target_dim = min(rotation_search_target_dim, 56)
        coarse_points = 20000
        coarse_quantile = 0.92
        full_points = 40000
        full_quantile = 0.90
    elif preset == "accurate":
        max_steps = 96
        ijk_step_size_min = 0.01
        ijk_step_size_max = 0.50
        coarse_angle_step_deg = min(coarse_angle_step_deg, 1.0)
        fine_angle_step_deg = min(fine_angle_step_deg, 0.1)
        rotation_search_target_dim = max(rotation_search_target_dim, 96)
        coarse_points = 80000
        coarse_quantile = 0.85
        full_points = 140000
        full_quantile = 0.80
    else:  # balanced
        max_steps = 64
        ijk_step_size_min = 0.03
        ijk_step_size_max = 0.48
        coarse_angle_step_deg = min(max(coarse_angle_step_deg, 1.0), 2.0)
        fine_angle_step_deg = min(max(fine_angle_step_deg, 0.2), 1.0)
        rotation_search_target_dim = min(max(rotation_search_target_dim, 64), 96)
        coarse_points = 50000
        coarse_quantile = 0.90
        full_points = 80000
        full_quantile = 0.86

    ref_fit, _ = _downsample_for_rotation_search(reference, target_max_dim=rotation_search_target_dim)
    mov_fit, stride = _downsample_for_rotation_search(moving, target_max_dim=rotation_search_target_dim)

    # Motion in downsampled coordinates should be scaled to preserve voxel-step intent.
    scale = float(stride)
    fit_rotation, fit_shift, refinement_steps, fit_score = _fit_rigid_real_space(
        ref_fit,
        mov_fit,
        optimize_translation=True,
        optimize_rotation=rotational_alignment,
        max_steps=max_steps,
        ijk_step_size_min=ijk_step_size_min / scale,
        ijk_step_size_max=ijk_step_size_max / scale,
        rotation_step_min_deg=max(fine_angle_step_deg, 1e-3),
        rotation_step_max_deg=max(coarse_angle_step_deg, fine_angle_step_deg),
        max_rotation_deg=max_rotation_deg,
        sample_max_points=coarse_points,
        sample_quantile=coarse_quantile,
    )

    # Full-resolution refinement from the downsampled fit result.
    full_rotation, full_shift, full_steps, full_score = _fit_rigid_real_space(
        reference,
        moving,
        optimize_translation=True,
        optimize_rotation=rotational_alignment,
        max_steps=max(8, max_steps // 4),
        ijk_step_size_min=ijk_step_size_min,
        ijk_step_size_max=ijk_step_size_max,
        rotation_step_min_deg=max(fine_angle_step_deg, 1e-3),
        rotation_step_max_deg=max(coarse_angle_step_deg, fine_angle_step_deg),
        max_rotation_deg=max_rotation_deg,
        sample_max_points=full_points,
        sample_quantile=full_quantile,
        init_rotation=fit_rotation,
        init_shift=fit_shift * scale,
    )

    center = 0.5 * (np.array(reference.shape, dtype=np.float64) - 1.0)
    aligned = _resample_rigid(moving, full_rotation, full_shift, center)

    shift_zyx = tuple(float(v) for v in full_shift.tolist())
    rotation_zyx_deg = tuple(float(v) for v in _matrix_to_euler_zyx_deg(full_rotation))
    post_corr = _normalized_correlation(reference, aligned)

    meta = {
        "shift_zyx": shift_zyx,
        "rotation_zyx_deg": rotation_zyx_deg,
        "peak_correlation": full_score if np.isfinite(full_score) else fit_score,
        "overall_correlation_before": pre_corr,
        "overall_correlation_after": post_corr,
        "overall_correlation_gain": post_corr - pre_corr if np.isfinite(pre_corr) and np.isfinite(post_corr) else np.nan,
        "rotation_search_correlation": fit_score,
        "rotation_refinement_steps": int(refinement_steps + full_steps),
        "speed_preset": speed_preset,
    }
    return aligned, meta


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

    # Vectorized shell aggregation is substantially faster than per-shell masking.
    flat_radius = radius.ravel()
    flat_bin = np.digitize(flat_radius, shell_edges, right=False) - 1
    valid = (flat_bin >= 0) & (flat_bin < num_shells)
    bin_idx = flat_bin[valid]

    cross = (f1 * np.conj(f2)).ravel()[valid]
    p1 = (f1.real * f1.real + f1.imag * f1.imag).ravel()[valid]
    p2 = (f2.real * f2.real + f2.imag * f2.imag).ravel()[valid]

    numerator = np.bincount(bin_idx, weights=np.real(cross), minlength=num_shells)
    power1 = np.bincount(bin_idx, weights=p1, minlength=num_shells)
    power2 = np.bincount(bin_idx, weights=p2, minlength=num_shells)

    denom = np.sqrt(power1 * power2)
    fsc_values = np.divide(
        numerator,
        denom,
        out=np.zeros(num_shells, dtype=np.float64),
        where=denom > 0,
    )

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
    align_map2: bool = False,
    subpixel_alignment: bool = True,
    rotational_alignment: bool = False,
    max_rotation_deg: float = 12.0,
    coarse_angle_step_deg: float = 6.0,
    fine_angle_step_deg: float = 2.0,
    rotation_search_target_dim: int = 96,
    speed_preset: str = "balanced",
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
    align_map2 : bool
        If True, estimate translation via phase correlation and align map 2 to
        map 1 before FSC computation.
    subpixel_alignment : bool
        If True, use quadratic subpixel peak refinement during alignment.
    rotational_alignment : bool
        If True, estimate a global Euler rotation (z, y, x) before translational
        alignment. No masking is applied.
    max_rotation_deg : float
        Maximum absolute rotation angle (degrees) per axis during search.
    coarse_angle_step_deg : float
        Coarse rotational search step size in degrees.
    fine_angle_step_deg : float
        Fine rotational search step size in degrees around the best coarse angle.
    rotation_search_target_dim : int
        Target max size for stride downsampling during rotational search.
    speed_preset : str
        Alignment speed/accuracy preset: ``"fast"``, ``"balanced"``, or
        ``"accurate"``. This controls rotational search density.

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

    alignment = {
        "applied": False,
        "method": None,
        "shift_zyx": (0.0, 0.0, 0.0),
        "rotation_zyx_deg": (0.0, 0.0, 0.0),
        "rotation_refinement_steps": 0,
        "peak_correlation": np.nan,
        "overall_correlation_before": np.nan,
        "overall_correlation_after": np.nan,
        "overall_correlation_gain": np.nan,
        "rotation_search_correlation": np.nan,
        "masking": False,
        "speed_preset": speed_preset,
    }

    if align_map2:
        vol2, align_meta = align_volume_to_reference(
            vol1,
            vol2,
            subpixel=subpixel_alignment,
            rotational_alignment=rotational_alignment,
            max_rotation_deg=max_rotation_deg,
            coarse_angle_step_deg=coarse_angle_step_deg,
            fine_angle_step_deg=fine_angle_step_deg,
            rotation_search_target_dim=rotation_search_target_dim,
            speed_preset=speed_preset,
        )
        alignment = {
            "applied": True,
            "method": (
                "rotation_translation_phase_correlation_no_mask"
                if rotational_alignment
                else "translation_phase_correlation_no_mask"
            ),
            "shift_zyx": align_meta["shift_zyx"],
            "rotation_zyx_deg": align_meta["rotation_zyx_deg"],
            "rotation_refinement_steps": align_meta.get("rotation_refinement_steps", 0),
            "peak_correlation": align_meta["peak_correlation"],
            "overall_correlation_before": align_meta["overall_correlation_before"],
            "overall_correlation_after": align_meta["overall_correlation_after"],
            "overall_correlation_gain": align_meta["overall_correlation_gain"],
            "rotation_search_correlation": align_meta["rotation_search_correlation"],
            "masking": False,
            "speed_preset": align_meta.get("speed_preset", speed_preset),
        }

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
        "alignment":     alignment,
    }
