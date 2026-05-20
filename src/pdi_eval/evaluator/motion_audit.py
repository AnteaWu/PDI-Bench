import numpy as np
from typing import Tuple, Optional


def _medfilt1d(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """1D median filter without scipy (numpy sliding window)."""
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1)


def audit_3d_trajectory_consistency(
    pointmaps: Optional[np.ndarray],
    masks: np.ndarray,
    fps: float = 24.0,
) -> np.ndarray:
    """3D kinematic trajectory audit in world coordinates.

    Samples foreground centroid from MegaSAM world point maps and scores
    motion smoothness (acceleration spikes + direction reversals).

    Compared with classic VP-based methods:
    - No dependence on h_pixel; orthogonal to scale term
    - Robust to camera motion (absolute position in world frame)
    - Works for straight and curved paths

    Args:
        pointmaps: (T, H, W, 3) world-coordinate point map (MegaSAM)
        masks:     (T, H, W) foreground masks (SAM2)
        fps:       frame rate for time normalization (m/s, m/s^2)

    Returns:
        (T-1,) trajectory residual sequence
    """
    if pointmaps is None or masks is None:
        return np.zeros(1)

    T = pointmaps.shape[0]
    if T < 3:
        return np.zeros(max(1, T - 1))

    # Validity: all-zero pointmaps means MegaSAM used fallback
    mask0 = masks[0]
    if mask0.shape[:2] != pointmaps.shape[1:3]:
        import cv2
        mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]),
                           interpolation=cv2.INTER_NEAREST)
    fg_pts0 = pointmaps[0][mask0 > 0]
    if fg_pts0.shape[0] == 0 or np.mean(np.any(fg_pts0 != 0, axis=-1)) < 0.5:
        return np.zeros(T - 1)

    # Step 1: robust 3D centroid
    # - Median instead of mean to reduce depth bleeding at SAM2 mask boundaries
    #   (edges often leak background depth; median resists outliers)
    # - If mask is lost, keep previous centroid to avoid jumping to world origin
    world_traj = np.zeros((T, 3))
    last_valid = np.zeros(3)

    for t in range(T):
        m = masks[t]
        pm = pointmaps[t]
        if m.shape[:2] != pm.shape[:2]:
            import cv2
            m = cv2.resize(m.astype(np.uint8), (pm.shape[1], pm.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        valid_pts = pm[m > 0]
        if len(valid_pts) > 10:
            centroid = np.median(valid_pts, axis=0)
            world_traj[t] = centroid
            last_valid = centroid
        else:
            world_traj[t] = last_valid

    # Step 1.5: light temporal median smoothing for MegaSAM depth flicker
    for i in range(3):
        world_traj[:, i] = _medfilt1d(world_traj[:, i], k=3)

    # Step 2: 3D velocity and acceleration (physical units via dt)
    # dt = 1/fps makes the metric frame-rate invariant
    dt = 1.0 / max(fps, 1.0)
    velocity = np.diff(world_traj, axis=0) / dt      # (T-1, 3), m/s
    speed = np.linalg.norm(velocity, axis=1)          # (T-1,), m/s
    acceleration = np.diff(velocity, axis=0) / dt    # (T-2, 3), m/s^2
    accel_mag = np.linalg.norm(acceleration, axis=1)  # (T-2,)

    # Normalize by global mean speed, not per-frame speed.
    # Per-frame tiny denominators blow up noise on slow frames.
    # Global reference measures acceleration vs typical motion scale in this clip;
    # static clips fall back via global_floor tied to accel magnitude.
    speed_median = float(np.median(speed))
    accel_median = float(np.median(accel_mag))
    global_floor = accel_median * 2.0   # noise floor from accel median
    speed_ref    = max(speed_median, global_floor, 1e-6)

    # Relative acceleration rate (1/s)
    relative_accel_raw = accel_mag / speed_ref          # (T-2,)

    # tanh compress unbounded values to [0, 2), matching angle_penalty scale
    # tanh(x/5)*2: x=5 -> 1.52, x=10 -> 1.93, x>=15 -> ~2
    relative_accel = 2.0 * np.tanh(relative_accel_raw / 5.0)  # (T-2,)

    # Step 3: adjacent velocity cosine (penalize sharp direction reversals)
    v1, v2 = velocity[:-1], velocity[1:]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9
    n2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9
    cos_angles = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=1), -1.0, 1.0)

    # Near-static: direction penalty is meaningless; use speed_ref gate
    moving = (speed[:-1] > speed_ref * 0.1) & (speed[1:] > speed_ref * 0.1)
    angle_penalty = np.zeros_like(cos_angles)
    angle_penalty[moving] = 1.0 - cos_angles[moving]   # same direction=0, reversal=2

    # Step 4: combine and pad to length (T-1,)
    eps_traj = relative_accel * 0.5 + angle_penalty * 0.5
    return np.insert(eps_traj, 0, eps_traj[0])


def audit_trajectory_consistency(
    h_seq: np.ndarray,
    xy_seq: np.ndarray,
    vanishing_point: Tuple[float, float],
) -> np.ndarray:
    """Generalized perspective trajectory audit (handles lateral motion).

    Auto mode:
    - Forward / oblique: log H-VP homography residual, log(h1/ht) vs log(d1/dt)
    - Lateral translation (VP at infinity): height stability |h(t)-h(0)|/h(0)

    Lateral if VP distance range ratio < 5%.

    Args:
        h_seq:           (T,) SAM2 pixel heights
        xy_seq:          (T, 2) Co-Tracker centroid positions
        vanishing_point: (vx, vy)

    Returns:
        (T-1,) trajectory residuals
    """
    T = len(h_seq)
    if T < 2:
        return np.zeros(1)

    vp = np.array(vanishing_point, dtype=np.float64)
    dist = np.linalg.norm(xy_seq.astype(np.float64) - vp, axis=1)  # (T,)

    dist_range_ratio = float(np.ptp(dist)) / (float(np.mean(dist)) + 1e-6)

    if dist_range_ratio < 0.05:
        # Lateral: depth ~ const, h should stay stable
        h0 = max(float(h_seq[0]), 1e-6)
        errors = np.abs(h_seq[1:] - h0) / h0
        return errors

    # Forward / oblique: log-space H-VP
    log_h = np.log(np.maximum(h_seq, 1e-6))
    log_d = np.log(np.maximum(dist, 1e-6))

    # First 5 frames median baseline to damp early noise
    n_ref = min(5, T)
    h_base = float(np.median(log_h[:n_ref]))
    d_base = float(np.median(log_d[:n_ref]))

    log_h_ratio = log_h - h_base
    log_d_ratio = log_d - d_base

    errors = np.abs(log_h_ratio - log_d_ratio)
    return errors[1:]
