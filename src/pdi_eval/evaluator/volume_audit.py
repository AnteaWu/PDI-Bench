import numpy as np
import cv2
from typing import Optional, Tuple
from ..utils.logger import pdi_logger


def audit_3d_rigidity_cv(
    pointmaps: np.ndarray,
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    masks: Optional[np.ndarray] = None,
    num_pairs: int = 30,
) -> Tuple[float, np.ndarray]:
    """3D rigidity audit via pointmap-sampled pairs (world-space distance invariance).

    Anchor selection uses three filters:
    1. Visibility: visibility > 0.5
    2. Depth gradient: drop points at sharp Z jumps (edge bleed / occluder boundaries);
       threshold = 75th percentile of visible-point gradients.
    3. Scoring: score = 3D_distance * min(boundary distance at i, at j)
       favors wide baselines (SNR) and interior points (reliability).
       distanceTransform uses SAM2 mask if present, else valid pointmap region.

    Args:
        pointmaps:   (T, H, W, 3) MegaSAM world point map
        tracks_2d:   (T, N, 2) Co-Tracker tracks
        visibility:  (T, N) visibility
        masks:       (T, H, W) SAM2 foreground masks for distanceTransform
        num_pairs:   number of anchor pairs to sample
    Returns:
        final_score: mean rigidity failure metric
        history:     per-frame rigidity score
    """
    T, N, _ = tracks_2d.shape
    H, W = pointmaps.shape[1], pointmaps.shape[2]

    # ==========================================
    # Step 1: sample 3D trajectory from pointmaps
    # ==========================================
    pts_3d = np.zeros((T, N, 3))
    for t in range(T):
        u = np.clip(np.round(tracks_2d[t, :, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(tracks_2d[t, :, 1]).astype(int), 0, H - 1)
        pts_3d[t] = pointmaps[t, v, u]

    # ==========================================
    # Step 2: depth-gradient filter + distanceTransform scoring
    #
    # score = 3D separation * min(edge distance i, j)
    # large separation -> better deformation SNR
    # large edge_min -> both points well inside the region
    # ==========================================

    # distanceTransform: prefer SAM2 mask, else pointmap validity
    if masks is not None:
        m0 = masks[0]
        if m0.shape != (H, W):
            m0 = cv2.resize(m0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask0_pt = m0.astype(np.uint8)
    else:
        mask0_pt = pointmaps[0].any(axis=-1).astype(np.uint8)
    dist_map = cv2.distanceTransform(mask0_pt, cv2.DIST_L2, 5)  # (H,W); larger = more interior

    # Depth gradient filter: adaptive 75th pct, covers outline + internal boundaries
    z0 = pointmaps[0, :, :, 2].astype(np.float32)
    gx = cv2.Sobel(z0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z0, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    all_u = np.clip(np.round(tracks_2d[0, :, 0]).astype(int), 0, W - 1)
    all_v = np.clip(np.round(tracks_2d[0, :, 1]).astype(int), 0, H - 1)
    vis_filter = visibility[0] > 0.5
    grad_at_all = grad_mag[all_v, all_u]
    vis_count = int(vis_filter.sum())
    grad_thresh = float(np.percentile(grad_at_all[vis_filter], 75)) if vis_count > 4 else np.inf
    # Relax: gradient+visible -> visible only
    valid_idx = np.array([], dtype=int)
    for filt in [vis_filter & (grad_at_all < grad_thresh), vis_filter]:
        valid_idx = np.where(filt)[0]
        if len(valid_idx) >= 5:
            break

    if len(valid_idx) < 5:
        return 1.0, np.full(T, 1.0)

    # distanceTransform at each track on frame 0
    u0 = np.clip(np.round(tracks_2d[0, valid_idx, 0]).astype(int), 0, W - 1)
    v0 = np.clip(np.round(tracks_2d[0, valid_idx, 1]).astype(int), 0, H - 1)
    edge_dist = dist_map[v0, u0]

    # Pairwise 3D distances
    pts_0 = pts_3d[0, valid_idx]
    diff = pts_0[:, np.newaxis, :] - pts_0[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    score_matrix = dist_matrix * np.minimum(
        edge_dist[:, np.newaxis],
        edge_dist[np.newaxis, :]
    )
    # Large dist_matrix: pairs are spread -> better deformation SNR
    # Large edge_min: both points are in the interior -> reliable
    
    i_upper, j_upper = np.triu_indices_from(score_matrix, k=1)
    scores    = score_matrix[i_upper, j_upper]
    distances = dist_matrix[i_upper, j_upper]
    actual_i  = valid_idx[i_upper]
    actual_j  = valid_idx[j_upper]

    sorted_args   = np.argsort(scores)[::-1]
    selected_args = sorted_args[:num_pairs]

    pair_i = actual_i[selected_args]
    pair_j = actual_j[selected_args]
    d_0    = distances[selected_args]

    # Drop tiny 3D distances (avoid Inf in ratio d_t/d_0)
    valid_mask = d_0 > 1e-3
    pair_i, pair_j, d_0 = pair_i[valid_mask], pair_j[valid_mask], d_0[valid_mask]

    if len(d_0) < 3:
        return 1.0, np.full(T, 1.0)

    # ==========================================
    # Step 3 & 4: per-frame distance ratio + robust MAD/median
    # ==========================================
    rigidity_history = [0.0]  # frame 0 baseline (perfect)

    for t in range(1, T):
        vis_mask = (visibility[t, pair_i] > 0.5) & (visibility[t, pair_j] > 0.5)

        if np.sum(vis_mask) < 3:
            # heavy occlusion: keep last score
            rigidity_history.append(rigidity_history[-1])
            continue

        cur_i, cur_j, cur_d0 = pair_i[vis_mask], pair_j[vis_mask], d_0[vis_mask]
        p_i, p_j = pts_3d[t, cur_i], pts_3d[t, cur_j]
        d_t = np.linalg.norm(p_i - p_j, axis=-1)
        ratios = d_t / cur_d0
        median_r = np.median(ratios)
        mad_r = np.median(np.abs(ratios - median_r))
        rigidity_history.append(float(mad_r / (median_r + 1e-6)))

    rigidity_history = np.array(rigidity_history)
    # Skip frame 0 (reference only) so short clips are not biased
    score_frames = rigidity_history[1:] if len(rigidity_history) > 1 else rigidity_history
    return float(np.mean(score_frames)), rigidity_history


def audit_rigidity_stability(
    tracks: np.ndarray,
    h_seq: np.ndarray,
    n_pairs: int = 30,
) -> Tuple[float, np.ndarray]:
    """Rotation-robust 2D rigidity from Co-Tracker pair ratios.

    No division by h(t) (rotation would change h and false-trigger).
    Uses pairwise distance ratio coherence: rigid scaling keeps all ratios aligned;
    non-physical stretch spreads ratios apart.

    Define:
        ratio_ij(t) = d_ij(t) / d_ij(0)
        score(t)    = std(ratios) / (mean(ratios) + 1e-6)  # coherence failure

    Args:
        tracks:  (T, N, 2) Co-Tracker tracks
        h_seq:   (T,) kept for API compatibility (unused)
        n_pairs: number of random anchor pairs

    Returns:
        (rigidity_cv, rigidity_history)
        rigidity_cv:      higher -> more "jello"
        rigidity_history: per-frame coherence failure
    """
    T, N, _ = tracks.shape
    if N < 2 or T < 2:
        return 0.0, np.zeros(T)

    rng = np.random.default_rng(42)
    actual_pairs = min(n_pairs, N * (N - 1) // 2)
    pairs = []
    seen = set()
    while len(pairs) < actual_pairs:
        i, j = rng.choice(N, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((int(i), int(j)))

    first_dists = np.array([
        np.linalg.norm(tracks[0, i] - tracks[0, j]) + 1e-6
        for i, j in pairs
    ])

    rigidity_history = [1.0]  # t=0 reference
    for t in range(1, T):
        curr_dists = np.array([
            np.linalg.norm(tracks[t, i] - tracks[t, j])
            for i, j in pairs
        ])
        ratios = curr_dists / first_dists
        mean_r = float(np.mean(ratios))
        score = float(np.std(ratios)) / (mean_r + 1e-6)
        rigidity_history.append(score)

    rigidity_history = np.array(rigidity_history)
    return float(np.mean(rigidity_history)), rigidity_history


def audit_3d_volume_stability(
    pointmaps: Optional[np.ndarray],
    masks: np.ndarray,
    tracks: Optional[np.ndarray] = None,
    h_seq: Optional[np.ndarray] = None,
    visibility: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, str]:
    """Volume / rigidity stability (three-strategy cascade).

    Priority:
    1. 3D rigid pairwise ratios (needs pointmaps + tracks + visibility) -- most robust
    2. 3D point-cloud "height" swing (needs pointmaps + masks)
    3. 2D Co-Tracker rigidity (needs tracks + h_seq) -- fallback without 3D

    Returns:
        (rigidity_cv, history, strategy_name)
    """
    T = len(masks)

    # --- Strategy 1: 3D pairwise ratio (MAD/median) ---
    if pointmaps is not None and tracks is not None and visibility is not None:
        mask0 = masks[0]
        if mask0.shape[:2] != pointmaps.shape[1:3]:
            mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]), interpolation=cv2.INTER_NEAREST)
        fg_pts0 = pointmaps[0][mask0 > 0]
        if fg_pts0.shape[0] > 0 and np.mean(np.any(fg_pts0 != 0, axis=-1)) > 0.5:
            pdi_logger.info("Rigidity: strategy 1 (3D rigid pairwise ratios)")
            cv, hist = audit_3d_rigidity_cv(pointmaps, tracks, visibility, masks)
            return cv, hist, "Strategy 1 (3D rigid pairwise ratios)"

    # --- Strategy 2: 3D point-cloud extent ---
    if pointmaps is not None:
        mask0 = masks[0]
        if mask0.shape[:2] != pointmaps.shape[1:3]:
            mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]), interpolation=cv2.INTER_NEAREST)
        fg_pts0 = pointmaps[0][mask0 > 0]
        fg_valid = (fg_pts0.shape[0] > 0) and (np.mean(np.any(fg_pts0 != 0, axis=-1)) > 0.5)
        if fg_valid:
            pdi_logger.info("Rigidity: strategy 2 (3D point-cloud extent)")
            vol_history = []
            for t in range(T):
                pm_t = pointmaps[t]
                m_t = masks[t]
                if m_t.shape[:2] != pm_t.shape[:2]:
                    m_t = cv2.resize(m_t.astype(np.uint8), (pm_t.shape[1], pm_t.shape[0]), interpolation=cv2.INTER_NEAREST)
                bool_mask = m_t > 0
                if np.any(bool_mask):
                    y_pts = pm_t[bool_mask][:, 1]
                    h_3d = np.percentile(y_pts, 95) - np.percentile(y_pts, 5)
                    vol_history.append(h_3d)
                else:
                    vol_history.append(vol_history[-1] if vol_history else 0.0)
            vol_history = np.array(vol_history)
            if np.mean(vol_history) > 1e-6:
                vol_cv = float(np.std(vol_history) / np.mean(vol_history))
                return vol_cv, vol_history, "Strategy 2 (3D point-cloud extent)"

    # --- Strategy 3: 2D Co-Tracker rigidity ---
    if tracks is not None and h_seq is not None:
        pdi_logger.info("Rigidity: strategy 3 (2D Co-Tracker pairwise distances)")
        cv, hist = audit_rigidity_stability(tracks, h_seq)
        return cv, hist, "Strategy 3 (2D Co-Tracker pairwise distances)"

    # --- Fallback ---
    pdi_logger.warning("Rigidity: no strategy available; returning zero")
    return 0.0, np.zeros(T), "Fallback (no usable data)"
