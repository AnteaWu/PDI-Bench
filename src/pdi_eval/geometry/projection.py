import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import RANSACRegressor

class ProjectionJudge:
    """Perspective-consistency checks (PDI rules).
    
    Features:
    1. H-X homography residual (core epsilon)
    2. RANSAC vanishing point from lines — dual-path fusion
    3. Depth-height evolution (pinhole h·Z consistency)
    """
    def __init__(self, cx: float, cy: float):
        """
        cx, cy: principal point, usually (W/2, H/2)
        """
        self.cx = cx
        self.cy = cy

    def calculate_hx_residue(self, h_seq: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        """epsilon = |(h1/hi) - (x1-cx)/(xi-cx)|
        
        Args:
            h_seq: pixel heights (T,)
            x_seq: centroid x (T,)
            
        Returns:
            residuals (T-1,)
        """
        if len(h_seq) < 2:
            return np.array([0.0])
            
        h1 = max(h_seq[0], 1.0)
        x1_relative = x_seq[0] - self.cx
        
        residues = []
        
        for i in range(1, len(h_seq)):
            hi = max(h_seq[i], 1.0)
            xi_relative = x_seq[i] - self.cx
            
            ratio_h = h1 / hi
            
            if np.abs(x1_relative) > 1.0 and np.abs(xi_relative) > 1.0:
                ratio_x = x1_relative / xi_relative
                epsilon = np.abs(ratio_h - ratio_x)
            else:
                epsilon = 0.0 
                
            residues.append(epsilon)
            
        return np.array(residues)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def _lines_from_tracks(
        self,
        tracks: np.ndarray,
        min_motion_px: float = 3.0,
        bottom_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Homogeneous motion lines (M, 3) from (N, T, 2) tracks.

        Args:
            tracks:        (N, T, 2)
            min_motion_px: min displacement filter
            bottom_ratio:  if set, keep tracks whose mean y is in top fraction (foreground)
        Returns:
            lines_h: (M, 3) normalized homogeneous lines
        """
        N, T, _ = tracks.shape
        if T < 2 or N < 2:
            return np.empty((0, 3))

        n_avg = max(1, T // 10)
        p_start = tracks[:, :n_avg, :].mean(axis=1)
        p_end   = tracks[:, -n_avg:, :].mean(axis=1)
        motion_len = np.linalg.norm(p_end - p_start, axis=1)

        if bottom_ratio is not None:
            mean_y = tracks[:, :, 1].mean(axis=1)
            y_thresh = np.percentile(mean_y, (1 - bottom_ratio) * 100)
            spatial_mask = mean_y >= y_thresh
            valid_idx = np.where(spatial_mask & (motion_len >= min_motion_px))[0]
            if len(valid_idx) < 4:
                valid_idx = np.where(motion_len >= min_motion_px)[0]
        else:
            valid_idx = np.where(motion_len >= min_motion_px)[0]

        if len(valid_idx) < 2:
            return np.empty((0, 3))

        def to_h(p):
            return np.concatenate([p, np.ones((len(p), 1))], axis=1)

        ps = to_h(p_start[valid_idx])
        pe = to_h(p_end[valid_idx])
        lines_h = np.cross(ps, pe)
        norms = np.linalg.norm(lines_h[:, :2], axis=1, keepdims=True) + 1e-8
        return lines_h / norms

    def _lines_from_lsd(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        min_len: float = 60.0,
        angle_tol: float = 12.0,
    ) -> np.ndarray:
        """LSD lines in background (outside mask), homogeneous (M, 3).

        Args:
            frame:     (H, W, 3) RGB or (H, W) gray
            mask:      (H, W) fg=1
            min_len:   min segment length in px
            angle_tol: drop near-horizontal / near-vertical (deg)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()

        bg_mask = (mask == 0).astype(np.uint8)
        if bg_mask.shape != gray.shape:
            bg_mask = cv2.resize(bg_mask, (gray.shape[1], gray.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        gray = (gray.astype(np.float32) * bg_mask).astype(np.uint8)

        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(gray)[0]
        if detected is None:
            return np.empty((0, 3))

        lines_h = []
        for seg in detected.reshape(-1, 4):
            x1, y1, x2, y2 = seg
            if np.hypot(x2 - x1, y2 - y1) < min_len:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            if angle < angle_tol or angle > (180 - angle_tol):
                continue
            l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
            norm = np.linalg.norm(l[:2]) + 1e-8
            lines_h.append(l / norm)

        return np.array(lines_h) if lines_h else np.empty((0, 3))

    def _ransac_vp(
        self,
        lines_h: np.ndarray,
        thresh: float = 20.0,
        iters: int = 500,
    ) -> Tuple[float, float]:
        """RANSAC intersection of homogeneous lines -> (vp_x, vp_y)."""
        M = len(lines_h)
        if M < 2:
            return self.cx, self.cy

        def dist(vp_h, lines):
            return np.abs(lines @ vp_h) / (np.linalg.norm(vp_h[:2]) + 1e-8)

        best_vp, best_n = None, 0
        rng = np.random.default_rng(42)
        for _ in range(iters):
            idx = rng.choice(M, 2, replace=False)
            vp_h = np.cross(lines_h[idx[0]], lines_h[idx[1]])
            if abs(vp_h[2]) < 1e-8:
                continue
            vp_h = vp_h / vp_h[2]
            n = int((dist(vp_h, lines_h) < thresh).sum())
            if n > best_n:
                best_n, best_vp = n, vp_h

        if best_vp is None:
            return self.cx, self.cy

        inliers = lines_h[dist(best_vp, lines_h) < thresh]
        if len(inliers) >= 2:
            A, b = inliers[:, :2], -inliers[:, 2]
            vp_xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return float(vp_xy[0]), float(vp_xy[1])
        return float(best_vp[0]), float(best_vp[1])

    # ------------------------------------------------------------------ #
    #  Vanishing point API                                             #
    # ------------------------------------------------------------------ #

    def estimate_vanishing_point(
        self,
        tracks: np.ndarray,
        bottom_ratio: float = 0.4,
        min_motion_px: float = 3.0,
        ransac_thresh: float = 20.0,
        ransac_iters: int = 500,
    ) -> Tuple[float, float]:
        """Backward compatible: VP from foreground tracks only."""
        lines_h = self._lines_from_tracks(tracks, min_motion_px, bottom_ratio)
        return self._ransac_vp(lines_h, ransac_thresh, ransac_iters)

    def estimate_vanishing_point_v2(
        self,
        fg_tracks: np.ndarray,
        bg_tracks: Optional[np.ndarray] = None,
        frames: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        min_motion_px: float = 3.0,
        ransac_thresh: float = 20.0,
        ransac_iters: int = 500,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Dual-path VP: foreground motion + background / LSD geometry.

        Path 1 (FG): object motion lines -> fg_vp
        Path 2 (BG): background motion + LSD diagonals -> bg_vp
        Global:      RANSAC on merged lines -> global_vp

        Args:
            fg_tracks:   (N_fg, T, 2)
            bg_tracks:   (N_bg, T, 2) or None
            frames:      (T, H, W, 3) for LSD
            masks:       (T, H, W) or None

        Returns:
            (global_vp, fg_vp, bg_vp)
        """
        fg_min_motion = max(min_motion_px, 10.0)
        fg_lines = self._lines_from_tracks(fg_tracks, fg_min_motion, bottom_ratio=0.4)
        fg_vp = self._ransac_vp(fg_lines, ransac_thresh, ransac_iters)

        bg_line_parts = []

        if bg_tracks is not None and len(bg_tracks) >= 2:
            bg_motion_lines = self._lines_from_tracks(bg_tracks, min_motion_px)
            if len(bg_motion_lines) > 0:
                bg_line_parts.append(bg_motion_lines)

        if frames is not None and masks is not None:
            lsd_all = []
            n_lsd_frames = min(len(frames), len(masks), 3)
            for i in range(n_lsd_frames):
                lsd_f = self._lines_from_lsd(frames[i], masks[i])
                if len(lsd_f) > 0:
                    lsd_all.append(lsd_f)
            if lsd_all:
                lsd_combined = np.concatenate(lsd_all, axis=0)
                bg_line_parts.append(np.tile(lsd_combined, (3, 1)))

        if bg_line_parts:
            bg_lines_all = np.concatenate(bg_line_parts, axis=0)
            bg_vp = self._ransac_vp(bg_lines_all, ransac_thresh, ransac_iters)
        else:
            bg_vp = (self.cx, self.cy)

        all_parts = [p for p in ([fg_lines] + bg_line_parts) if len(p) > 0]
        if all_parts:
            global_vp = self._ransac_vp(np.concatenate(all_parts, axis=0), ransac_thresh, ransac_iters)
        else:
            global_vp = (self.cx, self.cy)

        return global_vp, fg_vp, bg_vp

    def classify_fg_motion(
        self,
        fg_tracks: np.ndarray,
        min_motion_px: float = 5.0,
        parallel_angle_std_thresh: float = 15.0,
        horizontal_ratio_thresh: float = 2.5,
    ) -> str:
        """Classify foreground motion as transverse vs depth-dominant.

        Args:
            fg_tracks:                (N, T, 2)
            min_motion_px:            min motion filter
            parallel_angle_std_thresh: deg std below this -> parallel bundle
            horizontal_ratio_thresh:  |mean_dx|/|mean_dy| above -> horizontal

        Returns:
            "transverse" : lateral / parallel drift (H-VP model weak)
            "depth"      : depth-dominated motion (H-VP applies)
        """
        N, T, _ = fg_tracks.shape
        if T < 2 or N < 2:
            return "depth"

        n_avg = max(1, T // 10)
        p_start = fg_tracks[:, :n_avg, :].mean(axis=1)
        p_end   = fg_tracks[:, -n_avg:, :].mean(axis=1)
        disp    = p_end - p_start
        lengths = np.linalg.norm(disp, axis=1)
        valid   = lengths >= min_motion_px
        if valid.sum() < 3:
            return "depth"

        angles = np.degrees(np.arctan2(disp[valid, 1], disp[valid, 0]))
        angles = angles % 180.0
        angle_std = float(np.std(angles))

        mean_dx = float(np.mean(np.abs(disp[valid, 0])))
        mean_dy = float(np.mean(np.abs(disp[valid, 1]))) + 1e-6

        is_parallel   = angle_std < parallel_angle_std_thresh
        is_horizontal = (mean_dx / mean_dy) > horizontal_ratio_thresh
        if is_parallel and is_horizontal:
            return "transverse"
        return "depth"

    def compute_universal_trajectory_epsilon(
        self,
        h_seq: np.ndarray,
        xy_seq: np.ndarray,
        vanishing_point: Tuple[float, float],
    ) -> np.ndarray:
        """Generalized H-VP residual: h1/ht ~ Dist(p1,VP) / Dist(pt,VP)

        Works for off-center motion; detects "skating" when scale and VP convergence disagree.

        Args:
            h_seq:          (T,)
            xy_seq:         (T, 2) centroid (x,y)
            vanishing_point: (vx, vy)

        Returns:
            (T-1,) residuals
        """
        T = len(h_seq)
        if T < 2 or xy_seq.shape[0] != T:
            return np.zeros(1)

        vp = np.array(vanishing_point, dtype=np.float64)
        dist = np.linalg.norm(xy_seq.astype(np.float64) - vp, axis=1)

        n_ref = min(5, T)
        h_ref   = max(float(np.mean(h_seq[:n_ref])), 1.0)
        dist_ref = max(float(np.mean(dist[:n_ref])), 1.0)

        errors = []
        for t in range(1, T):
            ht   = max(float(h_seq[t]), 1.0)
            dt   = float(dist[t])

            ratio_scale = h_ref / ht
            if dt < 1.0 or dist_ref < 1.0:
                errors.append(0.0)
                continue
            ratio_dist = dist_ref / dt
            errors.append(abs(ratio_scale - ratio_dist))

        return np.array(errors) if errors else np.zeros(1)

    def check_scale_evolution(self, h_seq: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
        """Check h and Z follow pinhole scaling h2/h1 = z1/z2 via const h·Z.
        
        Pinhole: h * Z should be ~constant (up to f and physical height).
        """
        h_z_product = h_seq * z_seq
        normalized_product = h_z_product / h_z_product[0]
        
        return np.abs(normalized_product - 1.0)
