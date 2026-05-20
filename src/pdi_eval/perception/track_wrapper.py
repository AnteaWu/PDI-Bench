import numpy as np
import torch
import os
import cv2
from .base import BasePerceptor, PerceptionResult
from typing import Optional, Dict

from ..utils.logger import pdi_logger

try:
    from cotracker.predictor import CoTrackerPredictor
except ImportError:
    CoTrackerPredictor = None

class TrackWrapper(BasePerceptor):
    """Co-Tracker v3 wrapper for dense motion cues."""
    
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda"):
        super().__init__(device)
        self.model = self._load_model(checkpoint)

    def _load_model(self, checkpoint):
        """Load local checkpoint or fall back to torch.hub."""
        pdi_logger.info(f"Initializing Co-Tracker (device: {self.device})...")
        
        if checkpoint is None or not os.path.exists(checkpoint):
            pdi_logger.warning("No valid local checkpoint; loading cotracker3_offline via torch.hub...")
            return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

        try:
            model = CoTrackerPredictor(checkpoint=checkpoint).to(self.device)
            pdi_logger.success(f"Loaded local checkpoint: {checkpoint}")
            return model
        except RuntimeError as e:
            pdi_logger.warning(f"Local checkpoint load failed: {str(e)[:50]}...")
            pdi_logger.info("Falling back to torch.hub cotracker3_offline...")
            return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

    def infer(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        grid_size: int = 10,
        bg_grid_size: int = 15,
        **kwargs,
    ) -> PerceptionResult:
        """Joint foreground+background tracking in one forward pass.

        Foreground queries: SIFT -> Shi-Tomasi -> uniform grid (three-level fallback).
        Background: Shi-Tomasi -> uniform grid.
        Merged queries run once; split by n_fg afterward.
        Background tracks go to metadata['bg_tracks'] / metadata['bg_visibility'].
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        frames = []
        max_dim = 880
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        orig_h, orig_w = initial_mask.shape
        curr_h, curr_w = frames[0].shape[:2]
        scale_x, scale_y = orig_w / curr_w, orig_h / curr_h

        video_np = np.stack(frames)
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].to(self.device)

        small_mask = cv2.resize(
            initial_mask.astype(np.uint8), (curr_w, curr_h),
            interpolation=cv2.INTER_NEAREST,
        )
        small_mask = (small_mask > 0).astype(np.uint8)

        first_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        n_fg = grid_size * grid_size
        fg_queries_np = self._sift_sample_queries(first_frame_gray, small_mask, region=1, n=n_fg)
        if len(fg_queries_np) < n_fg // 2:
            pdi_logger.info(f"Few SIFT fg points ({len(fg_queries_np)}); adding Shi-Tomasi")
            extra = self._shi_tomasi_sample_queries(first_frame_gray, small_mask, region=1, n=n_fg - len(fg_queries_np))
            fg_queries_np = np.vstack([fg_queries_np, extra]) if len(fg_queries_np) > 0 else extra
        if len(fg_queries_np) < 2:
            pdi_logger.warning("Very few fg feature points; uniform grid fallback")
            fg_queries_np = self._grid_sample_queries(small_mask, region=1, n=n_fg)

        n_bg = bg_grid_size * bg_grid_size
        bg_queries_np = self._shi_tomasi_sample_queries(first_frame_gray, small_mask, region=0, n=n_bg)
        if len(bg_queries_np) < 2:
            pdi_logger.warning("Very few bg corners; uniform grid fallback")
            bg_queries_np = self._grid_sample_queries(small_mask, region=0, n=n_bg)

        n_fg_pts = len(fg_queries_np)
        all_queries_np = np.vstack([fg_queries_np, bg_queries_np]).astype(np.float32)

        pdi_logger.info(
            f"Co-Tracker ({curr_w}x{curr_h}, "
            f"fg:{n_fg_pts} pts, bg:{len(bg_queries_np)} pts)..."
        )

        if len(all_queries_np) >= 2:
            queries = torch.from_numpy(all_queries_np[None]).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        queries=queries,
                        grid_size=0,
                        grid_query_frame=0,
                    )
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        grid_size=grid_size,
                        grid_query_frame=0,
                    )
            n_fg_pts = tracks.shape[2]

        tracks_np = tracks[0].cpu().numpy()
        tracks_np[:, :, 0] *= scale_x
        tracks_np[:, :, 1] *= scale_y
        vis_np = visibility[0].cpu().numpy()

        fg_tracks = tracks_np[:, :n_fg_pts, :]
        bg_tracks = tracks_np[:, n_fg_pts:, :]
        fg_vis    = vis_np[:, :n_fg_pts]
        bg_vis    = vis_np[:, n_fg_pts:]

        fg_tracks, fg_vis = self._filter_tracks(fg_tracks, fg_vis)
        bg_tracks, bg_vis = self._filter_tracks(bg_tracks, bg_vis)

        breathing_metric = self.calculate_breathing_artifact(fg_tracks)

        del video_tensor
        torch.cuda.empty_cache()

        n_fg_kept = fg_tracks.shape[1]
        tracking_confidence = float(fg_vis.mean()) if fg_vis.size > 0 else 0.0
        pdi_logger.info(f"Tracking done: kept {n_fg_kept} fg points, mean visibility {tracking_confidence:.3f}")

        return PerceptionResult(
            video_id=os.path.basename(video_path),
            frames_count=len(fg_tracks),
            masks=np.zeros((1, 1, 1)),
            h_pixel=np.zeros(len(fg_tracks)),
            x_center=np.zeros(len(fg_tracks)),
            tracks_2d=fg_tracks,
            confidence=fg_vis,
            metadata={
                "breathing_metric": breathing_metric,
                "tracking_confidence": tracking_confidence,
                "bg_tracks": bg_tracks,
                "bg_visibility": bg_vis,
            },
        )

    def _sift_sample_queries(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """SIFT keypoints inside mask region (scale/rotation invariant).

        Returns top-n by response as (M, 3) -> [frame=0, x, y].
        """
        sift = cv2.SIFT_create(nfeatures=n * 4)
        kps = sift.detect(gray, None)
        if not kps:
            return np.empty((0, 3), dtype=np.float32)

        kps = sorted(kps, key=lambda k: k.response, reverse=True)
        h, w = mask.shape
        pts = []
        for kp in kps:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= y < h and 0 <= x < w and mask[y, x] == region:
                pts.append([0.0, float(kp.pt[0]), float(kp.pt[1])])
            if len(pts) >= n:
                break

        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)

    def _shi_tomasi_sample_queries(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """Shi-Tomasi corners inside mask region.

        Returns (M, 3) -> [frame=0, x, y].
        """
        region_mask = (mask == region).astype(np.uint8) * 255
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=n,
            qualityLevel=0.01,
            minDistance=5,
            mask=region_mask,
        )
        if corners is None:
            return np.empty((0, 3), dtype=np.float32)

        pts = [[0.0, float(c[0][0]), float(c[0][1])] for c in corners]
        return np.array(pts, dtype=np.float32)

    def _grid_sample_queries(
        self,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """Uniform spatial grid over region (1=foreground / 0=background).

        sqrt(n) x sqrt(n) cells; one random point per occupied cell.
        Returns (M, 3) [frame=0, x, y], M <= n.
        """
        yy, xx = np.where(mask == region)
        if len(yy) == 0:
            if region == 1:
                pdi_logger.warning("Initial mask has no foreground; bg-only grid tracking")
            return np.empty((0, 3), dtype=np.float32)

        n = min(n, len(yy))
        side = max(1, int(np.ceil(np.sqrt(n))))
        h, w = mask.shape
        cell_h = max(1, h // side)
        cell_w = max(1, w // side)

        rng = np.random.default_rng(42)
        pts = []
        for gy in range(side):
            for gx in range(side):
                y0, y1 = gy * cell_h, min((gy + 1) * cell_h, h)
                x0, x1 = gx * cell_w, min((gx + 1) * cell_w, w)
                in_cell = np.where((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1))[0]
                if len(in_cell) > 0:
                    pick = rng.choice(in_cell)
                    pts.append([0.0, float(xx[pick]), float(yy[pick])])
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break

        if not pts:
            idx = rng.choice(len(yy), min(n, len(yy)), replace=False)
            pts = [[0.0, float(xx[i]), float(yy[i])] for i in idx]

        return np.array(pts, dtype=np.float32)

    def _filter_tracks(
        self,
        tracks: np.ndarray,
        vis: np.ndarray,
        min_vis_ratio: float = 0.3,
        max_jump_px: float = 120.0,
    ) -> tuple:
        """Drop low-quality tracks.

        Args:
            tracks:        (T, N, 2)
            vis:           (T, N) Co-Tracker visibility
            min_vis_ratio: fraction of frames that must be visible
            max_jump_px:   max per-frame motion; larger treated as jump

        Returns:
            filtered_tracks (T, M, 2), filtered_vis (T, M)
        """
        T, N, _ = tracks.shape
        if N == 0:
            return tracks, vis

        vis_ratio = vis.mean(axis=0)
        vis_ok = vis_ratio >= min_vis_ratio

        if T > 1:
            delta = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
            jump_ok = delta.max(axis=0) < max_jump_px
        else:
            jump_ok = np.ones(N, dtype=bool)

        keep = vis_ok & jump_ok
        n_removed = int((~keep).sum())
        if n_removed > 0:
            pdi_logger.info(f"Track filter: removed {n_removed}/{N} low-quality points "
                            f"(low vis:{int((~vis_ok).sum())} jumps:{int((~jump_ok).sum())})")

        if keep.sum() < 2:
            pdi_logger.warning(f"All {N} tracks look bad; keeping all to avoid empty state")
            keep = np.ones(N, dtype=bool)

        return tracks[:, keep, :], vis[:, keep]

    def calculate_breathing_artifact(self, tracks: np.ndarray) -> float:
        """Relative distance CV between two track groups (internal spread)."""
        T, N, _ = tracks.shape
        if N < 2: return 0.0
        
        group_a = tracks[:, :min(5, N//2), :]
        group_b = tracks[:, -min(5, N//2):, :]
        
        centroid_a = np.mean(group_a, axis=1)
        centroid_b = np.mean(group_b, axis=1)
        
        dists = np.linalg.norm(centroid_a - centroid_b, axis=1)
        
        if np.mean(dists) < 1e-6: return 0.0
        return float(np.std(dists) / np.mean(dists))
