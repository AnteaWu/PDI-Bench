import subprocess
import os
import sys
import glob
import cv2
import numpy as np
import torch
from pathlib import Path
from .base import BasePerceptor, PerceptionResult
from ..utils.logger import pdi_logger

def _masks_to_h_pixel_x_center(masks: np.ndarray):
    """Compute height and x-center time series from (T,H,W) masks."""
    T = masks.shape[0]
    h_list, x_list = [], []
    for t in range(T):
        m = masks[t]
        if np.any(m > 0):
            ys, xs = np.where(m > 0)
            h_list.append(float(np.ptp(ys) + 1))
            x_list.append(float(np.mean(xs)))
        else:
            h_list.append(1.0)
            x_list.append(m.shape[1] / 2.0)
    return np.array(h_list, dtype=np.float64), np.array(x_list, dtype=np.float64)

def _extract_frames(video_path: str, out_dir: str) -> int:
    """Extract frames with 6-digit names; clear directory first to drop stale frames."""
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "*.jpg")) + glob.glob(os.path.join(out_dir, "*.png")):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        path = os.path.join(out_dir, f"{count:06d}.jpg")
        cv2.imwrite(path, frame)
        count += 1
    cap.release()
    return count

class MegaSamWrapper(BasePerceptor):
    def __init__(self, checkpoint=None, device="cuda"):
        super().__init__(device)
        self.mega_sam_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../third_party/mega_sam")
        )
        self.da_ckpt = os.path.join(
            self.mega_sam_root, "Depth-Anything", "checkpoints", "depth_anything_vitl14.pth"
        )
        self.megasam_weights = os.path.join(self.mega_sam_root, "checkpoints", "megasam_final.pth")
        self.raft_weights = os.path.join(self.mega_sam_root, "cvd_opt", "raft-things.pth")

    @staticmethod
    def _parse_intrinsic(K) -> tuple:
        """Robustly parse intrinsics from various K layouts."""
        K = np.asarray(K)
        try:
            if K.ndim == 1 and K.size >= 4:
                return float(K[0]), float(K[1]), float(K[2]), float(K[3])
            if K.ndim == 2 and K.shape[0] >= 3:
                return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        except Exception:
            pass
        pdi_logger.warning("Intrinsics parse failed; using defaults")
        return 1000.0, 1000.0, 320.0, 240.0

    @staticmethod
    def _depth_to_pointmaps(depths, cam_c2w, fx, fy, cx, cy):
        """Build world-coordinate point maps for volume_audit."""
        T, h, w = depths.shape
        # row_idx increases along y; col_idx along x
        row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pointmaps = np.zeros((T, h, w, 3), dtype=np.float32)
        
        for t in range(T):
            d = depths[t].astype(np.float32)
            # 1. Back-project to camera coordinates
            x_cam = (col_idx - cx) * d / fx
            y_cam = (row_idx - cy) * d / fy
            z_cam = d
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
            
            # 2. Transform to world: P_w = R * P_c + t
            R = cam_c2w[t, :3, :3]
            t_vec = cam_c2w[t, :3, 3]
            pts_world = (pts_cam @ R.T) + t_vec
            
            pointmaps[t] = pts_world.reshape(h, w, 3)
        return pointmaps

    def _mega_sam_env(self):
        parts = [
            self.mega_sam_root,
            os.path.join(self.mega_sam_root, "Depth-Anything"),
            os.path.join(self.mega_sam_root, "UniDepth"),
            os.path.join(self.mega_sam_root, "cvd_opt"),
            os.path.join(self.mega_sam_root, "cvd_opt", "core"),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(parts) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def infer(self, video_path: str, masks: np.ndarray, **kwargs) -> PerceptionResult:
        video_id = os.path.basename(video_path).split(".")[0]
        work = os.path.join(self.mega_sam_root, "work_space", video_id)
        frames_dir = os.path.join(work, "frames")
        
        # 1. Extract frames
        n_frames = _extract_frames(video_path, frames_dir)
        if n_frames == 0: return self._fallback_result(video_path, masks)

        # Align paths; clear old depth outputs to avoid stale files when frame count changes
        mono_depth_base = os.path.join(work, "da_depth")
        da_out_dir = os.path.join(mono_depth_base, video_id)
        metric_depth_base = os.path.join(work, "unidepth")
        metric_out_dir = os.path.join(metric_depth_base, video_id)

        for d, pat in [(da_out_dir, "*.npy"), (metric_out_dir, "*.npz")]:
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, pat)):
                    os.remove(f)
        os.makedirs(da_out_dir, exist_ok=True)
        os.makedirs(metric_depth_base, exist_ok=True)
        env = self._mega_sam_env()

        pdi_logger.info(f"Mega-SAM pipeline: processing {video_id} ({n_frames} frames)...")

        # 2. Depth-Anything
        r1 = subprocess.run([
            sys.executable, os.path.join(self.mega_sam_root, "Depth-Anything", "run_videos.py"),
            "--img-path", frames_dir, "--outdir", da_out_dir, "--encoder", "vitl", "--load-from", self.da_ckpt,
        ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
        if r1.returncode != 0:
            pdi_logger.error(f"Depth-Anything failed (code {r1.returncode}):\n{r1.stderr[-2000:]}")
            return self._fallback_result(video_path, masks)

        # 3. UniDepth
        r2 = subprocess.run([
            sys.executable, os.path.join(self.mega_sam_root, "UniDepth", "scripts", "demo_mega-sam.py"),
            "--img-path", frames_dir, "--outdir", metric_depth_base, "--scene-name", video_id,
        ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
        if r2.returncode != 0:
            pdi_logger.error(f"UniDepth failed (code {r2.returncode}):\n{r2.stderr[-2000:]}")
            return self._fallback_result(video_path, masks)

        # 4. DROID camera tracking
        r3 = subprocess.run([
            sys.executable, os.path.join(self.mega_sam_root, "camera_tracking_scripts", "test_demo.py"),
            "--datapath", frames_dir, "--mono_depth_path", mono_depth_base,
            "--metric_depth_path", metric_depth_base, "--scene_name", video_id,
            "--weights", self.megasam_weights, "--disable_vis",
        ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
        if r3.returncode != 0:
            pdi_logger.error(f"DROID tracking failed (code {r3.returncode}):\n{r3.stderr[-2000:]}")
            return self._fallback_result(video_path, masks)

        # 5a. RAFT flow (CVD prerequisite)
        cvd_npz_path = os.path.join(self.mega_sam_root, "outputs_cvd", f"{video_id}_sgd_cvd_hr.npz")
        use_cvd = False
        if os.path.exists(self.raft_weights):
            r4 = subprocess.run([
                sys.executable, os.path.join(self.mega_sam_root, "cvd_opt", "preprocess_flow.py"),
                "--datapath", frames_dir,
                "--model", self.raft_weights,
                "--scene_name", video_id,
                "--mixed_precision",
            ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
            if r4.returncode != 0:
                pdi_logger.warning(f"RAFT flow failed; skip CVD refinement:\n{r4.stderr[-1000:]}")
            else:
                # 5b. CVD depth refinement
                r5 = subprocess.run([
                    sys.executable, os.path.join(self.mega_sam_root, "cvd_opt", "cvd_opt.py"),
                    "--scene_name", video_id,
                    "--output_dir", "outputs_cvd",
                    "--w_grad", "2.0",
                    "--w_normal", "5.0",
                ], cwd=self.mega_sam_root, env=env, capture_output=True, text=True)
                if r5.returncode != 0:
                    pdi_logger.warning(f"CVD opt failed; falling back to raw DROID output:\n{r5.stderr[-1000:]}")
                elif os.path.exists(cvd_npz_path):
                    use_cvd = True
                    pdi_logger.info("CVD depth refinement done; using temporally consistent depths")
        else:
            pdi_logger.warning(
                f"RAFT weights missing ({self.raft_weights}); skip CVD. "
                "Download under third_party/mega_sam/cvd_opt/: gdown 1R8m_jMvCun-N45XkMvHlG0P38kXy-h6I"
            )

        # 6. Load result and lift to point maps
        npz_path = cvd_npz_path if use_cvd else os.path.join(self.mega_sam_root, "outputs", f"{video_id}_droid.npz")
        if not os.path.exists(npz_path): return self._fallback_result(video_path, masks)

        data = np.load(npz_path, allow_pickle=True)
        depths = data["depths"]     # (T, H, W)
        cam_c2w = data["cam_c2w"]   # (T, 4, 4)
        fx, fy, cx, cy = self._parse_intrinsic(data["intrinsic"])
        
        T_out = min(depths.shape[0], len(masks))
        pdi_logger.info("Running 3D reprojection and scale normalization...")
        
        pointmaps = self._depth_to_pointmaps(depths[:T_out], cam_c2w[:T_out], fx, fy, cx, cy)

        # Foreground depth Z sequence
        depth_z = []
        for t in range(T_out):
            d = depths[t]
            m = masks[t]
            if m.shape[:2] != d.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            val = np.median(d[m > 0]) if np.any(m > 0) else np.median(d)
            depth_z.append(val)

        depth_z_norm = np.array(depth_z) / (depth_z[0] + 1e-8)
        h_pixel, x_center = _masks_to_h_pixel_x_center(masks[:T_out])

        return PerceptionResult(
            video_id=video_id,
            frames_count=T_out,
            masks=masks[:T_out],
            h_pixel=h_pixel,
            x_center=x_center,
            depth_z=depth_z_norm,
            focal_length=fx,
            camera_poses=cam_c2w[:T_out],
            pointmaps=pointmaps,
            metadata={"engine": "Mega-SAM-Complete-Logic"}
        )

    def _fallback_result(self, video_path: str, masks: np.ndarray) -> PerceptionResult:
        """Safe fallback when Mega-SAM subprocesses fail."""
        T = len(masks)
        h_pixel, x_center = _masks_to_h_pixel_x_center(masks)
        return PerceptionResult(
            video_id=os.path.basename(video_path).split(".")[0],
            frames_count=T,
            masks=masks,
            h_pixel=h_pixel,
            x_center=x_center,
            depth_z=np.ones(T),
            focal_length=1000.0,
            camera_poses=np.eye(4)[None].repeat(T, axis=0),
            pointmaps=np.zeros((T, masks.shape[1], masks.shape[2], 3)),
            metadata={"engine": "Mega-SAM-Fallback"}
        )
