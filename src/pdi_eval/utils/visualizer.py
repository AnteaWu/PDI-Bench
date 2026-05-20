import os
import cv2
import matplotlib.pyplot as mp
import numpy as np
from typing import List, Dict, Any, Tuple

class EvidenceVisualizer:
    """Generate PDI audit plots and overlay videos.
    
    Features:
    1. Time-series residual plots for scale and trajectory terms.
    2. Perspective overlay: vanishing point and convergence lines on video.
    3. Side-by-side raw vs pseudo-colored depth.
    """
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = output_dir

    def draw_error_curves(self, scale_errors: np.ndarray, traj_errors: np.ndarray, video_id: str):
        """Plot geometric residuals over time."""
        mp.figure(figsize=(10, 4), dpi=100)
        
        mp.plot(scale_errors, label='Scale Residue (Volume Breathing)', color='blue', alpha=0.8)
        mp.plot(traj_errors, label='Trajectory Residue (Skating)', color='red', alpha=0.8)
        
        mp.title(f"PDI Geometric Residue Analysis: {video_id}")
        mp.xlabel("Frame Index")
        mp.ylabel("Error Magnitude")
        mp.grid(True, linestyle='--', alpha=0.6)
        mp.legend()
        
        save_path = f"{self.output_dir}/{video_id}_error_plot.png"
        mp.tight_layout()
        mp.savefig(save_path)
        mp.close()
        return save_path

    def draw_volume_stability(self, volume_history: np.ndarray, video_id: str):
        """Plot normalized 3D extent over time."""
        mp.figure(figsize=(10, 4), dpi=100)
        
        if len(volume_history) > 0 and np.mean(volume_history) > 1e-6:
            norm_vol = volume_history / np.mean(volume_history)
        else:
            norm_vol = volume_history
            
        mp.plot(norm_vol, label='Normalized 3D Height', color='green', alpha=0.8)
        mp.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        mp.title(f"3D Volume Stability Analysis: {video_id}")
        mp.xlabel("Frame Index")
        mp.ylabel("Relative Height (y/mean_y)")
        mp.grid(True, linestyle='--', alpha=0.6)
        mp.legend()
        
        save_path = f"{self.output_dir}/{video_id}_volume_plot.png"
        mp.tight_layout()
        mp.savefig(save_path)
        mp.close()
        return save_path

    def overlay_perspective_evidence(self, video_frames: np.ndarray, vanishing_point: Tuple[float, float], 
                                     tracks: np.ndarray, pdi_summary: Dict[str, Any]):
        """Draw VP and theoretical convergence lines on frames."""
        output_frames = []
        vp_x, vp_y = int(vanishing_point[0]), int(vanishing_point[1])
        
        for i, frame in enumerate(video_frames):
            canvas = frame.copy()
            
            cv2.circle(canvas, (vp_x, vp_y), 10, (0, 0, 255), -1)
            cv2.putText(canvas, "Vanishing Point", (vp_x + 15, vp_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if i < tracks.shape[1]:
                for p_idx in range(tracks.shape[0]):
                    start_pt = (int(tracks[p_idx, i, 0]), int(tracks[p_idx, i, 1]))
                    cv2.line(canvas, start_pt, (vp_x, vp_y), (0, 255, 0), 1, cv2.LINE_AA)
            
            score = pdi_summary['pdi_score']
            grade = pdi_summary['grade']
            cv2.rectangle(canvas, (10, 10), (450, 80), (0, 0, 0), -1)
            cv2.putText(canvas, f"PDI Score: {score}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f"PDI Grade: {grade}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            output_frames.append(canvas)
            
        return np.stack(output_frames)

    def generate_side_by_side(self, raw_video: np.ndarray, depth_map_video: np.ndarray) -> np.ndarray:
        """Concatenate raw RGB and depth (Jet) side by side."""
        T, H, W, C = raw_video.shape
        sbs_frames = []
        for i in range(T):
            frame_sbs = np.concatenate([raw_video[i], depth_map_video[i]], axis=1)
            sbs_frames.append(frame_sbs)
        return np.stack(sbs_frames)

    def save_mask_sample(self, masks: np.ndarray, video_frames: np.ndarray, video_id: str) -> str:
        """Random frame: overlay SAM2 mask and save PNG.

        Returns:
            output path
        """
        T = masks.shape[0]
        i = int(np.random.randint(0, T))
        mask = (masks[i] > 0).astype(np.uint8)
        if video_frames is not None and i < len(video_frames):
            overlay = video_frames[i].copy()
            fh, fw = overlay.shape[:2]
            if mask.shape != (fh, fw):
                mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
            overlay[mask == 1] = (overlay[mask == 1] * 0.4 + np.array([0, 255, 0]) * 0.6).clip(0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        else:
            out = mask * 255
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"{video_id}_mask_frame{i:04d}.png")
        cv2.imwrite(save_path, out)
        return save_path

    def save_video(self, frames: np.ndarray, filename: str, fps: float = 25.0) -> str:
        """
        Write RGB frame sequence to MP4; tries several fourcc backends.
        frames: (T, H, W, C) RGB, same as overlay_perspective_evidence output.
        """
        if frames is None or len(frames) == 0:
            return ""
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, filename)
        T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
        fourcc_candidates = [
            ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
            ("avc1", cv2.VideoWriter_fourcc(*"avc1")),
            ("X264", cv2.VideoWriter_fourcc(*"X264")),
        ]
        writer = None
        for name, fourcc in fourcc_candidates:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None or not writer.isOpened():
            ext = ".avi"
            out_path_avi = os.path.splitext(out_path)[0] + ext
            writer = cv2.VideoWriter(out_path_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))
            if not writer.isOpened():
                return ""
            out_path = out_path_avi
        try:
            for i in range(T):
                bgr = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
        return out_path
