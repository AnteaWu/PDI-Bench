import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple

class EvidenceVisualizer:
    """证据生成器 (Evidence Generator)
    
    核心功能：
    1. 时域残差图：绘制 ε_scale 和 ε_trajectory 波动。
    2. 透视线叠加：在原视频上叠加运动消失点和收敛线。
    3. 侧视对比：原视频与深度图的 side-by-side 生成。
    """
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = output_dir

    def draw_error_curves(self, scale_errors: np.ndarray, traj_errors: np.ndarray, video_id: str):
        """绘制几何残差随时间波动的曲线图"""
        plt.figure(figsize=(10, 4), dpi=100)
        
        # 绘制残差曲线
        plt.plot(scale_errors, label='Scale Residue (Volume Breathing)', color='blue', alpha=0.8)
        plt.plot(traj_errors, label='Trajectory Residue (Skating)', color='red', alpha=0.8)
        
        plt.title(f"PDI Geometric Residue Analysis: {video_id}")
        plt.xlabel("Frame Index")
        plt.ylabel("Error Magnitude")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # 保存图表
        save_path = f"{self.output_dir}/{video_id}_error_plot.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def draw_volume_stability(self, volume_history: np.ndarray, video_id: str):
        """绘制 3D 高度随时间波动的曲线图"""
        plt.figure(figsize=(10, 4), dpi=100)
        
        # 归一化高度以方便观察波动
        if len(volume_history) > 0 and np.mean(volume_history) > 1e-6:
            norm_vol = volume_history / np.mean(volume_history)
        else:
            norm_vol = volume_history
            
        plt.plot(norm_vol, label='Normalized 3D Height', color='green', alpha=0.8)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f"3D Volume Stability Analysis: {video_id}")
        plt.xlabel("Frame Index")
        plt.ylabel("Relative Height (y/mean_y)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        save_path = f"{self.output_dir}/{video_id}_volume_plot.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def overlay_perspective_evidence(self, video_frames: np.ndarray, vanishing_point: Tuple[float, float], 
                                     tracks: np.ndarray, pdi_summary: Dict[str, Any]):
        """在视频帧上叠加消失点 (VP) 和理论收敛线"""
        output_frames = []
        vp_x, vp_y = int(vanishing_point[0]), int(vanishing_point[1])
        
        for i, frame in enumerate(video_frames):
            # 深拷贝一帧用于绘图
            canvas = frame.copy()
            
            # 1. 绘制当前的消失点 (VP)
            cv2.circle(canvas, (vp_x, vp_y), 10, (0, 0, 255), -1) # 红色实心圆标记消失点
            cv2.putText(canvas, "Vanishing Point", (vp_x + 15, vp_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 2. 绘制从物体各个追踪点指向消失点的“理论收敛线”
            if i < tracks.shape[1]:
                for p_idx in range(tracks.shape[0]):
                    start_pt = (int(tracks[p_idx, i, 0]), int(tracks[p_idx, i, 1]))
                    # 绿色虚线表示理论一致性路径
                    cv2.line(canvas, start_pt, (vp_x, vp_y), (0, 255, 0), 1, cv2.LINE_AA)
            
            # 3. 在左上角标记 PDI 等级和总分
            score = pdi_summary['pdi_score']
            grade = pdi_summary['grade']
            cv2.rectangle(canvas, (10, 10), (450, 80), (0, 0, 0), -1) # 黑色底框
            cv2.putText(canvas, f"PDI Score: {score}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f"PDI Grade: {grade}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            output_frames.append(canvas)
            
        return np.stack(output_frames)

    def generate_side_by_side(self, raw_video: np.ndarray, depth_map_video: np.ndarray) -> np.ndarray:
        """生成原视频与深度图的对比视频（Side-by-Side）"""
        # 1. 确保尺寸对齐
        T, H, W, C = raw_video.shape
        # 2. 对深度图进行伪彩色映射 (Jet)
        # 3. 将原图和深度图横向拼接
        sbs_frames = []
        for i in range(T):
            frame_sbs = np.concatenate([raw_video[i], depth_map_video[i]], axis=1)
            sbs_frames.append(frame_sbs)
        return np.stack(sbs_frames)

    def save_mask_sample(self, masks: np.ndarray, video_frames: np.ndarray, video_id: str) -> str:
        """随机抽一帧，将 SAM2 mask 叠加到原始帧上保存为单张 PNG。

        Returns:
            保存文件路径
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
        将标注帧序列写入 MP4。处理编码器不可用、目录缺失、帧数不一致等问题。
        frames: (T, H, W, C) RGB，与 overlay_perspective_evidence 输出一致。
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
