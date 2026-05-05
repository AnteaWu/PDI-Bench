import gc
import torch
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .perception.sam_wrapper import Sam2Wrapper
from .perception.track_wrapper import TrackWrapper
from .perception.mega_sam_wrapper import MegaSamWrapper
from .geometry.camera import CameraModel
from .geometry.projection import ProjectionJudge
from .evaluator.motion_audit import audit_trajectory_consistency, audit_3d_trajectory_consistency
from .evaluator.scale_audit import audit_scale_consistency
from .evaluator.volume_audit import audit_3d_volume_stability
from .evaluator.reconstruction_audit import audit_reconstruction
from .metrics.pdi_index import PDIIndexCalculator
from .data.cache_manager import CacheManager
from .utils.logger import pdi_logger
from .utils.visualizer import EvidenceVisualizer

class PDIEvaluationPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = CacheManager(cache_dir=config.get("cache_dir", "output/cache/"))
        self.video_id = ""
        self.video_path = ""
        self.last_report = None
        self.last_res_tracks = None
        self.last_masks = None
        self.last_pointmaps = None

    def run(self, video_path: str, click_points: Optional[list] = None, text_query: Optional[str] = None, box_prompt: Optional[list] = None, render_output_dir: Optional[str] = None) -> Dict[str, Any]:
        self.video_path = video_path
        self.video_id = Path(video_path).stem
        pdi_logger.info(f"--- [bold blue]PDI-Eval Pipeline Start: {self.video_id}[/bold blue] ---")
        
        # 1. SAM2 Perception
        if self.cache.exists(self.video_id, "sam2"):
            pdi_logger.info("Found SAM2 cache, skipping...")
            res_2d = self.cache.load_step(self.video_id, "sam2")
        else:
            sam = Sam2Wrapper(checkpoint=self.config['sam_ckpt'], config=self.config.get('sam_cfg'))
            res_2d_raw = sam.infer(video_path, click_points=click_points, text_query=text_query, box_prompt=box_prompt)
            self.cache.save_step(self.video_id, "sam2", {
                "masks": res_2d_raw.masks,
                "h_pixel": res_2d_raw.h_pixel,
                "x_center": res_2d_raw.x_center,
                "is_truncated": res_2d_raw.is_truncated
            })
            res_2d = self.cache.load_step(self.video_id, "sam2")
            del res_2d_raw, sam
            gc.collect()
            torch.cuda.empty_cache()

        # 1.5 Co-Tracker (前景+背景一次性追踪)
        if self.cache.exists(self.video_id, "cotracker"):
            pdi_logger.info("Found Co-Tracker cache, skipping...")
            res_tracks = self.cache.load_step(self.video_id, "cotracker")
        else:
            tracker = TrackWrapper(checkpoint=self.config['tracker_ckpt'])
            res_tracks_raw = tracker.infer(video_path, initial_mask=res_2d['masks'][0])
            self.cache.save_step(self.video_id, "cotracker", {
                "tracks": res_tracks_raw.tracks_2d,
                "visibility": res_tracks_raw.confidence,
                "bg_tracks": res_tracks_raw.metadata.get("bg_tracks", np.empty((0, 0, 2))),
                "bg_visibility": res_tracks_raw.metadata.get("bg_visibility", np.empty((0, 0))),
            })
            res_tracks = self.cache.load_step(self.video_id, "cotracker")
            del res_tracks_raw, tracker
            gc.collect()
            torch.cuda.empty_cache()

        # 2. 3D Engine Selection
        engine_type = self.config.get('engine_3d', 'mega_sam')
        if self.cache.exists(self.video_id, engine_type):
            pdi_logger.info(f"Found {engine_type} cache, skipping...")
            res_3d = self.cache.load_step(self.video_id, engine_type)
        else:
            perceptor = MegaSamWrapper(checkpoint=self.config.get('mega_sam_ckpt'))
            res_3d_raw = perceptor.infer(video_path, masks=res_2d['masks'])
            
            self.cache.save_step(self.video_id, engine_type, {
                "depth_z": res_3d_raw.depth_z,
                "focal_length": res_3d_raw.focal_length,
                "camera_poses": res_3d_raw.camera_poses,
                "pointmaps": res_3d_raw.pointmaps
            })
            res_3d = self.cache.load_step(self.video_id, engine_type)
            del res_3d_raw, perceptor
            gc.collect()
            torch.cuda.empty_cache()

        # 3. Geometry Alignment
        cam = CameraModel(focal_length=res_3d['focal_length'], image_size=res_2d['masks'].shape[1:])
        z_aligned = cam.align_to_unit_scale(res_3d['depth_z'])

        # 3.5 帧数对齐：SAM2/Co-Tracker/Mega-SAM 可能输出不同 T，取最小避免广播错误
        T_masks = res_2d['masks'].shape[0]
        T_tracks = res_tracks['tracks'].shape[0]
        T_depth = res_3d['depth_z'].shape[0]
        T_use = min(T_masks, T_tracks, T_depth)
        if T_use < T_masks or T_use < T_tracks or T_use < T_depth:
            pdi_logger.info(
                f"帧数对齐: masks={T_masks} tracks={T_tracks} depth={T_depth} -> 使用 T={T_use}"
            )
        h_pixel_use = res_2d['h_pixel'][:T_use]
        z_aligned_use = z_aligned[:T_use]
        tracks_use = res_tracks['tracks'][:T_use]
        masks_use = res_2d['masks'][:T_use]
        pm = res_3d.get('pointmaps')
        pointmaps_use = pm[:T_use] if (pm is not None and pm.shape[0] >= T_use) else None

        # 4. Evaluation Layers
        pdi_logger.info("Running Audit Layers...")

        # 4.0 双路消失点估算 (Dual-Path VP)
        fg_tracks_NTD = tracks_use.transpose(1, 0, 2)       # (N_fg, T, 2)
        bg_tracks_raw = res_tracks.get('bg_tracks', np.empty((0, 0, 2)))
        bg_tracks_NTD = bg_tracks_raw.transpose(1, 0, 2) if bg_tracks_raw.ndim == 3 and bg_tracks_raw.shape[0] > 0 else None

        # 读取首帧用于 LSD 背景线检测，同时获取视频帧率
        _cap = cv2.VideoCapture(video_path)
        _video_fps = _cap.get(cv2.CAP_PROP_FPS)
        video_fps = float(_video_fps) if _video_fps > 0 else 24.0
        _first_frames = []
        for _ in range(min(3, int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, _f = _cap.read()
            if ret:
                _first_frames.append(cv2.cvtColor(_f, cv2.COLOR_BGR2RGB))
        _cap.release()
        lsd_frames = np.array(_first_frames) if _first_frames else None

        proj = ProjectionJudge(cx=cam.cx, cy=cam.cy)
        global_vp, fg_vp, bg_vp = proj.estimate_vanishing_point_v2(
            fg_tracks=fg_tracks_NTD,
            bg_tracks=bg_tracks_NTD,
            frames=lsd_frames,
            masks=res_2d['masks'][:len(lsd_frames)] if lsd_frames is not None else None,
        )
        # 轨迹审计 VP 选择：优先 fg_vp，但若 fg_vp 退化或落在物体内部则 fallback 到 bg_vp
        def _vp_in_object_bbox(vp_xy, masks, margin_ratio=0.1):
            """检查消失点是否落在前景物体的包围盒内（含 margin），若是则说明 VP 退化"""
            if masks is None or len(masks) == 0:
                return False
            combined = np.any(masks[:min(5, len(masks))], axis=0)
            ys, xs = np.where(combined)
            if len(xs) == 0:
                return False
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            mx = (x_max - x_min) * margin_ratio
            my = (y_max - y_min) * margin_ratio
            return (x_min - mx <= vp_xy[0] <= x_max + mx and
                    y_min - my <= vp_xy[1] <= y_max + my)

        fg_degenerate = (fg_vp == (cam.cx, cam.cy))
        fg_in_bbox = _vp_in_object_bbox(fg_vp, masks_use)
        if fg_degenerate or fg_in_bbox:
            vp = bg_vp
            reason = "落在物体包围盒内" if fg_in_bbox else "退化为主点"
            pdi_logger.info(f"fg_vp {reason}，fallback 使用 bg_vp ({bg_vp[0]:.1f},{bg_vp[1]:.1f})")
        else:
            vp = fg_vp

        # VP 方向一致性残差：检测 fg_vp 与 bg_vp 是否指向同一方向
        # 用余弦角度差代替欧式距离，对横向运动鲁棒（横向运动时两个 VP 都指向同侧极远处）
        img_h, img_w = res_2d['masks'].shape[1], res_2d['masks'].shape[2]
        fg_dir = np.array([fg_vp[0] - cam.cx, fg_vp[1] - cam.cy], dtype=np.float64)
        bg_dir = np.array([bg_vp[0] - cam.cx, bg_vp[1] - cam.cy], dtype=np.float64)
        fg_norm = float(np.linalg.norm(fg_dir))
        bg_norm = float(np.linalg.norm(bg_dir))
        # fg_vp 飞出画面之外时为浅角运动退化，方向比较无意义
        fg_offscreen = (fg_vp[0] < 0 or fg_vp[0] > img_w or
                        fg_vp[1] < 0 or fg_vp[1] > img_h)
        if fg_norm < 5.0 or bg_norm < 5.0 or fg_offscreen:
            eps_vp = 0.0
        else:
            cos_sim = float(np.dot(fg_dir, bg_dir)) / (fg_norm * bg_norm)
            eps_vp = (1.0 - float(np.clip(cos_sim, -1.0, 1.0))) / 2.0  # [0, 1]

        pdi_logger.info(
            f"VP — global:({global_vp[0]:.1f},{global_vp[1]:.1f})  "
            f"fg:({fg_vp[0]:.1f},{fg_vp[1]:.1f})  "
            f"bg:({bg_vp[0]:.1f},{bg_vp[1]:.1f})  "
            f"eps_vp:{eps_vp:.4f}"
        )

        # 4.1 尺度审计 (h * z)
        eps_scale_seq = audit_scale_consistency(h_pixel_use, z_aligned_use)

        # 4.2 3D 运动学轨迹审计
        # 优先用 MegaSAM 世界坐标系点图直接审查 3D 质心轨迹的物理平滑性，
        # 彻底解耦 h_pixel，免疫动镜头，兼容直线与曲线运动。
        # pointmaps 全零（MegaSAM fallback）时函数内部自动返回零序列。
        eps_traj_seq = audit_3d_trajectory_consistency(pointmaps_use, masks_use, fps=video_fps)

        # 4.3 体积/刚性审计
        vol_cv, vol_history, rigidity_strategy = audit_3d_volume_stability(
            pointmaps_use,
            masks_use,
            tracks=tracks_use,
            h_seq=h_pixel_use,
            visibility=res_tracks['visibility'][:T_use]
        )

        # 5. Metrics Synthesis
        w = self.config.get('weights', {})
        calculator = PDIIndexCalculator(
            w_scale=w.get('w_scale', 0.3),
            w_traj=w.get('w_trajectory', 0.3),
            w_rigidity=w.get('w_rigidity', 0.2),
            w_vp=w.get('w_vp', 0.2),
        )
        # 若背景线过少（bg_vp 退化为主点），eps_vp 置 0 以避免误判
        effective_eps_vp = eps_vp if bg_vp != (cam.cx, cam.cy) else 0.0
        final_report = calculator.compute_pdi(eps_scale_seq, eps_traj_seq, vol_cv, effective_eps_vp)

        # 注入绘图数据与 VP 信息
        final_report['breakdown'].update({
            'scale_history': eps_scale_seq,
            'traj_history': eps_traj_seq,
            'volume_history': vol_history,
            'rigidity_strategy': rigidity_strategy,
        })
        final_report['vanishing_point'] = global_vp
        final_report['fg_vp'] = fg_vp
        final_report['bg_vp'] = bg_vp

        # 6. Reconstruction Audit（可选，由 config.reconstruction_audit.enabled 控制）
        ra_cfg = self.config.get('reconstruction_audit', {})
        # pointmaps 全零说明 mega_sam 走了 fallback，3D 重建无效，跳过审计
        _pm_valid = (pointmaps_use is not None and np.any(pointmaps_use != 0))
        if ra_cfg.get('enabled', False) and _pm_valid:
            pdi_logger.info("Running Reconstruction Audit...")
            # pointmaps Z 分量作为 (T, H, W) 深度图，供数学层使用
            depth_z_3d = pointmaps_use[:, :, :, 2]

            mllm_cfg = ra_cfg.get('mllm', {})
            mllm_config_for_audit = None
            frames_for_audit = None
            save_render_path = None

            if mllm_cfg.get('enabled', False) and mllm_cfg.get('api_key'):
                mllm_config_for_audit = mllm_cfg
                # 用 ffmpeg pipe 读帧，绕过 GStreamer 编解码限制
                from .evaluator.reconstruction_audit import _load_video_frames_ffmpeg
                # max_frames 必须 >= T_use，否则索引越界导致全部走渐变兜底
                frames_for_audit = _load_video_frames_ffmpeg(
                    video_path, max_frames=pointmaps_use.shape[0]
                )
                pdi_logger.info(f"读取视频帧 {len(frames_for_audit)} 帧用于点云着色")
                # 渲染图保存目录：优先使用外部传入的 render_output_dir
                _render_dir = Path(render_output_dir) if render_output_dir else (Path.cwd() / "results" / self.video_id)
                _render_dir.mkdir(parents=True, exist_ok=True)
                save_render_path = str(_render_dir / f"{self.video_id}_recon_render.jpg")

            ra_result = audit_reconstruction(
                pointmaps=pointmaps_use,
                depth_z=depth_z_3d,
                masks=masks_use,
                residuals=None,
                mllm_config=mllm_config_for_audit,
                frames=frames_for_audit,
                save_render_path=save_render_path,
            )
            pdi_logger.info(
                f"Reconstruction Audit: math_pass={ra_result['math']['math_pass']}  "
                f"overall_pass={ra_result['overall_pass']}"
            )
        else:
            ra_result = None

        final_report['reconstruction_audit'] = ra_result

        self.last_report = final_report
        self.last_res_tracks = {**res_tracks, "tracks": tracks_use}
        self.last_masks = masks_use
        self.last_pointmaps = pointmaps_use
        
        pdi_logger.pdi_report(final_report)
        return final_report

    def get_annotated_video(self, output_dir: Optional[str] = None):
        """生成并返回可视化视频路径。对齐视频帧数与 tracks 帧数，避免维度冲突导致未封包。

        Args:
            output_dir: 标注视频保存目录，若为 None 则使用 results/visuals
        """
        if self.last_report is None or self.last_res_tracks is None:
            pdi_logger.warning("无审计结果或追踪数据，跳过标注视频生成")
            return ""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return ""
        frames = np.array(frames)
        tracks_T = self.last_res_tracks["tracks"]  # (T, N, 2)
        T_video, T_tracks = len(frames), tracks_T.shape[0]
        T_use = min(T_video, T_tracks)
        if T_use < T_video or T_use < T_tracks:
            pdi_logger.info(f"标注视频帧数对齐: 视频{T_video} vs 追踪{T_tracks} -> 使用{T_use}帧")
        frames = frames[:T_use]
        tracks_for_viz = tracks_T[:T_use].transpose(1, 0, 2)  # (N, T_use, 2)

        out_dir = output_dir if output_dir is not None else "results/visuals"
        viz = EvidenceVisualizer(output_dir=out_dir)
        annotated = viz.overlay_perspective_evidence(
            frames,
            self.last_report["vanishing_point"],
            tracks_for_viz,
            self.last_report,
        )
        out_path = viz.save_video(annotated, f"{self.video_id}_annotated.mp4")
        if out_path:
            pdi_logger.info(f"标注视频已保存: {out_path}")
        else:
            pdi_logger.warning(f"标注视频写入失败，请检查 {out_dir} 目录与 OpenCV/FFmpeg 编码器")
        return out_path

    def get_error_plot(self):
        viz = EvidenceVisualizer(output_dir="results/visuals")
        return viz.draw_error_curves(
            self.last_report['breakdown']['scale_history'],
            self.last_report['breakdown']['traj_history'],
            self.video_id
        )