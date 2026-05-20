import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

DEFAULT_BATCH_MODELS = ["Flow", "cogvideoX", "hunyuan", "seedance", "Sora", "wan22"]
DEFAULT_BATCH_SCENES = [
    "axial_rigid",
    "tracking_nonhuman_bio",
    "nonrigid_nonhuman_bio",
    "orbital_rotation",
    "partial_occlusion",
]


def _pick_cache_by_video_stem(candidates: List[Path], video_stem: str, tag: str) -> Path:
    target = f"{video_stem}_{tag}"
    exact = [p for p in candidates if p.stem == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(f"Multiple cache files matched {target}: {exact}")
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Cannot uniquely resolve *_{tag}.npz for video '{video_stem}'. Found: {candidates}")


def resolve_inputs(
    input_dir: Optional[Path],
    video: Optional[Path],
    sam2_cache: Optional[Path],
    track_cache: Optional[Path],
    megasam_cache: Optional[Path],
) -> Tuple[Path, Path, Path, Path]:
    if input_dir is None:
        if video is None or sam2_cache is None or track_cache is None or megasam_cache is None:
            raise RuntimeError("Provide --input-dir, or provide --video --sam2-cache --track-cache --megasam-cache.")
        return video, sam2_cache, track_cache, megasam_cache

    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Invalid input directory: {input_dir}")

    if video is None:
        mp4_candidates = sorted([p for p in input_dir.glob("*.mp4") if "_hero" not in p.stem.lower()])
        if len(mp4_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one source mp4 (excluding *_hero*.mp4) in {input_dir}, found: {mp4_candidates}"
            )
        video = mp4_candidates[0]

    video_stem = video.stem
    if sam2_cache is None:
        sam2_candidates = sorted(list(input_dir.glob("*_sam2.npz")))
        if not sam2_candidates:
            raise RuntimeError(f"No *_sam2.npz found in {input_dir}")
        sam2_cache = _pick_cache_by_video_stem(sam2_candidates, video_stem, "sam2")
    if track_cache is None:
        track_candidates = sorted(list(input_dir.glob("*_cotracker.npz")))
        if not track_candidates:
            raise RuntimeError(f"No *_cotracker.npz found in {input_dir}")
        track_cache = _pick_cache_by_video_stem(track_candidates, video_stem, "cotracker")
    if megasam_cache is None:
        megasam_candidates = sorted(list(input_dir.glob("*_mega_sam.npz")))
        if not megasam_candidates:
            raise RuntimeError(f"No *_mega_sam.npz found in {input_dir}")
        megasam_cache = _pick_cache_by_video_stem(megasam_candidates, video_stem, "mega_sam")

    return video, sam2_cache, track_cache, megasam_cache


def create_video_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(str(video_path))


def create_video_writer(output_path: Path, fps: int, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), cv2.CAP_FFMPEG, fourcc, float(fps), size)
    if writer.isOpened():
        return writer
    writer.release()
    return cv2.VideoWriter(str(output_path), fourcc, float(fps), size)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def put_title(frame_bgr: np.ndarray, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (24, 24), (min(24 + 980, w - 24), 108), (0, 0, 0), -1)
    cv2.rectangle(canvas, (24, 24), (min(24 + 980, w - 24), 108), (60, 60, 60), 2)
    cv2.putText(canvas, text, (42, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
    return canvas


def maybe_put_title(
    frame_bgr: np.ndarray,
    text: str,
    enabled: bool,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if not enabled:
        return frame_bgr
    return put_title(frame_bgr, text, color=color)


def cache_model_name(video_model_name: str) -> str:
    if video_model_name == "cogvideoX":
        return "cogvideox"
    return video_model_name


def find_scene_video(videos_root: Path, model_name: str, scene: str, object_name: str) -> Path:
    scene_dir = videos_root / model_name / scene
    if not scene_dir.exists():
        raise RuntimeError(f"Missing scene directory: {scene_dir}")
    candidates = sorted(scene_dir.rglob(f"{object_name}.mp4"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise RuntimeError(f"No video matched object '{object_name}' in {scene_dir}")
    raise RuntimeError(f"Multiple videos matched object '{object_name}' in {scene_dir}: {candidates}")


def run_batch_website_export(
    script_path: Path,
    videos_root: Path,
    cache_root: Path,
    website_root: Path,
    camera_objects_path: Path,
    models: List[str],
    scenes: List[str],
) -> None:
    with open(camera_objects_path, "r", encoding="utf-8") as f:
        camera_objects = json.load(f)

    selected_object_by_scene: Dict[str, str] = {}
    for scene in scenes:
        if scene not in camera_objects or len(camera_objects[scene]) == 0:
            raise RuntimeError(f"Scene '{scene}' missing in camera objects json: {camera_objects_path}")
        selected_object_by_scene[scene] = camera_objects[scene][0]

    website_root.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        object_name = selected_object_by_scene[scene]
        scene_out_dir = website_root / scene
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BATCH] Scene={scene}, object='{object_name}'")
        for model_name in models:
            c_model = cache_model_name(model_name)
            video_path = find_scene_video(videos_root, model_name, scene, object_name)
            cache_dir = cache_root / c_model
            sam2_cache = cache_dir / f"{object_name}_sam2.npz"
            cotracker_cache = cache_dir / f"{object_name}_cotracker.npz"
            megasam_cache = cache_dir / f"{object_name}_mega_sam.npz"
            for p in [sam2_cache, cotracker_cache, megasam_cache]:
                if not p.exists():
                    raise RuntimeError(f"Missing cache file: {p}")

            output_path = scene_out_dir / f"{model_name}.mp4"
            cmd = [
                sys.executable,
                str(script_path),
                "--video",
                str(video_path),
                "--sam2-cache",
                str(sam2_cache),
                "--track-cache",
                str(cotracker_cache),
                "--megasam-cache",
                str(megasam_cache),
                "--output",
                str(output_path),
            ]
            print(f"[BATCH] Render {scene}/{model_name} -> {output_path}")
            subprocess.run(cmd, check=True)


def apply_semantic_overlay(frame_bgr: np.ndarray, mask: np.ndarray, tick: int) -> np.ndarray:
    out = frame_bgr.copy()
    dark = (out.astype(np.float32) * 0.36).astype(np.uint8)
    out = cv2.addWeighted(out, 0.42, dark, 0.58, 0.0)

    m = (mask > 0).astype(np.uint8)
    if m.shape[:2] != out.shape[:2]:
        m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = out.copy()
    overlay[m > 0] = (35, 215, 255)
    out = cv2.addWeighted(out, 0.62, overlay, 0.38, 0.0)

    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((5, 5), dtype=np.uint8))
    glow = np.zeros_like(out)
    glow[edge > 0] = (255, 220, 20)
    glow = cv2.GaussianBlur(glow, (0, 0), 2.4)
    out = cv2.addWeighted(out, 1.0, glow, 0.9, 0.0)

    scan_y = int((tick * 6) % max(out.shape[0], 1))
    cv2.line(out, (0, scan_y), (out.shape[1] - 1, scan_y), (255, 255, 255), 1, cv2.LINE_AA)

    return out


def sample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    return rng.choice(n, size=max_n, replace=False)


def tracks_to_pointmap_indices(
    tracks_2d: np.ndarray,
    h_pm: int,
    w_pm: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    u = np.clip(np.round(tracks_2d[..., 0] * scale_x).astype(np.int32), 0, w_pm - 1)
    v = np.clip(np.round(tracks_2d[..., 1] * scale_y).astype(np.int32), 0, h_pm - 1)
    return u, v


def infer_track_to_pointmap_scale(
    tracks_2d: np.ndarray,
    video_h: int,
    video_w: int,
    h_pm: int,
    w_pm: int,
) -> Tuple[float, float]:
    max_x = float(np.nanmax(tracks_2d[..., 0]))
    max_y = float(np.nanmax(tracks_2d[..., 1]))
    need_scale = (max_x > (w_pm - 1 + 1e-3)) or (max_y > (h_pm - 1 + 1e-3))
    if not need_scale:
        return 1.0, 1.0
    if video_w <= 1 or video_h <= 1:
        return 1.0, 1.0
    scale_x = float((w_pm - 1) / max(video_w - 1, 1))
    scale_y = float((h_pm - 1) / max(video_h - 1, 1))
    return scale_x, scale_y


def tracks_to_3d(
    pointmaps: np.ndarray,
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    t, h, w, _ = pointmaps.shape
    _, n, _ = tracks_2d.shape
    pts = np.zeros((t, n, 3), dtype=np.float32)
    valid = np.zeros((t, n), dtype=bool)

    u_all, v_all = tracks_to_pointmap_indices(tracks_2d, h, w, scale_x=scale_x, scale_y=scale_y)
    for i in range(t):
        u = u_all[i]
        v = v_all[i]
        p = pointmaps[i, v, u]
        good = np.isfinite(p).all(axis=1) & (np.linalg.norm(p, axis=1) > 1e-8) & (visibility[i] > 0.5)
        pts[i] = p
        valid[i] = good
    return pts, valid


def track_on_mask_ratio(
    tracks_2d: np.ndarray,
    masks: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    t, n, _ = tracks_2d.shape
    hit = np.zeros((n,), dtype=np.float32)
    for i in range(t):
        u, v = tracks_to_pointmap_indices(tracks_2d[i : i + 1], h, w, scale_x=1.0, scale_y=1.0)
        u = u[0]
        v = v[0]
        hit += (masks[i, v, u] > 0).astype(np.float32)
    return hit / max(t, 1)


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points.astype(np.float32)
    pts = points.astype(np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (pts_h @ pose.T)[:, :3]
    return out.astype(np.float32)


def farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int32)
    selected = [0]
    dist = np.full(n, np.inf, dtype=np.float64)
    for _ in range(1, k):
        last = points[selected[-1]]
        d = np.linalg.norm(points - last[None, :], axis=1)
        dist = np.minimum(dist, d)
        selected.append(int(np.argmax(dist)))
    return np.array(selected, dtype=np.int32)


def build_anchor_pairs(anchor_xyz0: np.ndarray, k_nei: int = 3) -> np.ndarray:
    m = anchor_xyz0.shape[0]
    if m < 2:
        return np.zeros((0, 2), dtype=np.int32)
    pairs = set()
    for i in range(m):
        d = np.linalg.norm(anchor_xyz0 - anchor_xyz0[i][None, :], axis=1)
        order = np.argsort(d)
        for j in order[1 : 1 + k_nei]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            pairs.add((a, b))
    out = np.array(sorted(pairs), dtype=np.int32)
    return out


class O3DRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.use_offscreen = True
        self.renderer = None
        self.vis = None

        self.mat_fg = o3d.visualization.rendering.MaterialRecord()
        self.mat_fg.shader = "defaultUnlit"
        self.mat_fg.point_size = 3.0

        self.mat_bg = o3d.visualization.rendering.MaterialRecord()
        self.mat_bg.shader = "defaultUnlit"
        self.mat_bg.point_size = 1.5

        self.mat_lines = o3d.visualization.rendering.MaterialRecord()
        self.mat_lines.shader = "unlitLine"
        self.mat_lines.line_width = 2.0
        self.mat_velocity = o3d.visualization.rendering.MaterialRecord()
        self.mat_velocity.shader = "unlitLine"
        self.mat_velocity.line_width = 4.0
        self.mat_trail = o3d.visualization.rendering.MaterialRecord()
        self.mat_trail.shader = "unlitLine"
        self.mat_trail.line_width = 2.0

        self.mat_anchor = o3d.visualization.rendering.MaterialRecord()
        self.mat_anchor.shader = "defaultUnlit"
        self.mat_anchor.point_size = 8.0

        try:
            self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            self.renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        except Exception:
            self.use_offscreen = False
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Open3D Renderer", width=width, height=height, visible=True)
            opt = self.vis.get_render_option()
            opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            opt.point_size = 3.0
            self.vis_view_ready = False
            self.vis_fg = o3d.geometry.PointCloud()
            self.vis_bg = o3d.geometry.PointCloud()
            self.vis_anchor = o3d.geometry.PointCloud()
            self.vis_pairs = o3d.geometry.LineSet()
            self.vis_velocity = o3d.geometry.LineSet()
            self.vis_trail = o3d.geometry.LineSet()
            self.vis.add_geometry(self.vis_bg)
            self.vis.add_geometry(self.vis_fg)
            self.vis.add_geometry(self.vis_anchor)
            self.vis.add_geometry(self.vis_pairs)
            self.vis.add_geometry(self.vis_velocity)
            self.vis.add_geometry(self.vis_trail)

    def _remove(self, name: str) -> None:
        if not self.use_offscreen:
            return
        try:
            self.renderer.scene.remove_geometry(name)
        except RuntimeError:
            pass
        except Exception:
            pass

    def close(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

    def render(
        self,
        fg_xyz: np.ndarray,
        fg_rgb: np.ndarray,
        bg_xyz: np.ndarray,
        bg_rgb: np.ndarray,
        anchor_xyz: Optional[np.ndarray],
        anchor_colors: Optional[np.ndarray],
        pair_index: Optional[np.ndarray],
        pair_colors: Optional[np.ndarray],
        camera_lookat: np.ndarray,
        camera_eye: np.ndarray,
        camera_up: np.ndarray,
        velocity_start: Optional[np.ndarray] = None,
        velocity_end: Optional[np.ndarray] = None,
        trail_points: Optional[np.ndarray] = None,
        trail_lines: Optional[np.ndarray] = None,
        trail_colors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pcd_fg = o3d.geometry.PointCloud()
        pcd_fg.points = o3d.utility.Vector3dVector(fg_xyz.astype(np.float64))
        pcd_fg.colors = o3d.utility.Vector3dVector(fg_rgb.astype(np.float64))

        if bg_xyz.shape[0] > 0:
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(bg_xyz.astype(np.float64))
            pcd_bg.colors = o3d.utility.Vector3dVector(bg_rgb.astype(np.float64))
        else:
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            pcd_bg.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if anchor_xyz is not None and anchor_xyz.shape[0] > 0:
            pcd_anchor = o3d.geometry.PointCloud()
            pcd_anchor.points = o3d.utility.Vector3dVector(anchor_xyz.astype(np.float64))
            if anchor_colors is not None and anchor_colors.shape[0] == anchor_xyz.shape[0]:
                pcd_anchor.colors = o3d.utility.Vector3dVector(anchor_colors.astype(np.float64))
            else:
                pcd_anchor.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.1, 0.1]], dtype=np.float64), (anchor_xyz.shape[0], 1)))
        else:
            pcd_anchor = o3d.geometry.PointCloud()
            pcd_anchor.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            pcd_anchor.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if pair_index is not None and pair_index.shape[0] > 0 and anchor_xyz is not None and pair_colors is not None:
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(anchor_xyz.astype(np.float64))
            lines.lines = o3d.utility.Vector2iVector(pair_index.astype(np.int32))
            lines.colors = o3d.utility.Vector3dVector(pair_colors.astype(np.float64))
        else:
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            lines.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if velocity_start is not None and velocity_end is not None:
            vel_line = o3d.geometry.LineSet()
            vel_points = np.vstack([velocity_start, velocity_end]).astype(np.float64)
            vel_line.points = o3d.utility.Vector3dVector(vel_points)
            vel_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
            vel_line.colors = o3d.utility.Vector3dVector(np.array([[0.2, 1.0, 0.2]], dtype=np.float64))
        else:
            vel_line = o3d.geometry.LineSet()
            vel_line.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            vel_line.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            vel_line.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if (
            trail_points is not None
            and trail_lines is not None
            and trail_colors is not None
            and trail_points.shape[0] > 0
            and trail_lines.shape[0] > 0
        ):
            trail_line = o3d.geometry.LineSet()
            trail_line.points = o3d.utility.Vector3dVector(trail_points.astype(np.float64))
            trail_line.lines = o3d.utility.Vector2iVector(trail_lines.astype(np.int32))
            trail_line.colors = o3d.utility.Vector3dVector(trail_colors.astype(np.float64))
        else:
            trail_line = o3d.geometry.LineSet()
            trail_line.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            trail_line.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            trail_line.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        if self.use_offscreen:
            self._remove("fg")
            self._remove("bg")
            self._remove("anchors")
            self._remove("pairs")
            self._remove("velocity")
            self._remove("trail")
            self.renderer.scene.add_geometry("fg", pcd_fg, self.mat_fg)
            self.renderer.scene.add_geometry("bg", pcd_bg, self.mat_bg)
            self.renderer.scene.add_geometry("anchors", pcd_anchor, self.mat_anchor)
            self.renderer.scene.add_geometry("pairs", lines, self.mat_lines)
            self.renderer.scene.add_geometry("velocity", vel_line, self.mat_velocity)
            self.renderer.scene.add_geometry("trail", trail_line, self.mat_trail)
            self.renderer.setup_camera(58.0, camera_lookat, camera_eye, camera_up)
            img = np.asarray(self.renderer.render_to_image())
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            return img.astype(np.uint8)

        if self.vis is None:
            raise RuntimeError("Open3D visualizer is not initialized.")
        self.vis_fg.points = pcd_fg.points
        self.vis_fg.colors = pcd_fg.colors
        self.vis_bg.points = pcd_bg.points
        self.vis_bg.colors = pcd_bg.colors
        self.vis_anchor.points = pcd_anchor.points
        self.vis_anchor.colors = pcd_anchor.colors
        self.vis_pairs.points = lines.points
        self.vis_pairs.lines = lines.lines
        self.vis_pairs.colors = lines.colors
        self.vis_velocity.points = vel_line.points
        self.vis_velocity.lines = vel_line.lines
        self.vis_velocity.colors = vel_line.colors
        self.vis_trail.points = trail_line.points
        self.vis_trail.lines = trail_line.lines
        self.vis_trail.colors = trail_line.colors
        self.vis.update_geometry(self.vis_bg)
        self.vis.update_geometry(self.vis_fg)
        self.vis.update_geometry(self.vis_anchor)
        self.vis.update_geometry(self.vis_pairs)
        self.vis.update_geometry(self.vis_velocity)
        self.vis.update_geometry(self.vis_trail)
        ctr = self.vis.get_view_control()
        if not self.vis_view_ready:
            self.vis.reset_view_point(True)
            front = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            ctr.set_front(front)
            ctr.set_up(np.array([0.0, -1.0, 0.0], dtype=np.float64))
            self.vis_view_ready = True
        self.vis.poll_events()
        self.vis.update_renderer()
        img = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 4-act real-video PDI hero MP4 using Open3D")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing source mp4 and cache npz files")
    parser.add_argument("--video", type=Path, default=None, help="Input real-world video path")
    parser.add_argument("--sam2-cache", type=Path, default=None, help="Path to *_sam2.npz")
    parser.add_argument("--track-cache", type=Path, default=None, help="Path to *_cotracker.npz")
    parser.add_argument("--megasam-cache", type=Path, default=None, help="Path to *_mega_sam.npz")
    parser.add_argument("--output", type=Path, default=Path("show/pdi_hero.mp4"), help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30, help="Output fps")
    parser.add_argument("--duration-act1", type=float, default=1.0, help="Act1 duration in seconds")
    parser.add_argument("--duration-act2", type=float, default=3.0, help="Act2 duration in seconds")
    parser.add_argument("--duration-act3", type=float, default=4.0, help="Act3 duration in seconds")
    parser.add_argument("--duration-act4", type=float, default=3.0, help="Act4 duration in seconds")
    parser.add_argument(
        "--act3-uplift-ratio",
        type=float,
        default=0.05,
        help="Act3 fraction used to raise point cloud from flat to full 3D (0,1]",
    )
    parser.add_argument("--max-fg-points", type=int, default=45000, help="Max foreground points per frame")
    parser.add_argument("--max-bg-points", type=int, default=50000, help="Max background points per frame")
    parser.add_argument("--anchor-count", type=int, default=30, help="Tracked anchor count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
    parser.add_argument(
        "--use-camera-poses-transform",
        action="store_true",
        help="Use camera_poses to transform each frame point cloud into a shared world frame.",
    )
    parser.add_argument("--act4-ghost-alpha", type=float, default=1.0, help="Foreground color keep ratio in act4 ghost mode")
    parser.add_argument("--trail-length", type=int, default=-1, help="Trail history length for key anchors in act4 (-1 for full history)")
    parser.add_argument("--act4-track-points", type=int, default=350, help="Number of CoTracker points for Step3 trajectory visualization")
    parser.add_argument("--min-history-length", type=int, default=2, help="Minimum history length to draw one trajectory")
    parser.add_argument("--trail-line-width", type=float, default=7.0, help="Line width for Step3 trajectories")
    parser.add_argument("--step3-start-ratio", type=float, default=0.1, help="Start ratio of source video frames used in Step3 (0~1)")
    parser.add_argument(
        "--step3-end-ratio",
        type=float,
        default=0.85,
        help="End ratio of source video frames used in Step3 (0~1), default keeps first 3/4 of the video",
    )
    parser.add_argument(
        "--with-title",
        action="store_true",
        help="Overlay subtitles/titles on frames. Default is disabled.",
    )
    parser.add_argument(
        "--batch-website",
        action="store_true",
        help="Batch render 5 scenes x 6 models and store in website/<scene>/<model>.mp4",
    )
    parser.add_argument("--videos-root", type=Path, default=Path("videos"), help="Root directory of model videos")
    parser.add_argument("--cache-root", type=Path, default=Path("output/cache"), help="Root directory of cache folders")
    parser.add_argument("--website-root", type=Path, default=Path("website"), help="Website output root")
    parser.add_argument("--camera-objects", type=Path, default=Path("camera_objects.json"), help="Camera objects json path")
    parser.add_argument(
        "--batch-models",
        type=str,
        default=",".join(DEFAULT_BATCH_MODELS),
        help=f"Comma-separated model names, default: {','.join(DEFAULT_BATCH_MODELS)}",
    )
    parser.add_argument(
        "--batch-scenes",
        type=str,
        default=",".join(DEFAULT_BATCH_SCENES),
        help=f"Comma-separated scenes, default: {','.join(DEFAULT_BATCH_SCENES)}",
    )
    args = parser.parse_args()

    if args.batch_website:
        models = [x.strip() for x in args.batch_models.split(",") if x.strip()]
        scenes = [x.strip() for x in args.batch_scenes.split(",") if x.strip()]
        run_batch_website_export(
            script_path=Path(__file__).resolve(),
            videos_root=args.videos_root,
            cache_root=args.cache_root,
            website_root=args.website_root,
            camera_objects_path=args.camera_objects,
            models=models,
            scenes=scenes,
        )
        return

    video_path, sam2_cache_path, track_cache_path, megasam_cache_path = resolve_inputs(
        input_dir=args.input_dir,
        video=args.video,
        sam2_cache=args.sam2_cache,
        track_cache=args.track_cache,
        megasam_cache=args.megasam_cache,
    )
    if args.output == Path("show/pdi_hero.mp4") and args.input_dir is not None:
        args.output = args.input_dir / f"{video_path.stem}_hero.mp4"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sam_data = load_npz(sam2_cache_path)
    trk_data = load_npz(track_cache_path)
    ms_data = load_npz(megasam_cache_path)

    masks = sam_data["masks"]  # (T,H,W)
    tracks = trk_data["tracks"]  # (T,N,2)
    visibility = trk_data["visibility"]  # (T,N)
    pointmaps = ms_data["pointmaps"]  # (T,H,W,3)
    camera_poses = ms_data["camera_poses"] if "camera_poses" in ms_data else None

    cap = create_video_capture(video_path)
    video_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_frames.append(frame)
    cap.release()
    if not video_frames:
        raise RuntimeError(f"Failed to read video: {video_path}")
    video_np = np.stack(video_frames, axis=0)  # BGR

    t_use = min(video_np.shape[0], masks.shape[0], tracks.shape[0], pointmaps.shape[0], visibility.shape[0])
    video_np = video_np[:t_use]
    masks = masks[:t_use]
    tracks = tracks[:t_use]
    visibility = visibility[:t_use]
    pointmaps = pointmaps[:t_use]
    if camera_poses is not None and camera_poses.ndim == 3 and camera_poses.shape[1:] == (4, 4):
        camera_poses = camera_poses[:t_use]
    else:
        camera_poses = None
    use_pose_transform = bool(args.use_camera_poses_transform and camera_poses is not None)

    h_pm, w_pm = pointmaps.shape[1:3]
    video_pm = np.stack([cv2.resize(f, (w_pm, h_pm), interpolation=cv2.INTER_LINEAR) for f in video_np], axis=0)
    h_vid, w_vid = video_np.shape[1:3]
    track_scale_x, track_scale_y = infer_track_to_pointmap_scale(tracks, h_vid, w_vid, h_pm, w_pm)
    track_u_all, track_v_all = tracks_to_pointmap_indices(
        tracks,
        h_pm,
        w_pm,
        scale_x=track_scale_x,
        scale_y=track_scale_y,
    )
    if abs(track_scale_x - 1.0) > 1e-6 or abs(track_scale_y - 1.0) > 1e-6:
        print(
            "[INFO] Rescale CoTracker pixels to pointmap grid: "
            f"sx={track_scale_x:.6f}, sy={track_scale_y:.6f}, "
            f"video=({w_vid},{h_vid}), pointmap=({w_pm},{h_pm})"
        )

    track_3d, track_valid = tracks_to_3d(
        pointmaps,
        tracks,
        visibility,
        scale_x=track_scale_x,
        scale_y=track_scale_y,
    )
    vis_score = visibility.mean(axis=0)
    anchor_frame = None
    cand = np.zeros((0,), dtype=np.int64)
    for tt in range(t_use):
        mask_tt = masks[tt]
        if mask_tt.shape != (h_pm, w_pm):
            mask_tt = cv2.resize(mask_tt.astype(np.uint8), (w_pm, h_pm), interpolation=cv2.INTER_NEAREST)
        u_tt = track_u_all[tt]
        v_tt = track_v_all[tt]
        in_mask_tt = mask_tt[v_tt, u_tt] > 0
        cand_tt = np.where(track_valid[tt] & in_mask_tt)[0]
        if cand_tt.shape[0] > 0:
            anchor_frame = tt
            cand = cand_tt
            break
    if cand.shape[0] == 0:
        for tt in range(t_use):
            cand_tt = np.where(track_valid[tt])[0]
            if cand_tt.shape[0] > 0:
                anchor_frame = tt
                cand = cand_tt
                break
    if cand.shape[0] == 0 or anchor_frame is None:
        raise RuntimeError("No CoTracker anchors found in any frame.")

    cand_xyz = track_3d[anchor_frame, cand]
    picked_local = farthest_point_sample(cand_xyz, min(args.anchor_count, cand.shape[0]))
    anchor_idx = cand[picked_local]

    anchor_xyz0 = track_3d[anchor_frame, anchor_idx]
    pair_idx = build_anchor_pairs(anchor_xyz0, k_nei=3)
    # Step3: directly draw trajectories of all CoTracker points.
    trail_idx = np.arange(tracks.shape[1], dtype=np.int32)
    pair_idx_trail = np.zeros((0, 2), dtype=np.int32)

    n1 = int(round(args.duration_act1 * args.fps))
    n2 = int(round(args.duration_act2 * args.fps))
    n3 = int(round(args.duration_act3 * args.fps))
    n4 = int(round(args.duration_act4 * args.fps))
    n_total = n1 + n2 + n3 + n4
    src_idx_act12 = np.linspace(0, t_use - 1, max(n1 + n2, 1)).round().astype(np.int32)
    src_idx_act3 = np.linspace(0, t_use - 1, max(n3, 1)).round().astype(np.int32)
    step3_start_ratio = float(np.clip(args.step3_start_ratio, 0.0, 1.0))
    step3_end_ratio = float(np.clip(args.step3_end_ratio, 0.0, 1.0))
    if step3_end_ratio < step3_start_ratio:
        step3_start_ratio, step3_end_ratio = step3_end_ratio, step3_start_ratio
    step3_start = int(round(step3_start_ratio * max(t_use - 1, 0)))
    step3_end = int(round(step3_end_ratio * max(t_use - 1, 0)))
    src_idx_step3 = np.linspace(step3_start, max(step3_start, step3_end), max(n4, 1)).round().astype(np.int32)
    freeze_t = int(src_idx_act12[min(n1, src_idx_act12.shape[0] - 1)])
    uplift_ratio = float(np.clip(args.act3_uplift_ratio, 1e-3, 1.0))

    h_out, w_out = video_np[0].shape[:2]
    writer = create_video_writer(args.output, args.fps, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {args.output}")

    o3d_renderer = O3DRenderer(w_out, h_out)
    o3d_renderer.mat_trail.line_width = float(max(args.trail_line_width, 1.0))
    if use_pose_transform:
        center_series: List[np.ndarray] = []
        for tt in range(t_use):
            valid_anchor = track_valid[tt, anchor_idx]
            if not np.any(valid_anchor):
                continue
            anchor_pts = track_3d[tt, anchor_idx][valid_anchor]
            anchor_pts = transform_points(anchor_pts, camera_poses[tt])
            center_series.append(anchor_pts.mean(axis=0).astype(np.float64))
        if not center_series:
            raise RuntimeError("Cannot initialize fixed camera from 3D tracks.")
        center_np = np.stack(center_series, axis=0)
        cam_center = center_np.mean(axis=0).astype(np.float64)
        cam_extent = np.maximum(center_np.max(axis=0) - center_np.min(axis=0), 1e-4)
    else:
        pm0 = pointmaps[freeze_t]
        m0 = masks[freeze_t] > 0
        if m0.shape != (h_pm, w_pm):
            m0 = cv2.resize(m0.astype(np.uint8), (w_pm, h_pm), interpolation=cv2.INTER_NEAREST) > 0
        valid0 = np.isfinite(pm0).all(axis=2) & (np.linalg.norm(pm0, axis=2) > 1e-8)
        fg0 = pm0[m0 & valid0]
        if fg0.shape[0] < 100:
            fg0 = pm0[valid0]
        if fg0.shape[0] == 0:
            raise RuntimeError("Cannot initialize camera center from SAM2 foreground.")
        cam_center = fg0.mean(axis=0).astype(np.float64)
        cam_extent = np.maximum(fg0.max(axis=0) - fg0.min(axis=0), 1e-4)
    cam_radius = float(np.linalg.norm(cam_extent) * 1.5 + 1e-2)
    # Align offscreen camera direction with Visualizer convention:
    # front=[0,0,-1], up=[0,-1,0] -> eye should be placed on +Z from lookat.
    cam_eye = cam_center + np.array([0.0, 0.0, 1.2 * cam_radius], dtype=np.float64)
    cam_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cam_offset = cam_eye - cam_center
    lookat_state = cam_center.copy()
    act4_cam_locked = False
    act4_lookat = cam_center.copy()
    act4_eye = cam_eye.copy()
    prev_anchor_center = None
    last_fg_xyz = None
    last_fg_rgb = None
    last_bg_xyz = None
    last_bg_rgb = None
    trail_histories: List[List[np.ndarray]] = [[] for _ in range(trail_idx.shape[0])]
    trail_time_histories: List[List[int]] = [[] for _ in range(trail_idx.shape[0])]
    trail_palette = np.zeros((trail_idx.shape[0], 3), dtype=np.float32)
    y0 = tracks[anchor_frame, trail_idx, 1].astype(np.float32)
    y_order = np.argsort(y0)
    y_rank = np.zeros_like(y_order, dtype=np.float32)
    y_rank[y_order] = np.linspace(0.0, 1.0, y_order.shape[0], dtype=np.float32)
    rainbow7 = np.array(
        [
            [1.0, 0.0, 0.0],   # red
            [1.0, 0.5, 0.0],   # orange
            [1.0, 1.0, 0.0],   # yellow
            [0.0, 1.0, 0.0],   # green
            [0.0, 1.0, 1.0],   # cyan
            [0.0, 0.0, 1.0],   # blue
            [0.56, 0.0, 1.0],  # violet
        ],
        dtype=np.float32,
    )
    for j in range(trail_idx.shape[0]):
        # Top -> bottom: red, orange, yellow, green, cyan, blue, violet.
        pos = y_rank[j] * (rainbow7.shape[0] - 1)
        left = int(np.floor(pos))
        right = min(left + 1, rainbow7.shape[0] - 1)
        w = pos - left
        trail_palette[j] = (1.0 - w) * rainbow7[left] + w * rainbow7[right]
    last_track_xyz_all = None

    for i in range(n_total):
        if i < n1 + n2:
            t = int(src_idx_act12[min(i, src_idx_act12.shape[0] - 1)])
        elif i < n1 + n2 + n3:
            t = int(src_idx_act3[min(i - (n1 + n2), src_idx_act3.shape[0] - 1)])
        else:
            t = int(src_idx_step3[min(i - (n1 + n2 + n3), src_idx_step3.shape[0] - 1)])
        bgr = video_np[t].copy()

        if i < n1:
            frame = maybe_put_title(bgr, "Input: Real-world Video (Ground Truth)", enabled=args.with_title)

        elif i < n1 + n2:
            f2 = apply_semantic_overlay(bgr, masks[t], tick=i - n1)
            frame = maybe_put_title(
                f2,
                "Step 1: Semantic Targeting (SAM 2)",
                enabled=args.with_title,
                color=(255, 230, 80),
            )

        else:
            in_act3 = i < n1 + n2 + n3
            act34_idx = i - (n1 + n2)

            if in_act3:
                t_cloud = t
            else:
                act4_idx = i - (n1 + n2 + n3)
                t_cloud = int(src_idx_step3[np.clip(act4_idx, 0, src_idx_step3.shape[0] - 1)])
            if in_act3:
                act3_progress = float((act34_idx + 1) / max(n3, 1))
                z_scale = min(1.0, act3_progress / uplift_ratio)
            else:
                z_scale = 1.0

            pm_t = pointmaps[t_cloud]
            m_t = masks[t_cloud] > 0
            if m_t.shape != (h_pm, w_pm):
                m_t = cv2.resize(m_t.astype(np.uint8), (w_pm, h_pm), interpolation=cv2.INTER_NEAREST) > 0

            rgb_t = cv2.cvtColor(video_pm[t_cloud], cv2.COLOR_BGR2RGB)
            valid_pm = np.isfinite(pm_t).all(axis=2) & (np.linalg.norm(pm_t, axis=2) > 1e-8)

            fg_sel = m_t & valid_pm
            bg_sel = (~m_t) & valid_pm

            fg_xyz_raw = pm_t[fg_sel]
            fg_rgb = rgb_t[fg_sel].astype(np.float32) / 255.0
            bg_xyz_raw = pm_t[bg_sel]

            if fg_xyz_raw.shape[0] < 100:
                fg_xyz_raw = pm_t[valid_pm]
                fg_rgb = rgb_t[valid_pm].astype(np.float32) / 255.0
            if fg_xyz_raw.shape[0] == 0:
                if last_fg_xyz is not None:
                    fg_xyz = last_fg_xyz.copy()
                    fg_rgb = last_fg_rgb.copy()
                    bg_xyz = last_bg_xyz.copy() if last_bg_xyz is not None else np.zeros((0, 3), dtype=np.float32)
                    bg_rgb = last_bg_rgb.copy() if last_bg_rgb is not None else np.zeros((0, 3), dtype=np.float32)
                else:
                    frame = put_title(bgr, "Open3D render skipped: empty point cloud", color=(60, 60, 255))
                    writer.write(frame)
                    continue
            else:
                fg_pick = sample_indices(fg_xyz_raw.shape[0], args.max_fg_points, rng)
                bg_pick = sample_indices(bg_xyz_raw.shape[0], args.max_bg_points, rng) if bg_xyz_raw.shape[0] > 0 else np.zeros((0,), dtype=np.int32)
                fg_xyz = fg_xyz_raw[fg_pick]
                fg_rgb = fg_rgb[fg_pick]
                bg_xyz = bg_xyz_raw[bg_pick] if bg_pick.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
                bg_rgb = rgb_t[bg_sel].astype(np.float32) / 255.0
                bg_rgb = bg_rgb[bg_pick] if bg_pick.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

            if use_pose_transform:
                fg_xyz = transform_points(fg_xyz, camera_poses[t_cloud])
                bg_xyz = transform_points(bg_xyz, camera_poses[t_cloud])
            last_fg_xyz = fg_xyz.copy()
            last_fg_rgb = fg_rgb.copy()
            last_bg_xyz = bg_xyz.copy()
            last_bg_rgb = bg_rgb.copy()

            z_base = float(np.median(fg_xyz[:, 2]))
            fg_xyz_anim = fg_xyz.copy()
            bg_xyz_anim = bg_xyz.copy()
            fg_xyz_anim[:, 2] = z_base + (fg_xyz_anim[:, 2] - z_base) * z_scale
            if bg_xyz_anim.shape[0] > 0:
                bg_xyz_anim[:, 2] = z_base + (bg_xyz_anim[:, 2] - z_base) * max(0.35, z_scale * 0.7)
            # Keep camera centered on SAM2 foreground object each frame.
            frame_center = fg_xyz_anim.mean(axis=0).astype(np.float64)
            if in_act3:
                lookat_state = 0.85 * lookat_state + 0.15 * frame_center
                frame_lookat = lookat_state
                frame_eye = frame_lookat + cam_offset
            else:
                if not act4_cam_locked:
                    act4_lookat = lookat_state.copy()
                    act4_eye = act4_lookat + cam_offset
                    act4_cam_locked = True
                frame_lookat = act4_lookat
                frame_eye = act4_eye

            if in_act3:
                rendered_rgb = o3d_renderer.render(
                    fg_xyz=fg_xyz_anim,
                    fg_rgb=fg_rgb,
                    bg_xyz=bg_xyz_anim,
                    bg_rgb=bg_rgb,
                    anchor_xyz=None,
                    anchor_colors=None,
                    pair_index=None,
                    pair_colors=None,
                    camera_lookat=frame_lookat,
                    camera_eye=frame_eye,
                    camera_up=cam_up,
                    trail_points=None,
                    trail_lines=None,
                    trail_colors=None,
                )
                frame = maybe_put_title(
                    cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR),
                    "Step 2: 3D Geometric Uplifting (MegaSaM)",
                    enabled=args.with_title,
                    color=(255, 220, 80),
                )
            else:
                act4_idx = i - (n1 + n2 + n3)
                # Explicit 2D->3D lifting: q_t^n = P_world[t, v_t^n, u_t^n]
                u_tr = track_u_all[t_cloud, trail_idx]
                v_tr = track_v_all[t_cloud, trail_idx]
                track_xyz_meas = pm_t[v_tr, u_tr].astype(np.float32)
                track_alive = (
                    np.isfinite(track_xyz_meas).all(axis=1)
                    & (np.linalg.norm(track_xyz_meas, axis=1) > 1e-8)
                    & (visibility[t_cloud, trail_idx] > 0.2)
                )
                track_sample_ok = track_alive
                if use_pose_transform:
                    track_xyz_meas = transform_points(track_xyz_meas, camera_poses[t_cloud])
                if last_track_xyz_all is None:
                    last_track_xyz_all = track_xyz_meas.copy()
                else:
                    last_track_xyz_all[track_alive] = track_xyz_meas[track_alive]
                track_xyz_all = last_track_xyz_all.copy()

                ghost_alpha = float(np.clip(args.act4_ghost_alpha, 0.0, 1.0))
                # Keep body opaque by default (alpha=1.0), optional fade if user lowers it.
                fg_rgb = np.clip(fg_rgb * ghost_alpha + (1.0 - ghost_alpha), 0.0, 1.0)

                trail_points_list: List[np.ndarray] = []
                trail_lines_list: List[np.ndarray] = []
                trail_colors_list: List[np.ndarray] = []
                point_base = 0
                for j in range(trail_idx.shape[0]):
                    if track_sample_ok[j]:
                        trail_histories[j].append(track_xyz_all[j].copy())
                        trail_time_histories[j].append(int(act4_idx))
                    elif len(trail_histories[j]) > 0:
                        # Keep continuous trail by holding last known 3D point.
                        trail_histories[j].append(trail_histories[j][-1].copy())
                        trail_time_histories[j].append(int(act4_idx))

                    if args.trail_length > 0 and len(trail_histories[j]) > args.trail_length:
                        trail_histories[j] = trail_histories[j][-args.trail_length :]
                        trail_time_histories[j] = trail_time_histories[j][-args.trail_length :]
                    hist = np.asarray(trail_histories[j], dtype=np.float32)
                    if hist.shape[0] < max(2, args.min_history_length):
                        continue
                    trail_points_list.append(hist)
                    seg_count = hist.shape[0] - 1
                    seg_lines = np.stack(
                        [
                            np.arange(point_base, point_base + seg_count, dtype=np.int32),
                            np.arange(point_base + 1, point_base + seg_count + 1, dtype=np.int32),
                        ],
                        axis=1,
                    )
                    trail_lines_list.append(seg_lines)
                    fade = np.linspace(0.75, 1.0, seg_count, dtype=np.float32)
                    base_color = trail_palette[j]
                    seg_colors = np.clip(base_color[None, :] * fade[:, None] * 1.15, 0.0, 1.0)
                    trail_colors_list.append(seg_colors)
                    point_base += hist.shape[0]
                if trail_points_list:
                    trail_points = np.concatenate(trail_points_list, axis=0)
                    trail_lines = np.concatenate(trail_lines_list, axis=0)
                    trail_colors = np.concatenate(trail_colors_list, axis=0)
                else:
                    trail_points = np.zeros((0, 3), dtype=np.float32)
                    trail_lines = np.zeros((0, 2), dtype=np.int32)
                    trail_colors = np.zeros((0, 3), dtype=np.float32)

                visible_track = track_xyz_all[track_sample_ok]
                if visible_track.shape[0] > 0:
                    anchor_xyz = visible_track
                    anchor_colors = trail_palette[track_sample_ok]
                    anchor_center = visible_track.mean(axis=0).astype(np.float64)
                elif prev_anchor_center is not None:
                    anchor_xyz = np.zeros((0, 3), dtype=np.float32)
                    anchor_colors = np.zeros((0, 3), dtype=np.float32)
                    anchor_center = prev_anchor_center.copy()
                else:
                    anchor_xyz = np.zeros((0, 3), dtype=np.float32)
                    anchor_colors = np.zeros((0, 3), dtype=np.float32)
                    anchor_center = frame_center.copy()
                if prev_anchor_center is None:
                    velocity_vec = np.zeros((3,), dtype=np.float64)
                else:
                    velocity_vec = (anchor_center - prev_anchor_center) * 5.2
                prev_anchor_center = anchor_center.copy()

                if pair_idx_trail.shape[0] > 0 and track_sample_ok.sum() > 1:
                    map_global_to_local = np.full(track_sample_ok.shape[0], -1, dtype=np.int32)
                    visible_ids = np.where(track_sample_ok)[0]
                    map_global_to_local[visible_ids] = np.arange(visible_ids.shape[0], dtype=np.int32)
                    pair_visible = track_sample_ok[pair_idx_trail[:, 0]] & track_sample_ok[pair_idx_trail[:, 1]]
                    pair_idx_global = pair_idx_trail[pair_visible]
                    if pair_idx_global.shape[0] > 0:
                        pair_index_local = np.stack(
                            [
                                map_global_to_local[pair_idx_global[:, 0]],
                                map_global_to_local[pair_idx_global[:, 1]],
                            ],
                            axis=1,
                        ).astype(np.int32)
                        pair_colors = np.tile(np.array([[1.0, 1.0, 1.0]], dtype=np.float32), (pair_index_local.shape[0], 1))
                    else:
                        pair_index_local = np.zeros((0, 2), dtype=np.int32)
                        pair_colors = np.zeros((0, 3), dtype=np.float32)
                else:
                    pair_index_local = np.zeros((0, 2), dtype=np.int32)
                    pair_colors = np.zeros((0, 3), dtype=np.float32)

                rendered_rgb = o3d_renderer.render(
                    fg_xyz=fg_xyz_anim,
                    fg_rgb=fg_rgb,
                    bg_xyz=bg_xyz_anim,
                    bg_rgb=bg_rgb,
                    anchor_xyz=anchor_xyz,
                    anchor_colors=anchor_colors,
                    pair_index=pair_index_local,
                    pair_colors=pair_colors,
                    camera_lookat=frame_lookat,
                    camera_eye=frame_eye,
                    camera_up=cam_up,
                    velocity_start=None,
                    velocity_end=None,
                    trail_points=trail_points,
                    trail_lines=trail_lines,
                    trail_colors=trail_colors,
                )
                frame = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                frame = maybe_put_title(
                    frame,
                    "Step 3: 3D Structural Anchoring (CoTracker3)",
                    enabled=args.with_title,
                    color=(90, 255, 255),
                )

        writer.write(frame)

    o3d_renderer.close()
    writer.release()
    print(f"[OK] Hero video written: {args.output}")
    print(f"[INFO] Frames used: {n_total}, source frames: {t_use}, output fps: {args.fps}")


if __name__ == "__main__":
    main()
