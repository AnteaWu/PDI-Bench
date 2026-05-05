import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


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


def tracks_to_3d(
    pointmaps: np.ndarray,
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t, h, w, _ = pointmaps.shape
    _, n, _ = tracks_2d.shape
    pts = np.zeros((t, n, 3), dtype=np.float32)
    valid = np.zeros((t, n), dtype=bool)

    for i in range(t):
        u = np.clip(np.round(tracks_2d[i, :, 0]).astype(np.int32), 0, w - 1)
        v = np.clip(np.round(tracks_2d[i, :, 1]).astype(np.int32), 0, h - 1)
        p = pointmaps[i, v, u]
        good = np.isfinite(p).all(axis=1) & (np.linalg.norm(p, axis=1) > 1e-8) & (visibility[i] > 0.5)
        pts[i] = p
        valid[i] = good
    return pts, valid


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
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.renderer.scene.set_background([0.02, 0.02, 0.03, 1.0])
        self.width = width
        self.height = height

        self.mat_fg = o3d.visualization.rendering.MaterialRecord()
        self.mat_fg.shader = "defaultUnlit"
        self.mat_fg.point_size = 3.0

        self.mat_bg = o3d.visualization.rendering.MaterialRecord()
        self.mat_bg.shader = "defaultUnlit"
        self.mat_bg.point_size = 1.5

        self.mat_lines = o3d.visualization.rendering.MaterialRecord()
        self.mat_lines.shader = "unlitLine"
        self.mat_lines.line_width = 2.0

        self.mat_anchor = o3d.visualization.rendering.MaterialRecord()
        self.mat_anchor.shader = "defaultUnlit"
        self.mat_anchor.point_size = 8.0

    def _remove(self, name: str) -> None:
        try:
            self.renderer.scene.remove_geometry(name)
        except RuntimeError:
            pass
        except Exception:
            pass

    def render(
        self,
        fg_xyz: np.ndarray,
        fg_rgb: np.ndarray,
        bg_xyz: np.ndarray,
        anchor_xyz: Optional[np.ndarray],
        pair_index: Optional[np.ndarray],
        pair_colors: Optional[np.ndarray],
        orbit_angle_deg: float,
    ) -> np.ndarray:
        self._remove("fg")
        self._remove("bg")
        self._remove("anchors")
        self._remove("pairs")

        pcd_fg = o3d.geometry.PointCloud()
        pcd_fg.points = o3d.utility.Vector3dVector(fg_xyz.astype(np.float64))
        pcd_fg.colors = o3d.utility.Vector3dVector(fg_rgb.astype(np.float64))
        self.renderer.scene.add_geometry("fg", pcd_fg, self.mat_fg)

        if bg_xyz.shape[0] > 0:
            bg_color = np.full((bg_xyz.shape[0], 3), 0.40, dtype=np.float32)
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(bg_xyz.astype(np.float64))
            pcd_bg.colors = o3d.utility.Vector3dVector(bg_color.astype(np.float64))
            self.renderer.scene.add_geometry("bg", pcd_bg, self.mat_bg)

        if anchor_xyz is not None and anchor_xyz.shape[0] > 0:
            pcd_anchor = o3d.geometry.PointCloud()
            pcd_anchor.points = o3d.utility.Vector3dVector(anchor_xyz.astype(np.float64))
            pcd_anchor.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.1, 0.1]], dtype=np.float64), (anchor_xyz.shape[0], 1)))
            self.renderer.scene.add_geometry("anchors", pcd_anchor, self.mat_anchor)

        if pair_index is not None and pair_index.shape[0] > 0 and anchor_xyz is not None and pair_colors is not None:
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(anchor_xyz.astype(np.float64))
            lines.lines = o3d.utility.Vector2iVector(pair_index.astype(np.int32))
            lines.colors = o3d.utility.Vector3dVector(pair_colors.astype(np.float64))
            self.renderer.scene.add_geometry("pairs", lines, self.mat_lines)

        center = fg_xyz.mean(axis=0)
        extent = np.maximum(fg_xyz.max(axis=0) - fg_xyz.min(axis=0), 1e-4)
        radius = float(np.linalg.norm(extent) * 1.7 + 1e-2)
        theta = np.deg2rad(orbit_angle_deg)
        eye = center + np.array([np.sin(theta) * radius, 0.33 * radius, np.cos(theta) * radius], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.renderer.setup_camera(58.0, center, eye, up)

        img = np.asarray(self.renderer.render_to_image())
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 4-act PDI hero MP4 using Open3D")
    parser.add_argument("--video", type=Path, required=True, help="Input AI video path")
    parser.add_argument("--sam2-cache", type=Path, required=True, help="Path to *_sam2.npz")
    parser.add_argument("--track-cache", type=Path, required=True, help="Path to *_cotracker.npz")
    parser.add_argument("--megasam-cache", type=Path, required=True, help="Path to *_mega_sam.npz")
    parser.add_argument("--output", type=Path, default=Path("show/pdi_hero.mp4"), help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30, help="Output fps")
    parser.add_argument("--duration-act1", type=float, default=2.0, help="Act1 duration in seconds")
    parser.add_argument("--duration-act2", type=float, default=2.0, help="Act2 duration in seconds")
    parser.add_argument("--duration-act3", type=float, default=3.0, help="Act3 duration in seconds")
    parser.add_argument("--duration-act4", type=float, default=5.0, help="Act4 duration in seconds")
    parser.add_argument("--max-fg-points", type=int, default=45000, help="Max foreground points per frame")
    parser.add_argument("--max-bg-points", type=int, default=12000, help="Max background points per frame")
    parser.add_argument("--anchor-count", type=int, default=30, help="Tracked anchor count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sam_data = load_npz(args.sam2_cache)
    trk_data = load_npz(args.track_cache)
    ms_data = load_npz(args.megasam_cache)

    masks = sam_data["masks"]  # (T,H,W)
    tracks = trk_data["tracks"]  # (T,N,2)
    visibility = trk_data["visibility"]  # (T,N)
    pointmaps = ms_data["pointmaps"]  # (T,H,W,3)

    cap = cv2.VideoCapture(str(args.video))
    video_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_frames.append(frame)
    cap.release()
    if not video_frames:
        raise RuntimeError(f"Failed to read video: {args.video}")
    video_np = np.stack(video_frames, axis=0)  # BGR

    t_use = min(video_np.shape[0], masks.shape[0], tracks.shape[0], pointmaps.shape[0], visibility.shape[0])
    video_np = video_np[:t_use]
    masks = masks[:t_use]
    tracks = tracks[:t_use]
    visibility = visibility[:t_use]
    pointmaps = pointmaps[:t_use]

    h_pm, w_pm = pointmaps.shape[1:3]
    video_pm = np.stack([cv2.resize(f, (w_pm, h_pm), interpolation=cv2.INTER_LINEAR) for f in video_np], axis=0)

    track_3d, track_valid = tracks_to_3d(pointmaps, tracks, visibility)
    u0 = np.clip(np.round(tracks[0, :, 0]).astype(np.int32), 0, w_pm - 1)
    v0 = np.clip(np.round(tracks[0, :, 1]).astype(np.int32), 0, h_pm - 1)
    in_mask0 = masks[0, v0, u0] > 0
    cand = np.where(track_valid[0] & in_mask0)[0]
    if cand.shape[0] < args.anchor_count:
        cand = np.where(track_valid[0])[0]
    if cand.shape[0] == 0:
        raise RuntimeError("No valid 3D anchors found from cache.")

    cand_xyz = track_3d[0, cand]
    picked_local = farthest_point_sample(cand_xyz, min(args.anchor_count, cand.shape[0]))
    anchor_idx = cand[picked_local]

    anchor_xyz0 = track_3d[0, anchor_idx]
    pair_idx = build_anchor_pairs(anchor_xyz0, k_nei=3)
    d0 = np.linalg.norm(anchor_xyz0[pair_idx[:, 0]] - anchor_xyz0[pair_idx[:, 1]], axis=1) if pair_idx.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    n1 = int(round(args.duration_act1 * args.fps))
    n2 = int(round(args.duration_act2 * args.fps))
    n3 = int(round(args.duration_act3 * args.fps))
    n4 = int(round(args.duration_act4 * args.fps))
    n_total = n1 + n2 + n3 + n4
    src_idx = np.linspace(0, t_use - 1, n_total).round().astype(np.int32)

    h_out, w_out = video_np[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {args.output}")

    o3d_renderer = O3DRenderer(w_out, h_out)

    for i in range(n_total):
        t = int(src_idx[i])
        bgr = video_np[t].copy()

        if i < n1:
            frame = put_title(bgr, "Input: AI-Generated Video (Looks realistic...)")

        elif i < n1 + n2:
            f2 = apply_semantic_overlay(bgr, masks[t], tick=i - n1)
            frame = put_title(f2, "Step 1: 2D Semantic Targeting (SAM 2)", color=(255, 230, 80))

        else:
            pm_t = pointmaps[t]
            m_t = masks[t] > 0
            if m_t.shape != (h_pm, w_pm):
                m_t = cv2.resize(m_t.astype(np.uint8), (w_pm, h_pm), interpolation=cv2.INTER_NEAREST) > 0

            rgb_t = cv2.cvtColor(video_pm[t], cv2.COLOR_BGR2RGB)
            valid_pm = np.isfinite(pm_t).all(axis=2) & (np.linalg.norm(pm_t, axis=2) > 1e-8)

            fg_sel = m_t & valid_pm
            bg_sel = (~m_t) & valid_pm

            fg_xyz = pm_t[fg_sel]
            fg_rgb = rgb_t[fg_sel].astype(np.float32) / 255.0
            bg_xyz = pm_t[bg_sel]

            if fg_xyz.shape[0] < 100:
                fg_xyz = pm_t[valid_pm]
                fg_rgb = rgb_t[valid_pm].astype(np.float32) / 255.0
            if fg_xyz.shape[0] == 0:
                frame = put_title(bgr, "Open3D render skipped: empty point cloud", color=(60, 60, 255))
                writer.write(frame)
                continue

            fg_pick = sample_indices(fg_xyz.shape[0], args.max_fg_points, rng)
            bg_pick = sample_indices(bg_xyz.shape[0], args.max_bg_points, rng) if bg_xyz.shape[0] > 0 else np.zeros((0,), dtype=np.int32)
            fg_xyz = fg_xyz[fg_pick]
            fg_rgb = fg_rgb[fg_pick]
            bg_xyz = bg_xyz[bg_pick] if bg_pick.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

            act34_idx = i - (n1 + n2)
            orbit = 6.0 + act34_idx * 1.28

            if i < n1 + n2 + n3:
                rendered_rgb = o3d_renderer.render(
                    fg_xyz=fg_xyz,
                    fg_rgb=fg_rgb,
                    bg_xyz=bg_xyz,
                    anchor_xyz=None,
                    pair_index=None,
                    pair_colors=None,
                    orbit_angle_deg=orbit,
                )
                frame = put_title(cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR), "Step 2: 3D Geometric Uplifting (MegaSAM)", color=(255, 220, 80))
            else:
                anchor_xyz = track_3d[t, anchor_idx]
                anchor_ok = track_valid[t, anchor_idx]
                if not np.all(anchor_ok):
                    good_anchor = np.where(anchor_ok)[0]
                    if good_anchor.shape[0] >= 2:
                        anchor_xyz = anchor_xyz.copy()
                        for a in np.where(~anchor_ok)[0]:
                            anchor_xyz[a] = anchor_xyz[good_anchor[a % good_anchor.shape[0]]]

                if pair_idx.shape[0] > 0:
                    dt = np.linalg.norm(anchor_xyz[pair_idx[:, 0]] - anchor_xyz[pair_idx[:, 1]], axis=1)
                    delta = np.abs(dt - d0) / (d0 + 1e-6)
                    pair_colors = np.tile(np.array([[1.0, 1.0, 0.05]], dtype=np.float32), (pair_idx.shape[0], 1))
                    bad = delta > 0.22
                    pair_colors[bad] = np.array([1.0, 0.12, 0.12], dtype=np.float32)
                    fail_level = float(delta.mean()) if delta.size > 0 else 0.0
                else:
                    pair_colors = np.zeros((0, 3), dtype=np.float32)
                    fail_level = 0.0

                rendered_rgb = o3d_renderer.render(
                    fg_xyz=fg_xyz,
                    fg_rgb=fg_rgb,
                    bg_xyz=bg_xyz,
                    anchor_xyz=anchor_xyz,
                    pair_index=pair_idx,
                    pair_colors=pair_colors,
                    orbit_angle_deg=orbit,
                )
                frame = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                frame = put_title(frame, "Step 3: Kinematic & Rigidity Audit", color=(90, 255, 255))
                if fail_level > 0.15:
                    cv2.rectangle(frame, (24, 124), (820, 198), (0, 0, 70), -1)
                    cv2.putText(
                        frame,
                        "FAIL: Non-physical Jello Effect Detected!",
                        (40, 173),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (60, 60, 255),
                        2,
                        cv2.LINE_AA,
                    )

        writer.write(frame)

    writer.release()
    print(f"[OK] Hero video written: {args.output}")
    print(f"[INFO] Frames used: {n_total}, source frames: {t_use}, output fps: {args.fps}")


if __name__ == "__main__":
    main()
