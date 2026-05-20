"""Two-layer reconstruction quality audit.

Layer 1 (math): low cost, fast, strict checks
  - Ground flatness via RANSAC / plane fit
  - Scale jump (second derivative of Z)
  - Reprojection residual

Layer 2 (MLLM semantics): Open3D multi-view render + vision-language API
  - Renders mid / end / global sparse panels and stitches them
  - OpenAI-compatible API (e.g. Doubao, GPT-4o), structured JSON response
"""

import base64
import io
import json
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils.logger import pdi_logger


# ------------------------------------------------------------------ #
#  Layer 1: math self-check                                              #
# ------------------------------------------------------------------ #

def _fit_plane_svd(pts: np.ndarray) -> float:
    """SVD plane fit; works in any rotated frame; more robust than ordinary least squares."""
    if len(pts) < 3:
        return 0.0
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    d = -float(np.dot(normal, centroid))
    dists = np.abs(pts @ normal + d)
    return float(np.sqrt(np.mean(dists ** 2)))


def audit_ground_flatness(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    bottom_ratio: float = 0.3,
    rmse_threshold: float = 0.08,
) -> Tuple[float, bool]:
    """Ground flatness: sample early/mid/late segments, SVD plane fit, detect spoon-shaped warping."""
    T, H, W, _ = pointmaps.shape
    all_pts = []
    sample_t = [0, T // 2, T - 1]
    y_cut = int(H * (1 - bottom_ratio))

    for t in sample_t:
        mask_t = masks[t] if t < len(masks) else np.zeros((H, W), dtype=np.uint8)
        if mask_t.shape != (H, W):
            mask_t = cv2.resize(mask_t.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        region = np.zeros((H, W), dtype=bool)
        region[y_cut:, :] = True
        region &= (mask_t == 0)
        pts = pointmaps[t][region]
        valid = np.any(pts != 0, axis=-1)
        if valid.sum() > 50:
            all_pts.append(pts[valid])

    if not all_pts:
        return 0.0, True

    pts_all = np.concatenate(all_pts, axis=0)
    if len(pts_all) > 5000:
        idx = np.random.default_rng(42).choice(len(pts_all), 5000, replace=False)
        pts_all = pts_all[idx]

    rmse = _fit_plane_svd(pts_all)
    passed = rmse < rmse_threshold
    pdi_logger.info(f"Ground flatness SVD RMSE={rmse:.4f} ({'pass' if passed else 'FAIL'})")
    return rmse, passed


def audit_scale_jump(
    depth_z: np.ndarray,
    masks: np.ndarray,
    jump_threshold: float = 0.5,
) -> Tuple[float, bool]:
    """Scale jump: second difference of foreground median depth; detects sudden shifts."""
    T = depth_z.shape[0]
    if T < 3:
        return 0.0, True

    z_median = []
    for t in range(T):
        m = masks[t] if t < len(masks) else np.zeros(depth_z.shape[1:], dtype=np.uint8)
        if m.shape != depth_z[t].shape:
            m = cv2.resize(m.astype(np.uint8), (depth_z.shape[2], depth_z.shape[1]),
                           interpolation=cv2.INTER_NEAREST)
        fg_z = depth_z[t][m > 0]
        z_median.append(float(np.median(fg_z)) if len(fg_z) > 0 else 0.0)

    z_arr = np.array(z_median)
    z_range = float(np.ptp(z_arr)) + 1e-6
    z_norm = (z_arr - z_arr.min()) / z_range
    d2 = np.abs(np.diff(z_norm, n=2))
    max_jump = float(d2.max()) if len(d2) > 0 else 0.0
    passed = max_jump < jump_threshold
    pdi_logger.info(f"Scale jump max_d2Z={max_jump:.4f} ({'pass' if passed else 'FAIL'})")
    return max_jump, passed


def audit_reprojection_residual(
    residuals: Optional[np.ndarray],
    threshold: float = 2.0,
) -> Tuple[float, bool]:
    """Reprojection residual: mean of MegaSAM residual sequence."""
    if residuals is None or len(residuals) == 0:
        return 0.0, True
    mean_res = float(np.mean(residuals))
    passed = mean_res < threshold
    pdi_logger.info(f"Mean reprojection residual={mean_res:.4f}px ({'pass' if passed else 'FAIL'})")
    return mean_res, passed


def audit_reconstruction_math(
    pointmaps: Optional[np.ndarray],
    depth_z: np.ndarray,
    masks: np.ndarray,
    residuals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Layer-1 math audit: combine the three checks."""
    result: Dict[str, Any] = {}

    if pointmaps is not None and pointmaps.ndim == 4:
        g_rmse, g_pass = audit_ground_flatness(pointmaps, masks)
    else:
        g_rmse, g_pass = 0.0, True
    result["ground_rmse"]  = round(g_rmse, 4)
    result["ground_pass"]  = g_pass

    sj, sj_pass = audit_scale_jump(depth_z, masks)
    result["scale_jump"]      = round(sj, 4)
    result["scale_jump_pass"] = sj_pass

    rr, rr_pass = audit_reprojection_residual(residuals)
    result["reprojection_residual"] = round(rr, 4)
    result["reprojection_pass"]     = rr_pass

    result["math_pass"] = g_pass and sj_pass and rr_pass
    return result


# ------------------------------------------------------------------ #
#  Layer 2: Open3D render + MLLM semantic audit                         #
# ------------------------------------------------------------------ #

_MLLM_PROMPT = (
    "You are a Senior 3D Reconstruction Auditor. The image shows three diagnostic views of a 3D point cloud "
    "reconstructed from a monocular video by Mega-SAM.\n\n"
    "### SYSTEM CONTEXT ###\n"
    "EXPECTED artifacts: texture holes and sparse regions are NORMAL, do NOT penalize them.\n"
    "Sometimes Open3D might render failed, falling back to Matplotlib rendering. Please be tolerant of this if the quality of the rendering is still acceptable."
    "If it is Matplotlib renderning, the quality might be lower. Please be very tolerant!!!"
    "FATAL ERRORS to detect: (1) Cardboard effect - zero thickness, flat like paper. "
    "(2) Bowl effect - ground curves like a spoon.\n\n"
    "### DIAGNOSTIC PANELS ###\n"
    "1. LEFT [Single Frame, 45-degree view]: Object recognition check. "
    "Is the main subject clearly formed as a 3D shape (not a flat cutout)?\n"
    "2. MIDDLE [5-Frame Temporal Stack, SIDE VIEW]: Thickness check. "
    "Does the object occupy real 3D volume? A flat 1-pixel-thin line = FAIL.\n"
    "3. RIGHT [Global Sparse, TOP-DOWN view]: Ground flatness check. "
    "Is the ground/track a flat horizontal plane? Curved like a spoon or banana = FAIL.\n\n"
    "Reply ONLY in this JSON format (no extra text):\n"
    '{"reconstruction_success": <bool>, "reason": "<one sentence on geometry>", "score": <1-10>}'
    "overall_pass should be set to false only if you are 100% confident that the reconstruction is incorrect."
)


def _build_colored_pointcloud(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    max_pts: int = 100000,
    sample_frames: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse colored point cloud (global map) by evenly sampling sample_frames frames."""
    T, H, W, _ = pointmaps.shape
    frame_indices = np.linspace(0, T - 1, min(T, sample_frames), dtype=int).tolist()
    pts_list, rgb_list = [], []
    for t in frame_indices:
        m = masks[t] if t < len(masks) else np.zeros((H, W), dtype=np.uint8)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        valid = np.any(pointmaps[t] != 0, axis=-1)
        pts = pointmaps[t][valid]
        if not len(pts):
            continue
        pts_list.append(pts)
        if frames and t < len(frames) and frames[t] is not None:
            frame = frames[t]
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb_list.append(rgb[valid])
        else:
            z = pts[:, 2]
            t_val = (z - z.min()) / (z.ptp() + 1e-6)
            col = np.column_stack([t_val, 0.5 + t_val * 0.5, 1 - t_val])
            rgb_list.append(np.clip(col, 0, 1).astype(np.float32))

    if not pts_list:
        return np.empty((0, 3)), np.empty((0, 3))

    all_pts = np.concatenate(pts_list, axis=0)
    all_rgb = np.concatenate(rgb_list, axis=0)
    if len(all_pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(all_pts), max_pts, replace=False)
        all_pts, all_rgb = all_pts[idx], all_rgb[idx]
    return all_pts, all_rgb


def _render_views_matplotlib(
    all_pts: np.ndarray,
    all_rgb: np.ndarray,
    img_size: int = 512,
) -> Optional[np.ndarray]:
    """Matplotlib three-view scatter; reliable headless fallback when Open3D fails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    px_plot, py_plot, pz_plot = all_pts[:, 0], all_pts[:, 2], -all_pts[:, 1]

    # Per-axis scaling so a huge Z range does not squash X/Y into a needle
    cx, cy, cz = px_plot.mean(), py_plot.mean(), pz_plot.mean()
    hx = px_plot.ptp() / 2 + 1e-6
    hy = py_plot.ptp() / 2 + 1e-6
    hz = pz_plot.ptp() / 2 + 1e-6

    # Point size scales with count: more points -> smaller dots for balanced density
    n_pts = len(all_pts)
    pt_size = max(2.0, min(12.0, 800000.0 / max(n_pts, 1)))

    # Render at 2x then downsample for better anti-aliasing
    render_dpi = 150
    px = img_size / render_dpi
    view_params = [(90, -90, "Top"), (0, 0, "Side"), (30, 45, "45deg")]
    fig = plt.figure(figsize=(px * 3, px), dpi=render_dpi)
    for i, (elev, azim, title) in enumerate(view_params):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(px_plot, py_plot, pz_plot, c=all_rgb, s=pt_size, alpha=0.85, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_zlim(cz - hz, cz + hz)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=render_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (img_size * 3, img_size))


def _render_views_open3d(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    img_size: int = 512,
) -> Optional[np.ndarray]:
    """White-background three-panel point cloud (img_size*3 x img_size) for MLLM.

    LEFT  (view1): single-frame 45 deg frontal, subject close-up
    MIDDLE(view2): 5-frame side stack for thickness / depth
    RIGHT (view3): sparse global top-down for ground flatness and trajectory
    """
    import sys
    T, H, W, _ = pointmaps.shape

    # Preflight: all-zero pointmaps means fallback; cannot render
    sample_idx = np.linspace(0, T - 1, min(T, 5), dtype=int)
    if not any(np.any(pointmaps[t] != 0) for t in sample_idx):
        pdi_logger.warning("pointmaps all zero (mega_sam fallback); skip MLLM render")
        return None

    n_frames_avail = len(frames) if frames else 0
    pdi_logger.info(f"[render] pointmaps T={T}, frames available={n_frames_avail}")

    def _build_pts(t_indices, sparse_ratio: float = 1.0):
        pts_l, col_l = [], []
        for t in t_indices:
            valid = np.any(pointmaps[t] != 0, axis=-1)
            if not valid.any():
                continue
            p = pointmaps[t][valid]
            if frames and t < len(frames) and frames[t] is not None:
                fr = frames[t]
                fh, fw = fr.shape[:2]
                if (fh, fw) != (H, W):
                    # High-res frame: map coords to source pixels for color to avoid resize washing out saturated colors
                    ys, xs = np.where(valid)
                    ys_orig = np.clip((ys * fh / H).astype(int), 0, fh - 1)
                    xs_orig = np.clip((xs * fw / W).astype(int), 0, fw - 1)
                    fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    c = fr_rgb[ys_orig, xs_orig].astype(np.float32) / 255.0
                else:
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    c = rgb[valid]
                # sRGB -> linear: OffscreenRenderer expects linear radiance;
                # feeding sRGB as linear makes colors look washed out
                c = np.clip(c ** 2.2, 0.0, 1.0).astype(np.float32)
            else:
                z = p[:, 2]
                t_val = (z - z.min()) / (z.ptp() + 1e-6)
                c = np.clip(np.column_stack(
                    [t_val, 0.5 + t_val * 0.5, 1.0 - t_val]), 0, 1).astype(np.float32)
            pts_l.append(p)
            col_l.append(c)
        if not pts_l:
            return None, None
        all_p = np.concatenate(pts_l)
        all_c = np.concatenate(col_l)
        if sparse_ratio < 1.0:
            n = max(300, int(len(all_p) * sparse_ratio))
            if len(all_p) > n:
                idx = np.random.default_rng(42).choice(len(all_p), n, replace=False)
                all_p, all_c = all_p[idx], all_c[idx]
        return all_p, all_c

    mid = T // 2
    win = min(3, T // 4)
    single_idx = [min(mid, T - 1)]
    window_idx = np.linspace(max(0, mid - win), min(T - 1, mid + win), 5, dtype=int).tolist()
    global_idx  = np.linspace(0, T - 1, min(T, 30), dtype=int).tolist()

    pts_single, col_single = _build_pts(single_idx)
    pts_window, col_window = _build_pts(window_idx)
    pts_global, col_global = _build_pts(global_idx, sparse_ratio=0.05)

    if pts_single is None and pts_global is None:
        pdi_logger.warning("No valid points in any frame; cannot render")
        return None

    # Three views (MegaSAM coords: X=right, Y=down, Z=forward)
    view_specs = [
        dict(pts=pts_single, col=col_single, front=[0,   -0.2, -1  ], up=[0, -1,  0], zoom=0.5),
        dict(pts=pts_window, col=col_window, front=[1,   -0.2, -0.3], up=[0, -1,  0], zoom=0.5),
        dict(pts=pts_global, col=col_global, front=[0,   -1,    0.1], up=[0,  0, -1], zoom=0.8),
    ]

    try:
        import open3d as o3d
        is_win = sys.platform == "win32"
        rendered = []

        # Single OffscreenRenderer; reuse EGL context to avoid repeated init segfault
        renderer = None if is_win else o3d.visualization.rendering.OffscreenRenderer(img_size, img_size)
        mat = None
        if renderer is not None:
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 8.0
            renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        for spec in view_specs:
            p, c = spec["pts"], spec["col"]
            if p is None:
                rendered.append(np.full((img_size, img_size, 3), 255, dtype=np.uint8))
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p)
            pcd.colors = o3d.utility.Vector3dVector(c)
            center = np.asarray(pcd.get_center())

            if not is_win:
                renderer.scene.clear_geometry()
                renderer.scene.add_geometry("pcd", pcd, mat)
                extent = float(np.linalg.norm(
                    np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())))
                front = np.array(spec["front"], dtype=float)
                eye = (front / np.linalg.norm(front)) * extent * spec["zoom"] * 3 + center
                renderer.setup_camera(60.0, center.tolist(), eye.tolist(), spec["up"])
                rendered.append(np.asarray(renderer.render_to_image())[:, :, :3])
            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=img_size, height=img_size)
                vis.add_geometry(pcd)
                opt = vis.get_render_option()
                opt.point_size = 8.0
                opt.background_color = np.array([1.0, 1.0, 1.0])
                vis.poll_events(); vis.update_renderer()
                vis.reset_view_point(True)
                ctr = vis.get_view_control()
                ctr.set_lookat(center.tolist())
                ctr.set_front(spec["front"])
                ctr.set_up(spec["up"])
                ctr.set_zoom(spec["zoom"])
                vis.poll_events(); vis.update_renderer()
                img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                vis.destroy_window()
                rendered.append((img * 255).astype(np.uint8))

        pdi_logger.info("Open3D white-background three-view render succeeded")
        return np.concatenate(rendered, axis=1)

    except Exception as e:
        pdi_logger.warning(f"Open3D render failed; falling back to matplotlib: {e}")

    fb_pts, fb_rgb = _build_colored_pointcloud(pointmaps, masks, frames)
    return _render_views_matplotlib(fb_pts, fb_rgb, img_size)


def _encode_image_b64(img: np.ndarray) -> str:
    """Encode RGB ndarray as base64 JPEG string."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Image encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def audit_reconstruction_mllm(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    api_key: str,
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    model: str = "doubao-vision-pro-32k-250615",
    save_render_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Layer-2 MLLM semantic audit."""
    result: Dict[str, Any] = {
        "render_success": False,
        "reconstruction_success": None,
        "reason": "",
        "score": None,
        "mllm_raw": "",
    }

    composite = _render_views_open3d(pointmaps, masks, frames)
    if composite is None:
        pdi_logger.warning("MLLM audit skipped: render failed")
        return result
    result["render_success"] = True

    if save_render_path:
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_render_path, bgr)
        pdi_logger.info(f"Render image saved: {save_render_path}")

    try:
        from openai import OpenAI
    except ImportError:
        pdi_logger.warning("openai package not installed; skip MLLM API call")
        return result

    b64 = _encode_image_b64(composite)
    client = OpenAI(api_key=api_key, base_url=base_url)

    pdi_logger.info(f"Calling MLLM semantic audit ({model})...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _MLLM_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=256,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        result["mllm_raw"] = raw
        parsed = json.loads(raw)
        result["reconstruction_success"] = bool(parsed.get("reconstruction_success", True))
        result["reason"] = str(parsed.get("reason", ""))
        result["score"]  = int(parsed.get("score", 5))
        pdi_logger.info(
            f"MLLM verdict: success={result['reconstruction_success']} "
            f"score={result['score']} reason={result['reason']}"
        )
    except json.JSONDecodeError:
        pdi_logger.warning(f"MLLM returned non-JSON: {raw[:100]}")
    except Exception as e:
        pdi_logger.warning(f"MLLM API call failed: {e}")

    return result


# ------------------------------------------------------------------ #
#  Unified entry                                                         #
# ------------------------------------------------------------------ #

def audit_reconstruction(
    pointmaps: Optional[np.ndarray],
    depth_z: np.ndarray,
    masks: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    mllm_config: Optional[Dict[str, Any]] = None,
    save_render_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Unified entry for two-layer reconstruction audit.

    Args:
        pointmaps:        (T, H, W, 3) MegaSAM point map sequence, or None
        depth_z:          (T, H, W) depth maps
        masks:            (T, H, W) foreground masks
        residuals:        (T,) MegaSAM reprojection residuals, or None
        mllm_config:      dict with api_key / base_url / model / ...
                          If None, skip layer 2
        save_render_path: debug: save composite render here
        frames:           optional BGR frames for point coloring

    Returns:
        {
          "math": {ground_rmse, scale_jump, reprojection_residual, math_pass, ...},
          "mllm": {render_success, reconstruction_success, reason, score, ...} | None,
          "overall_pass": bool,
        }
    """
    math_result = audit_reconstruction_math(pointmaps, depth_z, masks, residuals)

    mllm_result = None
    if mllm_config and mllm_config.get("api_key") and pointmaps is not None:
        if math_result["math_pass"]:
            mllm_result = audit_reconstruction_mllm(
                pointmaps=pointmaps,
                masks=masks,
                api_key=mllm_config["api_key"],
                base_url=mllm_config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"),
                model=mllm_config.get("model", "doubao-vision-pro-32k-250615"),
                save_render_path=save_render_path,
                frames=frames,
            )
        else:
            pdi_logger.info("Math layer failed; skip MLLM call to save cost")

    mllm_pass = (
        mllm_result is None
        or mllm_result["reconstruction_success"] is None
        or mllm_result["reconstruction_success"]
    )

    return {
        "math": math_result,
        "mllm": mllm_result,
        "overall_pass": math_result["math_pass"] and mllm_pass,
    }


# ------------------------------------------------------------------ #
#  Load data from .npz (+ optional video)                                #
# ------------------------------------------------------------------ #

def _load_video_frames_ffmpeg(
    video_path: str,
    max_frames: int = 60,
) -> List[np.ndarray]:
    """Read video frames (BGR) via ffmpeg pipe; avoids some OpenCV/GStreamer decode limits."""
    import subprocess as _sp
    video_path = str(Path(video_path).resolve())
    probe = _sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pdi_logger.warning(f"ffprobe could not read video: {video_path}")
        return []
    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    total = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else None

    cmd = ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)
    all_frames: List[np.ndarray] = []
    frame_size = h * w * 3
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        all_frames.append(np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy())
    proc.stdout.close()
    proc.wait()

    if not all_frames:
        return []
    if len(all_frames) <= max_frames:
        return all_frames
    idx = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
    return [all_frames[i] for i in idx]


def load_from_npz(
    npz_path: str,
    video_path: Optional[str] = None,
    max_frames: int = 60,
) -> Dict[str, Any]:
    """Load audit inputs from a MegaSAM .npz (and optional video).

    Args:
        npz_path:   path to MegaSAM .npz; should contain pointmaps / camera_poses
        video_path: optional source video for RGB frames used for coloring
        max_frames: max frames to use; uniformly subsample if exceeded

    Returns:
        dict with pointmaps, depth_z, masks, residuals, frames (BGR)
    """
    data = np.load(npz_path, allow_pickle=True)
    pdi_logger.info(f"npz keys: {list(data.keys())}")

    # pointmaps
    pointmaps = None
    if "pointmaps" in data:
        pointmaps = np.asarray(data["pointmaps"])

    # depth_z
    if "depth_z" in data:
        depth_z = np.asarray(data["depth_z"])
        if depth_z.ndim < 3 and pointmaps is not None:
            pdi_logger.info("depth_z is 1-D; using pointmaps[:,:,:,2] as depth maps")
            depth_z = pointmaps[:, :, :, 2]
    elif pointmaps is not None:
        depth_z = pointmaps[:, :, :, 2]
    else:
        raise ValueError("npz has neither depth_z nor pointmaps; cannot run audit")

    # masks
    if "masks" in data:
        masks = np.asarray(data["masks"]).astype(np.uint8)
    elif "mask" in data:
        masks = np.asarray(data["mask"]).astype(np.uint8)
    else:
        T, H, W = depth_z.shape[:3]
        masks = np.zeros((T, H, W), dtype=np.uint8)
        pdi_logger.warning("npz has no masks field; using zeros (no foreground)")

    # residuals
    residuals = None
    for k in ("residuals", "reprojection_residuals", "reproj_err"):
        if k in data:
            residuals = np.asarray(data[k])
            break

    # Temporal subsample
    T = depth_z.shape[0]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames, dtype=int)
        depth_z = depth_z[idx]
        masks   = masks[idx]
        if pointmaps is not None:
            pointmaps = pointmaps[idx]
        if residuals is not None:
            residuals = residuals[idx]

    # Video BGR frames
    frames: List[np.ndarray] = []
    if video_path and Path(video_path).exists():
        frames = _load_video_frames_ffmpeg(video_path, max_frames=max_frames)
        pdi_logger.info(f"Read {len(frames)} frames from video")

    return dict(
        pointmaps=pointmaps,
        depth_z=depth_z,
        masks=masks,
        residuals=residuals,
        frames=frames,
    )


# ------------------------------------------------------------------ #
#  White-background three views (visualization helper)                   #
# ------------------------------------------------------------------ #

def render_three_views_white_bg(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    output_dir: Optional[str] = None,
    img_w: int = 800,
    img_h: int = 600,
    point_size: float = 6.0,
    sample_frames: int = 20,
) -> List[str]:
    """Render white-background three views; save under output_dir.

    Returns:
        list of saved paths [view1.png, view2.png, view3.png]
    """
    all_pts, all_col = _build_colored_pointcloud(
        pointmaps, masks, frames,
        max_pts=300000,
        sample_frames=sample_frames,
    )
    if len(all_pts) == 0:
        pdi_logger.warning("render_three_views_white_bg: no valid points; skip render")
        return []

    try:
        import open3d as o3d
    except ImportError:
        pdi_logger.warning("open3d not installed; cannot render white-background views")
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_col)
    center = np.asarray(pcd.get_center())

    view_specs = [
        dict(name="view1", front=[0.1, -0.9,  0.2], up=[0,  0, -1], zoom=0.4),
        dict(name="view2", front=[1.0, -0.2, -0.2], up=[0, -1,  0], zoom=0.35),
        dict(name="view3", front=[0.6,  0.25, 0.75], up=[0, -1,  0], zoom=0.35),
    ]

    if output_dir is None:
        output_dir = "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved = []
    for spec in view_specs:
        img = _render_single(
            pcd,
            spec["front"], spec["up"], center,
            spec["zoom"], max(img_w, img_h), point_size,
        )
        out_path = str(Path(output_dir) / f"{spec['name']}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved.append(out_path)
        pdi_logger.info(f"White-background view saved: {out_path}")

    return saved
