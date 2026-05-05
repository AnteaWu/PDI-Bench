import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_cache(npz_path: str) -> dict:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")
    with np.load(str(p), allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}
    if "tracks" not in data or "visibility" not in data:
        raise KeyError("npz must contain 'tracks' and 'visibility'")
    return data


def make_color_palette(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(60, 255, size=(n, 3)).tolist()


def render(
    video_path: str,
    tracks: np.ndarray,
    visibility: np.ndarray,
    output_path: str,
    trail: int = 20,
    show_bg: bool = False,
    bg_tracks: np.ndarray = None,
    bg_visibility: np.ndarray = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"cannot open writer: {output_path}")

    t_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_tracks = tracks.shape[0]
    t_vis = visibility.shape[0]
    t_use = min(t_video if t_video > 0 else t_tracks, t_tracks, t_vis)

    tracks = tracks[:t_use]
    visibility = visibility[:t_use]

    n_fg = tracks.shape[1]
    fg_colors = make_color_palette(n_fg, seed=42)

    has_bg = (
        show_bg
        and bg_tracks is not None
        and bg_visibility is not None
        and bg_tracks.ndim == 3
        and bg_visibility.ndim == 2
        and bg_tracks.shape[0] > 0
        and bg_tracks.shape[1] > 0
    )
    if has_bg:
        bg_tracks = bg_tracks[:t_use]
        bg_visibility = bg_visibility[:t_use]
        n_bg = bg_tracks.shape[1]
        bg_colors = make_color_palette(n_bg, seed=7)

    for t in tqdm(range(t_use), desc="Rendering"):
        ok, frame = cap.read()
        if not ok:
            break

        if has_bg:
            for i in range(n_bg):
                if bg_visibility[t, i] < 0.5:
                    continue
                for s in range(max(0, t - trail), t):
                    if bg_visibility[s, i] < 0.5:
                        continue
                    p1 = (int(bg_tracks[s, i, 0]), int(bg_tracks[s, i, 1]))
                    p2 = (int(bg_tracks[s + 1, i, 0]), int(bg_tracks[s + 1, i, 1]))
                    cv2.line(frame, p1, p2, bg_colors[i], 1, cv2.LINE_AA)
                curr = (int(bg_tracks[t, i, 0]), int(bg_tracks[t, i, 1]))
                cv2.circle(frame, curr, 2, bg_colors[i], -1)

        for i in range(n_fg):
            if visibility[t, i] < 0.5:
                continue
            color = fg_colors[i]
            for s in range(max(0, t - trail), t):
                if visibility[s, i] < 0.5:
                    continue
                alpha = (s - max(0, t - trail)) / max(1, trail)
                faded = [int(c * (0.3 + 0.7 * alpha)) for c in color]
                p1 = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                p2 = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                cv2.line(frame, p1, p2, faded, 2, cv2.LINE_AA)

            curr = (int(tracks[t, i, 0]), int(tracks[t, i, 1]))
            cv2.circle(frame, curr, 5, color, -1)
            cv2.circle(frame, curr, 5, (255, 255, 255), 1)

        cv2.putText(
            frame,
            f"CoTracker | FG: {n_fg} | Frame: {t}/{t_use - 1}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CoTracker cache on video")
    parser.add_argument("--npz", required=True, help="Path to *_cotracker.npz")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--output", default="cotracker_viz.mp4", help="Output video path")
    parser.add_argument("--trail", type=int, default=30, help="Trail length")
    parser.add_argument("--show_bg", action="store_true", help="Draw background tracks too")
    args = parser.parse_args()

    data = load_cache(args.npz)
    tracks = data["tracks"]
    visibility = data["visibility"]
    bg_tracks = data.get("bg_tracks")
    bg_visibility = data.get("bg_visibility")

    print(f"FG tracks shape: {tracks.shape}, visibility shape: {visibility.shape}")
    if bg_tracks is not None and bg_tracks.size > 0:
        print(f"BG tracks shape: {bg_tracks.shape}, bg visibility shape: {bg_visibility.shape}")

    render(
        video_path=args.video,
        tracks=tracks,
        visibility=visibility,
        output_path=args.output,
        trail=args.trail,
        show_bg=args.show_bg,
        bg_tracks=bg_tracks,
        bg_visibility=bg_visibility,
    )


if __name__ == "__main__":
    main()