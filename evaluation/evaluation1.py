"""
DAVIS batch evaluation for PDI-Eval.

Loads an image-sequence split from the DAVIS dataset, converts each sequence
to a temporary MP4 file, and runs the PDI evaluation pipeline in parallel
across multiple GPUs.  Results are collected and written as a formatted table
to a TXT file.

Usage:
    python evaluation.py [--davis_root DAVIS] [--split 2017/val]
                         [--config configs/default.yaml]
                         [--output_dir results/davis_eval]
                         [--gpus 4,5,6,7] [--fps 24]
"""

import argparse
import os
import sys
import multiprocessing as mp
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Helper: convert JPEG frames directory -> temp MP4
# ---------------------------------------------------------------------------

def _frames_to_mp4(frames_dir: str, out_path: str, fps: int = 24) -> bool:
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        return False
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for fp in frames:
        frame = cv2.imread(str(fp))
        if frame is not None:
            writer.write(frame)
    writer.release()
    return True


# ---------------------------------------------------------------------------
# Helper: derive click point from first DAVIS annotation mask
# ---------------------------------------------------------------------------

def _click_from_annotation(ann_dir: str, fallback_wh=None):
    masks = sorted(Path(ann_dir).glob("*.png"))
    if masks:
        mask = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
        if mask is not None:
            if mask.ndim > 2:
                mask = mask.max(axis=-1)
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                return [[float(xs.mean()), float(ys.mean())]]
    if fallback_wh:
        w, h = fallback_wh
        return [[w / 2.0, h / 2.0]]
    return None


# ---------------------------------------------------------------------------
# Worker: run a single DAVIS sequence on a designated GPU
# ---------------------------------------------------------------------------

def _run_sequence(args):
    """
    Executed in a child process (spawn).  Sets CUDA_VISIBLE_DEVICES before
    any CUDA code is imported, then runs the full PDI pipeline.

    Returns (seq_name, result_dict | None, error_str | None).
    """
    seq_name, gpu_id, config_path, davis_root, src_dir, output_dir, tmp_dir, fps = args

    # --- GPU assignment (must happen before torch / CUDA imports) ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import sys as _sys
    _sys.path.insert(0, src_dir)

    import yaml
    import cv2 as _cv2
    import numpy as _np
    from pathlib import Path as _Path
    from pdi_eval.pipeline import PDIEvaluationPipeline
    from pdi_eval.utils.visualizer import EvidenceVisualizer

    frames_dir = os.path.join(davis_root, "JPEGImages", "480p", seq_name)
    ann_dir    = os.path.join(davis_root, "Annotations", "480p", seq_name)
    tmp_video  = os.path.join(tmp_dir, f"{seq_name}.mp4")

    try:
        # 1. Build temporary video file from JPEG frames
        frames = sorted(_Path(frames_dir).glob("*.jpg"))
        if not frames:
            return seq_name, None, f"No JPEG frames found: {frames_dir}"

        first_frame = _cv2.imread(str(frames[0]))
        if first_frame is None:
            return seq_name, None, f"Cannot read first frame: {frames[0]}"
        h, w = first_frame.shape[:2]

        writer = _cv2.VideoWriter(
            tmp_video,
            _cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for fp in frames:
            f = _cv2.imread(str(fp))
            if f is not None:
                writer.write(f)
        writer.release()

        if not os.path.exists(tmp_video) or os.path.getsize(tmp_video) == 0:
            return seq_name, None, "Video write failed"

        # 2. Derive click point from GT annotation (first frame)
        ann_masks = sorted(_Path(ann_dir).glob("*.png"))
        click_points = None
        if ann_masks:
            mask = _cv2.imread(str(ann_masks[0]), _cv2.IMREAD_UNCHANGED)
            if mask is not None:
                # DAVIS palettized PNGs may load as (H,W,C); collapse to 2-D
                if mask.ndim > 2:
                    mask = mask.max(axis=-1)
                ys, xs = _np.where(mask > 0)
                if len(ys) > 0:
                    click_points = [[float(xs.mean()), float(ys.mean())]]
        if click_points is None:
            click_points = [[w / 2.0, h / 2.0]]

        # 3. Load config and run pipeline
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        pipeline = PDIEvaluationPipeline(config=config)
        report = pipeline.run(video_path=tmp_video, click_points=click_points)

        # --- Per-sequence result files (mirrors main.py behaviour) ---
        seq_out = _Path(output_dir) / seq_name
        seq_out.mkdir(parents=True, exist_ok=True)

        viz = EvidenceVisualizer(output_dir=str(seq_out))

        viz.draw_error_curves(
            report["breakdown"]["scale_history"],
            report["breakdown"]["traj_history"],
            seq_name,
        )
        viz.draw_volume_stability(
            report["breakdown"]["volume_history"],
            seq_name,
        )

        pipeline.get_annotated_video(output_dir=str(seq_out))

        if pipeline.last_masks is not None:
            cap = _cv2.VideoCapture(tmp_video)
            raw_frames = []
            while True:
                ret, frm = cap.read()
                if not ret:
                    break
                raw_frames.append(_cv2.cvtColor(frm, _cv2.COLOR_BGR2RGB))
            cap.release()
            raw_arr = _np.array(raw_frames) if raw_frames else None
            viz.save_mask_sample(pipeline.last_masks, raw_arr, seq_name)

        bd = report.get("breakdown", {})

        report_txt = seq_out / f"{seq_name}_pdi_report.txt"
        with open(report_txt, "w", encoding="utf-8") as fh:
            fh.write("=" * 50 + "\n")
            fh.write("        PDI-Eval Final Audit Report\n")
            fh.write("=" * 50 + "\n")
            fh.write(f"Sequence:  {seq_name}\n")
            fh.write(f"GPU:       {gpu_id}\n")
            fh.write("-" * 50 + "\n")
            fh.write(f"FINAL PDI SCORE: {report['pdi_score']:.4f}\n")
            fh.write(f"OVERALL GRADE:   {report['grade']}\n")
            fh.write("-" * 50 + "\n")
            fh.write("INDICATOR BREAKDOWN:\n")
            fh.write(f" - Scale Component (1/Z Law):      {bd.get('scale_component',    0):.4f}\n")
            fh.write(f" - Trajectory Component (H-X):     {bd.get('traj_component',     0):.4f}\n")
            fh.write(f" - Rigidity Component (Stability): {bd.get('rigidity_component', 0):.4f}\n")
            fh.write(f" - VP Component (View Consistency):{bd.get('vp_component',       0):.4f}\n")
            fh.write("-" * 50 + "\n")
            fh.write(f"Results saved to: {seq_out}\n")
            fh.write("=" * 50 + "\n")

        result = {
            "pdi_score":          report["pdi_score"],
            "grade":              report.get("grade", "N/A"),
            "scale_component":    float(bd.get("scale_component",    0.0)),
            "traj_component":     float(bd.get("traj_component",     0.0)),
            "rigidity_component": float(bd.get("rigidity_component", 0.0)),
            "vp_component":       float(bd.get("vp_component",       0.0)),
        }
        return seq_name, result, None

    except Exception:
        import traceback as _tb
        tb_str = _tb.format_exc()
        # Print to stderr so it appears in the parent terminal
        print(f"\n[ERROR] {seq_name} (GPU {gpu_id}):\n{tb_str}", flush=True)
        # Also save to an error log file
        try:
            err_dir = _Path(output_dir) / seq_name
            err_dir.mkdir(parents=True, exist_ok=True)
            with open(err_dir / "error.log", "w", encoding="utf-8") as _ef:
                _ef.write(tb_str)
        except Exception:
            pass
        return seq_name, None, tb_str

    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)


# ---------------------------------------------------------------------------
# Helper: load cached result from an existing per-sequence report file
# ---------------------------------------------------------------------------

def _load_cached_result(report_path: Path, seq_name: str):
    """
    Parse a previously written *_pdi_report.txt and return a result dict,
    or None if the file is missing / unreadable.
    """
    if not report_path.exists():
        return None
    try:
        text = report_path.read_text(encoding="utf-8")
        def _extract(label):
            for line in text.splitlines():
                if label in line:
                    return float(line.split(":")[-1].strip().split()[0])
            return 0.0

        def _extract_grade():
            for line in text.splitlines():
                if "OVERALL GRADE:" in line:
                    return line.split(":", 1)[-1].strip()
            return "N/A"

        return {
            "pdi_score":          _extract("FINAL PDI SCORE"),
            "grade":              _extract_grade(),
            "scale_component":    _extract("Scale Component"),
            "traj_component":     _extract("Trajectory Component"),
            "rigidity_component": _extract("Rigidity Component"),
            "vp_component":       _extract("VP Component"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Summary table writer
# ---------------------------------------------------------------------------

def _parse_grade(grade_str: str):
    """
    Split grade string into letter and index description.

    Example input:  "A (Physical Realism) - 物理逻辑严丝合缝"
    Returns:        ("A", "Physical Realism")
    """
    letter = grade_str[0] if grade_str else "?"
    try:
        desc = grade_str.split("(")[1].split(")")[0].strip()
    except (IndexError, AttributeError):
        desc = grade_str
    return letter, desc


def _write_table(rows: list, output_path: str) -> str:
    """
    rows: list of (seq_name, result_dict | None, error_str | None)
    Writes a formatted ASCII table and returns the content as a string.
    """
    rows = sorted(rows, key=lambda x: x[0])

    W = 110
    lines = []
    lines.append("=" * W)
    lines.append("  DAVIS Validation Set - PDI Evaluation Summary")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    col_header = (
        f"  {'Sequence':<22}"
        f"{'PDI':>8}"
        f"{'Scale':>9}"
        f"{'Traj':>9}"
        f"{'Rigid':>9}"
        f"{'VP':>9}"
        f"  {'Lv':<3}"
        f"  {'Index Grade':<22}"
    )
    lines.append(col_header)
    lines.append("-" * W)

    pdi_vals, sc_vals, tr_vals, ri_vals, vp_vals = [], [], [], [], []
    error_seqs = []

    for seq_name, result, err in rows:
        if result is None:
            short_err = (str(err).splitlines()[0])[:50] if err else "unknown"
            lines.append(f"  {seq_name:<22}{'ERROR':>8}  {short_err}")
            error_seqs.append(seq_name)
        else:
            pdi = result["pdi_score"]
            sc  = result["scale_component"]
            tr  = result["traj_component"]
            ri  = result["rigidity_component"]
            vp  = result["vp_component"]
            lv, idx_desc = _parse_grade(result["grade"])
            lines.append(
                f"  {seq_name:<22}"
                f"{pdi:>8.4f}"
                f"{sc:>9.4f}"
                f"{tr:>9.4f}"
                f"{ri:>9.4f}"
                f"{vp:>9.4f}"
                f"  {lv:<3}"
                f"  {idx_desc:<22}"
            )
            pdi_vals.append(pdi)
            sc_vals.append(sc)
            tr_vals.append(tr)
            ri_vals.append(ri)
            vp_vals.append(vp)

    lines.append("-" * W)
    if pdi_vals:
        def _row(label, fn):
            return (
                f"  {label:<22}"
                f"{fn(pdi_vals):>8.4f}"
                f"{fn(sc_vals):>9.4f}"
                f"{fn(tr_vals):>9.4f}"
                f"{fn(ri_vals):>9.4f}"
                f"{fn(vp_vals):>9.4f}"
            )
        lines.append(_row("MEAN", np.mean))
        lines.append(_row("STD",  np.std))
        lines.append(_row("MIN",  np.min))
        lines.append(_row("MAX",  np.max))
    lines.append("=" * W)

    lines.append(f"\n  Total sequences: {len(rows)}"
                 f"  |  Succeeded: {len(pdi_vals)}"
                 f"  |  Failed: {len(error_seqs)}")
    if error_seqs:
        lines.append(f"  Failed: {', '.join(error_seqs)}")

    content = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DAVIS Batch PDI Evaluation")
    parser.add_argument("--davis_root",  type=str, default="DAVIS",
                        help="Path to DAVIS dataset root")
    parser.add_argument("--split",       type=str, default="2017/val",
                        help="Dataset split, e.g. '2016/val' or '2017/val'")
    parser.add_argument("--config",      type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir",  type=str, default="results/davis_eval")
    parser.add_argument("--gpus",        type=str, default="4,5,6,7",
                        help="Comma-separated GPU IDs, e.g. '4,5,6,7'")
    parser.add_argument("--fps",         type=int, default=24,
                        help="FPS for temporary video files")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Resolve paths to absolute so child processes are not CWD-dependent
    davis_root  = str(Path(args.davis_root).resolve())
    config_path = str(Path(args.config).resolve())
    output_dir  = str(Path(args.output_dir).resolve())
    src_dir     = str(Path(__file__).parent / "src")

    # Temporary directory for intermediate MP4 files
    tmp_dir = str(Path(output_dir) / "_tmp_videos")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # Load sequence list
    split_year, split_name = args.split.split("/")
    imageset_txt = Path(davis_root) / "ImageSets" / split_year / f"{split_name}.txt"
    if not imageset_txt.exists():
        print(f"[ERROR] ImageSet file not found: {imageset_txt}")
        sys.exit(1)

    with open(imageset_txt, "r") as fh:
        sequences = [ln.strip() for ln in fh if ln.strip()]

    # Load cached results for already-finished sequences
    cached_rows = []
    pending_seqs = []
    for seq in sequences:
        report_path = Path(output_dir) / seq / f"{seq}_pdi_report.txt"
        result = _load_cached_result(report_path, seq)
        if result is not None:
            cached_rows.append((seq, result, None))
        else:
            pending_seqs.append(seq)

    print(f"[INFO] Split: {args.split}  |  Sequences: {len(sequences)}"
          f"  |  Cached: {len(cached_rows)}  |  Pending: {len(pending_seqs)}  |  GPUs: {gpus}")

    rows = list(cached_rows)

    if pending_seqs:
        # Build task arguments: assign GPUs in round-robin fashion
        tasks = [
            (seq, gpus[i % len(gpus)], config_path, davis_root, src_dir, output_dir, tmp_dir, args.fps)
            for i, seq in enumerate(pending_seqs)
        ]

        # Run in parallel using spawn to guarantee clean GPU env per worker
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(gpus)) as pool:
            rows += pool.map(_run_sequence, tasks)
    else:
        print("[INFO] All sequences already completed, skipping evaluation.")

    # Clean up tmp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    # Print full tracebacks for any failed sequences
    failed = [(s, e) for s, r, e in rows if r is None]
    if failed:
        print(f"\n[WARN] {len(failed)} sequence(s) failed. Full errors saved to <output_dir>/<seq>/error.log")
        if len(failed) <= 3:
            for s, e in failed:
                print(f"\n--- {s} ---\n{e}")

    # Write summary table
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_tag = args.split.replace("/", "_")
    table_path = os.path.join(output_dir, f"pdi_results_{split_tag}_{ts}.txt")
    content = _write_table(rows, table_path)

    print("\n" + content)
    print(f"\n[INFO] Table saved to: {table_path}")


if __name__ == "__main__":
    main()
