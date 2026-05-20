"""
Flow batch evaluation for PDI-Eval.

Scans the Flow folder (structure: <category>/<id>/<video>.mp4),
runs the PDI evaluation pipeline in parallel across multiple GPUs,
and writes a formatted summary table grouped by category.

Usage:
    python "evaluation_hunyuan copy.py" [--flow_dir ../videos/camera_video/Flow]
                                        [--config ../configs/default.yaml]
                                        [--output_dir new_results/Flow_eval]
                                        [--gpus 4,5,6,7]
"""

import argparse
import json
import os
import sys
import multiprocessing as mp
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Discover all videos in the Flow folder
# ---------------------------------------------------------------------------

def _discover_videos(flow_dir: str):
    """
    Returns a list of (category, seq_id, video_path) tuples.
    Directory structure: Flow/<category>/<id>/<video>.mp4
    """
    entries = []
    base = Path(flow_dir)
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        for id_dir in sorted(cat_dir.iterdir()):
            if not id_dir.is_dir():
                continue
            mp4s = sorted(id_dir.glob("*.mp4"))
            if mp4s:
                entries.append((cat_dir.name, id_dir.name, str(mp4s[0])))
    return entries


# ---------------------------------------------------------------------------
# Worker: run a single video on a designated GPU
# ---------------------------------------------------------------------------

def _run_sequence(args):
    """
    Executed in a child process (spawn). Sets CUDA_VISIBLE_DEVICES before
    any CUDA code is imported, then runs the full PDI pipeline.

    Returns (seq_key, category, result_dict | None, error_str | None).
    seq_key = "<category>/<id>"
    """
    seq_key, category, seq_id, gpu_id, config_path, video_path, output_dir, src_dir, text_query = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import sys as _sys
    _sys.path.insert(0, src_dir)

    import yaml
    import cv2 as _cv2
    import numpy as _np
    from pathlib import Path as _Path
    from pdi_eval.pipeline import PDIEvaluationPipeline
    from pdi_eval.utils.visualizer import EvidenceVisualizer

    seq_out = _Path(output_dir) / category / seq_id
    seq_out.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)
        
        # NOTE: Sets cache directory to generate the corresponding cache for Flow dataset
        config["cache_dir"] = str(_Path(config_path).parent.parent / "output" / "cache" / "flow")

        pipeline = PDIEvaluationPipeline(config=config)
        # text_query from camera_objects.json; triggers Florence-2 -> box -> SAM2 box prompt
        # Falls back to image center prompt when there is no matching entry
        if text_query:
            report = pipeline.run(video_path=video_path, text_query=text_query, render_output_dir=str(seq_out))
        else:
            cap = _cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            if not ret or first_frame is None:
                return seq_key, category, None, f"Cannot read first frame: {video_path}"
            h, w = first_frame.shape[:2]
            report = pipeline.run(video_path=video_path, click_points=[[w / 2.0, h / 2.0]], render_output_dir=str(seq_out))

        # Save visualizations
        viz = EvidenceVisualizer(output_dir=str(seq_out))
        viz.draw_error_curves(
            report["breakdown"]["scale_history"],
            report["breakdown"]["traj_history"],
            seq_key.replace("/", "_"),
        )
        viz.draw_volume_stability(
            report["breakdown"]["volume_history"],
            seq_key.replace("/", "_"),
        )
        # pipeline.get_annotated_video(output_dir=str(seq_out))

        if pipeline.last_masks is not None:
            cap2 = _cv2.VideoCapture(video_path)
            raw_frames = []
            while True:
                ok, frm = cap2.read()
                if not ok:
                    break
                raw_frames.append(_cv2.cvtColor(frm, _cv2.COLOR_BGR2RGB))
            cap2.release()
            raw_arr = _np.array(raw_frames) if raw_frames else None
            viz.save_mask_sample(pipeline.last_masks, raw_arr, seq_key.replace("/", "_"))

        bd = report.get("breakdown", {})

        # Extract reconstruction audit result
        ra = report.get("reconstruction_audit")
        ra_math_pass    = ra["math"]["math_pass"]              if ra else None
        ra_mllm_success = ra["mllm"]["reconstruction_success"] if (ra and ra.get("mllm")) else None
        ra_mllm_score   = ra["mllm"]["score"]                  if (ra and ra.get("mllm")) else None
        ra_overall_pass = ra["overall_pass"]                   if ra else None
        ra_reason       = ra["mllm"]["reason"]                 if (ra and ra.get("mllm")) else ""

        # Write per-sequence report
        report_txt = seq_out / f"{seq_key.replace('/', '_')}_pdi_report.txt"
        with open(report_txt, "w", encoding="utf-8") as fh:
            fh.write("=" * 50 + "\n")
            fh.write("        PDI-Eval Final Audit Report\n")
            fh.write("=" * 50 + "\n")
            fh.write(f"Sequence:  {seq_key}\n")
            fh.write(f"Video:     {video_path}\n")
            fh.write(f"GPU:       {gpu_id}\n")
            fh.write(f"Prompt:    {text_query or '(center click)'}\n")
            fh.write("-" * 50 + "\n")
            fh.write(f"FINAL PDI SCORE: {report['pdi_score']:.4f}\n")
            fh.write(f"OVERALL GRADE:   {report['grade']}\n")
            fh.write("-" * 50 + "\n")
            fh.write("INDICATOR BREAKDOWN:\n")
            fh.write(f" - Scale Component (1/Z Law):      {bd.get('scale_component',    0):.4f}\n")
            fh.write(f" - Trajectory Component (H-X):     {bd.get('traj_component',     0):.4f}\n")
            fh.write(f" - Epsilon Rigidity: {bd.get('epsilon_rigidity', 0):.4f}\n")
            fh.write(f" - Rigidity Strategy:              {bd.get('rigidity_strategy', 'N/A')}\n")
            fh.write(f" - VP Component (View Consistency):{bd.get('vp_component',       0):.4f}\n")
            fh.write("-" * 50 + "\n")
            fh.write("RECONSTRUCTION AUDIT:\n")
            fh.write(f" - RA Math Pass:    {ra_math_pass}\n")
            fh.write(f" - RA MLLM Success: {ra_mllm_success}\n")
            fh.write(f" - RA MLLM Score:   {ra_mllm_score}\n")
            fh.write(f" - RA Overall Pass: {ra_overall_pass}\n")
            if ra_reason:
                fh.write(f" - RA MLLM Reason:  {ra_reason}\n")
            fh.write("-" * 50 + "\n")
            fh.write(f"Results saved to: {seq_out}\n")
            fh.write("=" * 50 + "\n")

        result = {
            "pdi_score":          report["pdi_score"],
            "grade":              report.get("grade", "N/A"),
            "scale_component":    float(bd.get("scale_component",    0.0)),
            "traj_component":     float(bd.get("traj_component",     0.0)),
            "epsilon_rigidity": float(bd.get("epsilon_rigidity", 0.0)),
            "vp_component":       float(bd.get("vp_component",       0.0)),
            "ra_math_pass":       ra_math_pass,
            "ra_mllm_success":    ra_mllm_success,
            "ra_mllm_score":      ra_mllm_score,
            "ra_overall_pass":    ra_overall_pass,
        }
        return seq_key, category, result, None

    except Exception:
        import traceback as _tb
        tb_str = _tb.format_exc()
        print(f"\n[ERROR] {seq_key} (GPU {gpu_id}):\n{tb_str}", flush=True)
        try:
            with open(seq_out / "error.log", "w", encoding="utf-8") as ef:
                ef.write(tb_str)
        except Exception:
            pass
        return seq_key, category, None, tb_str


# ---------------------------------------------------------------------------
# Load cached result from existing per-sequence report
# ---------------------------------------------------------------------------

def _load_cached_result(report_path: Path):
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

        def _extract_bool(label):
            for line in text.splitlines():
                if label in line:
                    val = line.split(":", 1)[-1].strip()
                    if val == "True":  return True
                    if val == "False": return False
            return None

        def _extract_int(label):
            for line in text.splitlines():
                if label in line:
                    val = line.split(":", 1)[-1].strip()
                    try: return int(val)
                    except (ValueError, TypeError): pass
            return None

        return {
            "pdi_score":          _extract("FINAL PDI SCORE"),
            "grade":              _extract_grade(),
            "scale_component":    _extract("Scale Component"),
            "traj_component":     _extract("Trajectory Component"),
            "epsilon_rigidity": _extract("Epsilon Rigidity"),
            "vp_component":       _extract("VP Component"),
            "ra_math_pass":       _extract_bool("RA Math Pass"),
            "ra_mllm_success":    _extract_bool("RA MLLM Success"),
            "ra_mllm_score":      _extract_int("RA MLLM Score"),
            "ra_overall_pass":    _extract_bool("RA Overall Pass"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Summary table writer
# ---------------------------------------------------------------------------

def _parse_grade(grade_str: str):
    letter = grade_str[0] if grade_str else "?"
    try:
        desc = grade_str.split("(")[1].split(")")[0].strip()
    except (IndexError, AttributeError):
        desc = grade_str
    return letter, desc


def _fmt_recon(result: dict) -> str:
    """Format reconstruction audit fields as a compact string (~12 chars).

    Display rules:
      N/A        — audit disabled
      math:PASS  — math layer passed only
      math:FAIL  — math layer failed
      MLLM:P(8)  — MLLM passed, score 8
      MLLM:F(3)  — MLLM failed, score 3
      MLLM:?(-)  — MLLM call failed (no result)
    """
    if result.get("ra_overall_pass") is None:
        return "N/A"
    math_ok = result.get("ra_math_pass")
    mllm_ok = result.get("ra_mllm_success")
    score   = result.get("ra_mllm_score")
    if mllm_ok is None:
        return "math:PASS" if math_ok else "math:FAIL"
    score_str = str(score) if score is not None else "-"
    flag = "P" if mllm_ok else "F"
    return f"MLLM:{flag}({score_str})"


def _write_table(rows: list, output_path: str) -> str:
    """
    rows: list of (seq_key, category, result_dict | None, error_str | None)
    Groups by category and writes a formatted ASCII table.
    """
    W = 146
    lines = []
    lines.append("=" * W)
    lines.append("  Flow Video Set - PDI Evaluation Summary")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    col_header = (
        f"  {'Category':<26}"
        f"{'ID':<5}"
        f"{'PDI':>8}"
        f"{'Scale':>9}"
        f"{'Traj':>9}"
        f"{'Rigid':>9}"
        f"{'VP':>9}"
        f"  {'Lv':<3}"
        f"  {'Index Grade':<22}"
        f"  {'ReconAudit':<12}"
        f"  {'MLLM_OK':<7}"
    )
    lines.append(col_header)
    lines.append("-" * W)

    # Group by category
    from collections import defaultdict
    cat_rows = defaultdict(list)
    for row in rows:
        cat_rows[row[1]].append(row)

    all_pdi, all_sc, all_tr, all_ri, all_vp = [], [], [], [], []
    error_seqs = []

    for cat in sorted(cat_rows.keys()):
        cat_pdi, cat_sc, cat_tr, cat_ri, cat_vp = [], [], [], [], []
        cat_ra_pass_count, cat_ra_total = 0, 0
        for seq_key, category, result, err in sorted(cat_rows[cat], key=lambda x: x[0]):
            seq_id = seq_key.split("/")[-1]
            if result is None:
                short_err = (str(err).splitlines()[0])[:45] if err else "unknown"
                lines.append(f"  {category:<26}{seq_id:<5}{'ERROR':>8}  {short_err}")
                error_seqs.append(seq_key)
            else:
                pdi = result["pdi_score"]
                sc  = result["scale_component"]
                tr  = result["traj_component"]
                ri  = result["epsilon_rigidity"]
                vp  = result["vp_component"]
                lv, idx_desc = _parse_grade(result["grade"])
                ra_str  = _fmt_recon(result)
                mllm_ok = result.get("ra_mllm_success")
                mllm_ok_str = "N/A" if mllm_ok is None else str(mllm_ok)
                lines.append(
                    f"  {category:<26}{seq_id:<5}"
                    f"{pdi:>8.4f}"
                    f"{sc:>9.4f}"
                    f"{tr:>9.4f}"
                    f"{ri:>9.4f}"
                    f"{vp:>9.4f}"
                    f"  {lv:<3}"
                    f"  {idx_desc:<22}"
                    f"  {ra_str:<12}"
                    f"  {mllm_ok_str:<7}"
                )
                cat_pdi.append(pdi); cat_sc.append(sc); cat_tr.append(tr)
                cat_ri.append(ri);   cat_vp.append(vp)
                if result.get("ra_overall_pass") is not None:
                    cat_ra_total += 1
                    if result["ra_overall_pass"]:
                        cat_ra_pass_count += 1

        # Category mean row
        if cat_pdi:
            ra_rate = f"{cat_ra_pass_count}/{cat_ra_total} pass" if cat_ra_total > 0 else "N/A"
            lines.append(
                f"  {'  >> ' + cat + ' mean':<31}"
                f"{np.mean(cat_pdi):>8.4f}"
                f"{np.mean(cat_sc):>9.4f}"
                f"{np.mean(cat_tr):>9.4f}"
                f"{np.mean(cat_ri):>9.4f}"
                f"{np.mean(cat_vp):>9.4f}"
                f"{'':>7}"
                f"  {ra_rate:<14}"
            )
            all_pdi += cat_pdi; all_sc += cat_sc; all_tr += cat_tr
            all_ri  += cat_ri;  all_vp += cat_vp
        lines.append("  " + "-" * (W - 2))

    lines.append("-" * W)
    if all_pdi:
        def _row(label, fn):
            return (
                f"  {label:<31}"
                f"{fn(all_pdi):>8.4f}"
                f"{fn(all_sc):>9.4f}"
                f"{fn(all_tr):>9.4f}"
                f"{fn(all_ri):>9.4f}"
                f"{fn(all_vp):>9.4f}"
            )
        lines.append(_row("OVERALL MEAN", np.mean))
        lines.append(_row("OVERALL STD",  np.std))
        lines.append(_row("OVERALL MIN",  np.min))
        lines.append(_row("OVERALL MAX",  np.max))
    lines.append("=" * W)

    total = len(rows)
    succeeded = len(all_pdi)
    lines.append(f"\n  Total videos: {total}  |  Succeeded: {succeeded}  |  Failed: {len(error_seqs)}")
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
    parser = argparse.ArgumentParser(description="Flow Batch PDI Evaluation")
    parser.add_argument("--flow_dir", type=str, default=str(Path(__file__).parent.parent / "videos/Flow"),
                        help="Path to the Flow folder (contains category subfolders)")
    parser.add_argument("--config",      type=str, default=str(Path(__file__).parent.parent / "configs/default.yaml"))
    parser.add_argument("--output_dir",  type=str, default=str(Path(__file__).parent.parent / "new_results/flow_eval"))
    parser.add_argument("--gpus",        type=str, default="4,5,6,7",
                        help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    flow_dir = str(Path(args.flow_dir).resolve())
    config_path = str(Path(args.config).resolve())
    output_dir  = str(Path(args.output_dir).resolve())
    src_dir     = str(Path(__file__).parent.parent / "src")

    # Discover all videos
    entries = _discover_videos(flow_dir)
    if not entries:
        print(f"[ERROR] No videos found under: {flow_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(entries)} videos across {len(set(e[0] for e in entries))} categories")

    # Load object names and build (category, seq_id) -> text_query mapping
    prompts_path = Path(__file__).parent.parent / "camera_objects.json"
    all_prompts = {}
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as fh:
            all_prompts = json.load(fh)
        print(f"[INFO] Loaded object names from {prompts_path}")
    else:
        print(f"[WARN] camera_objects.json not found at {prompts_path}, falling back to center click")

    # Assign prompt by per-category sequential index (same sort order as _discover_videos)
    cat_counter = defaultdict(int)
    prompt_map = {}
    for category, seq_id, _ in entries:
        idx = cat_counter[category]
        cat_prompts = all_prompts.get(category, [])
        prompt_map[(category, seq_id)] = cat_prompts[idx] if idx < len(cat_prompts) else None
        cat_counter[category] += 1

    # Check for cached results
    cached_rows = []
    pending_entries = []
    for category, seq_id, video_path in entries:
        seq_key = f"{category}/{seq_id}"
        report_path = Path(output_dir) / category / seq_id / f"{category}_{seq_id}_pdi_report.txt"
        result = _load_cached_result(report_path)
        if result is not None:
            cached_rows.append((seq_key, category, result, None))
        else:
            pending_entries.append((category, seq_id, video_path))

    print(f"[INFO] Cached: {len(cached_rows)}  |  Pending: {len(pending_entries)}  |  GPUs: {gpus}")

    rows = list(cached_rows)

    if pending_entries:
        tasks = [
            (
                f"{cat}/{sid}",                    # seq_key
                cat,                               # category
                sid,                               # seq_id
                gpus[i % len(gpus)],               # gpu_id
                config_path,
                vpath,                             # video_path
                output_dir,
                src_dir,
                prompt_map.get((cat, sid)),        # text_query
            )
            for i, (cat, sid, vpath) in enumerate(pending_entries)
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(gpus)) as pool:
            rows += pool.map(_run_sequence, tasks)
    else:
        print("[INFO] All videos already completed, skipping evaluation.")

    # Print full tracebacks for failed sequences
    failed = [(s, e) for s, c, r, e in rows if r is None]
    if failed:
        print(f"\n[WARN] {len(failed)} video(s) failed. Errors saved to <output_dir>/<cat>/<id>/error.log")
        if len(failed) <= 3:
            for s, e in failed:
                print(f"\n--- {s} ---\n{e}")

    # Write summary table
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = os.path.join(output_dir, f"pdi_results_flow_{ts}.txt")
    content = _write_table(rows, table_path)

    print("\n" + content)
    print(f"\n[INFO] Table saved to: {table_path}")


if __name__ == "__main__":
    main()
