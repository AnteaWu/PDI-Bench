import argparse
import yaml
import sys
import os
from pathlib import Path

# Make the pdi_eval package under src/ importable
sys.path.append(os.path.join(os.getcwd(), "src"))

from pdi_eval.pipeline import PDIEvaluationPipeline
from pdi_eval.utils.logger import pdi_logger
from pdi_eval.utils.visualizer import EvidenceVisualizer

def main():
    parser = argparse.ArgumentParser(description="PDI-Eval CLI Runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--points", type=str, default=None, help="SAM2 initial click coordinates, e.g. [[500,500]]")
    parser.add_argument("--box",    type=str, default=None, help="SAM2 bounding box, format [x1,y1,x2,y2], highest priority")
    parser.add_argument("--text",   type=str, default=None, help="Text description of the target object, e.g. 'train'; used by Florence-2 to auto-generate a bounding box")
    args = parser.parse_args()

    # 1. Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        pdi_logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize pipeline
    pipeline = PDIEvaluationPipeline(config=config)

    # 3. Parse prompt arguments (priority: box > points > text; at least one required)
    pdi_logger.info(f"Starting audit: {args.input}")
    if args.points is None and args.text is None and args.box is None:
        pdi_logger.error("Error: must provide at least one of --points, --box, or --text")
        return

    click_points = None
    box_prompt = None

    if args.box is not None:
        try:
            box_prompt = eval(args.box)
        except Exception as e:
            pdi_logger.error(f"Invalid box format: {args.box}. Error: {e}")
            return
    elif args.points is not None:
        try:
            click_points = eval(args.points)
        except Exception as e:
            pdi_logger.error(f"Invalid points format: {args.points}. Error: {e}")
            return

    # Run pipeline and get results
    report = pipeline.run(video_path=args.input, click_points=click_points, text_query=args.text, box_prompt=box_prompt)

    # 4. Generate visual evidence
    video_stem = Path(args.input).stem
    output_path = Path(args.output_dir) / video_stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    viz = EvidenceVisualizer(output_dir=str(output_path))
    
    # A. Plot scale and trajectory residual curves
    viz.draw_error_curves(
        report['breakdown']['scale_history'],
        report['breakdown']['traj_history'],
        video_stem
    )
    
    # B. Plot 3D volume stability curve
    viz.draw_volume_stability(
        report['breakdown']['volume_history'],
        video_stem
    )
    

    # # B. Save annotated video
    # pipeline.get_annotated_video(output_dir=str(output_path))

    # C. Save SAM2 mask overlay image
    if pipeline.last_masks is not None:
        import cv2 as _cv2
        import numpy as _np
        _cap = _cv2.VideoCapture(args.input)
        raw_frames = []
        while True:
            ret, f = _cap.read()
            if not ret:
                break
            raw_frames.append(_cv2.cvtColor(f, _cv2.COLOR_BGR2RGB))
        _cap.release()
        raw_frames_arr = _np.array(raw_frames) if raw_frames else None
        mask_path = viz.save_mask_sample(pipeline.last_masks, raw_frames_arr, video_stem)
        pdi_logger.info(f"SAM2 mask sample saved to: {mask_path}")

    # --- Save text report ---
    report_txt_path = output_path / f"{video_stem}_pdi_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("        PDI-Eval Final Audit Report\n")
        f.write("="*50 + "\n")
        f.write(f"Video Source:  {args.input}\n")
        target_desc = f"text='{args.text}'" if args.text else (f"box={args.box}" if args.box else f"points={args.points}")
        f.write(f"Target: {target_desc}\n")
        f.write("-" * 50 + "\n")
        f.write(f"FINAL PDI SCORE: {report['pdi_score']:.4f}\n")
        f.write(f"OVERALL GRADE:   {report['grade']}\n")
        f.write("-" * 50 + "\n")
        f.write("INDICATOR BREAKDOWN:\n")
        f.write(f" - Scale Component (1/Z Law):      {report['breakdown'].get('scale_component', 0):.4f}\n")
        f.write(f" - Trajectory Component (H-X):     {report['breakdown'].get('traj_component', 0):.4f}\n")
        f.write(f" - Epsilon Rigidity: {report['breakdown'].get('epsilon_rigidity', 0):.4f}\n")
        f.write(f" - Rigidity Strategy:              {report['breakdown'].get('rigidity_strategy', 'N/A')}\n")
        f.write(f" - VP Component (View Consistency):{report['breakdown'].get('vp_component', 0):.4f}\n")
        ra = report.get("reconstruction_audit")
        if ra is not None:
            f.write("-" * 50 + "\n")
            f.write("RECONSTRUCTION AUDIT:\n")
            math = ra.get("math", {})
            f.write(f" - RA Math Pass:    {math.get('math_pass')}\n")
            f.write(f" - RA Ground RMSE:  {math.get('ground_rmse')}\n")
            f.write(f" - RA Scale Jump:   {math.get('scale_jump')}\n")
            f.write(f" - RA Reproj Err:   {math.get('reprojection_residual')}\n")
            mllm = ra.get("mllm")
            if mllm is not None:
                f.write(f" - RA MLLM Success: {mllm.get('reconstruction_success')}\n")
                f.write(f" - RA MLLM Score:   {mllm.get('score')}\n")
                if mllm.get("reason"):
                    f.write(f" - RA MLLM Reason:  {mllm.get('reason')}\n")
            f.write(f" - RA Overall Pass: {ra.get('overall_pass')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Results generated at: {output_path}\n")
        f.write("="*50 + "\n")

    pdi_logger.info(f"Audit finished. Final PDI score: {report['pdi_score']:.4f} [{report['grade']}]")
    pdi_logger.info(f"Text report written: {report_txt_path}")
    pdi_logger.info(f"Detailed outputs and artifacts saved to: {output_path}")

if __name__ == "__main__":
    main()