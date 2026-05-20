import numpy as np
import torch
import cv2
import os
from pathlib import Path
from .base import BasePerceptor, PerceptionResult
from typing import List, Tuple, Optional, Any
from PIL import Image
from ..utils.logger import pdi_logger

# Optional official SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

class Sam2Wrapper(BasePerceptor):
    def __init__(self, checkpoint: str, config: str, device: Optional[str] = None):
        super().__init__(device)
        self.checkpoint = os.path.abspath(checkpoint)
        self.config = config
        
        # Florence-2 auto-detector (lazy init)
        self.processor = None
        self.detector = None
        
        self._init_sam2()

    def _init_sam2(self):
        """Load SAM2 (Hydra config dir workaround)."""
        from pathlib import Path
        config_path = Path(self.config).resolve()
        config_dir = str(config_path.parent)
        config_name = config_path.stem
        try:
            self.model = build_sam2_video_predictor(str(config_path), self.checkpoint)
        except:
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                self.model = build_sam2_video_predictor(config_name, self.checkpoint)
    
    def _init_detector(self):
            """Lazy-load Florence-2; mock flash_attn if missing."""
            if self.detector is None:
                import sys
                from types import ModuleType
                from importlib.machinery import ModuleSpec

                if "flash_attn" not in sys.modules:
                    pdi_logger.info("flash_attn missing; injecting mock module for import checks...")
                    
                    mock_flash_attn = ModuleType("flash_attn")
                    mock_flash_attn.__spec__ = ModuleSpec("flash_attn", None)
                    mock_flash_attn.__version__ = "2.5.8"
                    
                    sys.modules["flash_attn"] = mock_flash_attn
                    
                    interface_name = "flash_attn.flash_attn_interface"
                    mock_interface = ModuleType(interface_name)
                    mock_interface.__spec__ = ModuleSpec(interface_name, None)
                    sys.modules[interface_name] = mock_interface

                from transformers import AutoProcessor, AutoModelForCausalLM
                pdi_logger.info("Loading Florence-2 detector...")
                model_id = 'microsoft/Florence-2-base'
                
                self.processor = AutoProcessor.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                
                self.detector = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    trust_remote_code=True,
                    attn_implementation="eager", 
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device).eval()

    def _auto_detect(self, frame_bgr: np.ndarray, text_query: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Florence-2 caption-to-box; accepts BGR frame in memory."""
        self._init_detector()
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{text_query}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        inputs = {k: v.to(self.detector.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.detector.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(results, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=image.size)
        
        try:
            boxes = parsed_answer["<CAPTION_TO_PHRASE_GROUNDING>"]["bboxes"]
            if len(boxes) > 0:
                box = np.array(boxes[0], dtype=np.float32)
                pdi_logger.info(f"Auto-selected target [{text_query}] box: {box.tolist()}")
                return None, box
        except Exception:
            pass
        
        pdi_logger.warning(f"No detection for [{text_query}]; falling back to image center.")
        center_pt = np.array([[image.size[0]/2, image.size[1]/2]], dtype=np.float32)
        return center_pt, None

    def infer(self, video_path: str, click_points: Optional[List] = None, text_query: Optional[str] = None, box_prompt: Optional[List] = None, **kwargs) -> PerceptionResult:
        """Manual points, external box_prompt, or text-driven Florence-2. For click_points use len(), not truthiness."""
        if self.model is None:
            raise RuntimeError("SAM2 not initialized")

        _no_points = click_points is None or (hasattr(click_points, "__len__") and len(click_points) == 0)

        box_np = None
        if box_prompt is not None:
            box_np = np.array(box_prompt, dtype=np.float32)
            _no_points = False

        elif _no_points and text_query:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None or frame.size == 0:
                raise ValueError(f"Cannot read first frame: {video_path}; check file path or codec")
            click_points, box_np = self._auto_detect(frame, text_query)
            _no_points = False

        if (_no_points or click_points is None) and box_np is None:
            raise ValueError("Provide one of: click_points, box_prompt, or text_query.")

        state = self.model.init_state(video_path=video_path, offload_video_to_cpu=True)

        if box_np is not None:
            self.model.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box_np)
        else:
            points_np = np.array(click_points, dtype=np.float32)
            if points_np.ndim == 1:
                points_np = points_np.reshape(1, -1)
            labels_np = np.ones(len(points_np), dtype=np.int32)
            self.model.add_new_points(state, frame_idx=0, obj_id=1, points=points_np, labels=labels_np)
        
        h_list, x_list, mask_list, truncated_list = [], [], [], []
        
        for frame_idx, obj_ids, mask_logits in self.model.propagate_in_video(state):
            mask = (mask_logits[0] > 0.0).cpu().numpy()
            if mask.ndim > 2: mask = mask[0]
            
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 10: 
                h = float(y_coords.max() - y_coords.min())
                x_c = float(x_coords.mean())
                H, W = mask.shape
                is_edge = (y_coords.min() < 5 or y_coords.max() > H - 5)
            else:
                h, x_c, is_edge = 0.0, x_list[-1] if x_list else 0.0, True 
            
            h_list.append(h)
            x_list.append(x_c)
            mask_list.append(mask)
            truncated_list.append(is_edge)
            
        self.model.reset_state(state)
        if self.detector:
            del self.detector, self.processor
            self.detector, self.processor = None, None
            torch.cuda.empty_cache()

        return PerceptionResult(
            video_id=video_path,
            frames_count=len(mask_list),
            masks=np.stack(mask_list),
            h_pixel=np.array(h_list),
            x_center=np.array(x_list),
            is_truncated=np.array(truncated_list),
            metadata={"auto_detected": text_query is not None}
        )
