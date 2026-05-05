import numpy as np
import torch
import cv2
import os
from pathlib import Path
from .base import BasePerceptor, PerceptionResult
from typing import List, Tuple, Optional, Any
from PIL import Image
from ..utils.logger import pdi_logger

# 尝试引入 SAM2 官方库
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

class Sam2Wrapper(BasePerceptor):
    def __init__(self, checkpoint: str, config: str, device: Optional[str] = None):
        super().__init__(device)
        self.checkpoint = os.path.abspath(checkpoint)
        self.config = config
        
        # --- [新增] Florence-2 自动检测引擎初始化 ---
        self.processor = None
        self.detector = None
        
        # SAM2 初始化逻辑 (保持你原有的 Hydra 修复逻辑)
        self._init_sam2()

    def _init_sam2(self):
        """保持原有的 SAM2 加载逻辑"""
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
            """延迟加载检测器，并使用完整的 Mock 绕过 flash_attn 检查"""
            if self.detector is None:
                import sys
                from types import ModuleType
                from importlib.machinery import ModuleSpec

                # --- 完善后的黑科技：提供完整的模块规范 ---
                if "flash_attn" not in sys.modules:
                    pdi_logger.info("检测到环境缺少 flash_attn，正在注入 Mock 模块以绕过硬编码检查...")
                    
                    # 1. 创建主模块
                    mock_flash_attn = ModuleType("flash_attn")
                    # 2. 【关键修复】补齐 __spec__，否则 find_spec 会报错
                    mock_flash_attn.__spec__ = ModuleSpec("flash_attn", None)
                    # 3. 模拟一些模型可能检查的版本号
                    mock_flash_attn.__version__ = "2.5.8"
                    
                    # 注入到系统路径
                    sys.modules["flash_attn"] = mock_flash_attn
                    
                    # 4. 针对 Florence-2 可能进一步调用的子路径进行二级 Mock
                    # 某些版本的模型会尝试 from flash_attn.flash_attn_interface import ...
                    interface_name = "flash_attn.flash_attn_interface"
                    mock_interface = ModuleType(interface_name)
                    mock_interface.__spec__ = ModuleSpec(interface_name, None)
                    sys.modules[interface_name] = mock_interface
                # --- 结束 ---

                from transformers import AutoProcessor, AutoModelForCausalLM
                pdi_logger.info("正在加载 Florence-2 自动化检测引擎...")
                model_id = 'microsoft/Florence-2-base'
                
                self.processor = AutoProcessor.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                
                # 必须指定 attn_implementation="sdpa"，否则它即便通过了 import 也会在运行时找算子
                self.detector = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    trust_remote_code=True,
                    attn_implementation="eager", 
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device).eval()

    def _auto_detect(self, frame_bgr: np.ndarray, text_query: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """使用 Florence-2 自动获取物体中心点或 Bounding Box，直接接受 BGR numpy 帧，无需写临时文件"""
        self._init_detector()
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # 构造检测指令：在图中寻找特定的物体
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
        
        # 提取第一个匹配到的 Bounding Box
        try:
            boxes = parsed_answer["<CAPTION_TO_PHRASE_GROUNDING>"]["bboxes"]
            if len(boxes) > 0:
                box = np.array(boxes[0], dtype=np.float32) # [x1, y1, x2, y2]
                print(f"✅ 自动锁定目标 [{text_query}] 区域: {box.tolist()}")
                return None, box
        except Exception:
            pass
        
        print(f"⚠️ 未能识别到 [{text_query}]，降级使用画面中心点。")
        center_pt = np.array([[image.size[0]/2, image.size[1]/2]], dtype=np.float32)
        return center_pt, None

    def infer(self, video_path: str, click_points: Optional[List] = None, text_query: Optional[str] = None, box_prompt: Optional[List] = None, **kwargs) -> PerceptionResult:
        """支持手动点击点、外部 box_prompt 或 文本自动识别。click_points 可为 list 或 np.ndarray，勿用 if not click_points 判断。"""
        if self.model is None:
            raise RuntimeError("SAM2 未初始化")

        # 统一为可安全用 len() 判断的序列，避免 numpy 数组的 "truth value ambiguous"
        _no_points = click_points is None or (hasattr(click_points, "__len__") and len(click_points) == 0)

        # --- 外部 box_prompt 优先使用 ---
        box_np = None
        if box_prompt is not None:
            box_np = np.array(box_prompt, dtype=np.float32)
            _no_points = False  # 有 box 就不需要点提示

        # --- 自动识别逻辑（仅在无 box_prompt 且无点提示时触发） ---
        elif _no_points and text_query:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None or frame.size == 0:
                raise ValueError(f"无法读取视频首帧: {video_path}，请检查文件是否损坏或格式是否支持")
            click_points, box_np = self._auto_detect(frame, text_query)
            _no_points = False

        if (_no_points or click_points is None) and box_np is None:
            raise ValueError("必须提供 click_points、box_prompt 或 text_query 其中的一个。")

        # --- SAM2 核心推理逻辑 ---
        state = self.model.init_state(video_path=video_path, offload_video_to_cpu=True)

        if box_np is not None:
            # 如果有 box 提示，优先使用 box
            self.model.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box_np)
        else:
            # 降级使用点提示
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
        # 释放检测器显存
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