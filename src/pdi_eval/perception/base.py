from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, Union, List, Dict, Any

@dataclass
class PerceptionResult:
    """标准感知结果数据类：连接感知模型与 PDI Evaluator 的桥梁"""
    video_id: str
    frames_count: int
    
    # --- 2D 维度 (用于像素级审计) ---
    masks: np.ndarray            # (T, H, W) 二值掩码序列
    h_pixel: np.ndarray          # (T,) 物体像素高度序列 h(t)
    x_center: np.ndarray         # (T,) 物体质心横坐标序列 x(t)
    tracks_2d: Optional[np.ndarray] = None # (T, N, 2) 亚像素追踪点轨迹
    
    # --- 3D 维度 (用于深度审计) ---
    depth_z: Optional[np.ndarray] = None   # (T,) 或 (T, H, W) 深度序列 Z(t)
    focal_length: Optional[float] = None   # 隐含焦距 f (由 Dust3R 推导)
    camera_poses: Optional[np.ndarray] = None # (T, 4, 4) 相机外参矩阵序列
    pointmaps: Optional[np.ndarray] = None # (T, H, W, 3) 场景点图 (Dust3R 特有)
    
    # --- 质量与状态分析 ---
    confidence: Optional[np.ndarray] = None   # (T,) 或 (T, N) 置信度得分
    is_truncated: Optional[np.ndarray] = None # (T,) 是否触碰边缘标志
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class BasePerceptor(ABC):
    """感知模块抽象基类"""
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def infer(self, video_input: Any, **kwargs) -> PerceptionResult:
        """所有子类必须实现的统一推理接口"""
        pass

    def scale_coords(self, coords: np.ndarray, current_res: tuple, target_res: tuple) -> np.ndarray:
        """坐标缩放工具：确保不同模型输出的 x, h 在同一分辨率下进行 PDI 计算"""
        h_ratio = target_res[0] / current_res[0]
        w_ratio = target_res[1] / current_res[1]
        scaled_coords = coords.copy().astype(float)
        scaled_coords[..., 0] *= w_ratio # x 轴
        scaled_coords[..., 1] *= h_ratio # y 轴
        return scaled_coords
