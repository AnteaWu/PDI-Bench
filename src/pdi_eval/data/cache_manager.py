import os
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional

class CacheManager:
    """感知结果缓存管理器 (.npz/.pt)
    
    功能：
    - 检查感知步骤是否已完成
    - 保存/读取中间掩码、深度图等大宗数据
    """
    def __init__(self, cache_dir: str = "output/cache/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, video_id: str, step_name: str) -> Path:
        """生成缓存文件名，例如 output/cache/video01_sam2.npz"""
        return self.cache_dir / f"{video_id}_{step_name}.npz"

    def exists(self, video_id: str, step_name: str) -> bool:
        """检查指定步骤是否有缓存"""
        return self.get_cache_path(video_id, step_name).exists()

    def save_step(self, video_id: str, step_name: str, data: dict):
        """将感知步骤数据保存为 npz"""
        cache_path = self.get_cache_path(video_id, step_name)
        # 将 numpy 或 tensor 转为 ndarray
        serializable_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                serializable_data[k] = v.cpu().numpy()
            else:
                serializable_data[k] = v
        
        np.savez_compressed(cache_path, **serializable_data)

    def load_step(self, video_id: str, step_name: str) -> Optional[dict]:
        """从缓存加载数据"""
        cache_path = self.get_cache_path(video_id, step_name)
        if not cache_path.exists():
            return None
        
        with np.load(cache_path, allow_pickle=True) as loader:
            return {k: loader[k] for k in loader.files}
