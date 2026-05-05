import numpy as np
from typing import Optional, Tuple, List

class CameraModel:
    """相机几何模型 (Intrinsic & Extrinsic Manager)
    
    核心功能：
    1. 3D 到 2D 投影 (World-to-Pixel)
    2. 多帧 3D 点云在世界坐标系下的尺度对齐
    3. 管理主点 (Principal Point) 与焦距 (Focal Length)
    """
    def __init__(self, focal_length: float, image_size: Tuple[int, int]):
        """
        Args:
            focal_length: 隐含焦距 (像素单位)
            image_size: (H, W) 视频分辨率
        """
        self.H, self.W = image_size
        self.f = focal_length
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        
        # 构造内参矩阵 K
        self.K = np.array([
            [self.f, 0.0,    self.cx],
            [0.0,    self.f, self.cy],
            [0.0,    0.0,    1.0]
        ], dtype=np.float32)

    def project_3d_to_2d(self, pts_3d: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
        """实现投影公式: p = K * [R|t] * P
        
        Args:
            pts_3d: (N, 3) 空间坐标
            extrinsic_matrix: (4, 4) 相机外参矩阵 [R|t]
            
        Returns:
            (N, 2) 像素坐标 (u, v)
        """
        # 1. 变换到相机坐标系: P_cam = R * P_world + t
        R = extrinsic_matrix[:3, :3]
        t = extrinsic_matrix[:3, 3:4]
        pts_cam = (R @ pts_3d.T + t).T  # (N, 3)
        
        # 2. 深度归一化 (防止除零)
        z = pts_cam[:, 2:3]
        z[np.abs(z) < 1e-6] = 1e-6
        
        # 3. 映射到像素平面
        pts_norm = pts_cam[:, :2] / z
        u = self.f * pts_norm[:, 0] + self.cx
        v = self.f * pts_norm[:, 1] + self.cy
        
        return np.stack([u, v], axis=1)

    def align_to_unit_scale(self, z_seq: np.ndarray) -> np.ndarray:
        """尺度归一化 (Scale Normalization)
        
        由于单目 3D 具有尺度不确定性，将首帧深度/位移设为 1.0 (Unit Scale)
        """
        if len(z_seq) == 0:
            return z_seq
        z0 = z_seq[0]
        if np.abs(z0) < 1e-6:
            return z_seq
        return z_seq / z0

    def get_world_pcd(self, pcd_cam: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """将相机坐标系下的点云转回世界坐标系
        
        pose: 相机位姿 [R|t]
        """
        # P_world = R.inv * (P_cam - t)
        R = pose[:3, :3]
        t = pose[:3, 3:4]
        pts_world = (np.linalg.inv(R) @ (pcd_cam.T - t)).T
        return pts_world
