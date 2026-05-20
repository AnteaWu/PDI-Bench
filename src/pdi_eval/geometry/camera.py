import numpy as np
from typing import Optional, Tuple, List

class CameraModel:
    """Camera geometry (intrinsic & extrinsic helper).

    Features:
    1. Project 3D to 2D (world to pixel)
    2. Scale-align multi-frame 3D in world coordinates
    3. Track principal point and focal length
    """
    def __init__(self, focal_length: float, image_size: Tuple[int, int]):
        """
        Args:
            focal_length: focal length in pixels
            image_size: (H, W) video resolution
        """
        self.H, self.W = image_size
        self.f = focal_length
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        
        # Intrinsic matrix K
        self.K = np.array([
            [self.f, 0.0,    self.cx],
            [0.0,    self.f, self.cy],
            [0.0,    0.0,    1.0]
        ], dtype=np.float32)

    def project_3d_to_2d(self, pts_3d: np.ndarray, extrinsic_matrix: np.ndarray) -> np.ndarray:
        """Project: p = K * [R|t] * P
        
        Args:
            pts_3d: (N, 3) world coordinates
            extrinsic_matrix: (4, 4) [R|t]
            
        Returns:
            (N, 2) pixel (u, v)
        """
        # 1. Camera frame: P_cam = R * P_world + t
        R = extrinsic_matrix[:3, :3]
        t = extrinsic_matrix[:3, 3:4]
        pts_cam = (R @ pts_3d.T + t).T  # (N, 3)
        
        # 2. Depth normalize (avoid div by zero)
        z = pts_cam[:, 2:3]
        z[np.abs(z) < 1e-6] = 1e-6
        
        # 3. Map to pixel plane
        pts_norm = pts_cam[:, :2] / z
        u = self.f * pts_norm[:, 0] + self.cx
        v = self.f * pts_norm[:, 1] + self.cy
        
        return np.stack([u, v], axis=1)

    def align_to_unit_scale(self, z_seq: np.ndarray) -> np.ndarray:
        """Normalize scale: monocular depth is ambiguous; set first-frame scale to 1."""
        if len(z_seq) == 0:
            return z_seq
        z0 = z_seq[0]
        if np.abs(z0) < 1e-6:
            return z_seq
        return z_seq / z0

    def get_world_pcd(self, pcd_cam: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Map camera-frame point cloud to world frame.
        
        pose: camera pose [R|t]
        """
        # P_world = R^-1 * (P_cam - t)
        R = pose[:3, :3]
        t = pose[:3, 3:4]
        pts_world = (np.linalg.inv(R) @ (pcd_cam.T - t)).T
        return pts_world
