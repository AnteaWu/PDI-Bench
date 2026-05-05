import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import RANSACRegressor

class ProjectionJudge:
    """透视一致性判别式 (PDI Rule)
    
    核心功能：
    1. 计算 H-X 齐次性残差 (灵魂指标 ε)
    2. RANSAC 鲁棒估算消失点 (Vanishing Point) — 支持双路融合
    3. 深度-高度演化审计 (1/Z^2 规律校验)
    """
    def __init__(self, cx: float, cy: float):
        """
        cx, cy: 画面主点 (Principal Point)，通常为 (W/2, H/2)
        """
        self.cx = cx
        self.cy = cy

    def calculate_hx_residue(self, h_seq: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        """实现公式: ε = |(h1/hi) - (x1-cx)/(xi-cx)|
        
        Args:
            h_seq: 像素高度序列 (T,)
            x_seq: 物体质心横坐标序列 (T,)
            
        Returns:
            residues: 全序列的一致性残差序列 (T-1,)
        """
        if len(h_seq) < 2:
            return np.array([0.0])
            
        h1 = max(h_seq[0], 1.0) # 最小高度保护
        x1_relative = x_seq[0] - self.cx
        
        # 结果数组
        residues = []
        
        # 遍历后续帧进行一致性审计
        for i in range(1, len(h_seq)):
            hi = max(h_seq[i], 1.0)
            xi_relative = x_seq[i] - self.cx
            
            # 计算尺度缩放比与位移收敛比
            # 引入 epsilon=1e-3 避免除零，并处理物体穿过画面中心线的情况
            ratio_h = h1 / hi
            
            # 只有当物体不在中心点附近时才计算轨迹收敛比 (x_rel = 0 意味着位移收敛到消失点)
            if np.abs(x1_relative) > 1.0 and np.abs(xi_relative) > 1.0:
                ratio_x = x1_relative / xi_relative
                epsilon = np.abs(ratio_h - ratio_x)
            else:
                # 处于画面中心或初始位置就在中心的物体，主要考察 h(t) 的节奏
                epsilon = 0.0 
                
            residues.append(epsilon)
            
        return np.array(residues)

    # ------------------------------------------------------------------ #
    #  内部工具方法                                                        #
    # ------------------------------------------------------------------ #

    def _lines_from_tracks(
        self,
        tracks: np.ndarray,
        min_motion_px: float = 3.0,
        bottom_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """从 (N, T, 2) 轨迹提取齐次运动方向线 (M, 3)。

        Args:
            tracks:        (N, T, 2)
            min_motion_px: 运动量过滤阈值
            bottom_ratio:  若非 None，只保留 y 最大的该比例点（用于前景）
        Returns:
            lines_h: (M, 3) 归一化齐次直线，M 可为 0
        """
        N, T, _ = tracks.shape
        if T < 2 or N < 2:
            return np.empty((0, 3))

        n_avg = max(1, T // 10)
        p_start = tracks[:, :n_avg, :].mean(axis=1)
        p_end   = tracks[:, -n_avg:, :].mean(axis=1)
        motion_len = np.linalg.norm(p_end - p_start, axis=1)

        if bottom_ratio is not None:
            mean_y = tracks[:, :, 1].mean(axis=1)
            y_thresh = np.percentile(mean_y, (1 - bottom_ratio) * 100)
            spatial_mask = mean_y >= y_thresh
            valid_idx = np.where(spatial_mask & (motion_len >= min_motion_px))[0]
            if len(valid_idx) < 4:
                valid_idx = np.where(motion_len >= min_motion_px)[0]
        else:
            valid_idx = np.where(motion_len >= min_motion_px)[0]

        if len(valid_idx) < 2:
            return np.empty((0, 3))

        def to_h(p):
            return np.concatenate([p, np.ones((len(p), 1))], axis=1)

        ps = to_h(p_start[valid_idx])
        pe = to_h(p_end[valid_idx])
        lines_h = np.cross(ps, pe)
        norms = np.linalg.norm(lines_h[:, :2], axis=1, keepdims=True) + 1e-8
        return lines_h / norms

    def _lines_from_lsd(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        min_len: float = 60.0,
        angle_tol: float = 12.0,
    ) -> np.ndarray:
        """在背景区域（mask 之外）用 LSD 检测斜线，返回齐次直线 (M, 3)。

        Args:
            frame:     (H, W, 3) RGB 或 (H, W) 灰度帧
            mask:      (H, W) 前景 mask，前景=1
            min_len:   保留线段的最小像素长度
            angle_tol: 过滤近水平/近垂直线的角度容忍度（度）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()

        bg_mask = (mask == 0).astype(np.uint8)
        if bg_mask.shape != gray.shape:
            bg_mask = cv2.resize(bg_mask, (gray.shape[1], gray.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        gray = (gray.astype(np.float32) * bg_mask).astype(np.uint8)

        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(gray)[0]
        if detected is None:
            return np.empty((0, 3))

        lines_h = []
        for seg in detected.reshape(-1, 4):
            x1, y1, x2, y2 = seg
            if np.hypot(x2 - x1, y2 - y1) < min_len:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            # 过滤近水平线（<angle_tol）和近垂直线（>180-angle_tol）
            if angle < angle_tol or angle > (180 - angle_tol):
                continue
            l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
            norm = np.linalg.norm(l[:2]) + 1e-8
            lines_h.append(l / norm)

        return np.array(lines_h) if lines_h else np.empty((0, 3))

    def _ransac_vp(
        self,
        lines_h: np.ndarray,
        thresh: float = 20.0,
        iters: int = 500,
    ) -> Tuple[float, float]:
        """对齐次直线集合做 RANSAC，返回 (vp_x, vp_y)。"""
        M = len(lines_h)
        if M < 2:
            return self.cx, self.cy

        def dist(vp_h, lines):
            return np.abs(lines @ vp_h) / (np.linalg.norm(vp_h[:2]) + 1e-8)

        best_vp, best_n = None, 0
        rng = np.random.default_rng(42)
        for _ in range(iters):
            idx = rng.choice(M, 2, replace=False)
            vp_h = np.cross(lines_h[idx[0]], lines_h[idx[1]])
            if abs(vp_h[2]) < 1e-8:
                continue
            vp_h = vp_h / vp_h[2]
            n = int((dist(vp_h, lines_h) < thresh).sum())
            if n > best_n:
                best_n, best_vp = n, vp_h

        if best_vp is None:
            return self.cx, self.cy

        inliers = lines_h[dist(best_vp, lines_h) < thresh]
        if len(inliers) >= 2:
            A, b = inliers[:, :2], -inliers[:, 2]
            vp_xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return float(vp_xy[0]), float(vp_xy[1])
        return float(best_vp[0]), float(best_vp[1])

    # ------------------------------------------------------------------ #
    #  消失点估算接口                                                      #
    # ------------------------------------------------------------------ #

    def estimate_vanishing_point(
        self,
        tracks: np.ndarray,
        bottom_ratio: float = 0.4,
        min_motion_px: float = 3.0,
        ransac_thresh: float = 20.0,
        ransac_iters: int = 500,
    ) -> Tuple[float, float]:
        """（向后兼容）仅用前景轨迹估算消失点。"""
        lines_h = self._lines_from_tracks(tracks, min_motion_px, bottom_ratio)
        return self._ransac_vp(lines_h, ransac_thresh, ransac_iters)

    def estimate_vanishing_point_v2(
        self,
        fg_tracks: np.ndarray,
        bg_tracks: Optional[np.ndarray] = None,
        frames: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        min_motion_px: float = 3.0,
        ransac_thresh: float = 20.0,
        ransac_iters: int = 500,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """双路 VP 融合估算：背景轨迹 + LSD 场景几何线

        路径一 (FG)：前景轨迹（物体运动线）→ fg_vp
        路径二 (BG)：背景轨迹（环境运动线）+ LSD 斜线 → bg_vp
        全局 VP：  两路线段合并后再跑一次 RANSAC → global_vp

        Args:
            fg_tracks:   (N_fg, T, 2) 前景（mask 内）轨迹
            bg_tracks:   (N_bg, T, 2) 背景（mask 外）轨迹，可为 None
            frames:      (T, H, W, 3) 视频帧，供 LSD 检测，可为 None
            masks:       (T, H, W) 前景 mask，可为 None
            ...

        Returns:
            (global_vp, fg_vp, bg_vp): 三个 (x, y) 元组
        """
        # --- 路径一：前景轨迹线 ---
        # 短向量在 RANSAC 中噪声占主导，提高前景最小运动阈值到 10px
        fg_min_motion = max(min_motion_px, 10.0)
        fg_lines = self._lines_from_tracks(fg_tracks, fg_min_motion, bottom_ratio=0.4)
        fg_vp = self._ransac_vp(fg_lines, ransac_thresh, ransac_iters)

        # --- 路径二：背景轨迹线 + LSD 几何线 ---
        bg_line_parts = []

        if bg_tracks is not None and len(bg_tracks) >= 2:
            bg_motion_lines = self._lines_from_tracks(bg_tracks, min_motion_px)
            if len(bg_motion_lines) > 0:
                bg_line_parts.append(bg_motion_lines)

        # 多帧 LSD：对前 3 帧分别检测，合并后重复 3 次以提升权重
        if frames is not None and masks is not None:
            lsd_all = []
            n_lsd_frames = min(len(frames), len(masks), 3)
            for i in range(n_lsd_frames):
                lsd_f = self._lines_from_lsd(frames[i], masks[i])
                if len(lsd_f) > 0:
                    lsd_all.append(lsd_f)
            if lsd_all:
                lsd_combined = np.concatenate(lsd_all, axis=0)
                # 重复 3 次：在 RANSAC 投票中给背景几何线更高的 inlier 票数
                bg_line_parts.append(np.tile(lsd_combined, (3, 1)))

        if bg_line_parts:
            bg_lines_all = np.concatenate(bg_line_parts, axis=0)
            bg_vp = self._ransac_vp(bg_lines_all, ransac_thresh, ransac_iters)
        else:
            bg_vp = (self.cx, self.cy)

        # --- 全局融合 ---
        all_parts = [p for p in ([fg_lines] + bg_line_parts) if len(p) > 0]
        if all_parts:
            global_vp = self._ransac_vp(np.concatenate(all_parts, axis=0), ransac_thresh, ransac_iters)
        else:
            global_vp = (self.cx, self.cy)

        return global_vp, fg_vp, bg_vp

    def classify_fg_motion(
        self,
        fg_tracks: np.ndarray,
        min_motion_px: float = 5.0,
        parallel_angle_std_thresh: float = 15.0,
        horizontal_ratio_thresh: float = 2.5,
    ) -> str:
        """分类前景物体的运动模式。

        Args:
            fg_tracks:                (N, T, 2) 前景轨迹
            min_motion_px:            过滤过短运动的阈值
            parallel_angle_std_thresh: 运动方向角度标准差小于此值（度）时判为平行运动
            horizontal_ratio_thresh:  |mean_dx| / |mean_dy| 超过此值时判为横向运动

        Returns:
            "transverse" : 横向/平行平移（H-VP 公式退化）
            "depth"      : 深度方向运动（H-VP 公式有效）
        """
        N, T, _ = fg_tracks.shape
        if T < 2 or N < 2:
            return "depth"

        n_avg = max(1, T // 10)
        p_start = fg_tracks[:, :n_avg, :].mean(axis=1)
        p_end   = fg_tracks[:, -n_avg:, :].mean(axis=1)
        disp    = p_end - p_start                              # (N, 2)
        lengths = np.linalg.norm(disp, axis=1)
        valid   = lengths >= min_motion_px
        if valid.sum() < 3:
            return "depth"

        angles = np.degrees(np.arctan2(disp[valid, 1], disp[valid, 0]))  # [-180, 180]
        # 将角度折叠到 [0, 180) 消除方向符号影响
        angles = angles % 180.0
        angle_std = float(np.std(angles))

        mean_dx = float(np.mean(np.abs(disp[valid, 0])))
        mean_dy = float(np.mean(np.abs(disp[valid, 1]))) + 1e-6

        # 两个条件必须同时满足才判为横向平移：
        # 1. 矢量方向高度集中（近似平行，角度标准差小）
        # 2. 主运动方向明显水平（|dx| >> |dy|）
        # 单独满足任一条件可能只是前向运动带轻微侧漂，不应误判
        is_parallel   = angle_std < parallel_angle_std_thresh
        is_horizontal = (mean_dx / mean_dy) > horizontal_ratio_thresh
        if is_parallel and is_horizontal:
            return "transverse"
        return "depth"

    def compute_universal_trajectory_epsilon(
        self,
        h_seq: np.ndarray,
        xy_seq: np.ndarray,
        vanishing_point: Tuple[float, float],
    ) -> np.ndarray:
        """广义 H-VP 齐次性残差：h1/ht = Dist(p1,VP) / Dist(pt,VP)

        适用于任意方向的直线运动，不再依赖画面中心假设。
        捕捉"滑步"：物体缩小的节奏与奔向消失点的节奏不匹配。

        Args:
            h_seq:          (T,) 像素高度序列
            xy_seq:         (T, 2) 物体质心坐标序列 (x, y)
            vanishing_point: (vx, vy) 由 estimate_vanishing_point 计算得到

        Returns:
            (T-1,) 残差序列
        """
        T = len(h_seq)
        if T < 2 or xy_seq.shape[0] != T:
            return np.zeros(1)

        vp = np.array(vanishing_point, dtype=np.float64)
        dist = np.linalg.norm(xy_seq.astype(np.float64) - vp, axis=1)  # (T,)

        # 用前 5 帧均值作为稳定基准
        n_ref = min(5, T)
        h_ref   = max(float(np.mean(h_seq[:n_ref])), 1.0)
        dist_ref = max(float(np.mean(dist[:n_ref])), 1.0)

        errors = []
        for t in range(1, T):
            ht   = max(float(h_seq[t]), 1.0)
            dt   = float(dist[t])

            ratio_scale = h_ref / ht
            # 距离极小时说明物体已到达消失点附近，跳过
            if dt < 1.0 or dist_ref < 1.0:
                errors.append(0.0)
                continue
            ratio_dist = dist_ref / dt
            errors.append(abs(ratio_scale - ratio_dist))

        return np.array(errors) if errors else np.zeros(1)

    def check_scale_evolution(self, h_seq: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
        """验证高度缩减与深度的演化是否符合 1/Z^2 规律
        
        公式: h2 / h1 = z1 / z2 (在针孔模型下)
        """
        # 计算 h * z 的乘积，理论上应为常数 (取决于焦距 f 和物理高度 H)
        # 若乘积波动过大，说明出现了“体积呼吸感”
        h_z_product = h_seq * z_seq
        normalized_product = h_z_product / h_z_product[0] # 首帧归一化
        
        # 返回偏离常数的偏差
        return np.abs(normalized_product - 1.0)
