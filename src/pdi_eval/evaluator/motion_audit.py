import numpy as np
from typing import Tuple, Optional


def _medfilt1d(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """1D 中值滤波（不依赖 scipy，用 numpy 滑动窗口实现）"""
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1)


def audit_3d_trajectory_consistency(
    pointmaps: Optional[np.ndarray],
    masks: np.ndarray,
    fps: float = 24.0,
) -> np.ndarray:
    """基于 3D 运动学的轨迹一致性审计（世界坐标系）

    直接从 MegaSAM 世界坐标系点图中采样前景质心，
    审查 3D 轨迹的运动平滑性（加速度异常 + 方向折返异常）。

    优于传统 VP 方法：
    - 完全不依赖 h_pixel，与 scale 指标正交
    - 免疫一切动镜头（世界坐标系绝对位置）
    - 兼容直线和曲线运动

    Args:
        pointmaps: (T, H, W, 3) 世界坐标系三维点图（MegaSAM 输出）
        masks:     (T, H, W) 前景掩码（SAM2 输出）
        fps:       视频帧率，用于时间归一化（m/s、m/s²），消除帧率差异影响

    Returns:
        (T-1,) 轨迹残差序列
    """
    if pointmaps is None or masks is None:
        return np.zeros(1)

    T = pointmaps.shape[0]
    if T < 3:
        return np.zeros(max(1, T - 1))

    # 有效性检验：pointmaps 全零说明 MegaSAM 走了 fallback
    mask0 = masks[0]
    if mask0.shape[:2] != pointmaps.shape[1:3]:
        import cv2
        mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]),
                           interpolation=cv2.INTER_NEAREST)
    fg_pts0 = pointmaps[0][mask0 > 0]
    if fg_pts0.shape[0] == 0 or np.mean(np.any(fg_pts0 != 0, axis=-1)) < 0.5:
        return np.zeros(T - 1)

    # Step 1: 鲁棒提取 3D 质心
    # - 用中位数替代均值，过滤 SAM2 mask 边缘的 Depth Bleeding
        #  在物体边缘，深度图经常会“溢出”到背景上产生极大误差。中位数比均值（mean）更能抵抗这些离群点。
    # - 掩码丢失时继承上一帧坐标，避免坐标飞到世界中心产生虚假抖动
    world_traj = np.zeros((T, 3))
    last_valid = np.zeros(3)

    for t in range(T):
        m = masks[t]
        pm = pointmaps[t]
        if m.shape[:2] != pm.shape[:2]:
            import cv2
            m = cv2.resize(m.astype(np.uint8), (pm.shape[1], pm.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        valid_pts = pm[m > 0]
        if len(valid_pts) > 10:
            centroid = np.median(valid_pts, axis=0)
            world_traj[t] = centroid
            last_valid = centroid
        else:
            world_traj[t] = last_valid

    # Step 1.5: 轻量时序中值平滑，抑制 MegaSAM 的深度闪烁噪声
    for i in range(3):
        world_traj[:, i] = _medfilt1d(world_traj[:, i], k=3)

    # Step 2: 3D 速度与加速度（物理量纲归一化：÷dt 得到 m/s、m/s²）
    # Fix 3: 引入 dt=1/fps，使指标与帧率无关，不同 FPS 的视频可公平比较
    dt = 1.0 / max(fps, 1.0)
    velocity = np.diff(world_traj, axis=0) / dt      # (T-1, 3), 单位 m/s
    speed = np.linalg.norm(velocity, axis=1)          # (T-1,),   单位 m/s
    acceleration = np.diff(velocity, axis=0) / dt    # (T-2, 3), 单位 m/s²
    accel_mag = np.linalg.norm(acceleration, axis=1)  # (T-2,)

    # 以视频全局平均速度作为归一化基准，而非逐帧速度。
    # 逐帧速度做分母会导致：物体在慢帧上即使有微小噪声加速度也被放大为极大值（÷1e-3 = ×1000）。
    # 全局均值基准使"相对加速度"变为"加速度相对于该视频的典型运动尺度"，
    # 对静止视频退化为绝对加速度评估（mean_speed ≈ 0 时由 global_floor 兜底，
    # 而 global_floor 由 accel_mag 的量纲决定，不会无谓放大噪声）。
    speed_median = float(np.median(speed))
    accel_median = float(np.median(accel_mag))
    global_floor = accel_median * 2.0   # 底噪自适应：加速度中位数×2，量纲自洽
    speed_ref    = max(speed_median, global_floor, 1e-6)

    # 相对加速度率：加速度 / 全局速度基准，单位 1/s
    relative_accel_raw = accel_mag / speed_ref          # (T-2,), 1/s

    # tanh 压缩，将无界量映射至 [0, 2)，与 angle_penalty 量纲对齐
    # tanh(x/5)*2: x=5→1.52, x=10→1.93, x≥15→≈2（异常运动饱和）
    relative_accel = 2.0 * np.tanh(relative_accel_raw / 5.0)  # (T-2,)

    # Step 3: 相邻速度方向余弦（惩罚锐角折返）
    v1, v2 = velocity[:-1], velocity[1:]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9
    n2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9
    cos_angles = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=1), -1.0, 1.0)

    # 物体近乎静止时方向惩罚无意义，用全局基准 speed_ref 判断是否在运动
    moving = (speed[:-1] > speed_ref * 0.1) & (speed[1:] > speed_ref * 0.1)
    angle_penalty = np.zeros_like(cos_angles)
    angle_penalty[moving] = 1.0 - cos_angles[moving]   # 同向=0，折返=2

    # Step 4: 合成误差，补齐首帧保持 (T-1,) 长度
    eps_traj = relative_accel * 0.5 + angle_penalty * 0.5
    return np.insert(eps_traj, 0, eps_traj[0])


def audit_trajectory_consistency(
    h_seq: np.ndarray,
    xy_seq: np.ndarray,
    vanishing_point: Tuple[float, float],
) -> np.ndarray:
    """广义透视轨迹审计（横向运动自适应版）

    两种场景自动切换：
    - 纵向/斜向运动：Log(H-VP 齐次性)残差，log(h1/ht) vs log(d1/dt)
    - 横向平移（VP 在无穷远）：高度稳定性残差，|h(t) - h(0)| / h(0)

    判断依据：VP 距离序列的极差比 < 5% 视为横向运动。

    Args:
        h_seq:           (T,) SAM2 像素高度序列
        xy_seq:          (T, 2) Co-Tracker 质心坐标序列
        vanishing_point: (vx, vy)

    Returns:
        (T-1,) 轨迹残差序列
    """
    T = len(h_seq)
    if T < 2:
        return np.zeros(1)

    vp = np.array(vanishing_point, dtype=np.float64)
    dist = np.linalg.norm(xy_seq.astype(np.float64) - vp, axis=1)  # (T,)

    dist_range_ratio = float(np.ptp(dist)) / (float(np.mean(dist)) + 1e-6)

    if dist_range_ratio < 0.05:
        # 横向平移场景：深度基本不变，h 也不应变
        h0 = max(float(h_seq[0]), 1e-6)
        errors = np.abs(h_seq[1:] - h0) / h0
        return errors

    # 纵向/斜向场景：Log 空间 H-VP 齐次性
    log_h = np.log(np.maximum(h_seq, 1e-6))
    log_d = np.log(np.maximum(dist, 1e-6))

    # 用前 5 帧中值作为基准，抑制初始帧噪声
    n_ref = min(5, T)
    h_base = float(np.median(log_h[:n_ref]))
    d_base = float(np.median(log_d[:n_ref]))

    log_h_ratio = log_h - h_base
    log_d_ratio = log_d - d_base

    errors = np.abs(log_h_ratio - log_d_ratio)
    return errors[1:]
