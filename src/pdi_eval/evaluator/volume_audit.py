import numpy as np
import cv2
from typing import Optional, Tuple
from ..utils.logger import pdi_logger


def audit_3d_rigidity_cv(
    pointmaps: np.ndarray,
    tracks_2d: np.ndarray,
    visibility: np.ndarray,
    masks: Optional[np.ndarray] = None,
    num_pairs: int = 30,
) -> Tuple[float, np.ndarray]:
    """基于 3D 相机坐标系(通过 pointmaps 采样)的刚体形变审计 (World Space Distance Invariant)

    锚点选取三重过滤：
    1. 可见度过滤：visibility > 0.5
    2. 深度梯度过滤：剔除局部深度突变处的点（边缘溢出 / 内部遮挡边界），
       阈值自适应取所有可见点梯度的第 75 百分位。
    3. 综合评分选对：score = 3D距离 × min(两端离掩码边界距离)，
       同时优化点对间距（信噪比）和内陆程度（可靠性）。
       distanceTransform 优先使用 SAM2 语义掩码，回退到 pointmaps 有效区域。

    Args:
        pointmaps:   (T, H, W, 3) MegaSAM 世界坐标系点图
        tracks_2d:   (T, N, 2) Co-Tracker 2D 轨迹
        visibility:  (T, N) 可见性标识
        masks:       (T, H, W) SAM2 前景掩码，用于 distanceTransform
        num_pairs:   采样点对数量
    Returns:
        final_score: float，平均刚性失败度
        history:     (T,) 每帧的刚性得分
    """
    T, N, _ = tracks_2d.shape
    H, W = pointmaps.shape[1], pointmaps.shape[2]

    # ==========================================
    # Step 1: 采样 3D 轨迹 (从 pointmaps 直接采样)
    # ==========================================
    pts_3d = np.zeros((T, N, 3))
    for t in range(T):
        u = np.clip(np.round(tracks_2d[t, :, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(tracks_2d[t, :, 1]).astype(int), 0, H - 1)
        pts_3d[t] = pointmaps[t, v, u]

    # ==========================================
    # Step 2: 深度梯度过滤 + distanceTransform 综合评分选对
    #
    # 评分 = 3D 点间距 × min(离边缘距离_i, 离边缘距离_j)
    # 同时优化：点对间距大（信噪比高） + 双端离边缘远（可靠性高）
    # ==========================================

    # distanceTransform：优先使用 SAM2 语义掩码，回退到 pointmaps 有效区域
    if masks is not None:
        m0 = masks[0]
        if m0.shape != (H, W):
            m0 = cv2.resize(m0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask0_pt = m0.astype(np.uint8)
    else:
        mask0_pt = pointmaps[0].any(axis=-1).astype(np.uint8)
    dist_map = cv2.distanceTransform(mask0_pt, cv2.DIST_L2, 5)  # (H, W)，值越大越"内陆"

    # 深度梯度过滤：自适应阈值（第 75 百分位），覆盖外轮廓和内部遮挡边界
    z0 = pointmaps[0, :, :, 2].astype(np.float32)
    gx = cv2.Sobel(z0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z0, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    all_u = np.clip(np.round(tracks_2d[0, :, 0]).astype(int), 0, W - 1)
    all_v = np.clip(np.round(tracks_2d[0, :, 1]).astype(int), 0, H - 1)
    vis_filter = visibility[0] > 0.5
    grad_at_all = grad_mag[all_v, all_u]
    vis_count = int(vis_filter.sum())
    grad_thresh = float(np.percentile(grad_at_all[vis_filter], 75)) if vis_count > 4 else np.inf
    # 逐步放宽：梯度+可见 → 仅可见
    valid_idx = np.array([], dtype=int)
    for filt in [vis_filter & (grad_at_all < grad_thresh), vis_filter]:
        valid_idx = np.where(filt)[0]
        if len(valid_idx) >= 5:
            break

    if len(valid_idx) < 5:
        return 1.0, np.full(T, 1.0)

    # 查询每个追踪点在首帧的 distanceTransform 值
    u0 = np.clip(np.round(tracks_2d[0, valid_idx, 0]).astype(int), 0, W - 1)
    v0 = np.clip(np.round(tracks_2d[0, valid_idx, 1]).astype(int), 0, H - 1)
    edge_dist = dist_map[v0, u0]  # (V,)，各点离边缘的像素距离

    # 计算两两 3D 欧氏距离矩阵
    pts_0 = pts_3d[0, valid_idx]
    diff = pts_0[:, np.newaxis, :] - pts_0[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)  # (V, V)

    # 综合评分：3D 距离 × min(两端离边缘距离)
    edge_min = np.minimum(
        edge_dist[:, np.newaxis],
        edge_dist[np.newaxis, :]
    )  # (V, V)
    score_matrix = dist_matrix * edge_min
    #dist_matrix 大：点对分得开，对形变的观测信噪比高。
    #edge_min 大：两个点都位于内陆，数据极可靠。
    
    i_upper, j_upper = np.triu_indices_from(score_matrix, k=1)
    scores    = score_matrix[i_upper, j_upper]
    distances = dist_matrix[i_upper, j_upper]
    actual_i  = valid_idx[i_upper]
    actual_j  = valid_idx[j_upper]

    sorted_args   = np.argsort(scores)[::-1]
    selected_args = sorted_args[:num_pairs]

    pair_i = actual_i[selected_args]
    pair_j = actual_j[selected_args]
    d_0    = distances[selected_args]

    # 过滤微小 3D 距离点对（防止 ratio = d_t / d_0 产生 Inf）
    valid_mask = d_0 > 1e-3
    pair_i, pair_j, d_0 = pair_i[valid_mask], pair_j[valid_mask], d_0[valid_mask]

    if len(d_0) < 3:
        return 1.0, np.full(T, 1.0)

    # ==========================================
    # Step 3 & 4: 逐帧计算 3D 距离比例与鲁棒 CV (MAD/Median)
    # ==========================================
    rigidity_history = [0.0]  # 第 0 帧默认 0 (完美)

    for t in range(1, T):
        vis_mask = (visibility[t, pair_i] > 0.5) & (visibility[t, pair_j] > 0.5)

        if np.sum(vis_mask) < 3:
            # 遮挡太严重，沿用上一帧得分
            rigidity_history.append(rigidity_history[-1])
            continue

        cur_i, cur_j, cur_d0 = pair_i[vis_mask], pair_j[vis_mask], d_0[vis_mask]
        p_i, p_j = pts_3d[t, cur_i], pts_3d[t, cur_j]
        d_t = np.linalg.norm(p_i - p_j, axis=-1)
        ratios = d_t / cur_d0
        median_r = np.median(ratios)
        mad_r = np.median(np.abs(ratios - median_r))
        rigidity_history.append(float(mad_r / (median_r + 1e-6)))

    rigidity_history = np.array(rigidity_history)
    # 跳过第 0 帧（基准帧，得分恒为 0，不含形变信息），避免在短视频中拉低均值
    score_frames = rigidity_history[1:] if len(rigidity_history) > 1 else rigidity_history
    return float(np.mean(score_frames)), rigidity_history


def audit_rigidity_stability(
    tracks: np.ndarray,
    h_seq: np.ndarray,
    n_pairs: int = 30,
) -> Tuple[float, np.ndarray]:
    """抗旋转的刚性稳定性审计

    改进点：不再除以 h(t)（旋转时 h 会变化导致误报）。
    改用「点对距离比值协同度」：刚体缩放时，所有点对的距离应等比例缩小，
    比值方差极小；发生非物理拉伸时，各点缩放不一致，方差升高。

    定义：
        ratio_ij(t) = d_ij(t) / d_ij(0)
        score(t)    = std(ratios) / (mean(ratios) + 1e-6)  ← 比例协同失败度

    Args:
        tracks:  (T, N, 2) Co-Tracker 追踪轨迹
        h_seq:   (T,) 保留接口兼容，不再用于归一化
        n_pairs: 随机采样锚点对数量

    Returns:
        (rigidity_cv, rigidity_history)
        rigidity_cv:      float，全时段协同失败均值，越高越「果冻」
        rigidity_history: (T,) 每帧的比例协同失败度
    """
    T, N, _ = tracks.shape
    if N < 2 or T < 2:
        return 0.0, np.zeros(T)

    rng = np.random.default_rng(42)
    actual_pairs = min(n_pairs, N * (N - 1) // 2)
    pairs = []
    seen = set()
    while len(pairs) < actual_pairs:
        i, j = rng.choice(N, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((int(i), int(j)))

    # 第 0 帧基准距离
    first_dists = np.array([
        np.linalg.norm(tracks[0, i] - tracks[0, j]) + 1e-6
        for i, j in pairs
    ])

    rigidity_history = [1.0]  # t=0 基准帧，协同度完美
    for t in range(1, T):
        curr_dists = np.array([
            np.linalg.norm(tracks[t, i] - tracks[t, j])
            for i, j in pairs
        ])
        ratios = curr_dists / first_dists
        mean_r = float(np.mean(ratios))
        score = float(np.std(ratios)) / (mean_r + 1e-6)
        rigidity_history.append(score)

    rigidity_history = np.array(rigidity_history)
    return float(np.mean(rigidity_history)), rigidity_history


def audit_3d_volume_stability(
    pointmaps: Optional[np.ndarray],
    masks: np.ndarray,
    tracks: Optional[np.ndarray] = None,
    h_seq: Optional[np.ndarray] = None,
    visibility: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, str]:
    """物理体积/刚性稳定性审计（三策略集成版）

    策略优先级：
    1. 3D 刚体比例协同法 (需 pointmaps + tracks + visibility) —— 鲁棒性最高，免疫单目尺度漂移
    2. 3D 点云身高法 (需 pointmaps + masks) —— 检测大范围纵向形变
    3. 2D Co-Tracker 刚性法 (需 tracks + h_seq) —— 无 3D 信息时的 fallback

    Returns:
        (rigidity_cv, history, strategy_name)
    """
    T = len(masks)

    # --- 策略 1: 3D 刚体比例协同法 (锚点对距离比值 MAD/Median) ---
    if pointmaps is not None and tracks is not None and visibility is not None:
        mask0 = masks[0]
        if mask0.shape[:2] != pointmaps.shape[1:3]:
            mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]), interpolation=cv2.INTER_NEAREST)
        fg_pts0 = pointmaps[0][mask0 > 0]
        if fg_pts0.shape[0] > 0 and np.mean(np.any(fg_pts0 != 0, axis=-1)) > 0.5:
            pdi_logger.info("Rigidity: 策略 1 (3D 刚体比例协同法)")
            cv, hist = audit_3d_rigidity_cv(pointmaps, tracks, visibility, masks)
            return cv, hist, "策略 1 (3D 刚体比例协同法)"

    # --- 策略 2: 3D 点云身高法 ---
    if pointmaps is not None:
        mask0 = masks[0]
        if mask0.shape[:2] != pointmaps.shape[1:3]:
            mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]), interpolation=cv2.INTER_NEAREST)
        fg_pts0 = pointmaps[0][mask0 > 0]
        fg_valid = (fg_pts0.shape[0] > 0) and (np.mean(np.any(fg_pts0 != 0, axis=-1)) > 0.5)
        if fg_valid:
            pdi_logger.info("Rigidity: 策略 2 (3D 点云身高法)")
            vol_history = []
            for t in range(T):
                pm_t = pointmaps[t]
                m_t = masks[t]
                if m_t.shape[:2] != pm_t.shape[:2]:
                    m_t = cv2.resize(m_t.astype(np.uint8), (pm_t.shape[1], pm_t.shape[0]), interpolation=cv2.INTER_NEAREST)
                bool_mask = m_t > 0
                if np.any(bool_mask):
                    y_pts = pm_t[bool_mask][:, 1]
                    h_3d = np.percentile(y_pts, 95) - np.percentile(y_pts, 5)
                    vol_history.append(h_3d)
                else:
                    vol_history.append(vol_history[-1] if vol_history else 0.0)
            vol_history = np.array(vol_history)
            if np.mean(vol_history) > 1e-6:
                vol_cv = float(np.std(vol_history) / np.mean(vol_history))
                return vol_cv, vol_history, "策略 2 (3D 点云身高法)"

    # --- 策略 3: 2D 刚性稳定性（Co-Tracker） ---
    if tracks is not None and h_seq is not None:
        pdi_logger.info("Rigidity: 策略 3 (2D Co-Tracker 点对距离法)")
        cv, hist = audit_rigidity_stability(tracks, h_seq)
        return cv, hist, "策略 3 (2D Co-Tracker 点对距离法)"

    # --- 兜底 ---
    pdi_logger.warning("Rigidity: 所有策略均不可用，返回零")
    return 0.0, np.zeros(T), "兜底 (无可用数据)"
