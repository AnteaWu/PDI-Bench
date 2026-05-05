import numpy as np


def audit_scale_consistency(h_seq: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
    """Log 空间尺度守恒审计

    理论：log(h) + log(z) = log(fH) = Constant
    改进点：Log 空间确保"放大 2x"与"缩小 0.5x"的误差权重对称。

    Args:
        h_seq: (T,) SAM2 像素高度序列
        z_seq: (T,) Mega-SAM 对齐后的深度序列

    Returns:
        (T-1,) 相对于前 5 帧中值基准的 log 偏移残差
    """
    T = len(h_seq)
    if T < 2 or len(z_seq) != T:
        return np.zeros(1)

    log_hz = np.log(np.maximum(h_seq, 1e-6)) + np.log(np.maximum(z_seq, 1e-6))

    n_ref = min(5, T)
    baseline = float(np.median(log_hz[:n_ref]))

    errors = np.abs(log_hz - baseline)
    return errors[1:]
