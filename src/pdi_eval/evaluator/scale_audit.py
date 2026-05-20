import numpy as np


def audit_scale_consistency(h_seq: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
    """Log-space scale-consistency audit.

    Theory: log(h) + log(z) = log(fH) = constant.
    Log domain makes errors for "2x larger" vs "0.5x smaller" symmetric in weight.

    Args:
        h_seq: (T,) SAM2 pixel height sequence.
        z_seq: (T,) Mega-SAM aligned depth sequence.

    Returns:
        (T-1,) log-offset residuals vs. median baseline from the first 5 frames.
    """
    T = len(h_seq)
    if T < 2 or len(z_seq) != T:
        return np.zeros(1)

    log_hz = np.log(np.maximum(h_seq, 1e-6)) + np.log(np.maximum(z_seq, 1e-6))

    n_ref = min(5, T)
    baseline = float(np.median(log_hz[:n_ref]))

    errors = np.abs(log_hz - baseline)
    return errors[1:]
