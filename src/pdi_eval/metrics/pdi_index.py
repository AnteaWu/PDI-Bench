import numpy as np
from typing import Dict, List, Any, Union

class PDIIndexCalculator:
    """PDI 总分合成模块 (The Judge)
    
    核心功能：
    1. 指标聚合：接收尺度、轨迹、3D 稳定性和 VP 耦合残差。
    2. 鲁棒性处理：兼容输入为序列(Array)或预计算好的 RMSE(Scalar)。
    3. 等级判定：提供直观的物理真实度等级。

    PDI v2.0 四项指标：
        epsilon_scale      : 缩放节奏对不对？
        epsilon_trajectory : 奔向消失点的路径对不对？
        sigma(rigidity)    : 物体自身稳不稳（刚性）？
        epsilon_vp         : 物体与场景是否在同一透视空间（视角耦合）？
    """
    def __init__(
        self,
        w_scale: float = 0.3,
        w_traj: float = 0.3,
        w_rigidity: float = 0.2,
        w_vp: float = 0.2,
    ):
        self.weights = {
            "scale": w_scale,
            "trajectory": w_traj,
            "rigidity": w_rigidity,
            "vp": w_vp,
        }

    def _ensure_scalar(self, val: Union[float, np.ndarray]) -> float:
        """辅助函数：确保输入被转化为标量误差值"""
        if isinstance(val, (np.ndarray, list)):
            if len(val) == 0: return 0.0
            # 如果传入的是序列，则计算其 RMSE；如果已经是单个值的数组，则取该值
            if len(val) > 1:
                return float(np.sqrt(np.mean(np.square(val))))
            return float(np.array(val).flatten()[0])
        return float(val)

    def compute_pdi(
        self,
        scale_errors: Union[float, np.ndarray],
        trajectory_errors: Union[float, np.ndarray],
        rigidity_cv: float,
        eps_vp: float = 0.0,
    ) -> Dict[str, Any]:
        """合成最终 PDI v2.0 分数与明细

        Args:
            scale_errors:       尺度残差序列或其 RMSE 值
            trajectory_errors:  轨迹残差序列或其 RMSE 值
            rigidity_cv:        刚性变异系数（点对距离比值的 std/mean）
            eps_vp:             VP 偏移归一化残差（视角耦合一致性），范围 [0, 1]
                                若背景线不足（LSD < 5 条）则外部传入 0.0 以降低权重影响。
        """
        rmse_scale = self._ensure_scalar(scale_errors)
        rmse_traj  = self._ensure_scalar(trajectory_errors)

        pdi_score = (
            self.weights["scale"]     * rmse_scale +
            self.weights["trajectory"] * rmse_traj  +
            self.weights["rigidity"]  * rigidity_cv +
            self.weights["vp"]        * eps_vp
        )

        grade = self.assign_grade(pdi_score)

        return {
            "pdi_score": round(pdi_score, 4),
            "grade": grade,
            "breakdown": {
                "scale_component":    round(rmse_scale,  4),
                "traj_component":     round(rmse_traj,   4),
                "rigidity_component": round(rigidity_cv, 4),
                "vp_component":       round(eps_vp,      4),
            },
        }

    def assign_grade(self, score: float) -> str:
        """物理真实度等级判定标准"""
        if score < 0.1: return "A (Physical Realism) - 物理逻辑严丝合缝"
        if score < 0.3: return "B (Minor Jitter) - 存在轻微几何抖动"
        if score < 0.6: return "C (Obvious Distortion) - 明显透视幻觉/滑步"
        return "F (Geometric Failure) - 物理逻辑彻底崩溃"
