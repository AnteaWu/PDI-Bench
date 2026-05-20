import numpy as np
from typing import Dict, List, Any, Union

class PDIIndexCalculator:
    """PDI aggregate score module (The Judge).

    Responsibilities:
    1. Aggregate metrics: scale, trajectory, 3D stability, and VP-coupling residuals.
    2. Robust inputs: accepts either error sequences (arrays) or precomputed RMSE (scalar).
    3. Grade assignment: maps score to an interpretable physical-realism grade.

    PDI v2.0 components:
        epsilon_scale      : Is scale change rhythm consistent with perspective?
        epsilon_trajectory : Does motion converge toward the correct vanishing point?
        epsilon_rigidity   : Is the object rigid / stable in 3D?
        epsilon_vp         : Do object and scene share the same perspective space (view coupling)?
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
        """Coerce input to a scalar error value."""
        if isinstance(val, (np.ndarray, list)):
            if len(val) == 0: return 0.0
            # Sequences -> RMSE; single-element arrays -> that value
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
        """Compute final PDI v2.0 score and breakdown.

        Args:
            scale_errors:       Scale residual sequence or its RMSE.
            trajectory_errors:  Trajectory residual sequence or its RMSE.
            rigidity_cv:        Rigidity residual (epsilon_rigidity): coefficient of variation of pairwise distance ratios.
            eps_vp:             Normalized VP offset residual (view coupling), in [0, 1].
                                If background lines are insufficient (LSD < 5), callers may pass 0.0 to reduce weight impact.
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
                "epsilon_rigidity": round(rigidity_cv, 4),
                "vp_component":       round(eps_vp,      4),
            },
        }

    def assign_grade(self, score: float) -> str:
        """Map score to physical-realism grade."""
        if score < 0.1: return "A (Physical Realism) - physically consistent and tight"
        if score < 0.3: return "B (Minor Jitter) - slight geometric jitter"
        if score < 0.6: return "C (Obvious Distortion) - obvious perspective illusion or foot sliding"
        return "F (Geometric Failure) - geometric coherence breaks down"
