"""Observed vs forecast-informed target selection; planner is shared."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Prediction:
    times: np.ndarray
    positions: np.ndarray

    def __post_init__(self):
        t=np.asarray(self.times,dtype=float)
        p=np.asarray(self.positions,dtype=float)
        if t.ndim!=1 or t.size<2 or p.shape!=(t.size,3):
            raise ValueError("Expected times (N,) and positions (N,3)")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))
                and np.all(t>0) and np.all(np.diff(t)>0)):
            raise ValueError("Forecast times must increase and data must be finite")

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path,allow_pickle=False) as data:
            return cls(data["times"],data["positions"])

def choose_target(expert: str, observed_xyz, ready: bool, forecast: Prediction | None):
    if expert not in ("observed","intent"):
        raise ValueError("Unknown expert")
    observed=np.asarray(observed_xyz,dtype=float)
    if expert=="intent" and forecast is not None:
        # Stable future wrist at end of horizon; future multi-joint gating is out of scope.
        if np.linalg.norm(forecast.positions[-1]-forecast.positions[-2]) < 0.08:
            return forecast.positions[-1].copy(), "prediction"
    if ready:
        return observed.copy(), "observed"
    return None, "waiting"
