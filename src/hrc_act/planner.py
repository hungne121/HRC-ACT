"""Common continuous planner for both experts."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Segment:
    q0: np.ndarray
    q1: np.ndarray
    duration: float

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        u = float(np.clip(elapsed / self.duration, 0.0, 1.0))
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = (30*u**2 - 60*u**3 + 30*u**4)/self.duration
        return self.q0 + s*(self.q1-self.q0), ds*(self.q1-self.q0)

class MotionPlanner:
    def __init__(self, lengths=(0.35, 0.30), max_joint_speed=0.8):
        self.lengths = np.asarray(lengths, dtype=float)
        self.max_joint_speed = max_joint_speed

    def fk(self, q):
        a, b = q
        l1, l2 = self.lengths
        return np.array([l1*np.cos(a)+l2*np.cos(a+b),
                         l1*np.sin(a)+l2*np.sin(a+b)])

    def ik(self, xy, seed, iterations=150):
        q = np.asarray(seed, dtype=float).copy()
        xy = np.asarray(xy, dtype=float)
        radius = np.linalg.norm(xy)
        max_reach = self.lengths.sum()-0.015
        if radius > max_reach:
            xy = xy * max_reach/radius
        for _ in range(iterations):
            err = xy-self.fk(q)
            if np.linalg.norm(err) < 1e-4:
                break
            a,b=q
            l1,l2=self.lengths
            J=np.array([[-l1*np.sin(a)-l2*np.sin(a+b), -l2*np.sin(a+b)],
                        [ l1*np.cos(a)+l2*np.cos(a+b),  l2*np.cos(a+b)]])
            damping=1e-2
            dq=J.T @ np.linalg.solve(J @ J.T + damping*np.eye(2), err)
            q += np.clip(dq,-0.08,0.08)
        return q

    def plan(self, q0, q1):
        q0=np.asarray(q0,dtype=float)
        q1=np.asarray(q1,dtype=float)
        # Peak derivative of quintic is 1.875 / duration.
        duration=max(0.5, 1.875*np.max(np.abs(q1-q0))/self.max_joint_speed)
        return Segment(q0.copy(),q1.copy(),float(duration))
