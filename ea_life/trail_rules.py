"""Replaceable trail rules for EA-Life (NCA experiment, 16 Oct 2026).

Every rule maps the two visible trail fields (food trail, danger trail; flat lists, row-major n x n) to their
next values. Deposits by organisms happen before the rule runs, exactly as in the hand-coded engine.
"""
from __future__ import annotations
import numpy as np

D0 = np.array([0.985, 0.96])          # hand-coded per-tick persistence of the food and danger trails


def perceive(X: np.ndarray) -> np.ndarray:
    """(C, n, n) -> (4C, n, n): identity, Sobel x, Sobel y, neighbour mean minus centre (zero padding)."""
    P = np.pad(X, ((0, 0), (1, 1), (1, 1)))
    c = P[:, 1:-1, 1:-1]
    n_, s_, w_, e_ = P[:, :-2, 1:-1], P[:, 2:, 1:-1], P[:, 1:-1, :-2], P[:, 1:-1, 2:]
    nw, ne, sw, se = P[:, :-2, :-2], P[:, :-2, 2:], P[:, 2:, :-2], P[:, 2:, 2:]
    sx = ((ne + 2 * e_ + se) - (nw + 2 * w_ + sw)) / 8.0
    sy = ((sw + 2 * s_ + se) - (nw + 2 * n_ + ne)) / 8.0
    lap = (n_ + s_ + w_ + e_ + nw + ne + sw + se) / 8.0 - c
    return np.concatenate([c, sx, sy, lap], axis=0)


def _finish(vis: np.ndarray):
    vis = np.clip(vis, 0.0, 1.0)
    vis[vis < 0.001] = 0.0
    return vis[0].ravel().tolist(), vis[1].ravel().tolist()


class DecayRule:
    """The hand-coded rule in NumPy (reference; must equal the engine's own loop bit for bit)."""
    def update(self, rt, dt, n):
        S = np.array([rt, dt], dtype=float).reshape(2, n, n)
        return _finish(D0[:, None, None] * S)


class FixedCARule:
    """Hand-designed CA: persistence D and diffusion kappa toward the 8-neighbour mean, on both visible channels."""
    def __init__(self, persistence=(0.985, 0.96), kappa=0.0):
        self.D = np.asarray(persistence, float); self.kappa = float(kappa)
    def update(self, rt, dt, n):
        S = np.array([rt, dt], dtype=float).reshape(2, n, n)
        lap = perceive(S)[6:8]
        return _finish(self.D[:, None, None] * S + self.kappa * lap)


class LinearRule:
    """Learned linear 3x3 rule on the 2 visible channels: next = (D0 + dD) * S + L . perceive(S) + b."""
    DIM = 2 + 2 * 8 + 2
    def __init__(self, theta):
        t = np.asarray(theta, float); assert t.size == self.DIM
        self.D = D0 + t[:2]; self.L = t[2:18].reshape(2, 8); self.b = t[18:20]
    def update(self, rt, dt, n):
        S = np.array([rt, dt], dtype=float).reshape(2, n, n)
        P = perceive(S).reshape(8, -1)
        new = self.D[:, None] * S.reshape(2, -1) + self.L @ P + self.b[:, None]
        return _finish(new.reshape(2, n, n))


class NCARule:
    """Learned neural CA: 4 channels (2 visible, 2 hidden), shared weights at every cell.
    next = D * S + L . p + W2 . tanh(W1 . p + b1) + b2, with p = perceive(S) (16 features).
    W1 is a fixed random projection plus a learned offset; L, W2 and b2 start at zero, so theta = 0 is the
    hand-coded rule (hidden channels stay at zero)."""
    C, H = 4, 8
    DIM = 4 + 4 * 16 + 8 * 16 + 8 + 4 * 8 + 4
    def __init__(self, theta, init_seed=0):
        t = np.asarray(theta, float); assert t.size == self.DIM
        C, H, F = self.C, self.H, 4 * self.C
        rng = np.random.default_rng(init_seed)
        i = 0
        def take(k):
            nonlocal i
            v = t[i:i + k]; i += k; return v
        self.D = np.array([0.985, 0.96, 0.9, 0.9]) + take(C)
        self.L = take(C * F).reshape(C, F)
        self.W1 = rng.normal(0.0, 0.5, (H, F)) + take(H * F).reshape(H, F)
        self.b1 = take(H)
        self.W2 = take(C * H).reshape(C, H)
        self.b2 = take(C)
        self.hidden = None
    def update(self, rt, dt, n):
        C = self.C
        S = np.zeros((C, n, n))
        S[0] = np.asarray(rt, float).reshape(n, n); S[1] = np.asarray(dt, float).reshape(n, n)
        if self.hidden is not None: S[2:] = self.hidden
        P = perceive(S).reshape(4 * C, -1)
        new = self.D[:, None] * S.reshape(C, -1) + self.L @ P + self.W2 @ np.tanh(self.W1 @ P + self.b1[:, None]) + self.b2[:, None]
        new = new.reshape(C, n, n)
        self.hidden = np.clip(new[2:], -1.0, 1.0)
        return _finish(new[:2])


def make_rule(spec):
    kind = spec["kind"]
    if kind in ("engine", "no_trails"): return None
    if kind == "decay": return DecayRule()
    if kind == "fixed_ca": return FixedCARule(spec["persistence"], spec["kappa"])
    if kind == "linear": return LinearRule(spec["theta"])
    if kind == "nca": return NCARule(spec["theta"])
    raise ValueError(kind)
