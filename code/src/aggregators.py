from __future__ import annotations

"""Robust-aggregation baselines used for the paper comparison.

Each implementation follows the published formulation:

- Krum / Multi-Krum: Blanchard et al., "Machine Learning with Adversaries:
  Byzantine Tolerant Gradient Descent", NeurIPS 2017.
- Coordinate-wise trimmed mean: Yin et al., "Byzantine-Robust Distributed
  Learning: Towards Optimal Statistical Rates", ICML 2018.
- Norm-clipped mean: standard norm-bounding defense (e.g. Sun et al., 2019,
  "Can You Really Backdoor Federated Learning?").
- FLTrust: Cao et al., "FLTrust: Byzantine-robust Federated Learning via
  Trust Bootstrapping", NDSS 2021.
- FoolsGold: Fung et al., "The Limitations of Federated Learning in Sybil
  Settings", RAID 2020.
"""

from typing import Iterable, Sequence

import numpy as np


def _stack(deltas: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.stack([np.asarray(d, dtype=np.float64) for d in deltas], axis=0)
    if matrix.ndim != 2:
        raise ValueError("Client deltas must be one-dimensional vectors")
    return matrix


def krum_select(deltas: Sequence[np.ndarray], f: int, m: int = 1) -> list[int]:
    """Return the indices of the m clients with the lowest Krum scores.

    The Krum score of client i is the sum of its n - f - 2 smallest squared
    distances to other clients.  Requires 2f + 2 < n.
    """
    matrix = _stack(deltas)
    n = matrix.shape[0]
    if f < 0 or 2 * f + 2 >= n:
        raise ValueError(f"Krum requires 2f + 2 < n (got n={n}, f={f})")
    if not 1 <= m <= n:
        raise ValueError(f"multi-krum m must be in [1, {n}], got {m}")
    closest = n - f - 2
    squared = np.sum((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=2)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(squared[i], i)
        scores[i] = np.sum(np.sort(others)[:closest])
    return [int(i) for i in np.argsort(scores, kind="stable")[:m]]


def krum(deltas: Sequence[np.ndarray], f: int) -> np.ndarray:
    return np.asarray(deltas[krum_select(deltas, f, m=1)[0]], dtype=np.float64)


def multi_krum(deltas: Sequence[np.ndarray], f: int, m: int) -> tuple[np.ndarray, list[int]]:
    selected = krum_select(deltas, f, m=m)
    matrix = _stack([deltas[i] for i in selected])
    return matrix.mean(axis=0), selected


def trimmed_mean(deltas: Sequence[np.ndarray], trim_k: int) -> np.ndarray:
    """Coordinate-wise mean after removing the trim_k lowest and highest values."""
    matrix = _stack(deltas)
    n = matrix.shape[0]
    if not 0 <= trim_k < n / 2:
        raise ValueError(f"trim_k must satisfy 0 <= k < n/2 (got n={n}, k={trim_k})")
    if trim_k == 0:
        return matrix.mean(axis=0)
    ordered = np.sort(matrix, axis=0)
    return ordered[trim_k : n - trim_k].mean(axis=0)


def clipped_mean(deltas: Sequence[np.ndarray], clip_norm: float) -> np.ndarray:
    """Mean of updates individually clipped to an L2 ball of radius clip_norm."""
    if clip_norm <= 0.0:
        raise ValueError("clip_norm must be positive")
    matrix = _stack(deltas)
    norms = np.linalg.norm(matrix, axis=1)
    factors = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    return (matrix * factors[:, None]).mean(axis=0)


def fltrust_aggregate(
    deltas: Sequence[np.ndarray],
    server_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FLTrust aggregation given the server's own root-dataset update.

    Trust score TS_i = ReLU(cos(delta_i, delta_0)); each client update is
    rescaled to the server update's norm before the trust-weighted average.
    Returns (aggregate, trust_scores); the aggregate is zero when every trust
    score vanishes, which preserves the previous global model.
    """
    matrix = _stack(deltas)
    server = np.asarray(server_delta, dtype=np.float64)
    server_norm = float(np.linalg.norm(server))
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(norms * server_norm, 1e-12)
    cosines = matrix @ server / denominator
    trust = np.maximum(0.0, cosines)
    total = float(trust.sum())
    if total <= 0.0 or server_norm <= 0.0:
        return np.zeros_like(server), trust
    rescaled = matrix * (server_norm / np.maximum(norms, 1e-12))[:, None]
    return (trust[:, None] * rescaled).sum(axis=0) / total, trust


class FoolsGoldState:
    """Historical-update state for FoolsGold's sybil-similarity weighting."""

    def __init__(self, devices: Iterable[str], dimension: int, kappa: float = 1.0) -> None:
        self.devices = list(devices)
        self.history = {device: np.zeros(dimension, dtype=np.float64) for device in self.devices}
        self.kappa = float(kappa)

    def weights(self, deltas: dict[str, np.ndarray]) -> dict[str, float]:
        """Update histories with this round's deltas and return per-client weights."""
        for device, delta in deltas.items():
            self.history[device] = self.history[device] + np.asarray(delta, dtype=np.float64)

        matrix = np.stack([self.history[device] for device in self.devices], axis=0)
        norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
        normalised = matrix / norms[:, None]
        cs = normalised @ normalised.T
        np.fill_diagonal(cs, -np.inf)
        # Floor at 0 as in the released reference implementation, so pardoning
        # never sign-flips similarities when all off-diagonal cosines are negative.
        max_cs = np.maximum(cs.max(axis=1), 0.0)
        np.fill_diagonal(cs, 0.0)

        # Pardoning: rescale similarities of clients that look "more sybil".
        n = len(self.devices)
        for i in range(n):
            for j in range(n):
                if i != j and max_cs[j] > max_cs[i] and max_cs[j] > 0:
                    cs[i, j] *= max_cs[i] / max_cs[j]

        wv = 1.0 - cs.max(axis=1)
        wv = np.clip(wv, 0.0, 1.0)
        top = wv.max()
        if top > 0:
            wv = wv / top
        # Logit sharpening as in the reference implementation.
        with np.errstate(divide="ignore", over="ignore"):
            sharpened = self.kappa * (np.log(wv / (1.0 - wv) + 1e-12) + 0.5)
        sharpened[np.isinf(sharpened) | (wv >= 1.0)] = 1.0
        sharpened = np.clip(sharpened, 0.0, 1.0)
        return {device: float(w) for device, w in zip(self.devices, sharpened)}


def resolve_defense_params(
    n_clients: int,
    n_malicious: int,
    defense_cfg: dict | None,
) -> dict:
    """Fill the standard defense hyper-parameters from config or conventions.

    ``byzantine_f`` defaults to the attack's true count (the customary
    strongest-knowledge evaluation setting), floored at 1 so clean runs stay
    valid; multi-krum keeps n - f - 2 clients; trimmed mean trims f per side.
    """
    cfg = defense_cfg or {}
    f = cfg.get("byzantine_f")
    f = max(1, int(n_malicious)) if f is None else int(f)
    f = min(f, (n_clients - 3) // 2)
    m = cfg.get("multikrum_m")
    m = max(1, n_clients - f - 2) if m is None else int(m)
    trim_k = cfg.get("trim_k")
    trim_k = f if trim_k is None else int(trim_k)
    trim_k = min(trim_k, (n_clients - 1) // 2)
    return {
        "byzantine_f": f,
        "multikrum_m": m,
        "trim_k": trim_k,
        "clip_to_median_multiplier": float(cfg.get("clip_to_median_multiplier", 1.0)),
        "fltrust_root_size": int(cfg.get("fltrust_root_size", 200)),
    }
