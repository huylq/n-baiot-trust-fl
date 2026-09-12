from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrustComponents:
    cosine: float
    a_k: float
    m_k: float
    v_k: float
    s_k: float


@dataclass
class ClientSecurityRecord:
    device: str
    train_samples: int
    val_loss: float
    delta_norm: float
    reference_norm: float
    cosine_to_reference: float
    a_k: float
    m_k: float
    s_k: float
    trust_score: float
    strike: int
    flagged: bool
    admitted: bool
    rejection_reason: str
    detected: bool
    is_malicious_ground_truth: bool
    aggregation_weight: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def coordinate_wise_median(list_of_deltas: Iterable[np.ndarray]) -> np.ndarray:
    """Return one independent median per model coordinate."""
    deltas = [np.asarray(delta, dtype=np.float64) for delta in list_of_deltas]
    if not deltas:
        raise ValueError("At least one client delta is required")
    shape = deltas[0].shape
    if any(delta.shape != shape for delta in deltas):
        raise ValueError("All client deltas must have the same shape")
    return np.median(np.stack(deltas, axis=0), axis=0)


def reference_statistics(
    list_of_deltas: Iterable[np.ndarray],
    reference: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute the configured vector reference and the separate norm reference."""
    deltas = [np.asarray(delta, dtype=np.float64) for delta in list_of_deltas]
    if not deltas:
        raise ValueError("At least one client delta is required")
    norms = np.array([np.linalg.norm(delta) for delta in deltas], dtype=np.float64)
    if reference == "median":
        return coordinate_wise_median(deltas), float(np.median(norms)), norms
    if reference == "mean":
        return np.mean(np.stack(deltas, axis=0), axis=0), float(np.mean(norms)), norms
    raise ValueError(f"Unknown trust reference: {reference!r}; expected 'median' or 'mean'")


def compute_legacy_trust_score(val_loss: float, delta_norm: float, cosine_to_mean: float) -> float:
    """Product of inverse-loss, cosine-similarity, and inverse-norm factors."""
    loss_term = 1.0 / (1.0 + val_loss)
    similarity_term = max(0.0, (cosine_to_mean + 1.0) / 2.0)
    norm_term = 1.0 / (1.0 + 0.1 * delta_norm)
    return float(loss_term * similarity_term * norm_term)


def robust_norm_zscores(norms: np.ndarray, security_cfg: dict) -> np.ndarray:
    """Robust z-scores of the update norms in log-space.

    Log-space keeps the statistic scale-free; the MAD floor stops the score
    from exploding when all clients happen to send near-identical norms.
    """
    eps = float(security_cfg.get("eps", 1e-6))
    mad_floor = float(security_cfg.get("log_norm_mad_floor", 0.05))
    log_norms = np.log(np.maximum(np.asarray(norms, dtype=np.float64), eps))
    median = float(np.median(log_norms))
    mad = float(np.median(np.abs(log_norms - median)))
    scale = max(1.4826 * mad, mad_floor)
    return (log_norms - median) / scale


def compute_trust_score(
    delta: np.ndarray,
    reference_delta: np.ndarray,
    norm_z: float,
    security_cfg: dict,
    val_loss: float | None = None,
) -> TrustComponents:
    """Compute the v2 multiplicative per-round trust evidence.

    ``a_k`` is the nonnegative cosine. ``m_k`` penalizes positive norm
    z-scores above ``norm_z_soft``. The default path ignores ``val_loss``.
    """
    z_soft = float(security_cfg.get("norm_z_soft", 2.0))
    cosine = cosine_similarity(delta, reference_delta)
    a_k = max(0.0, cosine)
    m_k = float(np.exp(-max(0.0, float(norm_z) - z_soft)))
    v_k = 1.0
    if security_cfg.get("use_self_reported_loss", False):
        if val_loss is None:
            raise ValueError("val_loss is required when use_self_reported_loss=true")
        v_k = 1.0 / (1.0 + float(val_loss))
    s_k = float(a_k * m_k * v_k)
    return TrustComponents(cosine=cosine, a_k=a_k, m_k=m_k, v_k=v_k, s_k=s_k)


class TrustState:
    """Persistent EMA and strike-window state for all clients."""

    def __init__(self, devices: Iterable[str], security_cfg: dict) -> None:
        self.security_cfg = security_cfg
        initial = float(security_cfg.get("T_init", 0.5))
        window_length = int(security_cfg.get("window_W", 5))
        if window_length <= 0:
            raise ValueError("window_W must be positive")
        self.trust = {device: initial for device in devices}
        self.windows = {device: deque(maxlen=window_length) for device in devices}

    def update(
        self,
        device: str,
        s_k: float,
        raw_cosine: float,
        norm_z: float,
        in_warmup: bool = False,
    ) -> tuple[float, int, bool, list[str]]:
        """Advance the per-client EMA and strike window.

        Violations are cosine < -cos_margin or norm z-score > norm_z_strike.
        Warm-up updates the EMA but does not add strikes.
        """
        cfg = self.security_cfg
        reasons: list[str] = []
        if raw_cosine < -float(cfg.get("cos_margin", 0.1)):
            reasons.append("negative_cosine")
        if float(norm_z) > float(cfg.get("norm_z_strike", 3.5)):
            reasons.append("norm_robust_outlier")

        if not cfg.get("use_memory", True):
            # Memory-off mode uses instantaneous scores and emits no strike flags.
            self.trust[device] = float(s_k)
            self.windows[device].clear()
            return float(s_k), 0, False, reasons

        beta = float(cfg.get("beta", 0.8))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        trust_score = beta * self.trust[device] + (1.0 - beta) * float(s_k)
        self.trust[device] = float(trust_score)

        if in_warmup:
            return float(trust_score), int(sum(self.windows[device])), False, reasons
        self.windows[device].append(bool(reasons))
        strike = int(sum(self.windows[device]))
        flagged = strike >= int(cfg.get("strike_W_neg", 3))
        return float(trust_score), strike, bool(flagged), reasons

    def round_trust_threshold(self, admitted_devices: Iterable[str]) -> float:
        """Return max(tau_abs, tau_rel * median admitted trust)."""
        cfg = self.security_cfg
        trusts = [self.trust[device] for device in admitted_devices]
        if not trusts:
            return float(cfg.get("tau_abs", 0.05))
        relative = float(cfg.get("tau_rel", 0.5)) * float(np.median(trusts))
        return max(float(cfg.get("tau_abs", 0.05)), relative)


def zero_trust_admission(
    train_samples: int,
    norm_z: float,
    security_cfg: dict,
) -> tuple[bool, str]:
    """Apply the sample-count minimum and robust norm z-score admission gate."""
    if train_samples < int(security_cfg.get("min_client_train_samples", 0)):
        return False, "insufficient_train_samples"
    if float(norm_z) > float(security_cfg.get("norm_z_admission", 4.5)):
        return False, "delta_norm_robust_outlier_gate"
    return True, "accepted"


def clip_delta(delta: np.ndarray, reference_norm: float, security_cfg: dict) -> np.ndarray:
    """Clip to clip_multiplier * reference_norm while preserving direction."""
    multiplier = float(security_cfg.get("clip_multiplier", 2.0))
    limit = multiplier * float(reference_norm)
    norm = float(np.linalg.norm(delta))
    if limit <= 0.0 or norm <= limit:
        return delta
    return delta * (limit / norm)


def hash_delta(delta: np.ndarray) -> str:
    canonical = np.asarray(delta, dtype="<f8").reshape(-1)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


class AuditChain:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.previous_hash = "GENESIS"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

    def append(self, payload: dict) -> str:
        record = dict(payload)
        record["previous_hash"] = self.previous_hash
        record_json = json.dumps(record, sort_keys=True, separators=(",", ":"))
        current_hash = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
        record["current_hash"] = current_hash
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.previous_hash = current_hash
        return current_hash

    def append_flag_event(
        self,
        round_index: int,
        client: str,
        delta: np.ndarray,
        reasons: list[str],
        **context: object,
    ) -> str:
        return self.append(
            {
                "event": "client_flagged",
                "round": int(round_index),
                "client": client,
                "reason": reasons,
                "delta_hash": hash_delta(delta),
                **context,
            }
        )

    def records(self) -> list[dict]:
        if not self.output_path.exists():
            return []
        with self.output_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def verify(self) -> bool:
        previous_hash = "GENESIS"
        for record in self.records():
            if record.get("previous_hash") != previous_hash:
                return False
            expected_hash = record.get("current_hash")
            payload = {key: value for key, value in record.items() if key != "current_hash"}
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            actual_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                return False
            previous_hash = expected_hash
        return True
