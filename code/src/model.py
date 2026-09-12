from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import stable_softmax


@dataclass
class ModelState:
    weights: np.ndarray
    bias: np.ndarray


class SoftmaxRegressor:
    def __init__(self, n_features: int, n_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
        self.bias = np.zeros(n_classes, dtype=np.float64)

    def copy(self) -> "SoftmaxRegressor":
        model = SoftmaxRegressor(self.n_features, self.n_classes)
        model.weights = self.weights.copy()
        model.bias = self.bias.copy()
        return model

    def set_state(self, state: ModelState) -> None:
        self.weights = state.weights.copy()
        self.bias = state.bias.copy()

    def get_state(self) -> ModelState:
        return ModelState(weights=self.weights.copy(), bias=self.bias.copy())

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias
        return stable_softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def cross_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        probabilities = self.predict_proba(x)
        return float(-np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean())

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        l2: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        targets = np.eye(self.n_classes, dtype=np.float64)[y]
        n_samples = x.shape[0]
        for _ in range(epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x[batch_idx]
                yb = targets[batch_idx]
                probabilities = self.predict_proba(xb)
                gradient = (probabilities - yb) / len(batch_idx)
                grad_w = xb.T @ gradient + l2 * self.weights
                grad_b = gradient.sum(axis=0)
                self.weights -= learning_rate * grad_w
                self.bias -= learning_rate * grad_b

    def flattened_delta(self, previous_state: ModelState) -> np.ndarray:
        weight_delta = (self.weights - previous_state.weights).ravel()
        bias_delta = self.bias - previous_state.bias
        return np.concatenate([weight_delta, bias_delta], axis=0)
