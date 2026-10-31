"""One-hidden-layer ReLU classifier with the experiment model-state interface."""
from __future__ import annotations

import numpy as np

from .model import ModelState
from .utils import stable_softmax


class MLPClassifier:
    def __init__(self, n_features: int, n_classes: int, seed: int = 42,
                 hidden_units: int = 32) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.first_weight_size = n_features * hidden_units
        rng = np.random.default_rng(seed)
        first = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden_units))
        second = rng.normal(0, np.sqrt(1.0 / hidden_units), (hidden_units, n_classes))
        self.weights = np.concatenate((first.ravel(), second.ravel()))
        self.bias = np.zeros(hidden_units + n_classes, dtype=np.float64)

    def _layers(self):
        w1 = self.weights[:self.first_weight_size].reshape(self.n_features, self.hidden_units)
        w2 = self.weights[self.first_weight_size:].reshape(self.hidden_units, self.n_classes)
        return w1, w2, self.bias[:self.hidden_units], self.bias[self.hidden_units:]

    def copy(self):
        model = MLPClassifier(self.n_features, self.n_classes, hidden_units=self.hidden_units)
        model.set_state(self.get_state())
        return model

    def get_state(self):
        return ModelState(weights=self.weights.copy(), bias=self.bias.copy())

    def set_state(self, state):
        if state.weights.shape != self.weights.shape or state.bias.shape != self.bias.shape:
            raise ValueError("MLP state dimensions do not match the architecture")
        self.weights = state.weights.copy()
        self.bias = state.bias.copy()

    def predict_proba(self, x):
        w1, w2, b1, b2 = self._layers()
        hidden = np.maximum(x @ w1 + b1, 0.0)
        return stable_softmax(hidden @ w2 + b2)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def cross_entropy(self, x, y):
        probabilities = self.predict_proba(x)
        return float(-np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean())

    def _gradients(self, x, targets, l2):
        w1, w2, b1, b2 = self._layers()
        preactivation = x @ w1 + b1
        hidden = np.maximum(preactivation, 0.0)
        error = (stable_softmax(hidden @ w2 + b2) - targets) / len(x)
        gw2 = hidden.T @ error + l2 * w2
        gb2 = error.sum(axis=0)
        hidden_error = (error @ w2.T) * (preactivation > 0)
        gw1 = x.T @ hidden_error + l2 * w1
        gb1 = hidden_error.sum(axis=0)
        return gw1, gw2, gb1, gb2

    def gradients(self, x, y, l2=0.0):
        gw1, gw2, gb1, gb2 = self._gradients(x, np.eye(self.n_classes)[y], l2)
        return np.concatenate((gw1.ravel(), gw2.ravel())), np.concatenate((gb1, gb2))

    def fit(self, x, y, epochs, learning_rate, batch_size, l2, seed):
        rng = np.random.default_rng(seed)
        targets = np.eye(self.n_classes, dtype=np.float64)[y]
        for _ in range(epochs):
            indices = np.arange(len(x))
            rng.shuffle(indices)
            for start in range(0, len(x), batch_size):
                selected = indices[start:start + batch_size]
                gw1, gw2, gb1, gb2 = self._gradients(x[selected], targets[selected], l2)
                self.weights[:self.first_weight_size] -= learning_rate * gw1.ravel()
                self.weights[self.first_weight_size:] -= learning_rate * gw2.ravel()
                self.bias[:self.hidden_units] -= learning_rate * gb1
                self.bias[self.hidden_units:] -= learning_rate * gb2
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise FloatingPointError("Non-finite MLP parameters")

    def flattened_delta(self, previous_state):
        return np.concatenate((self.weights - previous_state.weights,
                               self.bias - previous_state.bias))
