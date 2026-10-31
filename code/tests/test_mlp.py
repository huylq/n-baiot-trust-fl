"""Numerical and state checks for the MLP classifier."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from src.federated import ExperimentRunner
from src.mlp import MLPClassifier
from src.model import ModelState


class MLPTests(unittest.TestCase):
    def test_backpropagation_matches_finite_differences(self):
        model = MLPClassifier(3, 2, seed=13, hidden_units=4)
        x = np.random.default_rng(29).normal(size=(9, 3))
        y = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1])
        regularization = 0.07
        analytic = np.concatenate(model.gradients(x, y, regularization))
        numeric = []
        def loss():
            return model.cross_entropy(x, y) + regularization * np.sum(model.weights ** 2) / 2
        for array in (model.weights, model.bias):
            for i in range(array.size):
                before = array[i]
                array[i] = before + 1e-6
                plus = loss()
                array[i] = before - 1e-6
                minus = loss()
                array[i] = before
                numeric.append((plus - minus) / 2e-6)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)

    def test_all_parameters_roundtrip_and_copy_isolation(self):
        model = MLPClassifier(115, 2, seed=0)
        self.assertEqual(model.weights.size + model.bias.size, 3778)
        state = model.get_state()
        delta = np.linspace(-0.03, 0.03, 3778)
        ExperimentRunner._apply_flattened_delta(model, delta)
        np.testing.assert_allclose(model.flattened_delta(state), delta, atol=1e-16)
        duplicate = model.copy()
        duplicate.weights[0] += 1
        self.assertNotEqual(duplicate.weights[0], model.weights[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(path, weights=model.weights, bias=model.bias)
            loaded = MLPClassifier(115, 2, seed=19)
            with np.load(path) as saved:
                loaded.set_state(ModelState(saved["weights"], saved["bias"]))
            x = np.random.default_rng(17).normal(size=(8, 115))
            np.testing.assert_array_equal(model.predict_proba(x), loaded.predict_proba(x))

    def test_learns_nonlinear_xor(self):
        x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        y = np.array([0, 1, 1, 0])
        model = MLPClassifier(2, 2, seed=3, hidden_units=8)
        before = model.cross_entropy(x, y)
        model.fit(x, y, epochs=300, learning_rate=0.05, batch_size=4, l2=0, seed=0)
        np.testing.assert_array_equal(model.predict(x), y)
        self.assertLess(model.cross_entropy(x, y), before / 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
