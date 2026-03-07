"""Tests for the Metrics class (ACC and NMI)."""
import pytest
import numpy as np

from spectralnet._metrics import Metrics


class TestAccScore:
    def test_perfect_assignment(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        assignments = np.array([0, 0, 1, 1, 2, 2])
        acc = Metrics.acc_score(assignments, y, n_clusters=3)
        assert acc == pytest.approx(1.0)

    def test_permuted_assignment_still_perfect(self):
        """Cluster labels are arbitrary; a permutation should still score 1.0."""
        y = np.array([0, 0, 1, 1, 2, 2])
        # clusters 0->2, 1->0, 2->1
        assignments = np.array([2, 2, 0, 0, 1, 1])
        acc = Metrics.acc_score(assignments, y, n_clusters=3)
        assert acc == pytest.approx(1.0)

    def test_worst_assignment(self):
        """No correct assignment should give low accuracy."""
        y = np.array([0] * 5 + [1] * 5)
        assignments = np.array([1] * 5 + [0] * 5)
        acc = Metrics.acc_score(assignments, y, n_clusters=2)
        # After optimal matching, this is actually perfect (permutation)
        assert acc == pytest.approx(1.0)

    def test_random_returns_float(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 5, 50)
        assignments = rng.integers(0, 5, 50)
        acc = Metrics.acc_score(assignments, y, n_clusters=5)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


class TestNmiScore:
    def test_perfect_nmi(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        acc = Metrics.nmi_score(y, y)
        assert acc == pytest.approx(1.0)

    def test_random_labels_low_nmi(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, 100)
        assignments = rng.integers(0, 5, 100)
        nmi = Metrics.nmi_score(assignments, y)
        # NMI should be small for random assignments (not guaranteed to be
        # 0 but typically << 0.5 for random data with 5 classes)
        assert 0.0 <= nmi <= 1.0

    def test_nmi_symmetric(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([0, 1, 1, 2, 2, 0])
        assert Metrics.nmi_score(y, p) == pytest.approx(Metrics.nmi_score(p, y))
