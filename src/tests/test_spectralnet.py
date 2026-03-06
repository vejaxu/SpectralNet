"""Integration tests for the SpectralNet clustering pipeline."""
import pytest
import numpy as np
import torch
from sklearn.datasets import make_blobs

from spectralnet import SpectralNet


def _make_blobs_tensor(n_samples=200, n_features=4, centers=3, seed=0):
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers, random_state=seed
    )
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TestSpectralNetInit:
    def test_default_construction(self):
        # Default spectral_hiddens ends with 10, so n_clusters must match
        model = SpectralNet(n_clusters=10)
        assert model.n_clusters == 10

    def test_hidden_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            SpectralNet(n_clusters=5, spectral_hiddens=[64, 32, 4])  # last != 5

    def test_hidden_size_match_ok(self):
        SpectralNet(n_clusters=4, spectral_hiddens=[64, 4])

    def test_device_set(self):
        model = SpectralNet(n_clusters=10)
        assert isinstance(model.device, torch.device)


class TestSpectralNetFitPredict:
    """These tests use small networks / few epochs so they run quickly."""

    @pytest.fixture
    def small_model(self):
        return SpectralNet(
            n_clusters=3,
            spectral_hiddens=[32, 3],
            spectral_epochs=3,
            spectral_batch_size=64,
            spectral_n_nbg=5,
            spectral_scale_k=5,
        )

    def test_fit_runs_without_error(self, small_model):
        X, y = _make_blobs_tensor(n_samples=150, n_features=4, centers=3)
        small_model.fit(X, y)

    def test_predict_returns_correct_shape(self, small_model):
        X, y = _make_blobs_tensor(n_samples=150, n_features=4, centers=3)
        small_model.fit(X)
        assignments = small_model.predict(X)
        assert assignments.shape == (150,)

    def test_predict_integer_labels(self, small_model):
        X, _ = _make_blobs_tensor(n_samples=150, n_features=4, centers=3)
        small_model.fit(X)
        assignments = small_model.predict(X)
        assert assignments.dtype in (np.int32, np.int64, int)

    def test_predict_label_range(self, small_model):
        X, _ = _make_blobs_tensor(n_samples=150, n_features=4, centers=3)
        small_model.fit(X)
        assignments = small_model.predict(X)
        assert set(np.unique(assignments)).issubset(set(range(3)))

    def test_embeddings_stored_after_predict(self, small_model):
        X, _ = _make_blobs_tensor(n_samples=150, n_features=4, centers=3)
        small_model.fit(X)
        small_model.predict(X)
        assert hasattr(small_model, "embeddings_")
        assert small_model.embeddings_.shape == (150, 3)

    def test_fit_without_labels(self, small_model):
        X, _ = _make_blobs_tensor(n_samples=150, n_features=4, centers=3)
        small_model.fit(X)  # no y
        assignments = small_model.predict(X)
        assert assignments.shape == (150,)


class TestSpectralNetGetRandomBatch:
    def test_batch_size_respected(self):
        model = SpectralNet(
            n_clusters=3,
            spectral_hiddens=[16, 3],
            spectral_epochs=1,
            spectral_batch_size=32,
        )
        X, y = _make_blobs_tensor(n_samples=100, n_features=4, centers=3)
        model.fit(X, y)
        X_raw, X_enc = model.get_random_batch(batch_size=20)
        assert X_raw.shape[0] == 20
        assert X_enc.shape[0] == 20

    def test_batch_size_clamped_to_dataset(self):
        """batch_size larger than dataset should return the whole dataset."""
        model = SpectralNet(
            n_clusters=3,
            spectral_hiddens=[16, 3],
            spectral_epochs=1,
            spectral_batch_size=32,
        )
        X, y = _make_blobs_tensor(n_samples=50, n_features=4, centers=3)
        model.fit(X, y)
        X_raw, X_enc = model.get_random_batch(batch_size=9999)
        assert X_raw.shape[0] == 50
