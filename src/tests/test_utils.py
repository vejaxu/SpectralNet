"""Tests for spectralnet utility functions."""
import pytest
import numpy as np
import torch

from spectralnet._utils import (
    get_laplacian,
    get_nearest_neighbors,
    get_affinity_matrix,
    get_gaussian_kernel,
    get_t_kernel,
    compute_scale,
    get_grassman_distance,
    calculate_cost_matrix,
    get_cluster_labels_from_indices,
)


class TestGetLaplacian:
    def test_output_shape(self):
        W = torch.rand(10, 10)
        W = (W + W.T) / 2
        L = get_laplacian(W)
        assert L.shape == (10, 10)

    def test_row_sums_zero(self):
        """Laplacian rows must sum to zero."""
        W = torch.rand(8, 8)
        W = (W + W.T) / 2
        L = get_laplacian(W)
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-6)

    def test_symmetric(self):
        W = torch.rand(8, 8)
        W = (W + W.T) / 2
        L = get_laplacian(W)
        assert np.allclose(L, L.T, atol=1e-6)

    def test_positive_semidefinite(self):
        """All eigenvalues should be >= 0."""
        W = torch.rand(8, 8).abs()
        W = (W + W.T) / 2
        L = get_laplacian(W)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-6)


class TestGetNearestNeighbors:
    def test_output_shapes(self):
        X = torch.randn(20, 4)
        Dis, Ids = get_nearest_neighbors(X, k=3)
        assert Dis.shape == (20, 3)
        assert Ids.shape == (20, 3)

    def test_distances_non_negative(self):
        X = torch.randn(20, 4)
        Dis, _ = get_nearest_neighbors(X, k=3)
        assert np.all(Dis >= 0)

    def test_self_query_with_Y(self):
        """When Y is provided, neighbors should be found from X for each query in Y."""
        X = torch.randn(20, 4)
        Y = torch.randn(5, 4)
        Dis, Ids = get_nearest_neighbors(X, Y=Y, k=3)
        assert Dis.shape == (5, 3)
        assert Ids.shape == (5, 3)

    def test_k_clamped_to_dataset_size(self):
        """When k > len(X), it should clamp to len(X)."""
        X = torch.randn(5, 4)
        Dis, Ids = get_nearest_neighbors(X, k=20)
        assert Ids.shape[0] == 5


class TestComputeScale:
    def test_local_scale_shape(self):
        Dis = np.random.rand(20, 5)
        scale = compute_scale(Dis, is_local=True)
        assert scale.shape == (20,)

    def test_global_scale_scalar(self):
        Dis = np.random.rand(20, 5)
        scale = compute_scale(Dis, k=3, is_local=False)
        assert np.isscalar(scale) or scale.shape == ()

    def test_local_scale_median(self):
        Dis = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scale = compute_scale(Dis, is_local=True, med=True)
        expected = np.median(Dis, axis=1)
        np.testing.assert_allclose(scale, expected)

    def test_local_scale_max(self):
        Dis = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scale = compute_scale(Dis, is_local=True, med=False)
        expected = np.max(Dis, axis=1)
        np.testing.assert_allclose(scale, expected)


class TestGetGaussianKernel:
    def _make_data(self, n=16, d=4, k=3):
        X = torch.randn(n, d)
        Dis, Ids = get_nearest_neighbors(X, k=k)
        Dx = torch.cdist(X, X)
        scale = compute_scale(Dis, is_local=True)
        device = torch.device("cpu")
        return Dx, scale, Ids, device

    def test_output_shape(self):
        Dx, scale, Ids, device = self._make_data()
        W = get_gaussian_kernel(Dx, scale, Ids, device, is_local=True)
        assert W.shape == (16, 16)

    def test_symmetric(self):
        Dx, scale, Ids, device = self._make_data()
        W = get_gaussian_kernel(Dx, scale, Ids, device, is_local=True)
        assert torch.allclose(W, W.T, atol=1e-6)

    def test_values_in_01(self):
        Dx, scale, Ids, device = self._make_data()
        W = get_gaussian_kernel(Dx, scale, Ids, device, is_local=True)
        assert W.min().item() >= 0.0
        assert W.max().item() <= 1.0 + 1e-6

    def test_diagonal_is_one(self):
        """Self-similarity should be 1 (zero distance, exp(0)=1)."""
        Dx, _local_scale, Ids, device = self._make_data()
        global_scale = 1.0  # scalar, as expected by is_local=False
        W = get_gaussian_kernel(Dx, global_scale, None, device, is_local=False)
        diag = torch.diag(W)
        assert torch.allclose(diag, torch.ones(16, dtype=diag.dtype), atol=1e-5)


class TestGetTKernel:
    def _make_data(self, n=16, d=4, k=3):
        X = torch.randn(n, d)
        Dis, Ids = get_nearest_neighbors(X, k=k)
        Dx = torch.cdist(X, X)
        device = torch.device("cpu")
        return Dx, Ids, device

    def test_output_shape(self):
        Dx, Ids, device = self._make_data()
        W = get_t_kernel(Dx, Ids, device)
        assert W.shape == (16, 16)

    def test_symmetric(self):
        Dx, Ids, device = self._make_data()
        W = get_t_kernel(Dx, Ids, device)
        assert torch.allclose(W, W.T, atol=1e-6)

    def test_values_in_01(self):
        Dx, Ids, device = self._make_data()
        W = get_t_kernel(Dx, Ids, device)
        assert W.min().item() >= 0.0
        assert W.max().item() <= 1.0 + 1e-6


class TestGetAffinityMatrix:
    def test_output_shape(self):
        X = torch.randn(20, 4)
        W = get_affinity_matrix(X, n_neighbors=5, device=torch.device("cpu"))
        assert W.shape == (20, 20)

    def test_symmetric(self):
        X = torch.randn(20, 4)
        W = get_affinity_matrix(X, n_neighbors=5, device=torch.device("cpu"))
        assert torch.allclose(W, W.T, atol=1e-6)

    def test_non_negative(self):
        X = torch.randn(20, 4)
        W = get_affinity_matrix(X, n_neighbors=5, device=torch.device("cpu"))
        assert W.min().item() >= 0.0


class TestGetGrassmanDistance:
    def test_same_subspace_zero_distance(self):
        A = np.linalg.qr(np.random.randn(8, 3))[0]
        dist = get_grassman_distance(A, A)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_subspaces(self):
        """Orthogonal subspaces should have maximum Grassmann distance."""
        A = np.eye(6)[:, :3]
        B = np.eye(6)[:, 3:]
        dist = get_grassman_distance(A, B)
        # All singular values of A^T B are 0, so distance = sum of (1-0^2) = 3
        assert dist == pytest.approx(3.0, abs=1e-5)

    def test_non_negative(self):
        A = np.linalg.qr(np.random.randn(8, 3))[0]
        B = np.linalg.qr(np.random.randn(8, 3))[0]
        dist = get_grassman_distance(A, B)
        assert dist >= 0.0


class TestCalculateCostMatrix:
    def test_output_shape(self):
        C = np.random.randint(0, 100, (5, 5))
        cost = calculate_cost_matrix(C, n_clusters=5)
        assert cost.shape == (5, 5)

    def test_cost_non_negative(self):
        C = np.random.randint(0, 100, (4, 4))
        cost = calculate_cost_matrix(C, n_clusters=4)
        assert np.all(cost >= 0)

    def test_perfect_confusion_matrix_zero_diagonal_cost(self):
        """A diagonal confusion matrix should yield zero cost on the diagonal."""
        n = 4
        C = np.diag([100] * n)
        cost = calculate_cost_matrix(C, n_clusters=n)
        assert np.all(np.diag(cost) == 0)


class TestGetClusterLabelsFromIndices:
    def test_basic(self):
        indices = [(0, 2), (1, 0), (2, 3), (3, 1)]
        labels = get_cluster_labels_from_indices(indices)
        np.testing.assert_array_equal(labels, [2, 0, 3, 1])

    def test_length(self):
        indices = [(i, i) for i in range(6)]
        labels = get_cluster_labels_from_indices(indices)
        assert len(labels) == 6
