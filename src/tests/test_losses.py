"""Tests for SpectralNet and Siamese loss functions."""
import pytest
import torch

from spectralnet._losses._spectralnet_loss import SpectralNetLoss
from spectralnet._losses._siamese_loss import ContrastiveLoss


class TestSpectralNetLoss:
    def setup_method(self):
        self.criterion = SpectralNetLoss()

    def test_loss_is_scalar(self):
        W = torch.rand(16, 16)
        W = (W + W.T) / 2  # symmetric
        Y = torch.randn(16, 4)
        loss = self.criterion(W, Y)
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        W = torch.rand(16, 16)
        W = (W + W.T) / 2
        W = W.abs()
        Y = torch.randn(16, 4)
        loss = self.criterion(W, Y)
        assert loss.item() >= 0.0

    def test_zero_affinity_gives_zero_loss(self):
        W = torch.zeros(16, 16)
        Y = torch.randn(16, 4)
        loss = self.criterion(W, Y)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_identical_embeddings_give_low_loss(self):
        """If all embeddings are the same point, pairwise distances are 0,
        so the loss should be 0 regardless of W."""
        n, d = 16, 4
        W = torch.rand(n, n)
        W = (W + W.T) / 2
        Y = torch.ones(n, d)
        loss = self.criterion(W, Y)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_decreases_with_better_clustering(self):
        """Embeddings perfectly aligned with clusters should have lower loss
        than random embeddings for a block-diagonal affinity."""
        n = 32
        # Block-diagonal W: two equal clusters
        W = torch.zeros(n, n)
        W[:16, :16] = 1.0
        W[16:, 16:] = 1.0

        # Perfect 2-cluster embeddings (orthogonal)
        Y_good = torch.zeros(n, 2)
        Y_good[:16, 0] = 1.0
        Y_good[16:, 1] = 1.0

        Y_bad = torch.randn(n, 2)

        loss_good = self.criterion(W, Y_good)
        loss_bad = self.criterion(W, Y_bad)
        assert loss_good.item() < loss_bad.item()


class TestContrastiveLoss:
    def setup_method(self):
        self.criterion = ContrastiveLoss(margin=1.0)

    def test_loss_is_scalar(self):
        o1 = torch.randn(16, 8)
        o2 = torch.randn(16, 8)
        labels = torch.randint(0, 2, (16,)).float()
        loss = self.criterion(o1, o2, labels)
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        o1 = torch.randn(32, 8)
        o2 = torch.randn(32, 8)
        labels = torch.randint(0, 2, (32,)).float()
        loss = self.criterion(o1, o2, labels)
        assert loss.item() >= 0.0

    def test_identical_positive_pairs_zero_loss(self):
        """Same vectors as positive pairs should give 0 loss."""
        x = torch.randn(16, 8)
        labels = torch.ones(16)
        loss = self.criterion(x, x, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_far_negative_pairs_zero_loss(self):
        """Very far-apart vectors as negative pairs should give 0 loss
        (clamped margin term)."""
        o1 = torch.zeros(16, 8)
        o2 = torch.ones(16, 8) * 100  # far away
        labels = torch.zeros(16)
        loss = self.criterion(o1, o2, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_custom_margin(self):
        crit = ContrastiveLoss(margin=2.0)
        o1 = torch.zeros(8, 4)
        # distance = 1.0 (< margin=2), so negative loss should be (2-1)^2 = 1
        o2 = torch.ones(8, 4)
        labels = torch.zeros(8)
        loss = crit(o1, o2, labels)
        assert loss.item() > 0.0
