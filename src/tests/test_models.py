"""Tests for SpectralNet, SiameseNet, and AE model architectures."""
import pytest
import torch
import torch.nn as nn

from spectralnet._models._spectralnet_model import SpectralNetModel
from spectralnet._models._siamesenet_model import SiameseNetModel
from spectralnet._models._ae_model import AEModel


# ---------------------------------------------------------------------------
# SpectralNetModel
# ---------------------------------------------------------------------------

class TestSpectralNetModel:
    def _make_model(self, arch=(32, 16, 4), input_dim=8):
        return SpectralNetModel(architecture=list(arch), input_dim=input_dim)

    def test_orthonorm_weights_initialized_to_none(self):
        model = self._make_model()
        assert model.orthonorm_weights is None

    def test_forward_update_orth_weights(self):
        model = self._make_model(arch=(16, 4), input_dim=8)
        x = torch.randn(32, 8)
        y = model(x, should_update_orth_weights=True)
        assert y.shape == (32, 4)
        assert model.orthonorm_weights is not None

    def test_forward_no_update_uses_existing_weights(self):
        """Calling with should_update_orth_weights=False on a fresh model
        should still work by auto-initialising weights on first call."""
        model = self._make_model(arch=(16, 4), input_dim=8)
        x = torch.randn(32, 8)
        # should not raise AttributeError
        y = model(x, should_update_orth_weights=False)
        assert y.shape == (32, 4)

    def test_output_is_orthonormal(self):
        """Columns of Y should be approximately orthonormal (Y^T Y ~ I)."""
        model = self._make_model(arch=(32, 4), input_dim=8)
        x = torch.randn(128, 8)
        y = model(x, should_update_orth_weights=True)
        gram = (y.T @ y) / y.shape[0]
        eye = torch.eye(4)
        assert torch.allclose(gram, eye, atol=1e-4), f"Gram matrix not close to I:\n{gram}"

    def test_layer_count_matches_architecture(self):
        arch = [64, 32, 16, 4]
        model = SpectralNetModel(architecture=arch, input_dim=10)
        assert len(model.layers) == len(arch)

    def test_last_layer_uses_tanh(self):
        model = self._make_model(arch=(16, 4), input_dim=8)
        last_layer = model.layers[-1]
        activations = [m for m in last_layer.modules() if isinstance(m, nn.Tanh)]
        assert len(activations) == 1

    def test_hidden_layers_use_leaky_relu(self):
        model = self._make_model(arch=(16, 8, 4), input_dim=8)
        for layer in list(model.layers)[:-1]:
            activations = [m for m in layer.modules() if isinstance(m, nn.LeakyReLU)]
            assert len(activations) == 1


# ---------------------------------------------------------------------------
# SiameseNetModel
# ---------------------------------------------------------------------------

class TestSiameseNetModel:
    def _make_model(self, arch=(32, 16, 8), input_dim=10):
        return SiameseNetModel(architecture=list(arch), input_dim=input_dim)

    def test_forward_returns_two_outputs(self):
        model = self._make_model()
        x1 = torch.randn(16, 10)
        x2 = torch.randn(16, 10)
        o1, o2 = model(x1, x2)
        assert o1.shape == (16, 8)
        assert o2.shape == (16, 8)

    def test_forward_once_output_shape(self):
        model = self._make_model(arch=(32, 8), input_dim=10)
        x = torch.randn(20, 10)
        out = model.forward_once(x)
        assert out.shape == (20, 8)

    def test_shared_weights(self):
        """forward_once with x1 and x2 separately must equal forward(x1, x2)."""
        model = self._make_model()
        x1, x2 = torch.randn(8, 10), torch.randn(8, 10)
        o1_direct = model.forward_once(x1)
        o2_direct = model.forward_once(x2)
        o1, o2 = model(x1, x2)
        assert torch.allclose(o1, o1_direct)
        assert torch.allclose(o2, o2_direct)

    def test_identical_inputs_produce_identical_outputs(self):
        model = self._make_model()
        x = torch.randn(8, 10)
        o1, o2 = model(x, x)
        assert torch.allclose(o1, o2)


# ---------------------------------------------------------------------------
# AEModel
# ---------------------------------------------------------------------------

class TestAEModel:
    def _make_model(self, arch=(32, 16, 4), input_dim=64):
        return AEModel(architecture=list(arch), input_dim=input_dim)

    def test_encode_output_shape(self):
        model = self._make_model(arch=(32, 8), input_dim=64)
        x = torch.randn(16, 64)
        z = model.encode(x)
        assert z.shape == (16, 8)

    def test_decode_output_shape(self):
        model = self._make_model(arch=(32, 8), input_dim=64)
        z = torch.randn(16, 8)
        x_hat = model.decode(z)
        assert x_hat.shape == (16, 64)

    def test_forward_roundtrip_shape(self):
        model = self._make_model(arch=(32, 16, 8), input_dim=64)
        x = torch.randn(16, 64)
        x_hat = model(x)
        assert x_hat.shape == x.shape

    def test_encoder_decoder_symmetry(self):
        """Encoder + decoder should reconstruct the input dimension."""
        arch = [128, 64, 32, 8]
        input_dim = 256
        model = AEModel(architecture=arch, input_dim=input_dim)
        x = torch.randn(4, input_dim)
        assert model(x).shape == (4, input_dim)
