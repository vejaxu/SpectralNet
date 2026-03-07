import torch
import random
import numpy as np


def pytest_configure(config):
    """Set global seeds for reproducibility across all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
