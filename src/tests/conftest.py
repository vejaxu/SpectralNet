from spectralnet._utils import set_random_seed


def pytest_configure(config):
    """Set global seeds for reproducibility across all tests."""
    set_random_seed(42)
