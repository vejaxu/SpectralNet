"""Regression tests for the curated KBC example dataset registry."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

from examples.data import (
    DEFAULT_DATA_ROOT,
    KBC_DATASETS,
    SUPPORTED_KBC_DATASETS,
    get_kbc_mat_file,
    get_kbc_xy_keys,
)
from examples.kbc_utils import RESULTS_ROOT, preprocess_features


EXPECTED_DATASETS = (
    "spiral",
    "AC",
    "4C",
    "RingG",
    "complex9",
    "USPS",
    "STL-10",
    "Cifar-10",
    "ImageNet-10",
    "ImageNet-Dogs",
    "MNIST",
    "COIL20",
    "w1Gaussians",
    "w100Gaussians",
    "sparse_3_dense_3_dense_3",
    "sparse_8_dense_1_dense_1",
    "one_gaussian_10_one_line_5_2",
    "tutorial",
    "tonsil",
    "airway",
    "crohn",
    "dlpfc_151507",
    "non_spherical",
    "non_spherical_gap",
    "non_spherical_gap_0_5",
    "non_spherical_gap_0_8",
)


def test_dataset_registry_matches_curated_list():
    assert SUPPORTED_KBC_DATASETS == EXPECTED_DATASETS
    assert tuple(KBC_DATASETS) == EXPECTED_DATASETS


@pytest.mark.parametrize("key", EXPECTED_DATASETS)
def test_dataset_paths_and_keys_are_resolvable(key, tmp_path):
    relative_path, expected_x_key, expected_y_key = KBC_DATASETS[key]

    assert get_kbc_mat_file(key, tmp_path) == str(
        (tmp_path / relative_path).resolve()
    )
    assert get_kbc_xy_keys(key) == (expected_x_key, expected_y_key)


def test_default_paths_do_not_depend_on_working_directory():
    project_root = Path(__file__).resolve().parents[2]
    assert DEFAULT_DATA_ROOT == project_root.parent / "data"
    assert RESULTS_ROOT == project_root / "examples" / "results"


def test_dlpfc_preprocessing_preserves_original_feature_propagation():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(60, 55))
    positions = rng.normal(size=(60, 2))

    reduced = PCA(n_components=50, random_state=42).fit_transform(features)
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    neighbors = np.argsort(distances, axis=1)[:, 1:7]
    adjacency = np.zeros((len(positions), len(positions)))
    for row, columns in enumerate(neighbors):
        adjacency[row, columns] = 1

    expected_steps = [reduced]
    for _ in range(7):
        normalized = adjacency + np.eye(len(adjacency))
        normalized /= normalized.sum(axis=1, keepdims=True)
        expected_steps.append(normalized @ expected_steps[-1])
    expected = np.concatenate(expected_steps, axis=1)

    actual = preprocess_features("dlpfc_151507", features, positions)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("resolver", (get_kbc_mat_file, get_kbc_xy_keys))
def test_unknown_dataset_is_rejected(resolver):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolver("removed-dataset")
