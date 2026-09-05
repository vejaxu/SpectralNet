import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms


EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "data"

# Canonical dataset names and their locations under ../data. Keeping this in
# one registry prevents training and prediction from drifting apart.
KBC_DATASETS = {
    "spiral": ("spiral.mat", "data", "class"),
    "AC": ("AC.mat", "data", "class"),
    "4C": ("4C.mat", "data", "class"),
    "RingG": ("RingG.mat", "data", "class"),
    "complex9": ("complex9.mat", "data", "class"),
    "USPS": ("USPS.mat", "data", "class"),
    "STL-10": ("STL-10.mat", "data", "class"),
    "Cifar-10": ("Cifar-10.mat", "data", "class"),
    "ImageNet-10": ("ImageNet-10.mat", "data", "class"),
    "ImageNet-Dogs": ("ImageNet-Dogs.mat", "data", "class"),
    "MNIST": ("mnist.mat", "data", "class"),
    "COIL20": ("COIL20.mat", "X", "Y"),
    "w1Gaussians": ("wGaussians/w1Gaussians.mat", "data", "class"),
    "w100Gaussians": ("wGaussians/w100Gaussians.mat", "data", "class"),
    "sparse_3_dense_3_dense_3": (
        "sparse_3_dense_3_dense_3.mat",
        "data",
        "class",
    ),
    "sparse_8_dense_1_dense_1": (
        "sparse_8_dense_1_dense_1.mat",
        "data",
        "class",
    ),
    "one_gaussian_10_one_line_5_2": (
        "one_gaussian_10_one_line_5_2.mat",
        "data",
        "class",
    ),
    "tutorial": (
        "single_cell/SingleCell_Dataset/processed_tutorial.pkl",
        "expression_scaled",
        "ground_truth",
    ),
    "tonsil": (
        "single_cell/SingleCell_Dataset/processed_tonsil.pkl",
        "expression_scaled",
        "ground_truth",
    ),
    "airway": (
        "single_cell/SingleCell_Dataset/processed_airway.pkl",
        "expression_scaled",
        "ground_truth",
    ),
    "crohn": (
        "single_cell/SingleCell_Dataset/processed_crohn.pkl",
        "expression_scaled",
        "ground_truth",
    ),
    "dlpfc_151507": (
        "stdata/DLPFC_FINAL_PKL/151507_final.pkl",
        "expression_scaled",
        "ground_truth",
    ),
    "non_spherical": ("kmeans/non_spherical.mat", "data", "class"),
    "non_spherical_gap": ("kmeans/non_spherical_gap.mat", "data", "class"),
    "non_spherical_gap_0_5": (
        "kmeans/non_spherical_gap_0_5.mat",
        "data",
        "class",
    ),
    "non_spherical_gap_0_8": (
        "kmeans/non_spherical_gap_0_8.mat",
        "data",
        "class",
    ),
}
SUPPORTED_KBC_DATASETS = tuple(KBC_DATASETS)
SPATIAL_DATASETS = frozenset(
    {"tutorial", "tonsil", "airway", "crohn", "dlpfc_151507"}
)


def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root=DEFAULT_DATA_ROOT / "MNIST",
        train=True,
        download=True,
        transform=tensor_transform,
    )
    test_set = datasets.MNIST(
        root=DEFAULT_DATA_ROOT / "MNIST",
        train=False,
        download=True,
        transform=tensor_transform,
    )

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test


def load_twomoon() -> tuple:
    data, y = make_moons(n_samples=7000, shuffle=True, noise=0.075, random_state=42)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33, random_state=42
    )
    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    return x_train, y_train, x_test, y_test


def load_reuters() -> tuple:
    with h5py.File(DEFAULT_DATA_ROOT / "Reuters/reutersidf_total.h5", "r") as f:
        x = np.asarray(f.get("data"), dtype="float32")
        y = np.asarray(f.get("labels"), dtype="float32")

        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def load_from_path(dpath: str, lpath: str = None) -> tuple:
    X = np.loadtxt(dpath, delimiter=",", dtype=np.float32)
    n_train = int(0.9 * len(X))

    x_train, x_test = X[:n_train], X[n_train:]
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)

    if lpath is not None:
        y = np.loadtxt(lpath, delimiter=",", dtype=np.float32)
        y_train, y_test = y[:n_train], y[n_train:]
        y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    else:
        y_train, y_test = None, None

    return x_train, y_train, x_test, y_test


# ============================================================
# KBC dataset loading utilities
# ============================================================

def load_data_from_mat(
    key: str,
    filename: str | Path,
    x_key: str = "X",
    y_key: str = "y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load KBC-format dataset from .mat or .pkl file.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Ground-truth labels (ravelled).
    pos : np.ndarray
        Spatial coordinates (empty if unavailable).
    """
    if key not in KBC_DATASETS:
        raise ValueError(f"Unsupported dataset {key!r}")

    filename = Path(filename)
    if filename.suffix == ".pkl":
        with filename.open("rb") as f:
            data = pickle.load(f)
    else:
        data = scipy.io.loadmat(filename)

    positions = data.get("locations", np.array([]))
    return data[x_key], data[y_key].ravel(), positions


def get_kbc_mat_file(
    key: str, base_path: str | Path = DEFAULT_DATA_ROOT
) -> str:
    """Return the source file for a supported KBC dataset."""
    try:
        relative_path = KBC_DATASETS[key][0]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_KBC_DATASETS)
        raise ValueError(
            f"Unsupported dataset {key!r}. Choose one of: {supported}"
        ) from exc
    return str(Path(base_path).expanduser().resolve() / relative_path)


def get_kbc_xy_keys(key: str) -> tuple:
    """Return feature and label keys for a supported KBC dataset."""
    try:
        _, x_key, y_key = KBC_DATASETS[key]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_KBC_DATASETS)
        raise ValueError(
            f"Unsupported dataset {key!r}. Choose one of: {supported}"
        ) from exc
    return x_key, y_key


# ============================================================
# Legacy dispatcher
# ============================================================

def load_data(dataset: str) -> tuple:
    """
    This function loads the dataset specified in the config file.


    Args:
        dataset (str or dictionary):    In case you want to load your own dataset,
                                        you should specify the path to the data (and label if applicable)
                                        files in the config file in a dictionary fashion under the key "dataset".

    Raises:
        ValueError: If the dataset is not found in the config file.

    Returns:
        tuple: A tuple containing the train and test data and labels.
    """

    if dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == "twomoons":
        x_train, y_train, x_test, y_test = load_twomoon()
    elif dataset == "reuters":
        x_train, y_train, x_test, y_test = load_reuters()
    else:
        try:
            data_path = dataset["dpath"]
            if "lpath" in dataset:
                label_path = dataset["lpath"]
            else:
                label_path = None
        except:
            raise ValueError("Could not find dataset path. Check your config file.")
        x_train, y_train, x_test, y_test = load_from_path(data_path, label_path)

    return x_train, x_test, y_train, y_test
