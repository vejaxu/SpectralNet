import h5py
import torch
import numpy as np
import scipy.io
import pickle

from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, Subset
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root="../data", train=True, download=True, transform=tensor_transform
    )
    test_set = datasets.MNIST(
        root="../data", train=False, download=True, transform=tensor_transform
    )

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test


def load_twomoon() -> tuple:
    data, y = make_moons(n_samples=7000, shuffle=True, noise=0.075, random_state=None)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33, random_state=42
    )
    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    return x_train, y_train, x_test, y_test


def load_reuters() -> tuple:
    with h5py.File("../data/Reuters/reutersidf_total.h5", "r") as f:
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

def load_data_from_mat(key: str, filename: str, x_key: str = 'X', y_key: str = 'y'):
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
    pos = np.array([])

    if key in ["airway", "crohn", "tonsil", "tutorial", "151507_final"]:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        data = scipy.io.loadmat(filename)

    if key in ["airway", "crohn", "tonsil", "tutorial"]:
        X = data['expression_scaled']
        y = data['ground_truth'].ravel()
        pos = data['locations']
    elif key in ["151507_final"]:
        X = data['expression_scaled']
        y = data['ground_truth'].ravel()
        pos = data['locations']
    else:
        X = data[x_key]
        y = data[y_key].ravel()

    return X, y, pos


def get_kbc_mat_file(key: str, base_path: str = '../data') -> str:
    """Map dataset key to its file path."""
    if key in ["airway", "crohn", "tonsil", "tutorial"]:
        return f'{base_path}/single_cell/SingleCell_Dataset/processed_{key}.pkl'
    elif key in ["non_spherical", "non_spherical_gap", "non_spherical_gap_0_5", "non_spherical_gap_0_8"]:
        return f'{base_path}/kmeans/{key}.mat'
    elif key.startswith('w') and key.endswith('Gaussians'):
        return f'{base_path}/wGaussians/{key}.mat'
    elif key in ['151507_final']:
        return f'{base_path}/stdata/DLPFC_FINAL_PKL/{key}.pkl'
    else:
        return f'{base_path}/{key}.mat'


def get_kbc_xy_keys(key: str) -> tuple:
    """Map dataset key to its x_key / y_key for .mat files."""
    if key in ["pendigits", "YaleB", "reuters"]:
        return 'X', 'gtlabels'
    elif key in ["landsat", "waveform3", "cure-t2-4k"]:
        return 'data', 'label'
    elif key in ["COIL20"]:
        return 'X', 'Y'
    elif key in ["abalone", "drybean", "letters", "skin"]:
        return "fea", "gt"
    else:
        return 'data', 'class'


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
