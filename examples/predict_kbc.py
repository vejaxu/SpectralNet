import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR.parent))
sys.path.insert(0, str(EXAMPLES_DIR.parent / "src"))

from examples.data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    SUPPORTED_KBC_DATASETS,
    get_kbc_mat_file,
    get_kbc_xy_keys,
    load_data_from_mat,
)
from examples.kbc_utils import (  # noqa: E402
    RESULTS_ROOT,
    make_visualization_embedding,
    preprocess_features,
    save_visualization,
)
from spectralnet import Metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained SpectralNet model and predict a KBC dataset"
    )
    parser.add_argument("--key", choices=SUPPORTED_KBC_DATASETS, required=True)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--scaler_path", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    key = args.key
    result_dir = RESULTS_ROOT / key

    model_path = args.model_path or result_dir / f"{key}_model.pt"
    scaler_path = args.scaler_path or result_dir / f"{key}_scaler.pkl"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run training with --save_model first.")
        return
    if not scaler_path.exists():
        print(f"Error: Scaler not found at {scaler_path}")
        return

    print(f"Loading model from {model_path} ...")
    spectralnet = torch.load(model_path, map_location="cpu", weights_only=False)
    spectralnet.device = torch.device("cpu")

    print(f"Loading scaler from {scaler_path} ...")
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    print(f"Loading data for {key} ...")
    data_file = get_kbc_mat_file(key, base_path=args.data_root)
    x_key, y_key = get_kbc_xy_keys(key)
    raw_features, y_true, positions = load_data_from_mat(
        key, data_file, x_key, y_key
    )
    features = preprocess_features(key, raw_features, positions)

    X_normalized = scaler.transform(features)
    X_tensor = torch.from_numpy(X_normalized).float()

    print("Predicting cluster assignments ...")
    cluster_assignments = spectralnet.predict(X_tensor)

    output_path = args.output or result_dir / f"{key}_predictions.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, cluster_assignments)
    print(f"Cluster assignments saved to {output_path}")

    if args.eval and y_true is not None:
        nmi = normalized_mutual_info_score(y_true, cluster_assignments)
        ari = adjusted_rand_score(y_true, cluster_assignments)
        aligned = Metrics.align_labels(y_true, cluster_assignments)
        f1 = f1_score(y_true, aligned, average="macro")
        print("\nEvaluation Results:")
        print(f"  NMI: {nmi:.4f}")
        print(f"  ARI: {ari:.4f}")
        print(f"  F1 : {f1:.4f}")

    if y_true is not None:
        print("\nPreparing visualization ...")
        embedding = make_visualization_embedding(key, features, positions)
        aligned_labels = Metrics.align_labels(y_true, cluster_assignments)
        save_visualization(embedding, y_true, aligned_labels, key, result_dir)

    print("\nPrediction completed.")


if __name__ == "__main__":
    main()
