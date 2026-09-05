import os
import sys

# Limit thread counts to prevent parallel conflicts
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import MinMaxScaler

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
from spectralnet import Metrics, SpectralNet  # noqa: E402
from spectralnet._utils import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    set_random_seed,
)


def run_spectralnet(
    X_normalized: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    seed: int,
    result_dir: Path,
    key: str,
    args: argparse.Namespace,
) -> dict:
    """Run SpectralNet clustering and return metrics."""
    set_random_seed(seed)

    X_tensor = torch.from_numpy(X_normalized).float()
    y_tensor = torch.from_numpy(y_true).long() if y_true is not None else None

    spectral_hiddens = args.spectral_hiddens or [512, 512, n_clusters]

    spectralnet = SpectralNet(
        n_clusters=n_clusters,
        should_use_ae=args.use_ae,
        should_use_siamese=args.use_siamese,
        ae_hiddens=args.ae_hiddens,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        siamese_hiddens=args.siamese_hiddens,
        siamese_epochs=args.siamese_epochs,
        siamese_batch_size=args.siamese_batch_size,
        siamese_n_nbg=args.siamese_n_nbg,
        spectral_hiddens=spectral_hiddens,
        spectral_epochs=args.spectral_epochs,
        spectral_lr=args.spectral_lr,
        spectral_batch_size=args.spectral_batch_size,
        spectral_n_nbg=args.spectral_n_nbg,
        spectral_scale_k=args.spectral_scale_k,
        spectral_is_local_scale=args.spectral_is_local_scale,
        weights_dir=str(result_dir),
        random_state=seed,
    )

    start_time = time.time()
    spectralnet.fit(X_tensor, y_tensor)
    cluster_assignments = spectralnet.predict(X_tensor)
    elapsed = time.time() - start_time

    nmi = normalized_mutual_info_score(y_true, cluster_assignments)
    ari = adjusted_rand_score(y_true, cluster_assignments)
    aligned = Metrics.align_labels(y_true, cluster_assignments)
    f1 = f1_score(y_true, aligned, average="macro")

    # Save model if requested
    if args.save_model:
        model_path = result_dir / f"{key}_model.pt"
        torch.save(spectralnet, model_path)
        print(f"  Model saved: {model_path}")

    return {
        "seed": seed,
        "nmi": nmi,
        "ari": ari,
        "f1": f1,
        "time_sec": elapsed,
        "cluster_assignments": cluster_assignments,
        "embeddings": spectralnet.embeddings_,
    }


def process_dataset(key: str, args: argparse.Namespace) -> dict:
    """Main pipeline for a single KBC dataset."""
    print(f"\nProcessing {key} ...")
    result_dir = RESULTS_ROOT / key
    result_dir.mkdir(parents=True, exist_ok=True)

    data_file = get_kbc_mat_file(key, base_path=args.data_root)
    x_key, y_key = get_kbc_xy_keys(key)
    raw_features, y_true, positions = load_data_from_mat(
        key, data_file, x_key, y_key
    )
    features = preprocess_features(key, raw_features, positions)

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(features)
    n_clusters = np.unique(y_true).size

    scaler_path = result_dir / f"{key}_scaler.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")

    seed = DEFAULT_RANDOM_SEED
    print(f"  Running with seed={seed} ...")
    result = run_spectralnet(
        X_normalized, y_true, n_clusters, seed, result_dir, key, args
    )

    result_csv_path = result_dir / f"{key}_result.csv"
    with result_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "nmi", "ari", "f1", "time_sec"])
        writer.writerow(
            [
                seed,
                f"{result['nmi']:.4f}",
                f"{result['ari']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['time_sec']:.4f}",
            ]
        )
    print(f"Results saved: {result_csv_path}")

    print(f"\nVisualizing (seed={result['seed']}) ...")
    embedding = make_visualization_embedding(key, features, positions)
    aligned_labels = Metrics.align_labels(y_true, result["cluster_assignments"])
    save_visualization(embedding, y_true, aligned_labels, key, result_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpectralNet on KBC datasets")
    parser.add_argument("--key", choices=SUPPORTED_KBC_DATASETS, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Data root directory (default: {DEFAULT_DATA_ROOT})",
    )

    # SpectralNet architecture
    parser.add_argument("--use_ae", action="store_true", help="Use autoencoder")
    parser.add_argument("--use_siamese", action="store_true", help="Use Siamese net")
    parser.add_argument("--ae_hiddens", type=int, nargs="+", default=[512, 256, 64])
    parser.add_argument("--ae_epochs", type=int, default=30)
    parser.add_argument("--ae_batch_size", type=int, default=256)
    parser.add_argument(
        "--siamese_hiddens", type=int, nargs="+", default=[512, 512, 64]
    )
    parser.add_argument("--siamese_epochs", type=int, default=20)
    parser.add_argument("--siamese_batch_size", type=int, default=256)
    parser.add_argument("--siamese_n_nbg", type=int, default=5)
    parser.add_argument("--spectral_hiddens", type=int, nargs="+", default=None)
    parser.add_argument("--spectral_epochs", type=int, default=30)
    parser.add_argument("--spectral_lr", type=float, default=1e-3)
    parser.add_argument("--spectral_batch_size", type=int, default=1024)
    parser.add_argument("--spectral_n_nbg", type=int, default=30)
    parser.add_argument("--spectral_scale_k", type=int, default=15)
    parser.add_argument(
        "--spectral_is_local_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save_model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.key, args)
    print("\nProcess completed successfully.")


if __name__ == "__main__":
    main()
