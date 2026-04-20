import os
import sys
import csv
import time
import argparse
import warnings
import pickle
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

# Add parent directory to path so we can import spectralnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectralnet import SpectralNet, Metrics
from data import load_data_from_mat, get_kbc_mat_file, get_kbc_xy_keys

warnings.filterwarnings('ignore')

# Limit thread counts to prevent parallel conflicts
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


def run_spectralnet(X_normalized, y_true, n_clusters, seed, result_dir, key, args):
    """Run SpectralNet clustering and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tensor = torch.from_numpy(X_normalized).float()
    y_tensor = torch.from_numpy(y_true).long() if y_true is not None else None

    # Build SpectralNet config based on dataset size / dimensionality
    n_samples, n_features = X_normalized.shape
    use_ae = args.use_ae
    use_siamese = args.use_siamese

    # Default spectral_hiddens: last dim must equal n_clusters
    spectral_hiddens = args.spectral_hiddens
    if spectral_hiddens is None:
        spectral_hiddens = [512, 512, n_clusters]

    spectralnet = SpectralNet(
        n_clusters=n_clusters,
        should_use_ae=use_ae,
        should_use_siamese=use_siamese,
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
    )

    start_time = time.time()
    spectralnet.fit(X_tensor, y_tensor)
    cluster_assignments = spectralnet.predict(X_tensor)
    elapsed = time.time() - start_time

    nmi = normalized_mutual_info_score(y_true, cluster_assignments)
    ari = adjusted_rand_score(y_true, cluster_assignments)
    aligned = Metrics.align_labels(y_true, cluster_assignments)
    f1 = f1_score(y_true, aligned, average='macro')

    # Save model if requested
    if args.save_model:
        model_path = os.path.join(result_dir, f'{key}_model.pt')
        torch.save(spectralnet, model_path)
        print(f"  Model saved: {model_path}")

    return {
        'seed': seed,
        'nmi': nmi,
        'ari': ari,
        'f1': f1,
        'time_sec': elapsed,
        'cluster_assignments': cluster_assignments,
        'embeddings': spectralnet.embeddings_,
    }


def save_visualization(X_embedded, y_true, labels, key, base_dir="fig"):
    """Save ground-truth and clustering-result scatter plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    key_dir = os.path.join(base_dir, key)
    os.makedirs(key_dir, exist_ok=True)

    n_true = len(np.unique(y_true))
    n_pred = len(np.unique(labels))

    def get_colors(n):
        if n <= 10:
            cmap = plt.get_cmap('tab10')
            return [cmap(i) for i in range(n)]
        elif n <= 20:
            cmap = plt.get_cmap('tab20')
            return [cmap(i) for i in range(n)]
        else:
            return plt.cm.hsv(np.linspace(0, 1, n))

    true_colors = get_colors(n_true)
    pred_colors = get_colors(n_pred)

    # Ground truth
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true,
                cmap=ListedColormap(true_colors), alpha=0.7, s=15)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title('')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(key_dir, f"{key}_dataset.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    # Clustering result
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels,
                cmap=ListedColormap(pred_colors), alpha=0.7, s=15)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title('')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(key_dir, f"{key}_clustering_result.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {key_dir}/")


def process_dataset(key, args):
    """Main pipeline for a single KBC dataset."""
    print(f"\nProcessing {key} ...")
    result_dir = os.path.join("results", key)
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load data
    mat_file = get_kbc_mat_file(key, base_path=args.data_root)
    x_key, y_key = get_kbc_xy_keys(key)
    X_original, y_true, pos = load_data_from_mat(key, mat_file, x_key, y_key)

    # 2. Special preprocessing for 151507_final
    if key in ['151507_final']:
        from scipy.spatial.distance import cdist
        pca = PCA(n_components=50, random_state=42)
        x_pca = pca.fit_transform(X_original)
        tt_dis = cdist(pos, pos)
        neighbor_indices = np.argsort(tt_dis, axis=1)[:, 1:7]
        adjacency_matrix = np.zeros((pos.shape[0], pos.shape[0]))
        for i in range(pos.shape[0]):
            adjacency_matrix[i, neighbor_indices[i]] = 1
        node_features = [x_pca]
        adj_mat = [adjacency_matrix]
        labels_sequence = []
        for i in range(len(node_features)):
            graph_feat = []
            for it in range(7 + 1):
                if it == 0:
                    graph_feat.append(node_features[i])
                else:
                    adj_cur = adj_mat[i] + np.identity(adj_mat[i].shape[0])
                    deg = np.sum(adj_cur, axis=1).reshape(-1)
                    deg[deg == 0] = 1
                    deg = 1 / deg
                    deg_mat = np.diag(deg)
                    adj_cur_norm = deg_mat @ adj_cur
                    graph_feat_cur = adj_cur_norm @ graph_feat[it - 1]
                    graph_feat.append(graph_feat_cur)
            labels_sequence.append(np.concatenate(graph_feat, axis=1))
        X_original = labels_sequence[0]

    # 3. Decide visualization source
    if key in ['tutorial', 'tonsil', 'airway', 'crohn', '151507_final']:
        x_for_viz = pos
    else:
        x_for_viz = X_original

    # 4. Normalize
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_original)
    n_clusters = np.unique(y_true).size

    # Save scaler for later prediction
    scaler_path = os.path.join(result_dir, f'{key}_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved: {scaler_path}")

    # 5. Run once with fixed seed
    seed = 42
    print(f"  Running with seed={seed} ...")
    try:
        result = run_spectralnet(
            X_normalized, y_true, n_clusters, seed, result_dir, key, args
        )
    except Exception as e:
        print(f"    [Error] seed={seed}: {e}")
        return None

    # 6. Save result
    result_csv_path = os.path.join(result_dir, f'{key}_result.csv')
    with open(result_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'nmi', 'ari', 'f1', 'time_sec'])
        writer.writerow([
            seed,
            f"{result['nmi']:.4f}",
            f"{result['ari']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['time_sec']:.4f}",
        ])
    print(f"Results saved: {result_csv_path}")

    best_result = result
    best_result['run_idx'] = 1

    # 8. Visualization
    print(f"\nVisualizing (seed={best_result['seed']}) ...")
    if x_for_viz.shape[1] > 2:
        perplexity = min(30, x_for_viz.shape[0] - 1)
        X_embedded = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(x_for_viz)
    else:
        X_embedded = x_for_viz

    aligned_labels = Metrics.align_labels(y_true, best_result['cluster_assignments'])
    save_visualization(X_embedded, y_true, aligned_labels, key, base_dir="fig")

    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SpectralNet on KBC datasets')
    parser.add_argument('--key', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_root', type=str, default='../../data', help='Data root directory')

    # SpectralNet architecture
    parser.add_argument('--use_ae', action='store_true', help='Use autoencoder')
    parser.add_argument('--use_siamese', action='store_true', help='Use siamese network')
    parser.add_argument('--ae_hiddens', type=int, nargs='+', default=[512, 256, 64], help='AE hidden dims')
    parser.add_argument('--ae_epochs', type=int, default=30, help='AE epochs')
    parser.add_argument('--ae_batch_size', type=int, default=256, help='AE batch size')
    parser.add_argument('--siamese_hiddens', type=int, nargs='+', default=[512, 512, 64], help='Siamese hidden dims')
    parser.add_argument('--siamese_epochs', type=int, default=20, help='Siamese epochs')
    parser.add_argument('--siamese_batch_size', type=int, default=256, help='Siamese batch size')
    parser.add_argument('--siamese_n_nbg', type=int, default=5, help='Siamese nearest neighbors')
    parser.add_argument('--spectral_hiddens', type=int, nargs='+', default=None, help='SpectralNet hidden dims (last must equal n_clusters)')
    parser.add_argument('--spectral_epochs', type=int, default=30, help='SpectralNet epochs')
    parser.add_argument('--spectral_lr', type=float, default=1e-3, help='SpectralNet learning rate')
    parser.add_argument('--spectral_batch_size', type=int, default=1024, help='SpectralNet batch size')
    parser.add_argument('--spectral_n_nbg', type=int, default=30, help='SpectralNet nearest neighbors')
    parser.add_argument('--spectral_scale_k', type=int, default=15, help='Scale k for Gaussian kernel')
    parser.add_argument('--spectral_is_local_scale', action='store_true', default=True, help='Use local scale')
    parser.add_argument('--save_model', action='store_true', help='Save trained model for later prediction')

    args = parser.parse_args()

    result = process_dataset(args.key, args)
    if result:
        print("\nProcess completed successfully.")
    else:
        print("\nProcess failed.")
