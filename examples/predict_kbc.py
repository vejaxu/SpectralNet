import os
import sys
import argparse
import pickle
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# Add parent directory to path so we can import spectralnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectralnet import Metrics
from data import load_data_from_mat, get_kbc_mat_file, get_kbc_xy_keys


def apply_wl_preprocessing(X_original, pos):
    """Apply the same WL preprocessing as training for 151507_final."""
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
    return labels_sequence[0]


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


def main():
    parser = argparse.ArgumentParser(description='Load trained SpectralNet model and predict on KBC dataset')
    parser.add_argument('--key', type=str, required=True, help='Dataset name (same as training)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model .pt file')
    parser.add_argument('--scaler_path', type=str, default=None, help='Path to saved scaler .pkl file')
    parser.add_argument('--data_root', type=str, default='../../data', help='Data root directory')
    parser.add_argument('--output', type=str, default=None, help='Output path for cluster assignments .npy')
    parser.add_argument('--eval', action='store_true', help='Evaluate against ground truth labels')
    args = parser.parse_args()

    key = args.key
    result_dir = os.path.join("results", key)

    # Default paths
    model_path = args.model_path or os.path.join(result_dir, f'{key}_model.pt')
    scaler_path = args.scaler_path or os.path.join(result_dir, f'{key}_scaler.pkl')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run training with --save_model first.")
        return
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}")
        return

    # 1. Load model and scaler
    print(f"Loading model from {model_path} ...")
    spectralnet = torch.load(model_path, map_location='cpu', weights_only=False)
    spectralnet.device = torch.device('cpu')

    print(f"Loading scaler from {scaler_path} ...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 2. Load data
    print(f"Loading data for {key} ...")
    mat_file = get_kbc_mat_file(key, base_path=args.data_root)
    x_key, y_key = get_kbc_xy_keys(key)
    X_original, y_true, pos = load_data_from_mat(key, mat_file, x_key, y_key)

    # 3. Apply same preprocessing as training
    if key in ['151507_final']:
        X_original = apply_wl_preprocessing(X_original, pos)

    # 4. Decide visualization source (same as cluster_kbc)
    if key in ['tutorial', 'tonsil', 'airway', 'crohn', '151507_final']:
        x_for_viz = pos
    else:
        x_for_viz = X_original

    # 5. Normalize with saved scaler
    X_normalized = scaler.transform(X_original)
    X_tensor = torch.from_numpy(X_normalized).float()

    # 6. Predict
    print("Predicting cluster assignments ...")
    cluster_assignments = spectralnet.predict(X_tensor)

    # 7. Save or print results
    if args.output:
        np.save(args.output, cluster_assignments)
        print(f"Cluster assignments saved to {args.output}")
    else:
        output_path = os.path.join(result_dir, f'{key}_predictions.npy')
        np.save(output_path, cluster_assignments)
        print(f"Cluster assignments saved to {output_path}")

    # 8. Evaluate if requested
    if args.eval and y_true is not None:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
        nmi = normalized_mutual_info_score(y_true, cluster_assignments)
        ari = adjusted_rand_score(y_true, cluster_assignments)
        aligned = Metrics.align_labels(y_true, cluster_assignments)
        f1 = f1_score(y_true, aligned, average='macro')
        print(f"\nEvaluation Results:")
        print(f"  NMI: {nmi:.4f}")
        print(f"  ARI: {ari:.4f}")
        print(f"  F1 : {f1:.4f}")

    # 9. Visualization (same logic as cluster_kbc)
    if y_true is not None:
        print("\nPreparing visualization ...")
        if x_for_viz.shape[1] > 2:
            perplexity = min(30, x_for_viz.shape[0] - 1)
            X_embedded = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(x_for_viz)
        else:
            X_embedded = x_for_viz

        aligned_labels = Metrics.align_labels(y_true, cluster_assignments)
        save_visualization(X_embedded, y_true, aligned_labels, key, base_dir="fig")

    print("\nPrediction completed.")


if __name__ == '__main__':
    main()
