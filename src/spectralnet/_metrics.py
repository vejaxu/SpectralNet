import numpy as np
import sklearn.metrics as metrics

from munkres import Munkres
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_rand_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment

from spectralnet._utils import *


class Metrics:
    @staticmethod
    def acc_score(
        cluster_assignments: np.ndarray, y: np.ndarray, n_clusters: int
    ) -> float:
        """
        Compute the accuracy score of the clustering algorithm.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Cluster assignments for each data point.
        y : np.ndarray
            Ground truth labels.
        n_clusters : int
            Number of clusters.

        Returns
        -------
        float
            The computed accuracy score.

        Notes
        -----
        This function takes the `cluster_assignments` which represent the assigned clusters for each data point,
        the ground truth labels `y`, and the number of clusters `n_clusters`. It computes the accuracy score of the
        clustering algorithm by comparing the cluster assignments with the ground truth labels. The accuracy score
        is returned as a floating-point value.
        """

        confusion_matrix = metrics.confusion_matrix(y, cluster_assignments, labels=None)
        cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters=n_clusters)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
        print(metrics.confusion_matrix(y, y_pred))
        accuracy = np.mean(y_pred == y)
        return accuracy

    @staticmethod
    def nmi_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the normalized mutual information score of the clustering algorithm.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Cluster assignments for each data point.
        y : np.ndarray
            Ground truth labels.

        Returns
        -------
        float
            The computed normalized mutual information score.

        Notes
        -----
        This function takes the `cluster_assignments` which represent the assigned clusters for each data point
        and the ground truth labels `y`. It computes the normalized mutual information (NMI) score of the clustering
        algorithm. NMI measures the mutual dependence between the cluster assignments and the ground truth labels,
        normalized by the entropy of both variables. The NMI score ranges between 0 and 1, where a higher score
        indicates a better clustering performance. The computed NMI score is returned as a floating-point value.
        """
        return nmi(cluster_assignments, y)

    @staticmethod
    def ari_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """Compute Adjusted Rand Index."""
        return adjusted_rand_score(y, cluster_assignments)

    @staticmethod
    def f1_score(
        cluster_assignments: np.ndarray, y: np.ndarray, average: str = "macro"
    ) -> float:
        """
        Compute F1 score after Hungarian label alignment.

        Parameters
        ----------
        cluster_assignments : np.ndarray
            Cluster assignments for each data point.
        y : np.ndarray
            Ground truth labels.
        average : str, optional
            Averaging strategy for F1, by default "macro".

        Returns
        -------
        float
            The computed F1 score.
        """
        aligned = Metrics.align_labels(y, cluster_assignments)
        return f1_score(y, aligned, average=average)

    @staticmethod
    def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Align predicted cluster labels to ground-truth labels via the Hungarian algorithm.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Predicted cluster labels.

        Returns
        -------
        np.ndarray
            Re-mapped predicted labels that best match `y_true`.
        """
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        size = max(y_true.max(), y_pred.max()) + 1
        w = np.zeros((size, size), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        mapping = {row: col for row, col in zip(row_ind, col_ind)}
        return np.array([mapping.get(x, x) for x in y_pred])
