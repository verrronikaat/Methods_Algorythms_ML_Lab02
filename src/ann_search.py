"""Приближённый поиск ближайших соседей через Annoy."""
import time
import numpy as np
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors


def _annoy_metric(metric):
    if metric == "manhattan":
        return "manhattan"
    if metric == "euclidean":
        return "euclidean"
    return "angular"


def build_annoy_index(X, metric="manhattan", n_trees=11):
    X = np.asarray(X, dtype=np.float32)
    dim = X.shape[1]
    index = AnnoyIndex(dim, _annoy_metric(metric))
    for i, v in enumerate(X):
        index.add_item(i, v)
    index.build(n_trees)
    return index


def exact_search(X, x, k, metric="manhattan"):
    sk_metric = "manhattan" if metric == "manhattan" else "euclidean"
    nn = NearestNeighbors(n_neighbors=k, metric=sk_metric)
    nn.fit(X)
    _, idx = nn.kneighbors([x])
    return idx[0]


def recall_at_k(ann_idx, exact_idx):
    a = set(ann_idx)
    e = set(exact_idx)
    return len(a & e) / max(len(e), 1)


def compare_ann_vs_exact(X, query_vec, k=9, n_trees=11,
                         metric="manhattan"):
    X = np.asarray(X, dtype=float)
    index = build_annoy_index(X, metric=metric, n_trees=n_trees)

    t0 = time.time()
    ann_idx = index.get_nns_by_vector(
        query_vec.astype(np.float32), k, include_distances=False
    )
    ann_time = time.time() - t0

    t0 = time.time()
    exact_idx = exact_search(X, query_vec, k, metric=metric)
    exact_time = time.time() - t0

    return {
        "ann_indices": ann_idx,
        "exact_indices": list(map(int, exact_idx)),
        "recall": recall_at_k(ann_idx, exact_idx),
        "ann_time": ann_time,
        "exact_time": exact_time,
    }