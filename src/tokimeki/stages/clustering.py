"""Clustering on a precomputed distance matrix; pure numpy, so it is tested without models."""

from collections import deque

import numpy as np
from numpy.typing import NDArray

NOISE = -1


def dbscan(distances: NDArray[np.float32], eps: float, min_samples: int) -> NDArray[np.int64]:
    """DBSCAN labels (0, 1, … or `NOISE`) for a square distance matrix; a point counts itself."""
    n = len(distances)
    labels = np.full(n, NOISE, dtype=np.int64)
    neighbours = [np.flatnonzero(distances[i] <= eps) for i in range(n)]
    core = np.array([len(nb) >= min_samples for nb in neighbours], dtype=bool)
    next_label = 0
    for seed in range(n):
        if labels[seed] != NOISE or not core[seed]:
            continue
        labels[seed] = next_label
        queue = deque([seed])
        while queue:
            point = queue.popleft()
            if not core[point]:
                continue
            for other in neighbours[point]:
                if labels[other] == NOISE:
                    labels[other] = next_label
                    queue.append(int(other))
        next_label += 1
    return labels


def nearest_cluster(
    distances: NDArray[np.float32],
    exemplar_clusters: NDArray[np.int64],
    threshold: float,
    k: int = 3,
) -> NDArray[np.int64]:
    """For each row, the cluster whose exemplars are closest, or `NOISE` if none is close enough.

    `distances` is `(faces, exemplars)` and `exemplar_clusters` gives each exemplar's cluster.
    A cluster's distance to a face is the mean of its `k` nearest exemplars (fewer if the
    cluster is that small), which keeps a single stray exemplar from pulling faces in.
    """
    n = len(distances)
    out = np.full(n, NOISE, dtype=np.int64)
    if n == 0 or distances.shape[1] == 0:
        return out
    clusters = np.unique(exemplar_clusters)
    scores = np.empty((n, len(clusters)), dtype=np.float32)
    for j, cluster in enumerate(clusters):
        own = np.sort(distances[:, exemplar_clusters == cluster], axis=1)[:, :k]
        scores[:, j] = own.mean(axis=1)
    best = scores.argmin(axis=1)
    close = scores[np.arange(n), best] <= threshold
    out[close] = clusters[best[close]]
    return out


def evenly_spaced[T](items: list[T], limit: int) -> list[T]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]
