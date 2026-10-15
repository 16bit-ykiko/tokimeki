"""Clustering on a precomputed distance matrix; pure numpy, so it is tested without models."""

import numpy as np
from numpy.typing import NDArray


def average_linkage(
    distances: NDArray[np.float32],
    threshold: float,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.int64]:
    """Agglomerative clustering with average linkage, cut at `threshold`.

    Returns a label per item, numbered by first appearance. Average linkage does not chain
    the way DBSCAN does: two groups merge only if their faces are close on average, not
    when one ambiguous face sits between them. Items at an infinite distance never end up
    in one group. `weights` lets an item stand for a group of that many members.

    Nearest-neighbour chain algorithm: O(n^2) time and memory.
    """
    n = len(distances)
    d = distances.astype(np.float64)
    np.fill_diagonal(d, np.inf)
    size = np.ones(n) if weights is None else weights.astype(np.float64)
    active = np.ones(n, dtype=bool)
    merges: list[tuple[int, int]] = []
    chain: list[int] = []
    remaining = n
    while remaining > 1:
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        a = chain[-1]
        row = np.where(active, d[a], np.inf)
        b = int(row.argmin())
        if len(chain) > 1 and row[chain[-2]] <= row[b]:
            b = chain[-2]
        if not np.isfinite(row[b]):
            chain.pop()
            active[a] = False
            remaining -= 1
        elif len(chain) > 1 and b == chain[-2]:
            chain.pop()
            chain.pop()
            if row[b] <= threshold:
                merges.append((a, b))
            merged = (size[a] * d[a] + size[b] * d[b]) / (size[a] + size[b])
            d[a], d[:, a] = merged, merged
            d[a, a] = np.inf
            d[b], d[:, b] = np.inf, np.inf
            size[a] += size[b]
            active[b] = False
            remaining -= 1
        else:
            chain.append(b)
    root = np.arange(n)

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = int(root[x])
        return x

    for a, b in merges:
        root[find(b)] = find(a)
    roots = [find(i) for i in range(n)]
    numbering: dict[int, int] = {}
    return np.array([numbering.setdefault(r, len(numbering)) for r in roots], dtype=np.int64)


def evenly_spaced[T](items: list[T], limit: int) -> list[T]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]
