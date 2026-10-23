"""Exact filtered nearest-neighbor ground truth helpers for small workloads."""

from __future__ import annotations

import numpy as np


def exact_ground_truth(
    vectors: np.ndarray,
    attributes: np.ndarray,
    queries: np.ndarray,
    ranges: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return filtered squared-L2 top-k IDs, padding short rows with ``-1``."""
    vectors = np.asarray(vectors, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.float64)
    if vectors.ndim != 2 or attributes.ndim != 2 or len(vectors) != len(attributes):
        raise ValueError("vectors and attributes must be 2-D with equal row counts")
    if queries.ndim != 2 or queries.shape[1] != vectors.shape[1]:
        raise ValueError("queries must be 2-D with the same vector dimension")
    if ranges.shape != (len(queries), attributes.shape[1], 2):
        raise ValueError("ranges must have shape [query_count, attribute_count, 2]")
    if k < 1:
        raise ValueError("k must be positive")
    if not all(np.isfinite(x).all() for x in (vectors, attributes, queries, ranges)):
        raise ValueError("inputs must contain only finite values")
    if np.any(ranges[:, :, 0] > ranges[:, :, 1]):
        raise ValueError("range lower bounds must not exceed upper bounds")

    result = np.full((len(queries), k), -1, dtype=np.int64)
    for row, (query, query_range) in enumerate(zip(queries, ranges)):
        eligible = np.all(
            (attributes >= query_range[:, 0]) & (attributes <= query_range[:, 1]), axis=1
        )
        ids = np.flatnonzero(eligible)
        delta = vectors[ids] - query[None, :]
        distances = np.einsum("ij,ij->i", delta, delta, optimize=True)
        order = np.argsort(distances, kind="stable")[:k]
        result[row, : len(order)] = ids[order]
    return result
