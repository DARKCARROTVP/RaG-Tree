"""Generate the paper-style synthetic numeric range-query workload."""

from __future__ import annotations

import argparse

import numpy as np


def generate(vectors: np.ndarray, attributes: np.ndarray, count: int, seed: int, k: int):
    rng = np.random.default_rng(seed)
    n, m = attributes.shape
    ranges = np.empty((count, m, 2), dtype=np.float64)
    query_ids = np.empty(count, dtype=np.int64)
    selectivities = np.empty(count, dtype=np.float64)
    attempts = 0
    produced = 0
    while produced < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("could not generate enough queries returning at least k objects")
        p = int(rng.integers(1, m + 1))
        selected = rng.choice(m, size=p, replace=False)
        desired = float(10 ** rng.uniform(-4.0, 0.0))
        query_id = int(rng.integers(0, n))
        bounds = np.column_stack([attributes.min(axis=0), attributes.max(axis=0)]).astype(np.float64)
        query_range = bounds.copy()
        per_attr = desired ** (1.0 / p)
        for dim in selected:
            lo, hi = bounds[dim]
            width = max(0.0, hi - lo)
            span = width * per_attr
            center = attributes[query_id, dim]
            left = center - span * rng.uniform(0.35, 0.65)
            right = left + span
            shift = max(0.0, lo - left)
            left += shift
            right += shift
            if right > hi:
                left -= right - hi
                right = hi
            query_range[dim] = (max(lo, left), min(hi, right))
        eligible = np.all((attributes >= query_range[:, 0]) & (attributes <= query_range[:, 1]), axis=1)
        if int(eligible.sum()) < k:
            continue
        ranges[produced] = query_range
        query_ids[produced] = query_id
        selectivities[produced] = float(eligible.mean())
        produced += 1
    return vectors[query_ids].astype(np.float32), ranges, query_ids, selectivities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NPZ with vectors and attributes")
    parser.add_argument("output", help="output NPZ path")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    data = np.load(args.input)
    queries, ranges, query_ids, selectivities = generate(data["vectors"], data["attributes"], args.count, args.seed, args.k)
    np.savez_compressed(args.output, queries=queries, ranges=ranges, query_ids=query_ids, selectivities=selectivities)
    print(f"wrote {len(queries)} queries; selectivity range={selectivities.min():.6g}..{selectivities.max():.6g}")


if __name__ == "__main__":
    main()
