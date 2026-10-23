"""Build a benchmark bundle with generated filtered queries and exact ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_query_workload import generate
from rag_tree.ground_truth import exact_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NPZ containing vectors and attributes")
    parser.add_argument("output", help="output NPZ benchmark bundle")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1 or args.k < 1:
        raise SystemExit("--count and -k must be positive")

    source = np.load(args.input)
    if not {"vectors", "attributes"}.issubset(source.files):
        raise SystemExit("input NPZ must contain vectors and attributes")
    vectors = np.asarray(source["vectors"], dtype=np.float32)
    attributes = np.asarray(source["attributes"], dtype=np.float64)
    if (
        vectors.ndim != 2
        or attributes.ndim != 2
        or len(vectors) != len(attributes)
        or not vectors.shape[0]
        or not vectors.shape[1]
        or not attributes.shape[1]
    ):
        raise SystemExit("vectors and attributes must be 2-D with the same row count")
    if not np.isfinite(vectors).all() or not np.isfinite(attributes).all():
        raise SystemExit("vectors and attributes must contain only finite values")

    queries, ranges, query_ids, selectivities = generate(
        vectors, attributes, args.count, args.seed, args.k
    )
    ground_truth = exact_ground_truth(vectors, attributes, queries, ranges, args.k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vectors=vectors,
        attributes=attributes,
        queries=queries,
        ranges=ranges,
        query_ids=query_ids,
        selectivities=selectivities,
        ground_truth=ground_truth,
    )
    print(f"wrote {len(queries)} queries with exact top-{args.k} ground truth to {output}")


if __name__ == "__main__":
    main()
