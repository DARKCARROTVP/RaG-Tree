from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_tree import RaGTree  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(2_000, 32)).astype(np.float32)
    first = rng.uniform(0.0, 1.0, size=2_000)
    second = np.clip(0.7 * first + 0.3 * rng.uniform(0.0, 1.0, size=2_000), 0.0, 1.0)
    attributes = np.column_stack([first, second])
    index = RaGTree(vectors, attributes, leaf_capacity=64, backend="exact", seed=17)
    query_id = 31
    ranges = np.array([[0.2, 0.8], [0.3, 0.85]])
    ids, distances, stats = index.search(vectors[query_id], ranges, k=10, ef_search=64)
    truth_ids, truth_distances = index.exact_search(vectors[query_id], ranges, k=10)
    recall = len(set(ids.tolist()) & set(truth_ids.tolist())) / max(1, len(truth_ids))
    print(json.dumps({
        "index": index.stats(),
        "query": {
            "k": 10,
            "ef_search": 64,
            "returned": int(len(ids)),
            "exact_returned": int(len(truth_ids)),
            "recall_at_10": float(recall),
            "selected_graphs": stats.selected_graphs,
            "visited_tree_nodes": stats.visited_tree_nodes,
            "distance_work_estimate": stats.estimated_distance_computations,
            "selected_node_ids": stats.selected_node_ids,
        },
    }, indent=2))
    if not np.array_equal(ids, truth_ids) or not np.allclose(distances, truth_distances):
        raise SystemExit("exact backend differs from brute-force ground truth")


if __name__ == "__main__":
    main()
