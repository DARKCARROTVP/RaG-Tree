"""Run RaG-Tree QPS/recall sweeps on a prepared NPZ benchmark bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_tree import RaGTree  # noqa: E402


def recall_at_k(result: np.ndarray, truth: np.ndarray) -> float:
    truth_set = set(int(x) for x in truth if x >= 0)
    if not truth_set:
        return float("nan")
    return len(truth_set.intersection(int(x) for x in result)) / len(truth_set)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="NPZ containing vectors, attributes, queries, ranges, ground_truth")
    parser.add_argument("--backend", choices=["hnswlib", "exact"], default="hnswlib")
    parser.add_argument("--ef", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--leaf-capacity", type=int, default=None)
    parser.add_argument("--output", default="results/rag_tree_benchmark.json")
    args = parser.parse_args()
    data = np.load(args.dataset)
    required = {"vectors", "attributes", "queries", "ranges", "ground_truth"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"missing NPZ arrays: {sorted(missing)}")
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    attributes = np.asarray(data["attributes"], dtype=np.float64)
    queries = np.asarray(data["queries"], dtype=np.float32)
    ranges = np.asarray(data["ranges"], dtype=np.float64)
    truth = np.asarray(data["ground_truth"], dtype=np.int64)
    if len(queries) != len(ranges) or len(queries) != len(truth):
        raise SystemExit("queries, ranges and ground_truth must have equal row counts")
    k = int(truth.shape[1])
    rows = []
    for run in range(args.runs):
        build_start = time.perf_counter()
        index = RaGTree(
            vectors,
            attributes,
            leaf_capacity=args.leaf_capacity,
            backend=args.backend,
            seed=20261023 + run,
        )
        build_seconds = time.perf_counter() - build_start
        for ef in args.ef:
            latencies = []
            recalls = []
            distance_work = []
            node_visits = []
            graph_counts = []
            for q, r, gt in zip(queries, ranges, truth):
                start = time.perf_counter()
                ids, _, stats = index.search(q, r, k=k, ef_search=ef)
                latencies.append(time.perf_counter() - start)
                recalls.append(recall_at_k(ids, gt))
                distance_work.append(stats.estimated_distance_computations)
                node_visits.append(stats.visited_tree_nodes)
                graph_counts.append(stats.selected_graphs)
            elapsed = float(np.sum(latencies))
            rows.append({
                "run": run + 1,
                "ef_search": int(ef),
                "build_seconds": build_seconds,
                "qps": len(queries) / elapsed if elapsed else float("inf"),
                "recall_at_k": float(np.nanmean(recalls)),
                "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
                "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
                "latency_ms_p99": float(np.percentile(latencies, 99) * 1000),
                "estimated_distance_computations_mean": float(np.mean(distance_work)),
                "visited_tree_nodes_mean": float(np.mean(node_visits)),
                "selected_graphs_mean": float(np.mean(graph_counts)),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "dataset": str(Path(args.dataset).resolve()),
        "objects": int(len(vectors)),
        "vector_dimension": int(vectors.shape[1]),
        "attribute_count": int(attributes.shape[1]),
        "query_count": int(len(queries)),
        "k": k,
        "backend": args.backend,
        "runs": args.runs,
        "measurements": rows,
        "limitations": [
            "No paper baseline methods are implemented in this runner.",
            "hnswlib distance work is a degree-times-retrieval proxy, not instrumented internal distance count.",
            "The default tree split is a documented implementation choice because the PDF does not fully specify it.",
        ],
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
