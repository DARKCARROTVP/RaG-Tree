# RaG-Tree reproduction

This repository implements the central RaG-Tree ideas from Jiawei Liu et al.,
“RaG-Tree: Combining R-Tree and HNSW for Multi-Attribute Range Filtered
Approximate Nearest Neighbor Search” (arXiv:2608.01255v1): a binary
attribute-space tree, a local vector graph at every tree node, APD-based local
HNSW degree selection, selectivity-based `ef_search` allocation, and the
cost-based dynamic program that chooses a disjoint node cover for each query.

The paper's artifact URL (`github.com/rucjrliu/RaG-Tree_code`) returned HTTP
404 when checked on 2026-10-23. This code is an independent implementation of
the paper description; it does not copy the unavailable artifact.

## Status and scope

The current implementation is a small-data reference and development baseline.
It supports an exact local-graph backend for deterministic correctness tests
and an optional `hnswlib` backend for approximate search. It is not yet a
large-scale reproduction of the paper's C++ experiments. In particular, the
paper leaves its binary partition routine underspecified in the paper text;
this repository uses a deterministic median split on the attribute with the
largest normalized spread. That choice is explicit in the index metadata.

The core delta-HNSW insertion, lazy deletion, main-plus-delta query, and
cost-triggered merge path are implemented. The merge trigger uses the paper's
cost model with `omega_c = 8M` from its experiment setup. The paper does not
state the delta-degree scale parameter `alpha` in the PDF text, so the
implementation exposes it as a constructor option (default 2) rather than
claiming a paper-specified value. Benchmarks against DBLP, MSMarco, LAION,
DIGRA, KHI, and ACORN are not yet included. No paper performance claim is
reproduced by the included synthetic demo.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,hnsw]'
pytest
python scripts/synthetic_demo.py
```

To avoid compiling hnswlib, install `-e '.[test]'` and use the exact backend in
the demo. The exact backend is intended only for small datasets and tests.

## Prepared-data benchmark

`scripts/benchmark_npz.py` expects a compressed NumPy archive with these arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `vectors` | `[N, d]` | Dataset embeddings; converted to `float32` |
| `attributes` | `[N, m]` | Numeric filter attributes |
| `queries` | `[Q, d]` | Query embeddings |
| `ranges` | `[Q, m, 2]` | Inclusive lower/upper range for each query and attribute |
| `ground_truth` | `[Q, k]` | Exact filtered top-k object IDs; use `-1` for padding |

Prepare a bundle and run a sweep:

```bash
python scripts/prepare_npz_benchmark.py data/source.npz data/experiment.npz --count 1000 -k 10
python scripts/benchmark_npz.py data/experiment.npz --backend hnswlib --ef 32 64 128 256 --runs 3 --output results/rag_tree.json
```

The JSON records build time, recall, QPS, latency percentiles, and planner
work estimates. The preparation script generates the documented numeric-range
workload and computes exact filtered top-k IDs by brute force; this can be
expensive for large datasets. Keep source datasets and generated result files
outside version control.

## Minimal API

```python
import numpy as np
from rag_tree import RaGTree

index = RaGTree(
    vectors,                  # float32 [N, d]
    attributes,               # numeric [N, m]
    leaf_capacity=max(1, int(np.ceil(0.001 * len(vectors)))),
    degree_range=(24, 32),
    backend="hnswlib",
)
ids, squared_l2, stats = index.search(
    query_vector,
    np.array([[10, 40], [0.1, 0.8]]),
    k=10,
    ef_search=100,
)
```

`exact_search` returns filtered brute-force ground truth for small experiments.
For `backend="hnswlib"`, distances are hnswlib's squared L2 values. Search
statistics distinguish the paper's estimated distance-work cost from wall
clock latency; the hnswlib backend reports a degree-times-retrieved-count proxy,
not instrumented internal distance computations.

## Method mapping

| Paper mechanism | Implementation |
| --- | --- |
| R-tree node MBR pruning | Inclusive numeric range/MBR intersection |
| Per-node local HNSW | Optional hnswlib index built for each node |
| APD-guided degree (Eq. 1-3) | Sampled mean pairwise L2, mapped to `[M_L, M_H]` |
| Leaf/internal selectivity (Eq. 5-6) | MBR overlap product; child-cardinality weighted aggregation |
| Global search budget (Eq. 7) | Proportional ef allocation with deterministic rounding |
| Local cost (Eq. 4) and node-choice DP (Eq. 8) | Implemented in the query planner |
| Delta-HNSW updates (Section 5, Eq. 9-11) | Insert path, lazy tombstones, joint query, cost-triggered merge |

## Reproduction plan

1. Validate exact-backend correctness and the query planner on generated small
   data; this reference stage is covered by the included tests and demo.
2. Prepare the paper's processed datasets, query workloads, and exact filtered
   ground truth, then validate their dimensions and provenance.
3. Implement the paper baselines and port the construction/search kernels to
   the paper's C++ setup before making scale or performance comparisons.
4. Reproduce the hardware protocol and report QPS-recall curves, index size,
   build time, distance computations, update cost, and three-run averages.
   Label every implementation and dataset deviation from the paper.

## Citation

```bibtex
@article{liu2026ragtree,
  title={RaG-Tree: Combining R-Tree and HNSW for Multi-Attribute Range Filtered Approximate Nearest Neighbor Search},
  author={Liu, Jiawei and Zhang, Xiang and Zhang, Chao and Fan, Ju and Du, Xiaoyong},
  journal={arXiv preprint arXiv:2608.01255},
  year={2026}
}
```
