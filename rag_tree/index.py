"""A compact, auditable RaG-Tree research implementation.

The paper specifies the APD/degree mapping, selectivity estimates, ef allocation,
local graph cost, and tree DP. Its PDF does not fully pin down the binary split
routine, so this implementation uses a deterministic widest-normalized-spread
median split and labels it as an implementation choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log, prod
from typing import Any

import numpy as np

try:  # Optional; the exact backend keeps tests and small examples self-contained.
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover - depends on optional runtime package
    hnswlib = None


@dataclass
class SearchStats:
    visited_tree_nodes: int = 0
    selected_graphs: int = 0
    estimated_distance_computations: int = 0
    returned_candidates: int = 0
    selected_node_ids: list[int] = field(default_factory=list)
    per_graph_ef: dict[int, int] = field(default_factory=dict)


@dataclass
class _Node:
    node_id: int
    ids: np.ndarray
    low: np.ndarray
    high: np.ndarray
    degree: int
    apd: float
    left: "_Node | None" = None
    right: "_Node | None" = None
    graph: Any = None
    main_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    delta_ids: list[int] = field(default_factory=list)
    delta_graph: Any = None
    delta_capacity: int = 0
    delta_search_cost_since_merge: float = 0.0
    split_dim: int | None = None
    split_pivot: float | None = None
    selectivity: float = 1.0
    ef: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class RaGTree:
    """Binary R-tree with one local HNSW (or exact) index per tree node.

    Parameters follow the paper unless the paper leaves a choice open:
    ``leaf_capacity`` defaults to ceil(0.1% of N); ``degree_range`` defaults to
    [24, 32]. The ``split_strategy`` and APD sampling implementation are local
    reproducibility choices and are recorded on the instance.

    The exact backend is intended for tests and small datasets. For scale, use
    ``backend='hnswlib'`` and account for the paper's replicated per-node graph
    storage, which can be substantially larger than a single global HNSW.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        attributes: np.ndarray,
        *,
        leaf_capacity: int | None = None,
        degree_range: tuple[int, int] = (24, 32),
        construction_ef_factor: int = 8,
        apd_pair_samples: int = 4096,
        delta_degree_scale: float = 2.0,
        seed: int = 20261023,
        backend: str = "exact",
    ) -> None:
        x = np.asarray(vectors, dtype=np.float32)
        a = np.asarray(attributes, dtype=np.float64)
        if x.ndim != 2 or a.ndim != 2 or x.shape[0] != a.shape[0]:
            raise ValueError("vectors and attributes must be 2-D with equal row counts")
        if x.shape[0] == 0 or x.shape[1] == 0 or a.shape[1] == 0:
            raise ValueError("the index requires non-empty vectors and attributes")
        if not np.isfinite(x).all() or not np.isfinite(a).all():
            raise ValueError("vectors and attributes must be finite")
        if backend not in {"exact", "hnswlib"}:
            raise ValueError("backend must be 'exact' or 'hnswlib'")
        if backend == "hnswlib" and hnswlib is None:
            raise RuntimeError("backend='hnswlib' requires the hnswlib package")
        ml, mh = degree_range
        if ml < 2 or mh < ml:
            raise ValueError("degree_range must satisfy 2 <= M_L <= M_H")

        self.vectors = np.ascontiguousarray(x)
        self.attributes = np.ascontiguousarray(a)
        self.n, self.dimension = x.shape
        self.attribute_count = a.shape[1]
        self.degree_range = (int(ml), int(mh))
        self.construction_ef_factor = int(construction_ef_factor)
        self.apd_pair_samples = int(apd_pair_samples)
        if delta_degree_scale < 1:
            raise ValueError("delta_degree_scale must be >= 1")
        self.delta_degree_scale = float(delta_degree_scale)
        self.seed = int(seed)
        self.backend = backend
        self.split_strategy = "widest-normalized-spread stable median split"
        self._rng = np.random.default_rng(seed)
        self.deleted_ids: set[int] = set()
        if leaf_capacity is None:
            leaf_capacity = max(1, int(ceil(0.001 * self.n)))
        if leaf_capacity < 1:
            raise ValueError("leaf_capacity must be positive")
        self.leaf_capacity = int(leaf_capacity)
        self._next_node_id = 0
        self.global_apd = self._estimate_apd(np.arange(self.n, dtype=np.int64))
        self.root = self._build(np.arange(self.n, dtype=np.int64))

    def _estimate_apd(self, ids: np.ndarray) -> float:
        """Estimate mean unordered-pair Euclidean distance using sampled pairs."""
        count = len(ids)
        if count < 2:
            return 0.0
        pairs = min(self.apd_pair_samples, count * (count - 1) // 2)
        if pairs == count * (count - 1) // 2 and count <= 256:
            values = self.vectors[ids].astype(np.float64, copy=False)
            dists: list[np.ndarray] = []
            for start in range(0, count, 64):
                lhs = values[start : start + 64]
                sq = np.sum(lhs * lhs, axis=1)[:, None] + np.sum(values * values, axis=1)[None, :]
                sq -= 2.0 * lhs @ values.T
                rows = np.arange(start, min(start + 64, count))
                mask = np.arange(count)[None, :] > rows[:, None]
                dists.append(np.sqrt(np.maximum(sq[mask], 0.0)))
            all_dists = np.concatenate(dists)
            return float(all_dists.mean()) if all_dists.size else 0.0

        # Sample distinct unordered pairs. Rejection is avoided by allowing the
        # two endpoints to coincide then dropping those pairs.
        i = self._rng.integers(0, count, size=pairs * 2)
        j = self._rng.integers(0, count - 1, size=pairs * 2)
        j += j >= i
        keep = i != j
        if not np.any(keep):
            return 0.0
        lhs = self.vectors[ids[i[keep]]].astype(np.float64, copy=False)
        rhs = self.vectors[ids[j[keep]]].astype(np.float64, copy=False)
        return float(np.linalg.norm(lhs - rhs, axis=1).mean())

    def _new_node(self, ids: np.ndarray) -> _Node:
        node_id = self._next_node_id
        self._next_node_id += 1
        subset = self.vectors[ids]
        low = self.attributes[ids].min(axis=0)
        high = self.attributes[ids].max(axis=0)
        apd = self._estimate_apd(ids)
        delta = 0.0 if self.global_apd <= 0 else (self.global_apd - apd) / self.global_apd
        ml, mh = self.degree_range
        degree = int(np.floor(mh - max(0.0, delta) * (mh - ml) + 0.5))
        degree = min(mh, max(ml, degree))
        node = _Node(node_id=node_id, ids=ids.copy(), low=low, high=high, degree=degree, apd=apd)
        node.main_ids = ids.copy()
        self._build_local_graph(node)
        return node

    def _build_local_graph(self, node: _Node) -> None:
        if self.backend == "exact" or len(node.main_ids) < 2:
            node.graph = None
            return
        graph = hnswlib.Index(space="l2", dim=self.dimension)
        graph.init_index(
            max_elements=len(node.main_ids),
            ef_construction=max(node.degree, self.construction_ef_factor * node.degree),
            M=node.degree,
            random_seed=self.seed + node.node_id,
        )
        graph.add_items(self.vectors[node.main_ids], node.main_ids, num_threads=1)
        node.graph = graph

    def _split(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self.attributes[ids]
        spans = values.max(axis=0) - values.min(axis=0)
        root_spans = np.maximum(self.attributes.max(axis=0) - self.attributes.min(axis=0), 1e-12)
        dim = int(np.argmax(spans / root_spans))
        order = np.lexsort((ids, values[:, dim]))
        middle = len(ids) // 2
        return ids[order[:middle]], ids[order[middle:]]

    def _build(self, ids: np.ndarray) -> _Node:
        node = self._new_node(ids)
        if len(ids) > self.leaf_capacity and len(ids) > 1:
            left_ids, right_ids = self._split(ids)
            if len(left_ids) and len(right_ids):
                spans = self.attributes[ids].max(axis=0) - self.attributes[ids].min(axis=0)
                root_spans = np.maximum(self.attributes.max(axis=0) - self.attributes.min(axis=0), 1e-12)
                dim = int(np.argmax(spans / root_spans))
                order = np.lexsort((ids, self.attributes[ids, dim]))
                node.split_dim = dim
                node.split_pivot = float(self.attributes[ids[order[len(ids) // 2]], dim])
                node.left = self._build(left_ids)
                node.right = self._build(right_ids)
        return node

    @staticmethod
    def _range_intersects(low: np.ndarray, high: np.ndarray, ranges: np.ndarray) -> bool:
        return bool(np.all(high >= ranges[:, 0]) and np.all(low <= ranges[:, 1]))

    def _overlap_selectivity(self, node: _Node, ranges: np.ndarray) -> float:
        widths = node.high - node.low
        overlap = np.maximum(0.0, np.minimum(node.high, ranges[:, 1]) - np.maximum(node.low, ranges[:, 0]))
        ratios = np.ones(self.attribute_count, dtype=np.float64)
        nondegenerate = widths > 0
        ratios[nondegenerate] = overlap[nondegenerate] / widths[nondegenerate]
        # A zero-width attribute is either fully covered or excluded; MBR
        # pruning handles exclusion, so the overlap ratio is one if covered.
        ratios[~nondegenerate] = 1.0
        return float(np.clip(prod(ratios), 0.0, 1.0))

    def _estimate_selectivity(self, node: _Node, ranges: np.ndarray, stats: SearchStats) -> float:
        stats.visited_tree_nodes += 1
        if not self._range_intersects(node.low, node.high, ranges):
            node.selectivity = 0.0
            return 0.0
        if node.is_leaf:
            node.selectivity = max(self._overlap_selectivity(node, ranges), 1.0 / max(1, len(node.ids)))
            return node.selectivity
        left_sel = self._estimate_selectivity(node.left, ranges, stats) if node.left else 0.0
        right_sel = self._estimate_selectivity(node.right, ranges, stats) if node.right else 0.0
        child_n = (len(node.left.ids) if node.left else 0) + (len(node.right.ids) if node.right else 0)
        weighted = (
            (len(node.left.ids) * left_sel if node.left else 0.0)
            + (len(node.right.ids) * right_sel if node.right else 0.0)
        )
        node.selectivity = max(weighted / max(1, child_n), 1.0 / max(1, len(node.ids)))
        return node.selectivity

    def _allocate_budget(self, node: _Node, ef: int, ranges: np.ndarray) -> None:
        node.ef = max(1, int(ef))
        if node.is_leaf:
            return
        children = [c for c in (node.left, node.right) if c is not None and self._range_intersects(c.low, c.high, ranges)]
        if not children:
            return
        weights = np.array([len(c.ids) * c.selectivity for c in children], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.array([len(c.ids) for c in children], dtype=np.float64)
        raw = node.ef * weights / weights.sum()
        allocations = np.floor(raw + 0.5).astype(int)  # paper Eq. (7)
        allocations = np.maximum(allocations, 1)
        # Independent rounding in Eq. (7) can exceed the parent budget by one;
        # conserve the supplied budget while preserving deterministic priorities.
        while allocations.sum() > node.ef and len(children) > 1:
            eligible = np.flatnonzero(allocations > 1)
            if not len(eligible):
                break
            worst = min(eligible, key=lambda i: (raw[i] - np.floor(raw[i]), weights[i], -i))
            allocations[worst] -= 1
        for child, child_ef in zip(children, allocations):
            self._allocate_budget(child, int(child_ef), ranges)

    def _height(self, node: _Node) -> int:
        return max(1, int(ceil(log(max(2, len(node.ids)), max(2, node.degree)))))

    def _estimated_cost(self, node: _Node) -> float:
        sel = max(node.selectivity, 1.0 / max(1, len(node.ids)))
        upper = (node.degree / 2.0) ** 2 * self._height(node)
        bottom = min(len(node.ids), node.degree * node.ef / sel)
        return float(upper + bottom)

    def _choose_nodes(self, node: _Node, ranges: np.ndarray, stats: SearchStats) -> tuple[float, list[_Node]]:
        if not self._range_intersects(node.low, node.high, ranges):
            return 0.0, []
        own_cost = self._estimated_cost(node)
        if node.is_leaf:
            return own_cost, [node]
        left_cost, left_nodes = self._choose_nodes(node.left, ranges, stats) if node.left else (0.0, [])
        right_cost, right_nodes = self._choose_nodes(node.right, ranges, stats) if node.right else (0.0, [])
        child_cost = left_cost + right_cost
        # Paper Algorithm 2 chooses children only when strictly cheaper; ties use v.
        if child_cost < own_cost:
            return child_cost, left_nodes + right_nodes
        return own_cost, [node]

    def _matches(self, ids: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        if self.deleted_ids:
            ids = ids[~np.isin(ids, np.fromiter(self.deleted_ids, dtype=np.int64))]
            if not len(ids):
                return ids
        values = self.attributes[ids]
        keep = np.all((values >= ranges[:, 0]) & (values <= ranges[:, 1]), axis=1)
        return ids[keep]

    def _search_one_graph(
        self,
        node: _Node,
        graph: Any,
        ids: np.ndarray,
        query: np.ndarray,
        ranges: np.ndarray,
        k: int,
        ef: int,
        degree: int,
        stats: SearchStats,
    ) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0 or len(ids) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        sel = max(node.selectivity, 1.0 / max(1, len(node.ids)))
        retrieve = min(len(ids), max(k, ef, int(ceil(ef / sel))))
        if self.backend == "hnswlib" and graph is not None:
            graph.set_ef(max(retrieve, ef))
            labels, distances = graph.knn_query(query.reshape(1, -1), k=retrieve, num_threads=1)
            candidate_ids = labels[0].astype(np.int64)
            candidate_distances = distances[0].astype(np.float64)
            stats.estimated_distance_computations += int(degree * retrieve)
        else:
            candidate_ids = ids
            delta = self.vectors[candidate_ids] - query[None, :]
            candidate_distances = np.einsum("ij,ij->i", delta, delta, optimize=True)
            stats.estimated_distance_computations += len(candidate_ids)
        eligible_mask = np.all(
            (self.attributes[candidate_ids] >= ranges[:, 0])
            & (self.attributes[candidate_ids] <= ranges[:, 1]), axis=1
        )
        eligible = candidate_ids[eligible_mask]
        distances = candidate_distances[eligible_mask]
        if self.deleted_ids and len(eligible):
            keep = ~np.isin(eligible, np.fromiter(self.deleted_ids, dtype=np.int64))
            eligible, distances = eligible[keep], distances[keep]
        order = np.argsort(distances, kind="stable")[:k]
        return eligible[order], distances[order]

    def _local_search(self, node: _Node, query: np.ndarray, ranges: np.ndarray, k: int, stats: SearchStats) -> tuple[np.ndarray, np.ndarray]:
        ef = max(1, node.ef)
        main_ids, main_distances = self._search_one_graph(
            node, node.graph, node.main_ids, query, ranges, k, ef, node.degree, stats
        )
        delta_ids = np.asarray(node.delta_ids, dtype=np.int64)
        if not len(delta_ids):
            return main_ids, main_distances
        # Paper Eq. (9): allocate delta-graph ef in proportion to graph size.
        ef_delta = max(1, int(np.floor(ef * len(delta_ids) / max(1, len(node.main_ids)) + 0.5)))
        delta_degree = max(2, int(np.floor(node.degree / self.delta_degree_scale + 0.5)))
        delta_result_ids, delta_distances = self._search_one_graph(
            node, node.delta_graph, delta_ids, query, ranges, k, ef_delta, delta_degree, stats
        )
        delta_height = max(1, int(ceil(log(max(2, len(delta_ids)), max(2, delta_degree)))))
        selectivity = max(node.selectivity, 1.0 / max(1, len(node.ids)))
        node.delta_search_cost_since_merge += (
            (delta_degree / 2.0) ** 2 * delta_height
            + min(len(delta_ids), delta_degree * ef_delta / selectivity)
        )
        merged: dict[int, float] = {}
        for object_id, distance in zip(main_ids.tolist(), main_distances.tolist()):
            merged[object_id] = distance
        for object_id, distance in zip(delta_result_ids.tolist(), delta_distances.tolist()):
            merged[object_id] = min(distance, merged.get(object_id, float("inf")))
        ordered = sorted(merged.items(), key=lambda item: (item[1], item[0]))[:k]
        return np.asarray([x[0] for x in ordered], dtype=np.int64), np.asarray([x[1] for x in ordered], dtype=np.float64)

    def _append_delta(self, node: _Node, object_id: int) -> None:
        node.delta_ids.append(object_id)
        if self.backend != "hnswlib":
            return
        degree = max(2, int(np.floor(node.degree / self.delta_degree_scale + 0.5)))
        if node.delta_graph is None:
            node.delta_capacity = 16
            node.delta_graph = hnswlib.Index(space="l2", dim=self.dimension)
            node.delta_graph.init_index(
                max_elements=node.delta_capacity,
                ef_construction=max(degree, self.construction_ef_factor * degree),
                M=degree,
                random_seed=self.seed + 100000 + node.node_id,
            )
        elif len(node.delta_ids) > node.delta_capacity:
            node.delta_capacity *= 2
            node.delta_graph.resize_index(node.delta_capacity)
        node.delta_graph.add_items(self.vectors[object_id : object_id + 1], np.asarray([object_id]), num_threads=1)

    def _maybe_merge_delta(self, node: _Node) -> None:
        if not node.delta_ids:
            return
        # Paper Eq. (11), using omega_c,v = 8 M_v from Section 6.1.
        merge_cost = len(node.delta_ids) * (
            (node.degree / 2.0) ** 2 * self._height(node)
            + node.degree * (8 * node.degree)
        )
        if node.delta_search_cost_since_merge <= merge_cost:
            return
        delta = np.asarray(node.delta_ids, dtype=np.int64)
        node.main_ids = np.concatenate([node.main_ids, delta])
        if self.backend == "hnswlib":
            # Rebuild from the merged membership. In particular, a singleton
            # main graph has no hnswlib object yet, so building it before
            # appending the delta would still leave ``node.graph`` as None.
            self._build_local_graph(node)
        node.delta_ids.clear()
        node.delta_graph = None
        node.delta_capacity = 0
        node.delta_search_cost_since_merge = 0.0

    def search(
        self,
        query: np.ndarray,
        ranges: np.ndarray,
        k: int,
        *,
        ef_search: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, SearchStats]:
        """Search a vector with inclusive numeric ranges; returns IDs, L2², stats."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        r = np.asarray(ranges, dtype=np.float64)
        if q.shape != (self.dimension,):
            raise ValueError(f"query must have dimension {self.dimension}")
        if r.shape != (self.attribute_count, 2) or not np.isfinite(r).all():
            raise ValueError(f"ranges must have shape ({self.attribute_count}, 2)")
        if np.any(r[:, 0] > r[:, 1]):
            raise ValueError("each range lower bound must be <= its upper bound")
        if k < 1 or ef_search < 1:
            raise ValueError("k and ef_search must be positive")
        stats = SearchStats()
        self._estimate_selectivity(self.root, r, stats)
        self._allocate_budget(self.root, ef_search, r)
        _, nodes = self._choose_nodes(self.root, r, stats)
        if not nodes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), stats
        found: dict[int, float] = {}
        for node in nodes:
            stats.selected_graphs += 1
            stats.selected_node_ids.append(node.node_id)
            stats.per_graph_ef[node.node_id] = node.ef
            ids, distances = self._local_search(node, q, r, k, stats)
            for object_id, distance in zip(ids.tolist(), distances.tolist()):
                if object_id not in found or distance < found[object_id]:
                    found[object_id] = distance
            self._maybe_merge_delta(node)
        ordered = sorted(found.items(), key=lambda item: (item[1], item[0]))[:k]
        stats.returned_candidates = len(ordered)
        return (
            np.asarray([item[0] for item in ordered], dtype=np.int64),
            np.asarray([item[1] for item in ordered], dtype=np.float64),
            stats,
        )

    def insert(self, vector: np.ndarray, attributes: np.ndarray) -> int:
        """Insert one object via per-node delta graphs along its attribute path."""
        x = np.asarray(vector, dtype=np.float32).reshape(-1)
        a = np.asarray(attributes, dtype=np.float64).reshape(-1)
        if x.shape != (self.dimension,) or a.shape != (self.attribute_count,):
            raise ValueError("inserted vector/attributes have the wrong dimension")
        if not np.isfinite(x).all() or not np.isfinite(a).all():
            raise ValueError("inserted vector and attributes must be finite")
        object_id = self.n
        self.vectors = np.ascontiguousarray(np.vstack([self.vectors, x]))
        self.attributes = np.ascontiguousarray(np.vstack([self.attributes, a]))
        self.n += 1
        node = self.root
        while True:
            node.ids = np.append(node.ids, object_id)
            node.low = np.minimum(node.low, a)
            node.high = np.maximum(node.high, a)
            self._append_delta(node, object_id)
            if node.is_leaf:
                break
            if node.split_dim is None or node.split_pivot is None:
                raise RuntimeError("internal tree node has no split metadata")
            node = node.left if a[node.split_dim] < node.split_pivot else node.right
            if node is None:
                raise RuntimeError("corrupt binary tree")
        return object_id

    def delete(self, object_id: int) -> None:
        """Lazily delete an object; graph navigation remains available."""
        if object_id < 0 or object_id >= self.n:
            raise IndexError("object_id out of range")
        self.deleted_ids.add(int(object_id))

    def exact_search(self, query: np.ndarray, ranges: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force filtered top-k ground truth for evaluation."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        r = np.asarray(ranges, dtype=np.float64)
        ids = np.arange(self.n, dtype=np.int64)
        ids = self._matches(ids, r)
        if not len(ids):
            return ids, np.empty(0, dtype=np.float64)
        delta = self.vectors[ids] - q[None, :]
        dists = np.einsum("ij,ij->i", delta, delta, optimize=True)
        order = np.argsort(dists, kind="stable")[:k]
        return ids[order], dists[order]

    def stats(self) -> dict[str, int | float | str]:
        nodes = 0
        graph_memberships = 0
        leaves = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes += 1
            graph_memberships += len(node.ids)
            leaves += node.is_leaf
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return {
            "objects": self.n,
            "dimension": self.dimension,
            "attributes": self.attribute_count,
            "nodes": nodes,
            "leaves": leaves,
            "replicated_graph_memberships": graph_memberships,
            "leaf_capacity": self.leaf_capacity,
            "backend": self.backend,
            "split_strategy": self.split_strategy,
        }
