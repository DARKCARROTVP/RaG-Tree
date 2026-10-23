import numpy as np
import pytest

from rag_tree import RaGTree
from rag_tree.ground_truth import exact_ground_truth


def fixture_data():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(240, 12)).astype(np.float32)
    first = rng.uniform(size=240)
    second = 0.8 * first + 0.2 * rng.uniform(size=240)
    return vectors, np.column_stack([first, second])


def test_exact_backend_matches_filtered_ground_truth():
    vectors, attrs = fixture_data()
    index = RaGTree(vectors, attrs, leaf_capacity=24, backend="exact", seed=4)
    ranges = np.array([[0.25, 0.85], [0.2, 0.8]])
    ids, distances, stats = index.search(vectors[13], ranges, k=12, ef_search=48)
    truth_ids, truth_distances = index.exact_search(vectors[13], ranges, k=12)
    np.testing.assert_array_equal(ids, truth_ids)
    np.testing.assert_allclose(distances, truth_distances)
    assert stats.selected_graphs >= 1
    assert stats.visited_tree_nodes <= index.stats()["nodes"]


def test_out_of_domain_query_returns_no_candidates():
    vectors, attrs = fixture_data()
    index = RaGTree(vectors, attrs, leaf_capacity=24, backend="exact")
    ids, distances, _ = index.search(vectors[0], np.array([[2, 3], [2, 3]]), 5)
    assert ids.size == distances.size == 0


def test_invalid_range_and_shape_rejected():
    vectors, attrs = fixture_data()
    index = RaGTree(vectors, attrs, leaf_capacity=24)
    with pytest.raises(ValueError):
        index.search(vectors[0], np.array([[0.8, 0.2], [0.0, 1.0]]), 5)
    with pytest.raises(ValueError):
        index.search(vectors[0], np.array([[0.0, 1.0]]), 5)


def test_leaf_capacity_and_degree_range_validation():
    vectors, attrs = fixture_data()
    with pytest.raises(ValueError):
        RaGTree(vectors, attrs, leaf_capacity=0)
    with pytest.raises(ValueError):
        RaGTree(vectors, attrs, degree_range=(32, 24))


def test_delta_insert_and_lazy_delete_match_ground_truth():
    vectors, attrs = fixture_data()
    index = RaGTree(vectors, attrs, leaf_capacity=24, backend="exact", seed=8)
    new_vector = np.linspace(-1, 1, vectors.shape[1], dtype=np.float32)
    new_attrs = np.array([0.51, 0.55])
    new_id = index.insert(new_vector, new_attrs)
    assert new_id == len(vectors)
    ranges = np.array([[0.45, 0.6], [0.4, 0.7]])
    ids, distances, _ = index.search(new_vector, ranges, k=7, ef_search=32)
    truth_ids, truth_distances = index.exact_search(new_vector, ranges, k=7)
    np.testing.assert_array_equal(ids, truth_ids)
    np.testing.assert_allclose(distances, truth_distances)


def test_prepared_ground_truth_respects_filter_and_distance_order():
    vectors = np.array([[0, 0], [2, 0], [1, 0], [5, 5]], dtype=np.float32)
    attrs = np.array([[0], [1], [1], [2]], dtype=np.float64)
    queries = np.array([[0.9, 0]], dtype=np.float32)
    ranges = np.array([[[1, 1]]], dtype=np.float64)
    gt = exact_ground_truth(vectors, attrs, queries, ranges, k=3)
    np.testing.assert_array_equal(gt, [[2, 1, -1]])


def test_hnsw_backend_filters_and_handles_delta_updates():
    pytest.importorskip("hnswlib")
    vectors, attrs = fixture_data()
    index = RaGTree(vectors, attrs, leaf_capacity=24, backend="hnswlib", seed=21)
    ranges = np.array([[0.25, 0.85], [0.2, 0.8]])
    ids, _, _ = index.search(vectors[13], ranges, k=8, ef_search=64)
    assert len(ids) > 0
    assert np.all((attrs[ids] >= ranges[:, 0]) & (attrs[ids] <= ranges[:, 1]))
    new_vector = np.ones(vectors.shape[1], dtype=np.float32)
    new_id = index.insert(new_vector, np.array([0.5, 0.5]))
    ids, _, _ = index.search(new_vector, ranges, k=1, ef_search=64)
    assert new_id in ids.tolist()
    index.delete(new_id)
    ids, _, _ = index.search(new_vector, ranges, k=1, ef_search=64)
    assert new_id not in ids.tolist()

    index.delete(new_id)
    ids, distances, _ = index.search(new_vector, ranges, k=7, ef_search=32)
    truth_ids, truth_distances = index.exact_search(new_vector, ranges, k=7)
    np.testing.assert_array_equal(ids, truth_ids)
    np.testing.assert_allclose(distances, truth_distances)


def test_hnsw_delta_merge_rebuilds_singleton_main_graph():
    pytest.importorskip("hnswlib")
    vectors = np.array([[0.0, 0.0]], dtype=np.float32)
    attrs = np.array([[0.5]], dtype=np.float64)
    index = RaGTree(vectors, attrs, leaf_capacity=1, backend="hnswlib", seed=3)
    node = index.root
    assert node.graph is None

    object_id = index.insert(np.array([0.1, 0.1], dtype=np.float32), np.array([0.5]))
    assert node.delta_ids == [object_id]
    node.delta_search_cost_since_merge = float("inf")
    index._maybe_merge_delta(node)

    assert node.delta_ids == []
    np.testing.assert_array_equal(node.main_ids, [0, object_id])
    assert node.graph is not None
    ids, _, _ = index.search(np.array([0.1, 0.1]), np.array([[0.4, 0.6]]), k=2, ef_search=8)
    assert set(ids.tolist()) == {0, object_id}
