import numpy as np
import pytest

from gta2real.match_data import nearest_neighbors


def test_chunked_retrieval_matches_dense_cosine():
    random = np.random.default_rng(4)
    source = random.normal(size=(7, 9)).astype(np.float32)
    target = random.normal(size=(17, 9)).astype(np.float32)
    query = source / np.linalg.norm(source, axis=1, keepdims=True)
    keys = target / np.linalg.norm(target, axis=1, keepdims=True)
    expected = np.argsort(-(query @ keys.T), axis=1)[:, :5]
    actual = nearest_neighbors(source, target, 5, query_chunk=3, target_chunk=4)
    np.testing.assert_array_equal(actual, expected)


def test_retrieval_ties_keep_target_order_and_clip_topk():
    source = np.ones((2, 3), np.float32)
    target = np.ones((3, 3), np.float32)
    actual = nearest_neighbors(source, target, 8, query_chunk=1, target_chunk=1)
    np.testing.assert_array_equal(actual, [[0, 1, 2], [0, 1, 2]])


def test_retrieval_rejects_nonfinite():
    with pytest.raises(ValueError, match="NaN"):
        nearest_neighbors(np.array([[np.nan]], np.float32), np.ones((1, 1), np.float32), 1)
