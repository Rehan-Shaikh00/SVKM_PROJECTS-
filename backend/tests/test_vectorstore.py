"""Local vector store: mutation, persistence and orphan reconciliation.

The retriever prunes embeddings whose chunk no longer exists in the database;
without that the index drifts (it once held 156 vectors for 155 chunks) and a
stale vector can outrank a live one.

Vectors here are one-hot, so cosine similarity is exactly 1 for a matching id
and 0 otherwise — every assertion is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.kb.vectorstore import LocalVectorStore

DIM = 8


def _one_hot(i: int) -> list[float]:
    vector = [0.0] * DIM
    vector[i % DIM] = 1.0
    return vector


def _query(i: int) -> np.ndarray:
    return np.array(_one_hot(i), dtype=np.float32)


def _items(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"c{i}",
            "vector": _one_hot(i),
            "record_id": f"r{i % 2}",
            "category": "course",
            "language": "en-IN",
            "verified": i % 2 == 0,
            "title": f"chunk {i}",
        }
        for i in range(count)
    ]


async def test_upsert_count_and_ids(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    assert await store.upsert(_items(3)) == 3
    assert sorted(await store.ids()) == ["c0", "c1", "c2"]
    assert await store.count() == 3


async def test_empty_upsert_is_a_noop(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    assert await store.upsert([]) == 0
    assert await store.ids() == []


async def test_reupsert_replaces_in_place(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(3))
    await store.upsert(_items(3))
    assert await store.count() == 3, "an update must not duplicate the vector"


async def test_search_finds_the_matching_chunk(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(4))
    hits = await store.search(_query(2), top_k=1)
    assert hits and hits[0].chunk_id == "c2"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


async def test_delete_removes_only_the_requested_ids(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(3))
    assert await store.delete(["c1"]) == 1
    assert sorted(await store.ids()) == ["c0", "c2"]
    assert await store.count() == 2


async def test_delete_unknown_ids_is_a_noop(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(2))
    assert await store.delete(["does-not-exist"]) == 0
    assert await store.count() == 2


async def test_delete_by_record(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(4))  # r0: c0,c2 / r1: c1,c3
    assert await store.delete_by_record(["r0"]) == 2
    assert sorted(await store.ids()) == ["c1", "c3"]


async def test_orphan_reconciliation(tmp_index) -> None:
    """Exactly what retriever.load_from_db does after reloading chunks."""
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(3))

    live_chunk_ids = {"c0", "c2"}  # c1 was deleted from the knowledge base
    orphans = [cid for cid in await store.ids() if cid not in live_chunk_ids]
    assert orphans == ["c1"]
    assert await store.delete(orphans) == 1
    assert sorted(await store.ids()) == ["c0", "c2"]


async def test_search_is_correct_after_a_delete(tmp_index) -> None:
    """Deleting must not corrupt the id -> row position index."""
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(4))
    await store.delete(["c2"])

    survivors = await store.search(_query(3), top_k=5)
    assert all(hit.chunk_id != "c2" for hit in survivors), "deleted id resurfaced"

    still_exact = await store.search(_query(1), top_k=1)
    assert still_exact[0].chunk_id == "c1"
    assert still_exact[0].score == pytest.approx(1.0, abs=1e-5)


async def test_persist_and_reload_roundtrip(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.upsert(_items(3))
    await store.persist()

    reloaded = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await reloaded.load()
    assert sorted(await reloaded.ids()) == ["c0", "c1", "c2"]

    hits = await reloaded.search(_query(1), top_k=1)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


async def test_load_without_files_is_safe(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    await store.load()  # nothing on disk yet
    assert await store.count() == 0
    assert await store.search(_query(0), top_k=3) == []


async def test_dimension_mismatch_is_rejected(tmp_index) -> None:
    store = LocalVectorStore(dimensions=DIM, path=tmp_index)
    with pytest.raises(ValueError, match="vector dim"):
        await store.upsert([{"id": "bad", "vector": [1.0, 2.0]}])
