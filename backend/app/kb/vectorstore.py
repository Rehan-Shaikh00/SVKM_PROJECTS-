"""Vector stores.

* `LocalVectorStore` — in-process numpy matrix persisted to `data/index/`.
  Zero infrastructure, ~sub-millisecond search for tens of thousands of chunks.
  Perfect for the pilot and for CI.
* `PgVectorStore` — same Postgres that holds the KB records (production default;
  one less moving part, transactional with the KB tables).
* `QdrantVectorStore` — REST client, no extra dependency.

All three expose the same interface so `HybridRetriever` does not care.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings

logger = logging.getLogger("nims.vectorstore")


def _json_default(obj: Any) -> Any:
    """Metadata may carry datetimes from the KB rows; keep persist() robust."""
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "value"):  # enums
        return obj.value
    return str(obj)


@dataclass
class VectorHit:
    chunk_id: str
    score: float
    record_id: str = ""
    category: str = ""
    language: str = "en-IN"
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    name = "base"

    @abstractmethod
    async def upsert(self, items: list[dict[str, Any]]) -> int: ...

    @abstractmethod
    async def search(
        self, vector: np.ndarray, top_k: int = 6, filters: dict[str, Any] | None = None
    ) -> list[VectorHit]: ...

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int: ...

    @abstractmethod
    async def delete_by_record(self, record_ids: list[str]) -> int: ...

    @abstractmethod
    async def count(self) -> int: ...

    async def ids(self) -> list[str]:
        """Chunk ids currently held in the store.

        Optional. Stores that cannot enumerate cheaply return ``[]``, and the
        caller then skips orphan reconciliation rather than deleting everything.
        """
        return []

    async def stats(self) -> dict[str, Any]:  # pragma: no cover
        return {"store": self.name, "count": await self.count()}


# --------------------------------------------------------------------------- #
# Local (numpy)
# --------------------------------------------------------------------------- #


class LocalVectorStore(VectorStore):
    name = "local"

    def __init__(self, dimensions: int, path: Path | None = None) -> None:
        self.dimensions = dimensions
        self.path = path or (settings.index_dir / "vectors")
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._meta: list[dict[str, Any]] = []
        self._matrix: np.ndarray = np.zeros((0, dimensions), dtype=np.float32)
        self._dirty = False
        self._lock = asyncio.Lock()

    # -- persistence ------------------------------------------------------- #
    async def load(self) -> None:
        npz_path = Path(str(self.path) + ".npz")
        meta_path = Path(str(self.path) + ".jsonl")
        if not npz_path.exists() or not meta_path.exists():
            return
        try:
            # numpy + JSONL IO is blocking; keep it off the event loop so a cold
            # index load cannot stall live calls that are already in flight.
            loaded = await asyncio.to_thread(self._read, npz_path, meta_path)
        except Exception as exc:
            logger.warning("could not load local vector index: %s", exc)
            return
        if loaded is None:
            return
        self._matrix, self._ids, self._index, self._meta = loaded
        logger.info("local vector store loaded: %d vectors", len(self._ids))

    def _read(
        self, npz_path: Path, meta_path: Path
    ) -> tuple[np.ndarray, list[str], dict[str, int], list[dict[str, Any]]] | None:
        data = np.load(npz_path)
        matrix = data["vectors"].astype(np.float32)
        if matrix.shape[1] != self.dimensions:
            logger.warning(
                "vector dimension change (%s -> %s); rebuilding index",
                matrix.shape[1], self.dimensions,
            )
            return None
        ids: list[str] = []
        index: dict[str, int] = {}
        meta: list[dict[str, Any]] = []
        with meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                index[row["id"]] = len(ids)
                ids.append(row["id"])
                meta.append(row.get("meta", {}))
        return matrix, ids, index, meta

    async def persist(self) -> None:
        async with self._lock:
            if not self._dirty:
                return
            await asyncio.to_thread(self._write)
            self._dirty = False

    def _write(self) -> None:  # pragma: no cover - IO
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.path) + ".npz", vectors=self._matrix)
        with Path(str(self.path) + ".jsonl").open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(
                        {"id": chunk_id, "meta": meta},
                        ensure_ascii=False,
                        default=_json_default,
                    )
                    + "\n" for chunk_id, meta in zip(self._ids, self._meta))

    # -- mutation ---------------------------------------------------------- #
    async def upsert(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        async with self._lock:
            vectors = np.asarray(
                [item["vector"] for item in items], dtype=np.float32
            ).reshape(len(items), -1)
            if vectors.shape[1] != self.dimensions:
                raise ValueError(
                    f"vector dim {vectors.shape[1]} != store dim {self.dimensions}"
                )
            for item, vector in zip(items, vectors):
                chunk_id = item["id"]
                meta = {
                    "record_id": item.get("record_id", ""),
                    "category": item.get("category", ""),
                    "language": item.get("language", "en-IN"),
                    "verified": bool(item.get("verified", False)),
                    "status": item.get("status", "published"),
                    "academic_year": item.get("academic_year"),
                    "title": item.get("title", ""),
                    "updated_at": item.get("updated_at"),
                }
                if chunk_id in self._index:
                    position = self._index[chunk_id]
                    self._matrix[position] = vector
                    self._meta[position] = meta
                else:
                    if self._matrix.size == 0:
                        self._matrix = vector.reshape(1, -1)
                    else:
                        self._matrix = np.vstack([self._matrix, vector.reshape(1, -1)])
                    self._index[chunk_id] = len(self._ids)
                    self._ids.append(chunk_id)
                    self._meta.append(meta)
            self._dirty = True
            return len(items)

    async def ids(self) -> list[str]:
        async with self._lock:
            return list(self._ids)

    async def delete(self, chunk_ids: list[str]) -> int:
        async with self._lock:
            positions = [self._index[c] for c in chunk_ids if c in self._index]
            if not positions:
                return 0
            await self._drop(positions)
            return len(positions)

    async def delete_by_record(self, record_ids: list[str]) -> int:
        wanted = set(record_ids)
        async with self._lock:
            positions = [
                i for i, meta in enumerate(self._meta) if meta.get("record_id") in wanted
            ]
            if not positions:
                return 0
            await self._drop(positions)
            return len(positions)

    async def _drop(self, positions: list[int]) -> None:
        keep = np.ones(len(self._ids), dtype=bool)
        keep[positions] = False
        self._matrix = self._matrix[keep]
        self._ids = [i for i, k in zip(self._ids, keep) if k]
        self._meta = [m for m, k in zip(self._meta, keep) if k]
        self._index = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        self._dirty = True

    async def count(self) -> int:
        return len(self._ids)

    async def stats(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_language: dict[str, int] = {}
        for meta in self._meta:
            by_category[meta.get("category", "?")] = by_category.get(meta.get("category", "?"), 0) + 1
            by_language[meta.get("language", "?")] = by_language.get(meta.get("language", "?"), 0) + 1
        return {
            "store": self.name,
            "count": len(self._ids),
            "dimensions": self.dimensions,
            "by_category": by_category,
            "by_language": by_language,
            "path": str(self.path),
        }

    # -- search ------------------------------------------------------------ #
    async def search(
        self, vector: np.ndarray, top_k: int = 6, filters: dict[str, Any] | None = None
    ) -> list[VectorHit]:
        if self._matrix.size == 0:
            return []
        started = time.perf_counter()
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dimensions:
            raise ValueError("query dimension mismatch")
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        # cosine similarity (matrix rows are stored L2-normalised)
        scores = self._matrix @ query
        mask = np.ones(len(self._ids), dtype=bool)
        if filters:
            for key, allowed in filters.items():
                if allowed in (None, [], ()):
                    continue
                values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
                column = np.array(
                    [meta.get(key) in values for meta in self._meta], dtype=bool
                )
                mask &= column
        if not mask.any():
            return []
        masked = np.where(mask, scores, -np.inf)
        k = min(top_k * 3, len(self._ids))  # over-fetch: reranker narrows it down
        top_positions = np.argpartition(-masked, k - 1)[:k] if k < len(masked) else np.argsort(-masked)
        top_positions = top_positions[np.argsort(-masked[top_positions])]

        hits = [
            VectorHit(
                chunk_id=self._ids[position],
                score=float(masked[position]),
                record_id=str(self._meta[position].get("record_id", "")),
                category=str(self._meta[position].get("category", "")),
                language=str(self._meta[position].get("language", "en-IN")),
                verified=bool(self._meta[position].get("verified", False)),
                metadata=dict(self._meta[position]),
            )
            for position in top_positions
            if np.isfinite(masked[position])
        ][: top_k * 2]
        logger.debug("local vector search %d hits in %.1f ms", len(hits),
                     (time.perf_counter() - started) * 1000)
        return hits


# --------------------------------------------------------------------------- #
# pgvector
# --------------------------------------------------------------------------- #


class PgVectorStore(VectorStore):  # pragma: no cover - requires Postgres
    name = "pgvector"

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    async def ensure_schema(self) -> None:
        from sqlalchemy import text

        from ..db import engine

        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS kb_embeddings (
                        id TEXT PRIMARY KEY,
                        record_id TEXT NOT NULL,
                        category TEXT,
                        language TEXT,
                        verified BOOLEAN DEFAULT FALSE,
                        status TEXT DEFAULT 'published',
                        academic_year TEXT,
                        title TEXT,
                        updated_at TIMESTAMP,
                        embedding vector({self.dimensions})
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS kb_embeddings_vector_idx
                    ON kb_embeddings USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )
            await conn.execute(
                text("CREATE INDEX IF NOT EXISTS kb_embeddings_record_idx ON kb_embeddings(record_id)")
            )

    async def upsert(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        from sqlalchemy import text

        from ..db import engine

        async with engine.begin() as conn:
            for item in items:
                await conn.execute(
                    text(
                        """
                        INSERT INTO kb_embeddings
                          (id, record_id, category, language, verified, status,
                           academic_year, title, updated_at, embedding)
                        VALUES (:id, :record_id, :category, :language, :verified, :status,
                                :academic_year, :title, :updated_at, CAST(:embedding AS vector))
                        ON CONFLICT (id) DO UPDATE SET
                          record_id = EXCLUDED.record_id,
                          category = EXCLUDED.category,
                          language = EXCLUDED.language,
                          verified = EXCLUDED.verified,
                          status = EXCLUDED.status,
                          academic_year = EXCLUDED.academic_year,
                          title = EXCLUDED.title,
                          updated_at = EXCLUDED.updated_at,
                          embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "id": item["id"],
                        "record_id": item.get("record_id", ""),
                        "category": item.get("category", ""),
                        "language": item.get("language", "en-IN"),
                        "verified": bool(item.get("verified", False)),
                        "status": item.get("status", "published"),
                        "academic_year": item.get("academic_year"),
                        "title": item.get("title", ""),
                        "updated_at": item.get("updated_at"),
                        "embedding": "[" + ",".join(f"{v:.6f}" for v in item["vector"]) + "]",
                    },
                )
        return len(items)

    async def search(
        self, vector: np.ndarray, top_k: int = 6, filters: dict[str, Any] | None = None
    ) -> list[VectorHit]:
        from sqlalchemy import text

        from ..db import engine

        clauses: list[str] = []
        params: dict[str, Any] = {
            "embedding": "[" + ",".join(f"{v:.6f}" for v in np.asarray(vector).reshape(-1)) + "]",
            "limit": top_k * 2,
        }
        for index, (key, allowed) in enumerate((filters or {}).items()):
            if allowed in (None, [], ()):
                continue
            values = list(allowed) if isinstance(allowed, (list, tuple, set)) else [allowed]
            params[f"p{index}"] = tuple(values)
            clauses.append(f"{key} = ANY(:p{index})")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT id, record_id, category, language, verified,
                               1 - (embedding <=> CAST(:embedding AS vector)) AS score
                        FROM kb_embeddings {where}
                        ORDER BY embedding <=> CAST(:embedding AS vector)
                        LIMIT :limit
                        """
                    ),
                    params,
                )
            ).mappings().all()
        return [
            VectorHit(
                chunk_id=row["id"], score=float(row["score"]), record_id=row["record_id"],
                category=row["category"] or "", language=row["language"] or "en-IN",
                verified=bool(row["verified"]),
            )
            for row in rows
        ]

    async def delete(self, chunk_ids: list[str]) -> int:
        from sqlalchemy import text

        from ..db import engine

        async with engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM kb_embeddings WHERE id = ANY(:ids)"), {"ids": tuple(chunk_ids)}
            )
            return int(result.rowcount or 0)

    async def delete_by_record(self, record_ids: list[str]) -> int:
        from sqlalchemy import text

        from ..db import engine

        async with engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM kb_embeddings WHERE record_id = ANY(:ids)"),
                {"ids": tuple(record_ids)},
            )
            return int(result.rowcount or 0)

    async def count(self) -> int:
        from sqlalchemy import text

        from ..db import engine

        async with engine.connect() as conn:
            row = (await conn.execute(text("SELECT count(*) FROM kb_embeddings"))).scalar()
            return int(row or 0)


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #


class QdrantVectorStore(VectorStore):  # pragma: no cover - network dependent
    name = "qdrant"
    COLLECTION = "nims_kb"

    def __init__(self, dimensions: int, url: str | None = None) -> None:
        self.dimensions = dimensions
        self.url = (url or settings.qdrant_url).rstrip("/")

    async def ensure_schema(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{self.url}/collections/{self.COLLECTION}")
            if resp.status_code == 404:
                await client.put(
                    f"{self.url}/collections/{self.COLLECTION}",
                    json={
                        "vectors": {"size": self.dimensions, "distance": "Cosine"},
                        "hnsw_config": {"m": 16, "ef_construct": 128},
                    },
                )

    async def upsert(self, items: list[dict[str, Any]]) -> int:
        import httpx

        points = [
            {
                "id": abs(hash(item["id"])) % (2**63),
                "vector": [float(v) for v in item["vector"]],
                "payload": {
                    "chunk_id": item["id"],
                    "record_id": item.get("record_id"),
                    "category": item.get("category"),
                    "language": item.get("language"),
                    "verified": bool(item.get("verified")),
                    "status": item.get("status", "published"),
                },
            }
            for item in items
        ]
        async with httpx.AsyncClient(timeout=30) as client:
            await client.put(
                f"{self.url}/collections/{self.COLLECTION}/points", json={"points": points}
            )
        return len(items)

    async def search(
        self, vector: np.ndarray, top_k: int = 6, filters: dict[str, Any] | None = None
    ) -> list[VectorHit]:
        import httpx

        must = []
        for key, allowed in (filters or {}).items():
            if allowed in (None, [], ()):
                continue
            values = list(allowed) if isinstance(allowed, (list, tuple, set)) else [allowed]
            must.append({"key": key, "match": {"any": values}})
        body: dict[str, Any] = {
            "vector": [float(v) for v in np.asarray(vector).reshape(-1)],
            "limit": top_k * 2,
            "with_payload": True,
        }
        if must:
            body["filter"] = {"must": must}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{self.url}/collections/{self.COLLECTION}/points/search", json=body
            )
            resp.raise_for_status()
            payload = resp.json().get("result", [])
        hits = []
        for item in payload:
            data = item.get("payload", {})
            hits.append(
                VectorHit(
                    chunk_id=str(data.get("chunk_id")),
                    score=float(item.get("score", 0.0)),
                    record_id=str(data.get("record_id") or ""),
                    category=str(data.get("category") or ""),
                    language=str(data.get("language") or "en-IN"),
                    verified=bool(data.get("verified")),
                    metadata=data,
                )
            )
        return hits

    async def delete(self, chunk_ids: list[str]) -> int:  # pragma: no cover
        import httpx

        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(
                f"{self.url}/collections/{self.COLLECTION}/points/delete",
                json={"filter": {"must": [{"key": "chunk_id", "match": {"any": chunk_ids}}]}},
            )
        return len(chunk_ids)

    async def delete_by_record(self, record_ids: list[str]) -> int:  # pragma: no cover
        import httpx

        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(
                f"{self.url}/collections/{self.COLLECTION}/points/delete",
                json={"filter": {"must": [{"key": "record_id", "match": {"any": record_ids}}]}},
            )
        return len(record_ids)

    async def count(self) -> int:
        import httpx

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{self.url}/collections/{self.COLLECTION}")
            if resp.status_code != 200:
                return 0
            return int(resp.json().get("result", {}).get("points_count", 0))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


async def build_vector_store(dimensions: int) -> VectorStore:
    backend = (settings.vector_store or "local").lower()
    if backend == "pgvector":
        from ..db import is_postgres

        if not is_postgres():
            logger.warning("VECTOR_STORE=pgvector but DATABASE_URL is not Postgres; using local")
            store: VectorStore = LocalVectorStore(dimensions)
            await store.load()
            return store
        store = PgVectorStore(dimensions)
        await store.ensure_schema()  # type: ignore[assignment]
        return store
    if backend == "qdrant":
        store = QdrantVectorStore(dimensions)
        try:
            await store.ensure_schema()  # type: ignore[assignment]
            return store
        except Exception as exc:
            logger.warning("qdrant unavailable (%s); falling back to local store", exc)
    local = LocalVectorStore(dimensions)
    await local.load()
    return local
