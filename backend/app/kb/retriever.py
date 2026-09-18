"""Hybrid retrieval: BM25 + dense vectors, fused with reciprocal rank fusion.

Why hybrid? Course and fee questions are dominated by *exact* tokens ("B.Tech
CSE", "NEET", "₹1.5 lakh") where lexical search beats embeddings, while
paraphrases ("how much do I have to pay for the computer engineering degree")
need semantics. RRF fusion gets both without a trained reranker, and the whole
thing runs in a few milliseconds.

Also decides **grounding**: if the best fused score is below the threshold the
answer is "not in the knowledge base" and the call escalates instead of the LLM
inventing a fee.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai.intents import IntentResult, detect_intent
from ..config import settings
from .embeddings import Embedder, expand, resolve_embedder
from .vectorstore import VectorStore, build_vector_store

logger = logging.getLogger("nims.retriever")

RRF_K = 60


@dataclass
class ChunkDoc:
    chunk_id: str
    record_id: str
    text: str
    title: str
    category: str
    subcategory: str | None
    language: str
    verified: bool
    status: str
    academic_year: str | None
    source: str | None
    source_uri: str | None
    structured: dict[str, Any]
    updated_at: Any | None
    slug: str = ""
    tokens: list[str] = field(default_factory=list)
    length: int = 0

    @property
    def is_stale(self) -> bool:
        if not self.updated_at:
            return False
        from datetime import datetime

        reference = self.updated_at
        if isinstance(reference, datetime):
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - reference).days
            return age_days > settings.kb_staleness_days
        return False

    @property
    def citation(self) -> str:
        year = f" ({self.academic_year})" if self.academic_year else ""
        flag = "" if self.verified else " [unverified]"
        return f"{self.title}{year} — {self.category.replace('_', ' ')}{flag}"


_TOKEN_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}

#: Degree names, folded for comparison. Devanagari is kept so a Marathi alias
#: ("\u090f\u092e\u092c\u0940\u090f") can match a Marathi query; dots, spaces and punctuation are
#: dropped so "B.Com" == "bcom" and "BBA LL.B." == "bballb".
_PROGRAMME_FOLD_RE = re.compile(r"[^0-9a-z\u0900-\u097f]+")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")


def _fold_programme(value: str) -> str:
    return _PROGRAMME_FOLD_RE.sub("", str(value).lower())


def _token_present(haystack: str, token: str) -> bool:
    """Whole-token match. Substring matching is unsafe here: the specialisation
    list contains short acronyms ("it", "ai", "ml", "hr") that would otherwise
    match inside ordinary words such as "with" or "digital"."""
    token = (token or "").strip().lower()
    if not token or not haystack:
        return False
    pattern = _TOKEN_BOUNDARY_CACHE.get(token)
    if pattern is None:
        pattern = re.compile(rf"(?<![\w]){re.escape(token)}(?![\w])")
        _TOKEN_BOUNDARY_CACHE[token] = pattern
    return pattern.search(haystack) is not None


#: Intra-word punctuation is dropped while word boundaries survive, so an alias
#: recorded as "wifi" matches a caller who says "Wi-Fi", and "btech" matches both
#: "B.Tech" and "B Tech". Whatever is left collapses to single spaces.
_INTRA_WORD_RE = re.compile(r"[.\-_/·'’&+]+")
_MATCH_TOKEN_RE = re.compile(r"[0-9a-z\u0900-\u097f]{3,}")


def _fold_for_match(value: str) -> str:
    """Lowercase and fold punctuation so spoken variants compare equal.

    Distinct from :func:`_fold_programme`, which also removes spaces: that is
    right for degree codes but would let a short alias match inside an ordinary
    word, so alias and title-term matching keeps the boundaries.
    """
    folded = _INTRA_WORD_RE.sub("", str(value).lower())
    return _PROGRAMME_FOLD_RE.sub(" ", folded).strip()


@dataclass
class RetrievedChunk:
    chunk_id: str
    record_id: str
    text: str
    title: str
    category: str
    language: str
    verified: bool
    academic_year: str | None
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0
    dense_rank: int = 0
    lexical_rank: int = 0
    source: str | None = None
    source_uri: str | None = None
    structured: dict[str, Any] = field(default_factory=dict)
    updated_at: Any | None = None
    citation: str = ""
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    items: list[RetrievedChunk]
    intent: IntentResult
    latency_ms: float = 0.0
    best_score: float = 0.0
    grounded: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def citations(self) -> list[dict[str, Any]]:
        return [
            {
                "record_id": item.record_id,
                "chunk_id": item.chunk_id,
                "title": item.title,
                "category": item.category,
                "score": round(item.score, 4),
                "verified": item.verified,
                "academic_year": item.academic_year,
                "source": item.source,
                "citation": item.citation,
            }
            for item in self.items
        ]

    def context_block(self, max_items: int | None = None, max_chars: int = 3600) -> str:
        """The text handed to the LLM as grounded context."""
        lines: list[str] = []
        used = 0
        for index, item in enumerate(self.items[: max_items or len(self.items)], start=1):
            verified = "verified" if item.verified else "UNVERIFIED"
            year = item.academic_year or "n/a"
            block = (
                f"[{index}] {item.title} | category={item.category} | "
                f"academic_year={year} | {verified} | source={item.source or 'internal'}\n"
                f"{item.text.strip()}"
            )
            if used + len(block) > max_chars and lines:
                break
            lines.append(block)
            used += len(block)
        return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


class BM25Index:
    """Okapi BM25 over expanded tokens (synonym-group aware)."""

    def __init__(self, k1: float = 1.6, b: float = 0.72) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_len = 0.0
        self.doc_lengths: dict[str, int] = {}
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: dict[str, dict[str, int]] = {}
        self.doc_ids: list[str] = []

    def build(self, docs: list[tuple[str, list[str]]]) -> None:
        self.doc_count = len(docs)
        self.doc_lengths = {}
        self.doc_freqs = {}
        self.term_freqs = {}
        self.doc_ids = []
        total_length = 0
        for doc_id, tokens in docs:
            self.doc_ids.append(doc_id)
            length = len(tokens) or 1
            self.doc_lengths[doc_id] = length
            total_length += length
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_freqs[doc_id] = counts
            for token in counts:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.avg_len = (total_length / self.doc_count) if self.doc_count else 1.0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        if doc_id in self.doc_lengths:
            self.remove(doc_id)
        self.doc_ids.append(doc_id)
        length = len(tokens) or 1
        self.doc_lengths[doc_id] = length
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        self.term_freqs[doc_id] = counts
        for token in counts:
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.doc_count = len(self.doc_ids)
        total = sum(self.doc_lengths.values())
        self.avg_len = (total / self.doc_count) if self.doc_count else 1.0

    def remove(self, doc_id: str) -> None:
        if doc_id not in self.doc_lengths:
            return
        counts = self.term_freqs.pop(doc_id, {})
        for token in counts:
            if token in self.doc_freqs:
                self.doc_freqs[token] -= 1
                if self.doc_freqs[token] <= 0:
                    self.doc_freqs.pop(token, None)
        self.doc_lengths.pop(doc_id, None)
        if doc_id in self.doc_ids:
            self.doc_ids.remove(doc_id)
        self.doc_count = len(self.doc_ids)
        total = sum(self.doc_lengths.values())
        self.avg_len = (total / self.doc_count) if self.doc_count else 1.0

    def score(self, query_tokens: list[str], top_k: int = 20) -> list[tuple[str, float]]:
        if not self.doc_count:
            return []
        unique_tokens = list(dict.fromkeys(query_tokens))
        scores: dict[str, float] = {}
        for token in unique_tokens:
            df = self.doc_freqs.get(token)
            if not df:
                continue
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for doc_id in self.doc_ids:
                tf = self.term_freqs.get(doc_id, {}).get(token)
                if not tf:
                    continue
                length = self.doc_lengths.get(doc_id, 1) or 1
                denominator = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                value = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / denominator
                scores[doc_id] = value
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[: max(top_k, top_k)]


# --------------------------------------------------------------------------- #
# Hybrid retriever
# --------------------------------------------------------------------------- #


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        min_score: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.min_score = min_score if min_score is not None else settings.retrieval_min_score
        self.top_k = top_k or settings.retrieval_top_k
        self.chunks: dict[str, ChunkDoc] = {}
        self.bm25 = BM25Index()
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()

    # -- lifecycle --------------------------------------------------------- #
    @property
    def embedder_dimensions(self) -> int:
        return self.embedder.dimensions if self.embedder else settings.vector_dimensions

    async def initialise(self) -> None:
        if self.embedder is None:
            self.embedder, info = resolve_embedder()
            logger.info("embeddings provider=%s degraded=%s", info["active"], info["degraded"])
        if self.store is None:
            self.store = await build_vector_store(self.embedder.dimensions)
        self._ready.set()

    async def wait_ready(self, timeout: float = 30) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except TimeoutError:  # pragma: no cover
            logger.warning("retriever not ready after %.0fs", timeout)

    async def load_from_db(self, session: AsyncSession) -> int:
        """(Re)build the in-memory lexical index from published chunks."""
        from ..models import KBChunk, KBRecord

        started = time.perf_counter()
        async with self._lock:
            rows = (
                await session.execute(
                    select(KBChunk, KBRecord)
                    .join(KBRecord, KBChunk.record_id == KBRecord.id)
                    .where(KBRecord.status == "published")
                )
            ).all()
            docs: list[tuple[str, list[str]]] = []
            self.chunks.clear()
            for chunk, record in rows:
                doc = ChunkDoc(
                    chunk_id=chunk.id,
                    record_id=record.id,
                    text=chunk.text,
                    title=record.title,
                    category=record.category,
                    subcategory=record.subcategory,
                    language=record.language,
                    verified=bool(record.verified),
                    status=record.status,
                    academic_year=record.academic_year,
                    source=record.source,
                    source_uri=record.source_uri,
                    structured=record.structured or {},
                    updated_at=record.updated_at,
                    slug=record.slug,
                )
                doc.tokens = expand(chunk.text)
                doc.length = len(doc.tokens)
                self.chunks[chunk.id] = doc
                docs.append((chunk.id, doc.tokens))
            self.bm25.build(docs)

        # Reconcile the vector store. Embeddings for chunks that were deleted or
        # unpublished otherwise linger forever, and a stale vector can outrank a
        # live one (this is how the index drifted to 156 vectors for 155 chunks).
        try:
            known = await self.store.ids()
            if known:
                live = set(self.chunks)
                orphans = [cid for cid in known if cid not in live]
                if orphans:
                    removed = await self.store.delete(orphans)
                    logger.info("vector store pruned %d orphaned embedding(s)", removed)
        except Exception as exc:  # pragma: no cover - store specific
            logger.warning("vector orphan reconciliation skipped: %s", exc)

        elapsed = (time.perf_counter() - started) * 1000
        logger.info("retriever loaded %d chunks in %.0f ms", len(docs), elapsed)
        return len(docs)

    async def upsert_chunks(
        self, session: AsyncSession, records: list[dict[str, Any]]
    ) -> int:
        """Embed + store + index a set of freshly written chunk rows."""
        await self.wait_ready()
        assert self.embedder and self.store
        if not records:
            return 0
        vectors = await self.embedder.embed([r["text"] for r in records])
        items = []
        for record, vector in zip(records, vectors):
            items.append(
                {
                    "id": record["id"],
                    "record_id": record["record_id"],
                    "category": record.get("category", ""),
                    "language": record.get("language", "en-IN"),
                    "verified": record.get("verified", False),
                    "status": record.get("status", "published"),
                    "academic_year": record.get("academic_year"),
                    "title": record.get("title", ""),
                    "updated_at": record.get("updated_at"),
                    "vector": vector,
                }
            )
        await self.store.upsert(items)
        for record, vector in zip(records, vectors):
            doc = ChunkDoc(
                chunk_id=record["id"],
                record_id=record["record_id"],
                text=record["text"],
                title=record.get("title", ""),
                category=record.get("category", ""),
                subcategory=record.get("subcategory"),
                language=record.get("language", "en-IN"),
                verified=bool(record.get("verified", False)),
                status=record.get("status", "published"),
                academic_year=record.get("academic_year"),
                source=record.get("source"),
                source_uri=record.get("source_uri"),
                structured=record.get("structured") or {},
                updated_at=record.get("updated_at"),
                slug=record.get("slug", ""),
            )
            doc.tokens = expand(record["text"])
            doc.length = len(doc.tokens)
            self.chunks[record["id"]] = doc
            self.bm25.add(record["id"], doc.tokens)
        if hasattr(self.store, "persist"):
            await self.store.persist()  # type: ignore[attr-defined]
        return len(records)

    async def delete_record(self, record_id: str) -> None:
        assert self.store
        chunk_ids = [cid for cid, doc in self.chunks.items() if doc.record_id == record_id]
        for chunk_id in chunk_ids:
            self.bm25.remove(chunk_id)
            self.chunks.pop(chunk_id, None)
        if chunk_ids:
            await self.store.delete(chunk_ids)
        await self.store.delete_by_record([record_id])
        if hasattr(self.store, "persist"):
            await self.store.persist()  # type: ignore[attr-defined]

    async def count(self) -> int:
        return len(self.chunks)

    # -- search ------------------------------------------------------------ #
    async def search(
        self,
        query: str,
        *,
        language: str = "en-IN",
        intent: IntentResult | None = None,
        top_k: int | None = None,
        categories: tuple[str, ...] | None = None,
        verified_only: bool = False,
        course_tokens: list[str] | None = None,
    ) -> RetrievalResult:
        await self.wait_ready()
        assert self.embedder and self.store
        started = time.perf_counter()
        top_k = top_k or self.top_k
        intent = intent or detect_intent(query)
        if categories is None:
            categories = intent.categories or None
        course_tokens = course_tokens or intent.course_tokens

        if not self.chunks:
            return RetrievalResult(
                query=query, items=[], intent=intent,
                latency_ms=(time.perf_counter() - started) * 1000,
                debug={"reason": "index empty"},
            )

        filters: dict[str, Any] = {}
        if verified_only:
            filters["verified"] = [True]
        # NOTE: category filtering is applied as a *boost*, not a hard filter —
        # hard filters silently miss cross-category answers (a fee question is
        # often answered by the course record).

        query_vector = await self.embedder.embed_one(query)
        dense_hits = await self.store.search(
            query_vector, top_k=max(top_k * 4, 24), filters=filters or None
        )
        lexical_hits = self.bm25.score(expand(query), top_k=max(top_k * 4, 24))

        fused = self._fuse(dense_hits, lexical_hits)
        reranked = self._rerank(
            fused,
            intent=intent,
            language=language,
            categories=categories,
            course_tokens=course_tokens,
            query=query,
        )
        items = reranked[:top_k]
        best_score = items[0].score if items else 0.0
        latency = (time.perf_counter() - started) * 1000
        grounded = best_score >= self.min_score and bool(items)

        return RetrievalResult(
            query=query,
            items=items,
            intent=intent,
            latency_ms=latency,
            best_score=round(best_score, 4),
            grounded=grounded,
            filters=filters,
            debug={
                "dense_candidates": len(dense_hits),
                "lexical_candidates": len(lexical_hits),
                "fused": len(fused),
                "intent": intent.to_dict(),
                "index_size": len(self.chunks),
                "min_score": self.min_score,
                "embedding_provider": self.embedder.name,
                "vector_store": self.store.name,
            },
        )

    # -- fusion ------------------------------------------------------------ #
    def _fuse(self, dense_hits: list[Any], lexical_hits: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
        fused: dict[str, dict[str, float]] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            entry = fused.setdefault(hit.chunk_id, {"rrf": 0.0, "dense": 0.0, "lexical": 0.0,
                                                     "dense_rank": 999, "lexical_rank": 999})
            entry["rrf"] += 1.0 / (RRF_K + rank)
            entry["dense"] = float(hit.score)
            entry["dense_rank"] = rank
        for rank, (chunk_id, score) in enumerate(lexical_hits, start=1):
            entry = fused.setdefault(chunk_id, {"rrf": 0.0, "dense": 0.0, "lexical": 0.0,
                                                "dense_rank": 999, "lexical_rank": 999})
            entry["rrf"] += 1.0 / (RRF_K + rank)
            entry["lexical"] = float(score)
            entry["lexical_rank"] = rank
        # normalise the RRF component so scores stay in a readable 0..1 range
        for entry in fused.values():
            entry["rrf_norm"] = entry["rrf"] / (2.0 / (RRF_K + 1))
        return fused

    def _rerank(
        self,
        fused: dict[str, dict[str, float]],
        *,
        intent: IntentResult,
        language: str,
        categories: tuple[str, ...] | None,
        course_tokens: list[str],
        query: str,
    ) -> list[RetrievedChunk]:
        query_lower = query.lower()
        # Aliases live in a record's "Also known as:" chunk, which is penalised
        # below because it cannot be spoken. Credit the *record* for an alias hit
        # instead, so an exact match in the caller's own language ("प्रवेश
        # प्रक्रिया") still beats a sibling record whose prose happens to score
        # well on dense similarity.
        aliases_by_record: dict[str, list[str]] = {}
        for doc in self.chunks.values():
            body = doc.text.partition("\n")[2].lower()
            for line in body.splitlines():
                line = line.strip()
                if not line.startswith("also known as:"):
                    continue
                listing = line[len("also known as:"):].strip().rstrip(".")
                aliases_by_record.setdefault(doc.record_id, []).extend(
                    alias.strip() for alias in listing.split(",") if len(alias.strip()) >= 4
                )

        # Which of the caller's words actually *name* something in this KB?
        # Document frequency over record titles decides, so the rule keeps
        # working as the KB grows instead of leaning on a hand-kept word list:
        # "campus" sits in eight of the fifty-four record titles and says
        # nothing, "wi-fi" sits in one and says everything. A word that appears
        # in no title at all ("about", "how") counts zero and drops out, which
        # is what keeps ordinary stopwords from earning a bonus.
        query_folded = _fold_for_match(query)
        # The caller's own words, not the augmented retrieval query. augment_query
        # appends cross-script bridges and course tokens -- a Marathi question
        # arrives as "... B.Tech Engineering Computer Science CSE BTECH" -- which
        # is what lets it reach English records at all, but those glosses are not
        # what the caller said. Crediting "science" from them lifted B.Tech CSE
        # (Data Science) above the B.Tech Computer Engineering record the caller
        # had actually named in Marathi, and reported 120 seats instead of 180.
        spoken_folded = _fold_for_match((intent.text or "").strip() or query)
        title_surfaces: dict[str, str] = {}
        for doc in self.chunks.values():
            title_surfaces[doc.record_id] = _fold_for_match(doc.title)
        title_df: dict[str, int] = {}
        for surface in title_surfaces.values():
            for token in set(_MATCH_TOKEN_RE.findall(surface)):
                title_df[token] = title_df.get(token, 0) + 1
        rare_cap = max(2, round(0.06 * len(title_surfaces)))
        # Two things this bonus must stay out of. Degree codes are the intent
        # detector's business: "bba" already earns course_match and degree_exact
        # below, and paying for it a third time let the BBA course record
        # outrank the admission-dates record for "has the merit list come out for
        # BBA?", so a question the KB answers escalated to a human instead. And
        # titles only -- an abbreviation parked in a record's aliases ("cse") is
        # weaker evidence than the caller's own words; crediting it let "B.Tech
        # CSE (Data Science)" beat "B.Tech Computer Engineering" for a Marathi
        # caller who had said संगणक अभियांत्रिकी and was told 120 seats, not 180.
        claimed = {_fold_programme(token) for token in course_tokens if token}
        distinctive_terms = [
            token
            for token in dict.fromkeys(_MATCH_TOKEN_RE.findall(spoken_folded))
            if 1 <= title_df.get(token, 0) <= rare_cap
            and _fold_programme(token) not in claimed
        ]

        out: list[RetrievedChunk] = []
        for chunk_id, entry in fused.items():
            doc = self.chunks.get(chunk_id)
            if doc is None:
                continue
            score = float(entry.get("rrf_norm", 0.0))
            # blend absolute signals so a chunk with a strong lexical hit but a
            # weak RRF still surfaces (common for exact course-code matches)
            score = 0.75 * score + 0.25 * min(1.0, float(entry.get("dense", 0.0)))
            score += 0.12 * min(1.0, float(entry.get("lexical", 0.0)) / 8.0)

            signals: dict[str, Any] = {}
            if categories and doc.category in categories:
                score += 0.10
                signals["category_match"] = True
                # The first category is the caller's actual subject. A record
                # that only shares a secondary one ("facilities" for a hostel
                # question, whose intent categories are hostel+facilities) must
                # not outrank the record the question is really about.
                if doc.category == categories[0]:
                    score += 0.08
                    signals["primary_category_match"] = True
            if doc.language and doc.language == language:
                score += 0.06
                signals["language_match"] = True
            if doc.verified:
                score += 0.05
                signals["verified"] = True
            else:
                score -= 0.06
                signals["unverified"] = True
            if doc.is_stale:
                score -= 0.07
                signals["stale"] = True
            title_lower = doc.title.lower()
            title_compact = title_lower.replace(".", "").replace(" ", "")
            body_lower = doc.text.lower()
            for token in course_tokens:
                # tokens are upper-cased by the intent detector; compare folded
                compact = token.replace(".", "").lower()
                if compact and compact in title_compact:
                    score += 0.14
                    signals[f"course_match:{token}"] = True

            # Degree-prefix exactness. "What is the fee for BBA" must land on the
            # BBA record, not on "BBA LL.B. (Honours) — integrated law" — the law
            # record's facts chunk literally contains "total fee", so plain BM25
            # prefers it even though the caller did not ask about law. The prefix
            # is the title up to the first bracket/dash/comma, dot- and
            # space-folded: "BBA LL.B. (Honours) — …" -> "bballb".
            degree_prefix = re.split(r"[(\u2014\u2013,]", title_lower, maxsplit=1)[0]
            degree_prefix = degree_prefix.replace(".", "").replace(" ", "").strip()
            # A record that spells its degree out -- "Bachelor of Pharmacy
            # (B.Pharm)" -- carries the code the caller actually says inside
            # brackets, and the spoken prefix alone never matched it. So a bare
            # "how many seats in B.Pharm?" landed on "B.Pharm + MBA (Pharma Tech)",
            # a five year dual degree, and the caller was told forty seats.
            bracketed = [
                match.replace(".", "").replace(" ", "").strip().lower()
                for match in re.findall(r"\(([^)]+)\)", title_lower)
            ]
            exact_candidates = [c for c in (degree_prefix, *bracketed) if c]
            folded_tokens = [t.replace(".", "").lower() for t in course_tokens if t]
            if exact_candidates and folded_tokens:
                # Every degree the caller named must be the record's own degree.
                # "Is BBA LL.B. available here?" carries tokens for BBA *and* LL.B.,
                # and the plain BBA record matches one of them -- but it is not the
                # programme asked about, so it must not win as an exact match.
                if all(
                    any(cand == token for cand in exact_candidates)
                    for token in folded_tokens
                ):
                    score += 0.25
                    signals["degree_exact"] = degree_prefix or bracketed[0]
                elif degree_prefix and all(degree_prefix.startswith(token) for token in folded_tokens):
                    # `startswith`, not `in`: "Lateral entry to B.Tech for diploma
                    # holders" contains "btech" but is not a B.Tech programme
                    # record, and it used to outrank the entrance-exam record for
                    # "which entrance exam for B.Tech".
                    #
                    # A dual or integrated degree is a *different* programme from
                    # the one the caller named, so a prefix match alone must not
                    # outrank the record whose degree code matches exactly.
                    combined = bool(re.search(r"\+|dual|integrated", f"{degree_prefix} {title_lower}"))
                    score += 0.04 if combined else 0.18
                    signals["degree_prefix_match"] = degree_prefix

            # "We do not run that programme" records exist to intercept the exact
            # question a caller asks ("do you have MBBS?", "is B.Com there?"). The
            # intent layer tags those queries as `courses`, so without a decisive
            # signal the call goes to whichever degree record happens to score best
            # -- a Ph.D. record for MBBS, the BBA record for B.Com -- and the
            # assistant confidently describes a programme the university does not
            # offer. The `not_offered` list is the record's own declaration of what
            # it intercepts, so a token match there outranks any degree record.
            not_offered = (
                doc.structured.get("not_offered")
                if isinstance(doc.structured, dict) else None
            )
            if not_offered and folded_tokens:
                if isinstance(not_offered, str):
                    not_offered = [not_offered]
                # Match the *degree names* the record intercepts, never its prose.
                # A substring hit is far too loose: the list reads "BBA LL.B. (Hons)
                # or any law degree", and a caller asking about BBA eligibility was
                # told the university does not run BBA at all.
                targets = {
                    _fold_programme(a) for a in aliases_by_record.get(doc.record_id, ())
                }
                # `listing`, not `entry`: the enclosing loop iterates
                # `fused.items()` as `entry`, and shadowing it here turned the
                # RetrievedChunk built below into an AttributeError on a str.
                for listing in not_offered:
                    head = re.split(
                        r"[(,/]|\s+(?:or|and|including|such as|with)\s+",
                        str(listing).lower(),
                        maxsplit=1,
                    )[0]
                    targets.add(_fold_programme(head))
                    words = head.split()
                    if words:
                        # "A standalone MBA": the degree is the last word of the head
                        targets.add(_fold_programme(words[-1]))
                targets.discard("")
                hit = next(
                    (t for t in folded_tokens if _fold_programme(t) in targets), None
                )
                if hit is None:
                    # A Devanagari query yields no Latin course token, so match the
                    # folded query against the record's own Devanagari names:
                    # "\u090f\u092e\u092c\u0940\u090f \u091a\u093e \u0915\u094b\u0930\u094d\u0938 \u0906\u0939\u0947 \u0915\u093e?" has to reach the interceptor
                    # too, or the caller is told about the B.Tech + MBA dual degree
                    # instead of hearing that no standalone MBA exists here.
                    # Restricted to Devanagari targets: a folded Latin substring
                    # match has no word boundary to protect it.
                    query_folded = _fold_programme(query)
                    indic = [t for t in targets if _DEVANAGARI_RE.search(t) and len(t) >= 3]
                    if query_folded and indic:
                        hit = next((t for t in indic if t in query_folded), None)
                if hit:
                    score += 0.55
                    signals["programme_not_offered"] = hit

            # A specialisation named in the *title* is decisive. One that merely
            # appears in the body is weak evidence -- "B.Tech IT" prose says
            # "…a stronger systems orientation than the CSE programme", and that
            # must not let the IT record outrank the CSE record for a CSE query.
            title_spec = next(
                (s for s in intent.specialisations if _token_present(title_lower, s)), None
            )
            if title_spec:
                score += 0.45
                signals["specialisation_title"] = title_spec
            else:
                body_spec = next(
                    (s for s in intent.specialisations if _token_present(body_lower, s)), None
                )
                if body_spec:
                    score += 0.04
                    signals["specialisation_body"] = body_spec
            # "Also known as: …" chunks exist to widen recall, not to be spoken:
            # they are label dumps the extractive composer filters out entirely.
            # Ranking one first starves the answer (a Hindi hostel question used
            # to land on the facilities alias chunk and escalate).
            chunk_body = body_lower.partition("\n")[2].lstrip()
            if chunk_body.startswith("also known as:"):
                score -= 0.12
                signals["alias_dump"] = True
            else:
                # The longest matching alias wins the boost: "हॉस्टल की सुविधा"
                # (hostel) is a real match for "हॉस्टल की सुविधा है क्या", while
                # "सुविधा" alone (facilities) is only a fragment of it. Weighting
                # by length stops a one-word alias hijacking a specific question.
                # Folded, so an alias stored as "wifi" still matches a caller who
                # says "Wi-Fi". The old literal substring test quietly missed
                # every hyphenated, dotted or spaced variant -- which is how a
                # brand new "Campus Wi-Fi account activation" record lost to a
                # facilities record that never mentioned Wi-Fi at all.
                alias_hit = max(
                    (
                        alias
                        for alias in aliases_by_record.get(doc.record_id, ())
                        if len(_fold_for_match(alias)) >= 4
                        and _fold_for_match(alias) in query_folded
                    ),
                    key=len, default=None,
                )
                if alias_hit:
                    words = len(alias_hit.split())
                    score += min(0.30, 0.12 + 0.06 * (words - 1))
                    signals["alias_record_match"] = alias_hit
                elif distinctive_terms:
                    # No alias matched, but the caller used one of the KB's rare
                    # words and this record's own title says it back. Coarse
                    # category affinity (+0.10, plus +0.08 for the caller's
                    # primary category) must not outvote that, or staff cannot
                    # make a new record reachable simply by titling it accurately.
                    surface = title_surfaces.get(doc.record_id, "")
                    term_hits = [t for t in distinctive_terms if _token_present(surface, t)]
                    if term_hits:
                        # 0.12 a term, two terms maximum. Sized against the
                        # boosts it has to overcome: a category match (+0.10),
                        # the caller's primary category (+0.08) and the
                        # verified/unverified preference (0.11 the other way),
                        # less the lexical cap the same rare word already earns.
                        score += 0.12 * min(2, len(term_hits))
                        signals["title_term_match"] = ",".join(sorted(term_hits))[:80]

            # exact-title substring match is a very strong signal
            if title_lower and len(title_lower) > 4 and title_lower in query_lower:
                score += 0.18
                signals["title_in_query"] = True
            if doc.category == "important_dates" and intent.intent == "important_dates":
                score += 0.08

            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    record_id=doc.record_id,
                    text=doc.text,
                    title=doc.title,
                    category=doc.category,
                    language=doc.language,
                    verified=doc.verified,
                    academic_year=doc.academic_year,
                    score=round(max(0.0, score), 4),
                    dense_score=round(float(entry.get("dense", 0.0)), 4),
                    lexical_score=round(float(entry.get("lexical", 0.0)), 4),
                    dense_rank=int(entry.get("dense_rank", 0)),
                    lexical_rank=int(entry.get("lexical_rank", 0)),
                    source=doc.source,
                    source_uri=doc.source_uri,
                    structured=doc.structured,
                    updated_at=doc.updated_at,
                    citation=doc.citation,
                    signals=signals,
                )
            )
        out.sort(key=lambda item: -item.score)
        return out

    async def stats(self) -> dict[str, Any]:
        store_stats = await self.store.stats() if self.store else {}
        by_category: dict[str, int] = {}
        verified = 0
        for doc in self.chunks.values():
            by_category[doc.category] = by_category.get(doc.category, 0) + 1
            verified += 1 if doc.verified else 0
        return {
            "chunks": len(self.chunks),
            "records": len({d.record_id for d in self.chunks.values()}),
            "verified_chunks": verified,
            "by_category": by_category,
            "embedding": self.embedder.name if self.embedder else "none",
            "dimensions": self.embedder_dimensions,
            "store": store_stats,
            "min_score": self.min_score,
            "top_k": self.top_k,
        }
