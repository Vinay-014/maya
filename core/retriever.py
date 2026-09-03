"""Hybrid memory retrieval with decay and token budgeting."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from config import (
    MEMORY_DECAY_HALF_LIFE_DAYS,
    MEMORY_TOKEN_BUDGET,
    MEMORY_TOP_K,
)
from core.database import MemoryDatabase
from core.embeddings import embed_text, estimate_tokens
from core.schema import Fact, FactStatus, MemoryContext, RetrievedMemory, utc_now

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves active memories using hybrid vector + metadata scoring."""

    def __init__(
        self,
        db: MemoryDatabase,
        top_k: int = MEMORY_TOP_K,
        token_budget: int = MEMORY_TOKEN_BUDGET,
        decay_half_life_days: float = MEMORY_DECAY_HALF_LIFE_DAYS,
    ) -> None:
        self.db = db
        self.top_k = top_k
        self.token_budget = token_budget
        self.decay_half_life_days = decay_half_life_days

    def retrieve(self, query: str, category: str | None = None, top_k: int | None = None) -> MemoryContext:
        if not query.strip():
            return MemoryContext()

        result_limit = top_k or self.top_k

        query_embedding = embed_text(query)
        vector_results = self.db.search_memories(
            query_embedding=query_embedding,
            top_k=result_limit * 2,
            status=FactStatus.ACTIVE,
            category=category,
        )

        # Keyword boost from query tokens
        keywords = [w for w in query.lower().split() if len(w) > 3][:8]
        keyword_hits = self.db.keyword_search_memories(keywords, top_k=result_limit)

        merged: dict[str, RetrievedMemory] = {}

        for fact, vector_score in vector_results:
            decay = self._decay_score(fact)
            confidence = fact.confidence
            combined = 0.55 * vector_score + 0.25 * decay + 0.20 * confidence
            merged[fact.id] = RetrievedMemory(
                fact=fact,
                score=combined,
                vector_score=vector_score,
                decay_score=decay,
                confidence_score=confidence,
            )

        for fact in keyword_hits:
            if fact.id in merged:
                merged[fact.id].score += 0.15
            else:
                decay = self._decay_score(fact)
                merged[fact.id] = RetrievedMemory(
                    fact=fact,
                    score=0.35 + 0.25 * decay + 0.20 * fact.confidence,
                    vector_score=0.0,
                    decay_score=decay,
                    confidence_score=fact.confidence,
                )

        ranked = sorted(merged.values(), key=lambda m: m.score, reverse=True)[:result_limit]
        formatted, tokens, truncated = self._format_within_budget(ranked)

        for mem in ranked[:result_limit]:
            self.db.touch_fact_access(mem.fact.id)

        return MemoryContext(
            memories=ranked,
            formatted_block=formatted,
            estimated_tokens=tokens,
            truncated=truncated,
        )

    def search(self, query: str, top_k: int = 20) -> list[RetrievedMemory]:
        """Return active memories using the same vector and keyword ranking as retrieval."""
        context = MemoryRetriever(
            self.db,
            top_k=top_k,
            token_budget=self.token_budget,
            decay_half_life_days=self.decay_half_life_days,
        ).retrieve(query)
        return context.memories[:top_k]

    def _decay_score(self, fact: Fact) -> float:
        """Exponential decay based on recency of access/update."""
        reference = fact.last_accessed_at or fact.updated_at or fact.created_at
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        now = utc_now()
        age_days = max(0.0, (now - reference).total_seconds() / 86400.0)
        if self.decay_half_life_days <= 0:
            return 1.0
        return math.exp(-0.693 * age_days / self.decay_half_life_days)

    def _format_within_budget(
        self,
        memories: list[RetrievedMemory],
    ) -> tuple[str, int, bool]:
        if not memories:
            return "", 0, False

        lines: list[str] = []
        total_tokens = 0
        truncated = False

        header = "Relevant memories about the user:"
        lines.append(header)
        total_tokens += estimate_tokens(header)

        for mem in memories:
            fact = mem.fact
            line = (
                f"- [{fact.category.value}] {fact.text} "
                f"(confidence={fact.confidence:.2f}, relevance={mem.score:.2f})"
            )
            line_tokens = estimate_tokens(line)
            if total_tokens + line_tokens > self.token_budget:
                truncated = True
                break
            lines.append(line)
            total_tokens += line_tokens

        return "\n".join(lines), total_tokens, truncated
