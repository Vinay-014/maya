"""Memory reconciliation and contradiction resolution engine."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import BaseModel, field_validator

from config import (
    CONFIDENCE_BOOST_ON_REINFORCE,
    LLM_MODEL,
    MAX_CONFIDENCE,
    RECONCILIATION_CANDIDATE_K,
)
from core.database import MemoryDatabase
from core.embeddings import embed_text
from core.llm_client import is_rate_limit_error, structured_completion
from core.prompts import RECONCILIATION_SYSTEM_PROMPT
from core.schema import (
    Fact,
    ReconciliationAction,
    ReconciliationDecision,
    ReconciliationResult,
    utc_now,
)

logger = logging.getLogger(__name__)


class ReconciliationPayload(BaseModel):
    action: ReconciliationAction
    target_fact_id: str | None = None
    reasoning: str = ""
    updated_confidence: float | None = None

    @field_validator("updated_confidence", mode="before")
    @classmethod
    def parse_confidence(cls, v: Any) -> float | None:
        if v is None or v == "" or str(v).lower() == "none" or str(v).lower() == "null":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @field_validator("target_fact_id", mode="before")
    @classmethod
    def parse_target_fact_id(cls, v: Any) -> str | None:
        if v is None or v == "" or str(v).lower() == "none" or str(v).lower() == "null":
            return None
        return str(v)


class ReconciliationError(Exception):
    """Raised when reconciliation fails irrecoverably."""


class MemoryReconciler:
    """Resolves contradictions between new and existing facts."""

    def __init__(
        self,
        db: MemoryDatabase,
        model: str = LLM_MODEL,
        candidate_k: int = RECONCILIATION_CANDIDATE_K,
    ) -> None:
        self.db = db
        self.model = model
        self.candidate_k = candidate_k

    def reconcile_extraction(self, facts: list[Fact]) -> list[ReconciliationResult]:
        results: list[ReconciliationResult] = []
        for fact in facts:
            result = self.reconcile_fact(fact)
            results.append(result)
        return results

    def reconcile_fact(self, new_fact: Fact) -> ReconciliationResult:
        if new_fact.predicate == "lives_in":
            active_locations = [fact for fact in self.db.get_active_candidates(subject=new_fact.subject or None, limit=200) if fact.predicate == "lives_in" and fact.object.lower() != new_fact.object.lower()]
            if active_locations:
                self.db.upsert_fact(new_fact, embedding=embed_text(new_fact.to_memory_text()))
                try:
                    retired_ids = self.db.supersede_active_locations(new_fact.id, new_fact.subject or "user")
                except Exception as exc:
                    print(f"[Reconciler Warning] Supersession error: {exc}")
                    retired_ids = []
                return ReconciliationResult(
                    new_fact=new_fact,
                    decisions=[
                        ReconciliationDecision(
                            target_fact_id=active_locations[0].id,
                            action=ReconciliationAction.SUPERSEDE,
                            reasoning="New location supersedes all active location memories.",
                        )
                    ],
                    applied_actions=[
                        *[f"superseded:{fact_id}" for fact_id in retired_ids],
                        f"inserted:{new_fact.id}",
                    ],
                )

        candidates = self._find_candidates(new_fact)
        if not candidates:
            embedding = embed_text(new_fact.to_memory_text())
            self.db.upsert_fact(new_fact, embedding=embedding)
            return ReconciliationResult(
                new_fact=new_fact,
                decisions=[
                    ReconciliationDecision(
                        action=ReconciliationAction.KEEP,
                        reasoning="No similar active memories; stored as novel fact.",
                    )
                ],
                applied_actions=["inserted_novel"],
            )

        best_candidate = candidates[0]
        location_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.subject == new_fact.subject
                and candidate.predicate == "lives_in"
                and new_fact.predicate == "lives_in"
                and candidate.object.lower() != new_fact.object.lower()
            ),
            None,
        )
        decision = (
            self._heuristic_reconcile(new_fact, location_candidate)
            if location_candidate
            else self._llm_reconcile(new_fact, best_candidate, candidates)
        )

        applied: list[str] = []
        if decision.action == ReconciliationAction.SUPERSEDE and decision.target_fact_id:
            embedding = embed_text(new_fact.to_memory_text())
            self.db.upsert_fact(new_fact, embedding=embedding)
            self._supersede_related(new_fact, decision.target_fact_id)
            applied.append(f"superseded:{decision.target_fact_id}")
            applied.append(f"inserted:{new_fact.id}")
        elif decision.action == ReconciliationAction.UPDATE and decision.target_fact_id:
            existing = self.db.get_fact(decision.target_fact_id)
            if existing:
                new_conf = decision.updated_confidence or min(
                    MAX_CONFIDENCE,
                    existing.confidence + CONFIDENCE_BOOST_ON_REINFORCE,
                )
                self.db.update_fact_confidence(decision.target_fact_id, new_conf)
                applied.append(f"updated_confidence:{decision.target_fact_id}")
        elif decision.action == ReconciliationAction.NOOP:
            applied.append("noop_duplicate")
        else:
            embedding = embed_text(new_fact.to_memory_text())
            self.db.upsert_fact(new_fact, embedding=embedding)
            applied.append(f"inserted:{new_fact.id}")

        return ReconciliationResult(
            new_fact=new_fact,
            decisions=[decision],
            applied_actions=applied,
        )

    def _supersede_related(self, new_fact: Fact, primary_id: str) -> None:
        """Supersede primary and any other active facts with same subject+predicate."""
        to_retire: set[str] = {primary_id}
        if new_fact.subject and new_fact.predicate:
            for candidate in self.db.get_active_candidates(
                subject=new_fact.subject,
                limit=50,
            ):
                if candidate.id != new_fact.id and candidate.predicate == new_fact.predicate:
                    to_retire.add(candidate.id)
        for fact_id in to_retire:
            self.db.supersede_fact(fact_id, new_fact.id)

    def _find_candidates(self, new_fact: Fact) -> list[Fact]:
        embedding = embed_text(new_fact.to_memory_text())
        vector_hits = self.db.search_memories(
            query_embedding=embedding,
            top_k=self.candidate_k,
        )
        candidates = [fact for fact, _ in vector_hits]

        # Also pull same predicate/subject matches
        if new_fact.predicate:
            metadata_matches = self.db.get_active_candidates(
                subject=new_fact.subject or None,
                limit=self.candidate_k,
            )
            seen = {c.id for c in candidates}
            for fact in metadata_matches:
                if fact.id not in seen and (
                    fact.predicate == new_fact.predicate
                    or fact.category == new_fact.category
                ):
                    candidates.append(fact)
                    seen.add(fact.id)

        return candidates[: self.candidate_k]

    def _llm_reconcile(
        self,
        new_fact: Fact,
        primary_existing: Fact,
        all_candidates: list[Fact],
    ) -> ReconciliationDecision:
        candidate_block = "\n".join(
            f"- ID={f.id} | {f.text} (subject={f.subject}, predicate={f.predicate}, "
            f"object={f.object}, confidence={f.confidence})"
            for f in all_candidates
        )
        user_prompt = (
            f"NEW FACT:\n{new_fact.model_dump_json()}\n\n"
            f"EXISTING CANDIDATES:\n{candidate_block}\n\n"
            f"Primary candidate to compare: ID={primary_existing.id}\n"
            "Return reconciliation decision JSON."
        )

        try:
            payload = self._call_llm(user_prompt)
            if payload.action == ReconciliationAction.KEEP:
                payload.target_fact_id = None
            elif payload.target_fact_id is None and payload.action in (
                ReconciliationAction.SUPERSEDE,
                ReconciliationAction.UPDATE,
            ):
                payload.target_fact_id = primary_existing.id
            return ReconciliationDecision(
                target_fact_id=payload.target_fact_id,
                action=payload.action,
                reasoning=payload.reasoning,
                updated_confidence=payload.updated_confidence,
            )
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.info("LLM rate limit reached; using heuristic reconciliation")
            else:
                logger.warning("LLM reconciliation failed; using heuristic (%s)", type(exc).__name__)
            return self._heuristic_reconcile(new_fact, primary_existing)

    def _call_llm(self, user_prompt: str) -> ReconciliationPayload:
        try:
            messages = [
                {"role": "system", "content": RECONCILIATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            model = os.getenv("LLM_MODEL") or os.getenv("COMPANION_LLM_MODEL") or self.model
            if "/" not in model:
                model = f"gemini/{model}"
            return structured_completion(
                messages=messages,
                schema=ReconciliationPayload,
                model=model,
                temperature=0.1,
            )
        except json.JSONDecodeError as exc:
            raise ReconciliationError(f"Invalid reconciliation JSON: {exc}") from exc

    def _heuristic_reconcile(self, new_fact: Fact, existing: Fact) -> ReconciliationDecision:
        """Rule-based fallback for offline tests."""
        same_predicate = (
            new_fact.predicate
            and existing.predicate
            and new_fact.predicate == existing.predicate
        )
        same_subject = new_fact.subject == existing.subject

        if same_predicate and same_subject and new_fact.object and existing.object:
            if new_fact.object.lower() != existing.object.lower():
                return ReconciliationDecision(
                    target_fact_id=existing.id,
                    action=ReconciliationAction.SUPERSEDE,
                    reasoning="Same predicate with different object values (heuristic).",
                )
            return ReconciliationDecision(
                target_fact_id=existing.id,
                action=ReconciliationAction.UPDATE,
                reasoning="Reinforcing duplicate predicate (heuristic).",
                updated_confidence=min(MAX_CONFIDENCE, existing.confidence + CONFIDENCE_BOOST_ON_REINFORCE),
            )

        # Preference reversal: "quit/stopped X" supersedes "likes X"
        if new_fact.predicate in ("stopped", "quit") and existing.predicate == "likes":
            if new_fact.object and new_fact.object.lower() in existing.object.lower():
                return ReconciliationDecision(
                    target_fact_id=existing.id,
                    action=ReconciliationAction.SUPERSEDE,
                    reasoning="Stopped/quit supersedes prior like (heuristic).",
                )
            if existing.object and existing.object.lower() in new_fact.text.lower():
                return ReconciliationDecision(
                    target_fact_id=existing.id,
                    action=ReconciliationAction.SUPERSEDE,
                    reasoning="Stopped/quit supersedes prior like (heuristic).",
                )

        if new_fact.text.lower().strip() == existing.text.lower().strip():
            return ReconciliationDecision(
                target_fact_id=existing.id,
                action=ReconciliationAction.NOOP,
                reasoning="Duplicate text (heuristic).",
            )

        return ReconciliationDecision(
            action=ReconciliationAction.KEEP,
            reasoning="No contradiction detected (heuristic).",
        )
