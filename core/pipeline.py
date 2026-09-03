"""Orchestrates the full companion turn pipeline."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from config import DEFAULT_SESSION_LOG_DIR
from core.database import MemoryDatabase
from core.extractor import ExtractionError, FactExtractor
from core.generator import MODE_GREETING, MODE_RECALL, ResponseGenerator, classify_intent
from core.persona import DEFAULT_PERSONA
from core.reconciler import MemoryReconciler
from core.retriever import MemoryRetriever
from core.schema import MemoryContext, MemoryExtractionResult, PersonaConfig, ReconciliationResult

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    user_message: str
    assistant_response: str
    extraction: MemoryExtractionResult
    reconciliation: list[ReconciliationResult] = field(default_factory=list)
    memory_context: MemoryContext = field(default_factory=MemoryContext)


class CompanionPipeline:
    """End-to-end pipeline: extract → reconcile → retrieve → generate → persist."""

    def __init__(
        self,
        db: MemoryDatabase | None = None,
        persona: PersonaConfig | None = None,
        session_id: str | None = None,
        session_log_dir: Path | None = None,
    ) -> None:
        self.db = db or MemoryDatabase()
        self.persona = persona or DEFAULT_PERSONA
        self.session_id = session_id or str(uuid.uuid4())
        self.session_log_dir = session_log_dir or DEFAULT_SESSION_LOG_DIR
        self.session_log_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = FactExtractor()
        self.reconciler = MemoryReconciler(self.db)
        self.retriever = MemoryRetriever(self.db)
        self.generator = ResponseGenerator(persona=self.persona)

        self._history: list[dict[str, str]] = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def process_turn(self, user_message: str, stream: bool = True) -> TurnResult | Generator[str, None, TurnResult]:
        if stream:
            return self._process_turn_stream(user_message)
        return self._process_turn_sync(user_message)

    def _process_turn_sync(self, user_message: str) -> TurnResult:
        extraction, reconciliation, memory_context = self._pre_generate(user_message)
        response = self.generator.generate(
            user_message=user_message,
            memory_context=memory_context,
            conversation_history=self._history,
            emotional_state=extraction.emotional_state,
        )
        self._persist_turn(user_message, response, extraction, reconciliation)
        return TurnResult(
            user_message=user_message,
            assistant_response=response,
            extraction=extraction,
            reconciliation=reconciliation,
            memory_context=memory_context,
        )

    def _process_turn_stream(self, user_message: str) -> Generator[str, None, TurnResult]:
        extraction, reconciliation, memory_context = self._pre_generate(user_message)
        parts: list[str] = []
        for chunk in self.generator.stream(
            user_message=user_message,
            memory_context=memory_context,
            conversation_history=self._history,
            emotional_state=extraction.emotional_state,
        ):
            parts.append(chunk)
            yield chunk
        response = "".join(parts)
        self._persist_turn(user_message, response, extraction, reconciliation)
        return TurnResult(
            user_message=user_message,
            assistant_response=response,
            extraction=extraction,
            reconciliation=reconciliation,
            memory_context=memory_context,
        )

    def _pre_generate(
        self,
        user_message: str,
    ) -> tuple[MemoryExtractionResult, list[ReconciliationResult], MemoryContext]:
        if classify_intent(user_message) == MODE_GREETING:
            return MemoryExtractionResult(), [], MemoryContext()

        try:
            extraction = self.extractor.extract(user_message, conversation_context=self._history)
        except ExtractionError as exc:
            logger.error("Extraction failed: %s", exc)
            extraction = MemoryExtractionResult()

        reconciliation: list[ReconciliationResult] = []
        if extraction.facts:
            try:
                reconciliation = self.reconciler.reconcile_extraction(extraction.facts)
            except Exception as exc:
                logger.error("Reconciliation failed: %s", exc)

        retrieval_limit = 50 if classify_intent(user_message) == MODE_RECALL else None
        memory_context = self.retriever.retrieve(user_message, top_k=retrieval_limit)
        return extraction, reconciliation, memory_context

    def _persist_turn(
        self,
        user_message: str,
        response: str,
        extraction: MemoryExtractionResult,
        reconciliation: list[ReconciliationResult],
    ) -> None:
        meta = {
            "extracted_facts": len(extraction.facts),
            "reconciliation_actions": [r.applied_actions for r in reconciliation],
        }
        self.db.save_turn(self.session_id, "user", user_message)
        self.db.save_turn(self.session_id, "assistant", response, metadata=meta)
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": response})
        self._append_jsonl(user_message, response, extraction, reconciliation)

    def _append_jsonl(
        self,
        user_message: str,
        response: str,
        extraction: MemoryExtractionResult,
        reconciliation: list[ReconciliationResult],
    ) -> None:
        log_path = self.session_log_dir / f"{self.session_id}.jsonl"
        record = {
            "session_id": self.session_id,
            "user": user_message,
            "assistant": response,
            "facts": [f.model_dump(mode="json") for f in extraction.facts],
            "reconciliation": [r.model_dump(mode="json") for r in reconciliation],
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def reset_session(self) -> None:
        self.db.clear_session_turns(self.session_id)
        self._history.clear()

    def get_active_memories(self) -> list:
        return self.db.list_active_facts()

    def get_session_history(self) -> list[dict]:
        return self.db.get_turns(self.session_id)
