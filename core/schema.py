"""Pydantic schemas for facts, memory operations, and persona configuration."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class FactCategory(StrEnum):
    PREFERENCE = "preference"
    BIOGRAPHICAL = "biographical"
    EMOTIONAL = "emotional"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    TEMPORAL = "temporal"
    TECHNICAL = "technical"
    OTHER = "other"


class ReconciliationAction(StrEnum):
    KEEP = "KEEP"
    SUPERSEDE = "SUPERSEDE"
    UPDATE = "UPDATE"
    NOOP = "NOOP"


class Fact(BaseModel):
    """A discrete memory unit extracted from conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, description="Natural language statement of the fact")
    subject: str = Field(default="user", description="Entity the fact is about")
    predicate: str = Field(default="", description="Relationship or attribute key")
    object: str = Field(default="", description="Value or target of the predicate")
    category: FactCategory = FactCategory.OTHER
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    status: FactStatus = FactStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Fact text cannot be empty")
        return cleaned

    def to_memory_text(self) -> str:
        """Compact representation for embedding and retrieval."""
        parts = [self.text]
        if self.subject and self.predicate:
            parts.append(f"({self.subject} {self.predicate} {self.object})")
        return " ".join(parts)


class MemoryExtractionResult(BaseModel):
    """Structured output from the extraction engine."""

    facts: list[Fact] = Field(default_factory=list)
    emotional_state: str | None = None
    temporal_markers: list[str] = Field(default_factory=list)
    extraction_notes: str | None = None


class ReconciliationDecision(BaseModel):
    """LLM decision for reconciling a new fact against an existing one."""

    target_fact_id: str | None = None
    action: ReconciliationAction
    reasoning: str
    updated_confidence: float | None = None


class ReconciliationResult(BaseModel):
    """Aggregate outcome after reconciling one extracted fact."""

    new_fact: Fact
    decisions: list[ReconciliationDecision] = Field(default_factory=list)
    applied_actions: list[str] = Field(default_factory=list)


class PersonaConfig(BaseModel):
    """Immutable companion persona definition."""

    name: str
    tone: str
    background: str
    core_beliefs: list[str]
    forbidden_phrases: list[str]
    conversational_limits: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        beliefs = "; ".join(self.core_beliefs)
        limits = "; ".join(self.conversational_limits) if self.conversational_limits else "none"
        return (
            f"Name: {self.name}\n"
            f"Tone: {self.tone}\n"
            f"Background: {self.background}\n"
            f"Core beliefs: {beliefs}\n"
            f"Conversational limits: {limits}"
        )


class ConversationTurn(BaseModel):
    session_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    fact: Fact
    score: float
    vector_score: float = 0.0
    decay_score: float = 0.0
    confidence_score: float = 0.0


class MemoryContext(BaseModel):
    """Token-budgeted memory block for prompt injection."""

    memories: list[RetrievedMemory] = Field(default_factory=list)
    formatted_block: str = ""
    estimated_tokens: int = 0
    truncated: bool = False
