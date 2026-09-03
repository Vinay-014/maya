"""Fact extraction engine using LLM structured outputs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from config import LLM_MODEL, TEMPERATURE
from core.llm_client import is_rate_limit_error, structured_completion
from core.prompts import EXTRACTION_SYSTEM_PROMPT
from core.schema import Fact, FactCategory, MemoryExtractionResult

logger = logging.getLogger(__name__)


class ExtractedFactPayload(BaseModel):
    text: str
    subject: str = "user"
    predicate: str = ""
    object: str = ""
    category: FactCategory = FactCategory.OTHER
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class ExtractionPayload(BaseModel):
    facts: list[ExtractedFactPayload] = Field(default_factory=list)
    emotional_state: str | None = None
    temporal_markers: list[str] = Field(default_factory=list)
    extraction_notes: str | None = None


class ExtractionError(Exception):
    """Raised when fact extraction fails."""


class FactExtractor:
    """Extracts structured facts from user input via LLM."""

    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    def extract(self, user_message: str, conversation_context: list[dict[str, str]] | None = None) -> MemoryExtractionResult:
        normalized = user_message.strip()
        if not normalized:
            return MemoryExtractionResult()
        if self._is_greeting(normalized) or self._is_meta_query(normalized):
            return MemoryExtractionResult()

        context_block = ""
        if conversation_context:
            recent = conversation_context[-6:]
            lines = [f"{m['role']}: {m['content']}" for m in recent]
            context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

        user_prompt = (
            f"{context_block}"
            f"User message to analyze:\n{user_message}\n\n"
            "Return JSON matching the schema with extracted facts."
        )

        try:
            payload = self._call_llm_structured(user_prompt, user_message)
            facts = [
                Fact(
                    text=item.text,
                    subject=item.subject,
                    predicate=item.predicate,
                    object=item.object,
                    category=item.category,
                    confidence=item.confidence,
                )
                for item in payload.facts
                if not self._is_meta_fact(item.text, item.predicate, item.category)
            ]
            facts = self._merge_heuristic_facts(user_message, facts)
            return MemoryExtractionResult(
                facts=facts,
                emotional_state=payload.emotional_state,
                temporal_markers=payload.temporal_markers,
                extraction_notes=payload.extraction_notes,
            )
        except ExtractionError:
            raise
        except Exception as exc:
            logger.exception("Unexpected extraction failure")
            raise ExtractionError(f"Extraction failed: {exc}") from exc

    def _merge_heuristic_facts(self, user_message: str, facts: list[Fact]) -> list[Fact]:
        """Supplement valid LLM output with deterministic facts for known update forms."""
        heuristic = self._heuristic_extract(user_message).facts
        known = {(fact.predicate, fact.object.lower()) for fact in facts}
        for item in heuristic:
            key = (item.predicate, item.object.lower())
            if key not in known:
                facts.append(
                    Fact(
                        text=item.text,
                        subject=item.subject,
                        predicate=item.predicate,
                        object=item.object,
                        category=item.category,
                        confidence=item.confidence,
                    )
                )
                known.add(key)
        return facts

    def _call_llm_structured(self, user_prompt: str, user_message: str) -> ExtractionPayload:
        try:
            return structured_completion(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                schema=ExtractionPayload,
                model=self.model,
                temperature=self.temperature,
                max_tokens=256,
            )
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Invalid JSON from LLM: {exc}") from exc
        except ExtractionError:
            raise
        except Exception as exc:
            fallback = self._heuristic_extract(user_message)
            if is_rate_limit_error(exc):
                logger.info("LLM rate limit reached; used heuristic extraction")
            else:
                logger.warning("LLM unavailable; used heuristic extraction (%s)", type(exc).__name__)
            return fallback

    def _heuristic_extract(self, user_prompt: str) -> ExtractionPayload:
        """Simple pattern-based fallback when no API is available."""
        facts: list[ExtractedFactPayload] = []
        lower = user_prompt.lower()

        patterns: list[tuple[str, str, str, FactCategory]] = [
            (r"(?:fixing|working on|debugging) ([\w\s]+?)(?:,?\s+and\s+|\s+but\s+|\.|$)", "user", "works_on", FactCategory.TECHNICAL),
            (r"i live in ([\w\s,]+?)(?:\.|$|\s+and)", "user", "lives_in", FactCategory.BIOGRAPHICAL),
            (r"i moved to ([\w\s,]+?)(?:\.|$|\s+last)", "user", "lives_in", FactCategory.BIOGRAPHICAL),
            (r"moved to ([\w\s,]+?)(?:\.|$|\s+last)", "user", "lives_in", FactCategory.BIOGRAPHICAL),
            (r"moved back to (?:my\s+)?([\w\s]+?)(?:\s+for\s+work|\.|$)", "user", "lives_in", FactCategory.BIOGRAPHICAL),
            (r"i drink .{0,40}?(coffee)", "user", "likes", FactCategory.PREFERENCE),
            (r"i (?:love|like) ([\w\s]+?)(?:\.|$)", "user", "likes", FactCategory.PREFERENCE),
            (r"i (?:hate|dislike) ([\w\s]+?)(?:\.|$)", "user", "dislikes", FactCategory.PREFERENCE),
            (r"my name is ([\w\s]+?)(?:\.|$)", "user", "name", FactCategory.BIOGRAPHICAL),
            (r"i'm ([\w]+)(?:\.|,|$)", "user", "name", FactCategory.BIOGRAPHICAL),
            (r"i work (?:at|for) ([\w\s]+?)(?:\.|$)", "user", "works_at", FactCategory.BIOGRAPHICAL),
            (r"(?:i|and) work (?:in|on) ([\w\s]+?)(?:\.|$)", "user", "works_at", FactCategory.BIOGRAPHICAL),
            (r"i (?:quit|stopped) ([\w\s]+?)(?:\.|$|\s+(?:entirely|last))", "user", "stopped", FactCategory.BIOGRAPHICAL),
            (r"my dog'?s name is (\w+)", "user", "name", FactCategory.BIOGRAPHICAL),
            (r"accepted an offer at ([\w\s]+?)(?:\.|$)", "user", "works_at", FactCategory.BIOGRAPHICAL),
            (r"my favorite food is ([\w\s]+?)(?:\.|$)", "user", "likes", FactCategory.PREFERENCE),
            (r"i(?:'m| am) vegetarian", "user", "diet", FactCategory.PREFERENCE),
            (r"my birthday is ([\w\s\d]+?)(?:\.|$)", "user", "birthday", FactCategory.TEMPORAL),
            (r"my partner (\w+)", "user", "partner", FactCategory.RELATIONSHIP),
        ]

        for pattern, subject, predicate, category in patterns:
            match = re.search(pattern, lower)
            if match:
                obj = match.group(1).strip().rstrip(".") if match.lastindex else predicate
                if predicate == "lives_in":
                    obj = self._normalize_location(obj)
                elif predicate == "works_on":
                    obj = self._normalize_work_topic(obj)
                snippet = match.group(0).strip()
                if predicate == "works_on":
                    snippet = f"Fixing {obj}"
                elif predicate == "lives_in":
                    snippet = f"Moved to {obj}" if "moved" in snippet.lower() else f"Lives in {obj}"
                facts.append(
                    ExtractedFactPayload(
                        text=snippet.capitalize(),
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        category=category,
                        confidence=0.75 if predicate == "diet" else 0.7,
                    )
                )

        emotional_state = None
        for word in ("sad", "happy", "anxious", "stressed", "excited"):
            if word in lower:
                emotional_state = word
                facts.append(
                    ExtractedFactPayload(
                        text=f"User is feeling {word}",
                        subject="user",
                        predicate="emotional_state",
                        object=word,
                        category=FactCategory.EMOTIONAL,
                        confidence=0.75,
                    )
                )
                break

        return ExtractionPayload(facts=facts, emotional_state=emotional_state)

    @staticmethod
    def _normalize_location(value: str) -> str:
        aliases = {"sf": "San Francisco", "s.f.": "San Francisco"}
        cleaned = " ".join(value.split()).strip(" .,!?;:")
        cleaned = re.split(r"\s+(?:to|for|because|since|so|and)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = cleaned.strip(" .,!?;:")
        return aliases.get(cleaned.lower(), cleaned.title())

    @staticmethod
    def _normalize_work_topic(value: str) -> str:
        cleaned = " ".join(value.split()).strip(" .,!?;:")
        cleaned = re.split(r"\s+(?:and|but|because|since|so)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = re.sub(r"\s+in\s+(?:our|the|my)\s+backend$", " in backend", cleaned, flags=re.IGNORECASE)
        return cleaned.lower()

    @staticmethod
    def _is_meta_query(message: str) -> bool:
        lower = message.lower().strip()
        recall_markers = (
            "where do i", "where did i", "what is my", "what did i tell", "what drains my energy",
            "who am i", "remind me",
            "didn't i say", "did i say", "isn't it", "aren't i", "thought i",
        )
        return "?" in lower or lower.startswith(recall_markers)

    @staticmethod
    def _is_greeting(message: str) -> bool:
        return message.lower().strip(" !.,") in {
            "hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening",
        }

    @staticmethod
    def _is_meta_fact(text: str, predicate: str, category: FactCategory) -> bool:
        lower = text.lower()
        meta_markers = (
            "wants to be reminded", "want to be reminded", "asked to be reminded",
            "the user wants to know", "the user asked", "user asked", "recall request",
        )
        return any(marker in lower for marker in meta_markers) or predicate in {"wants_to_be_reminded", "wants_to_know"}
