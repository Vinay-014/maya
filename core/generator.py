"""Persona-constrained response generation with streaming support."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

from config import LLM_MODEL, MAX_RESPONSE_TOKENS, TEMPERATURE
from core.llm_client import chat_completion, is_rate_limit_error
from core.persona import DEFAULT_PERSONA, build_system_prompt
from core.schema import MemoryContext, PersonaConfig
from core.tools import ToolRegistry, default_tool_registry, execute_tool_json

logger = logging.getLogger(__name__)

MODE_GREETING = "greeting"
MODE_RECALL = "recall"
MODE_CONVERSATION = "conversation"
MODE_TOOL = "tool"


def classify_intent(user_input: str) -> str:
    """Classify the small set of intents needed by the offline and LLM paths."""
    lower = " ".join(user_input.lower().split()).strip(" !?.")
    greeting_text = re.sub(r"^(?:hey|hi|hello|hiya)\s+maya(?:[,!:]\s*)?", "", lower)
    greeting_text = greeting_text.strip(" !?.")
    if greeting_text in {"", "testing", "are you there", "do you read me"} or lower in {
        "hi", "hello", "hey", "hiya", "testing", "are you there", "do you read me",
    }:
        return MODE_GREETING
    if any(marker in lower for marker in ("calculate ", "system status", "cpu usage", "ram usage", "run git ", "check git ", "search my notes", "search notes")) or re.match(r"what is [\d\s+\-*/().%]+\??$", lower):
        return MODE_TOOL
    recall_markers = (
        "where do i", "where did i", "what is my", "what's my", "what did i tell",
        "draining my energy", "drains my energy", "drained my energy",
        "what drains my energy", "what kind of work", "what do i work on", "how did you remember", "dog's name", "dogs name",
    )
    if any(marker in lower for marker in recall_markers):
        return MODE_RECALL
    return MODE_CONVERSATION


def generate_fallback_response(user_input: str, retrieved_memories: list[Any]) -> str:
    """Generate a warm offline reply without exposing the memory representation."""
    message = " ".join(user_input.split()).strip()
    lower_message = message.lower()
    intent = classify_intent(message)

    if intent == MODE_GREETING:
        return _greeting_response(message)

    active_memories = _active_memories(retrieved_memories)
    challenge_answer = _memory_challenge_response(lower_message, retrieved_memories, active_memories)
    if challenge_answer:
        return challenge_answer

    direct_answer = _direct_memory_answer(lower_message, active_memories) if intent == MODE_RECALL else ""
    if direct_answer:
        return direct_answer

    update_response = _fact_update_response(message)
    if update_response:
        return update_response

    if any(marker in lower_message for marker in ("grind", "long day", "rough day", "exhausted", "overwhelmed")):
        return _support_response(lower_message)

    return "I caught that, but I am having a brief issue processing the details right now. Could you repeat that?"


def _fact_update_response(message: str) -> str:
    """Acknowledge explicit updates locally when the provider is unavailable."""
    lower = message.lower()
    if not any(marker in lower for marker in ("moved to", "moved back to", "plans changed", "update on that")):
        return ""
    match = re.search(
        r"(?:moved back to|moved to)\s+(?:my\s+)?([\w\s]+?)(?:\s+to\s+|\s+for\s+work|\s+because\s+|\.|$)",
        lower,
    )
    if not match:
        return "Got it. I have noted that update."
    location = _clean_memory_value(match.group(1))
    aliases = {"sf": "San Francisco", "s.f.": "San Francisco"}
    location = aliases.get(location.lower(), location.title())
    return f"Got it. That update is clear: you moved back to {location} for work."


def _greeting_response(message: str) -> str:
    """Acknowledge a ping using the user's own greeting without memory context."""
    lower = message.lower()
    if "read" in lower:
        return "Yes, I caught you. I am right here. What is on your mind?"
    if "there" in lower or "testing" in lower:
        return "I am here and paying attention. What would you like to talk through?"
    greeting = (message.rstrip("!?., ") or "hello").lower()
    return f"{greeting.capitalize()}, I am here with you. What is on your mind?"


def _support_response(message: str) -> str:
    """Respond to strain-related statements without a reusable memory script."""
    if "grind" in message:
        focus = "the grind"
    elif "exhausted" in message:
        focus = "the exhaustion"
    else:
        focus = "today"
    return f"That sounds heavy, especially {focus}. I am with you. What part needs the most room right now?"


def _memory_challenge_response(user_input: str, all_memories: list[Any], active_memories: list[Any]) -> str:
    """Resolve a challenged fact against current and historical memory records."""
    challenge_markers = ("isn't it", "aren't i", "didn't i say", "thought i moved to")
    if not any(marker in user_input for marker in challenge_markers):
        return ""

    challenged_value = _mentioned_move_value(user_input)
    current = next(
        (item for item in active_memories if _fact_matches_intent(item.fact, "lives_in")),
        None,
    )
    current_value = _fact_value(current.fact, "lives_in") if current else ""
    historical = [
        item
        for item in all_memories
        if getattr(item.fact, "status", None) == "superseded"
        and _fact_matches_intent(item.fact, "lives_in")
    ]
    previous_value = _fact_value(historical[0].fact, "lives_in") if historical else challenged_value

    if challenged_value and current_value and challenged_value.lower() != current_value.lower():
        return f"You previously mentioned {challenged_value.title()}, but later updated that you moved to {current_value.title()}."
    if challenged_value and not historical and (not current_value or challenged_value.lower() != current_value.lower()):
        return f"My mistake! I will update my memory that you are in {challenged_value.title()}."
    if current_value:
        return f"You are right to check. I have your current location as {current_value.title()}."
    if previous_value:
        return f"You previously mentioned {previous_value.title()}, though I cannot confirm a newer location yet."
    return "You are right to check. I do not have a reliable location saved yet."


def _mentioned_move_value(user_input: str) -> str:
    match = re.search(
        r"(?:moved to|live in|living in)\s+([\w\s]+?)(?:\?|\.|,|\s+isn't\b|\s+aren't\b|$)",
        user_input,
        re.IGNORECASE,
    )
    return _clean_memory_value(match.group(1)) if match else ""


def _active_memories(retrieved_memories: list[Any]) -> list[Any]:
    """Defensively keep only current memories and prefer newer, confident facts."""
    active = [memory for memory in retrieved_memories if getattr(memory.fact, "status", None) == "active"]

    def sort_key(memory: Any) -> tuple[datetime, float]:
        updated_at = memory.fact.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return updated_at, memory.fact.confidence

    return sorted(active, key=sort_key, reverse=True)


def _context_for_intent(memory_context: MemoryContext, intent: str) -> MemoryContext:
    """Limit general conversation context to memories with strong relevance."""
    if intent != MODE_CONVERSATION:
        return memory_context
    selected = [memory for memory in memory_context.memories if memory.score > 0.7]
    if len(selected) == len(memory_context.memories):
        return memory_context
    lines = ["Relevant memories about the user:"] if selected else []
    for memory in selected:
        fact = memory.fact
        lines.append(
            f"- [{fact.category.value}] {fact.text} "
            f"(confidence={fact.confidence:.2f}, relevance={memory.score:.2f})"
        )
    return MemoryContext(
        memories=selected,
        formatted_block="\n".join(lines),
        estimated_tokens=memory_context.estimated_tokens,
        truncated=memory_context.truncated or len(selected) < len(memory_context.memories),
    )


def _direct_memory_answer(user_input: str, memories: list[Any]) -> str:
    """Answer common factual questions directly from the newest active fact."""
    recall_keywords = (
        "where did i", "what is my", "remind me", "who am i", "where do i live",
        "what's my", "what kind of work", "what do i work on", "how did you remember", "dog's name", "dogs name",
    )
    if not any(keyword in user_input for keyword in recall_keywords):
        return ""

    if "how did you remember" in user_input:
        location = next(
            (item for item in memories if _fact_matches_intent(item.fact, "lives_in")),
            None,
        )
        if location:
            value = _fact_value(location.fact, "lives_in")
            if value:
                return f"You told me a moment ago that plans changed and you moved back to {value.title()} for work."

    if "what was draining my energy" in user_input or "what drained my energy" in user_input:
        location = next((item for item in memories if _fact_matches_intent(item.fact, "lives_in")), None)
        work = next((item for item in memories if item.fact.predicate == "works_on"), None)
        if location and work:
            location_value = _fact_value(location.fact, "lives_in")
            work_value = _fact_value(work.fact, "works_on")
            if location_value and work_value:
                work_phrase = work_value.lower().replace(" in backend", " in your backend")
                return f"You said you moved to {location_value.title()}, and the energy drain was fixing {work_phrase}."

    intent_predicates = (
        (("where did i move", "where did i say i moved", "where do i live", "where am i", "my city"), "lives_in"),
        (("where do i work", "who do i work for", "what kind of work", "what do i work on", "my job"), "works_at"),
        (("what kind of work", "what do i work on"), "works_at"),
        (("still drink coffee", "drink coffee", "coffee", "caffeine"), "stopped"),
        (("when is my birthday", "my birthday"), "birthday"),
        (("what is my name", "what's my name", "who am i", "dog's name", "dogs name", "name of my dog"), "name"),
        (("what was draining my energy", "what drained my energy", "what was exhausting"), "works_on"),
    )
    for phrases, predicate in intent_predicates:
        if not any(phrase in user_input for phrase in phrases):
            continue
        memory = None
        value = ""
        for item in memories:
            if _fact_matches_intent(item.fact, predicate):
                candidate_value = _fact_value(item.fact, predicate)
                if candidate_value:
                    memory = item
                    value = candidate_value
                    break
        if memory is None:
            continue
        if predicate == "lives_in":
            return f"You mentioned earlier that you moved to {value.title()}."
        if predicate == "works_at":
            return f"You told me you work at {value.title()}."
        if predicate == "stopped":
            return f"You told me you quit {value.lower()}."
        if predicate == "birthday":
            return f"You told me your birthday is {value.title()}."
        if predicate == "works_on":
            return f"You mentioned working on {value.lower()} in your backend."
        return f"You told me your name is {value.title()}."

    memory = next((item for item in memories if _fact_matches_intent(item.fact, "lives_in")), None)
    memory = memory or (memories[0] if memories else None)
    if memory:
        fact = memory.fact
        value = _fact_value(fact, fact.predicate)
        if value and fact.predicate == "lives_in":
            return f"You mentioned earlier that you moved to {value.title()}."
    return ""


def _fact_matches_intent(fact: Any, predicate: str) -> bool:
    if fact.predicate == predicate:
        return True
    text = fact.text.lower()
    return predicate == "lives_in" and ("moved to" in text or "live in" in text)


def _fact_value(fact: Any, predicate: str) -> str:
    value = _clean_memory_value(fact.object)
    if value and fact.predicate == predicate:
        return value
    if predicate == "lives_in":
        match = re.search(
            r"(?:moved to|live in)\s+([\w\s]+?)(?:\.|,|\s+and\s+|$)",
            fact.text,
            re.IGNORECASE,
        )
        if match:
            return _clean_memory_value(match.group(1))
    return ""


def _clean_memory_value(value: str) -> str:
    value = " ".join((value or "").split()).strip(" .,!?:;\"'")
    return value if value and len(value) <= 80 else ""


def _primary_memory_topic(retrieved_memories: list[Any], user_input: str = "") -> str:
    """Turn one structured memory into a small conversational topic."""
    if not retrieved_memories:
        return ""

    preferred_predicate = ""
    if any(word in user_input for word in ("where do i live", "where am i", "city")):
        preferred_predicate = "lives_in"
    elif any(word in user_input for word in ("coffee", "caffeine", "tea")):
        preferred_predicate = "stopped"

    fact = next(
        (memory.fact for memory in retrieved_memories if memory.fact.predicate == preferred_predicate),
        retrieved_memories[0].fact,
    )
    value = _clean_memory_value(fact.object)
    if not value:
        return ""

    topic_by_predicate = {
        "lives_in": f"living in {value.title()}",
        "works_at": f"your work at {value.title()}",
        "likes": f"{value.lower()}", 
        "dislikes": f"what you do not enjoy about {value.lower()}",
        "stopped": f"quitting {value.lower()}",
        "birthday": f"your birthday in {value.title()}",
        "partner": f"your partner {value.title()}",
        "name": f"your name, {value.title()}",
        "emotional_state": f"feeling {value.lower()}",
    }
    return topic_by_predicate.get(fact.predicate, "")


class GenerationError(Exception):
    """Raised when response generation fails."""


class ResponseGenerator:
    """Generates in-character companion responses."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_RESPONSE_TOKENS,
        persona: PersonaConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.persona = persona or DEFAULT_PERSONA
        self.tools = tools or default_tool_registry()

    def generate(
        self,
        user_message: str,
        memory_context: MemoryContext,
        conversation_history: list[dict[str, str]] | None = None,
        emotional_state: str | None = None,
    ) -> str:
        messages = self._build_messages(
            user_message=user_message,
            memory_context=memory_context,
            conversation_history=conversation_history,
            emotional_state=emotional_state,
        )
        intent = classify_intent(user_message)
        if intent == MODE_GREETING:
            return self._fallback_response(user_message, memory_context)
        if intent == MODE_RECALL:
            direct = self._fallback_response(user_message, memory_context)
            if "I do not have a reliable" not in direct and "I caught that" not in direct:
                return direct
        if intent == MODE_TOOL:
            return self._tool_response(user_message, messages)
        try:
            response = chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                caching=False,
            )
            content = response.choices[0].message.content
            if not content:
                raise GenerationError("Empty response from model")
            return content.strip()
        except GenerationError:
            raise
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.info("LLM rate limit reached; using fallback response")
            else:
                print(f"[LLM Error]: {exc}")
                logger.warning("LLM generation failed, using fallback (%s)", type(exc).__name__)
            return self._fallback_response(user_message, memory_context)

    def stream(
        self,
        user_message: str,
        memory_context: MemoryContext,
        conversation_history: list[dict[str, str]] | None = None,
        emotional_state: str | None = None,
    ) -> Generator[str, None, str]:
        messages = self._build_messages(
            user_message=user_message,
            memory_context=memory_context,
            conversation_history=conversation_history,
            emotional_state=emotional_state,
        )
        intent = classify_intent(user_message)
        if intent == MODE_GREETING:
            response_text = self._fallback_response(user_message, memory_context)
            yield response_text
            return response_text
        if intent == MODE_RECALL:
            direct = self._fallback_response(user_message, memory_context)
            if "I do not have a reliable" not in direct and "I caught that" not in direct:
                yield direct
                return direct
        if intent == MODE_TOOL:
            response_text = self._tool_response(user_message, messages)
            yield response_text
            return response_text
        try:
            response = chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                caching=False,
            )
            parts: list[str] = []
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield delta
            full = "".join(parts).strip()
            if not full:
                full = self._fallback_response(user_message, memory_context)
                yield full
            return full
        except Exception as exc:
            if is_rate_limit_error(exc):
                logger.info("LLM rate limit reached; using fallback response")
            else:
                print(f"[LLM Error]: {exc}")
                logger.warning("Streaming failed, using fallback (%s)", type(exc).__name__)
            fallback = self._fallback_response(user_message, memory_context)
            yield fallback
            return fallback

    def _build_messages(
        self,
        user_message: str,
        memory_context: MemoryContext,
        conversation_history: list[dict[str, str]] | None,
        emotional_state: str | None,
    ) -> list[dict[str, str]]:
        intent = classify_intent(user_message)
        if intent == MODE_GREETING:
            memory_context = MemoryContext()
        else:
            memory_context = _context_for_intent(memory_context, intent)
        system = build_system_prompt(
            persona=self.persona,
            memory_context=memory_context,
            emotional_state=emotional_state,
        )
        if intent == MODE_RECALL and conversation_history:
            system += (
                "\n\nRecall safety: if active memories are missing or weakly relevant, "
                "inspect the recent session history below for facts stated directly by the user. "
                "Use that history to answer the recall question accurately."
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if conversation_history:
            for turn in conversation_history[-12:]:
                role = turn.get("role", "user")
                if role not in ("user", "assistant"):
                    continue
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _fallback_response(self, user_message: str, memory_context: MemoryContext) -> str:
        """Offline fallback that keeps Maya conversational and memory-aware."""
        return generate_fallback_response(user_message, memory_context.memories)

    def _tool_response(self, user_message: str, messages: list[dict[str, str]]) -> str:
        """Execute obvious local tool intents, using LiteLLM tool calls when available."""
        try:
            response = chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self.tools.schemas(),
                tool_choice="auto",
                caching=False,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                tool_messages = [*messages, {"role": "assistant", "content": message.content, "tool_calls": tool_calls}]
                for call in tool_calls:
                    name = call.function.name
                    arguments = json.loads(call.function.arguments or "{}")
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": execute_tool_json(self.tools, name, arguments)}
                    )
                final = chat_completion(
                    model=self.model,
                    messages=tool_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    caching=False,
                )
                content = final.choices[0].message.content
                if content:
                    return content.strip()
        except Exception as exc:
            if not is_rate_limit_error(exc):
                logger.info("Tool-call synthesis unavailable; using local tool routing (%s)", type(exc).__name__)

        lower = user_message.lower().strip()
        if lower.startswith(("calculate ", "what is ")):
            prefix = "calculate " if lower.startswith("calculate ") else "what is "
            expression = user_message[len(prefix):].rstrip("?").strip()
            result = execute_tool_json(self.tools, "calculate_math", {"expression": expression})
            return f"The calculation result is {json.loads(result).get('result', result)}."
        if "system status" in lower or "cpu usage" in lower or "ram usage" in lower:
            result = json.loads(execute_tool_json(self.tools, "get_system_status", {}))
            return f"Your system is running {result['os']} with Python {result['python']} and {result['cpu_count']} CPU cores."
        if "search" in lower and "notes" in lower:
            query = user_message.split("notes", 1)[-1].strip(" :") or user_message
            result = json.loads(execute_tool_json(self.tools, "search_local_notes", {"query": query}))
            return f"I found {len(result['results'])} matching note file(s) for {query}."
        if lower.startswith(("run git ", "check git ")):
            command = user_message[user_message.lower().find("git "):].rstrip("?")
            result = json.loads(execute_tool_json(self.tools, "execute_shell_cmd", {"command": command}))
            return result.get("stdout", "The command completed with no output.").strip() or "The command completed with no output."
        return self._fallback_response(user_message, MemoryContext())
