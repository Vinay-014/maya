"""Unified LLM client: LiteLLM when available, else native Groq/Gemini/OpenAI SDKs."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from config import GROQ_API_KEY, LLM_MODEL, OPENAI_API_KEY, TEMPERATURE

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
_rate_limit_until = 0.0
_RATE_LIMIT_COOLDOWN_SECONDS = 60.0


def _litellm_available() -> bool:
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def _configure_litellm(litellm: Any) -> None:
    """Keep provider diagnostics out of the interactive CLI."""
    litellm.suppress_debug_info = True
    litellm.set_verbose = False


def _rate_limit_active() -> bool:
    return time.monotonic() < _rate_limit_until


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "429" in message
        or "ratelimit" in message
        or "quota" in message
        or "rate limit cooldown" in message
    )


def _mark_rate_limited(exc: Exception) -> None:
    global _rate_limit_until
    if is_rate_limit_error(exc):
        _rate_limit_until = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
        logger.info("LLM rate limit reached; using local fallbacks for 60 seconds")


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini/") or model.startswith("gemini-")


def _is_groq_model(model: str) -> bool:
    return model.startswith("groq/") or model.startswith("groq-")


def _strip_provider_prefix(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = 1024,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    caching: bool = False,
) -> Any:
    """Return completion response (streaming or full)."""
    model = model or LLM_MODEL
    if _litellm_available():
        import litellm

        _configure_litellm(litellm)
        if _rate_limit_active():
            raise RuntimeError("LLM rate limit cooldown is active")
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "caching": caching,
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            return litellm.completion(
                **kwargs,
            )
        except Exception as exc:
            _mark_rate_limited(exc)
            raise

    if stream:
        raise RuntimeError("Native SDK fallback does not support streaming yet.")
    return _native_completion(messages, model, temperature, max_tokens)


def structured_completion(
    messages: list[dict[str, str]],
    schema: type[T],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    caching: bool = False,
) -> T:
    """Request JSON matching a Pydantic schema."""
    model = model or LLM_MODEL
    if _litellm_available():
        import litellm

        _configure_litellm(litellm)
        if _rate_limit_active():
            raise RuntimeError("LLM rate limit cooldown is active")
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                caching=caching,
                response_format=schema,
            )
        except Exception as exc:
            _mark_rate_limited(exc)
            raise
        content = response.choices[0].message.content
        if isinstance(content, str):
            return schema.model_validate(json.loads(content))
        if hasattr(content, "model_dump"):
            return schema.model_validate(content.model_dump())
        raise ValueError("Unexpected structured response")

    # Native fallback: append JSON instruction
    schema_json = json.dumps(schema.model_json_schema())
    augmented = list(messages)
    augmented[-1] = {
        **augmented[-1],
        "content": augmented[-1]["content"]
        + f"\n\nRespond with valid JSON only matching this schema:\n{schema_json}",
    }
    text = _native_completion_text(augmented, model, temperature, max_tokens)
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return schema.model_validate(json.loads(text))


def _native_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    text = _native_completion_text(messages, model, temperature, max_tokens)

    class _Choice:
        class _Msg:
            content = text

        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return _Resp()


def _native_completion_text(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if _is_groq_model(model) or (GROQ_API_KEY and not _is_gemini_model(model) and not OPENAI_API_KEY):
        return _groq_complete(messages, _strip_provider_prefix(model.replace("groq-", "")), temperature, max_tokens)

    if _is_gemini_model(model):
        return _gemini_complete(messages, _strip_provider_prefix(model), temperature, max_tokens)

    if OPENAI_API_KEY:
        return _openai_complete(messages, _strip_provider_prefix(model), temperature, max_tokens)

    if GROQ_API_KEY:
        return _groq_complete(messages, "qwen/qwen3.6-27b", temperature, max_tokens)

    raise RuntimeError("No LLM provider available. Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")


def _groq_complete(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty Groq response")
    return content


def _gemini_complete(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import google.generativeai as genai
    from config import GEMINI_API_KEY

    genai.configure(api_key=GEMINI_API_KEY)
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    history = [m for m in messages if m["role"] != "system"]
    user_msg = history[-1]["content"] if history else ""
    prompt = f"{system}\n\n{user_msg}" if system else user_msg
    gm = genai.GenerativeModel(model)
    response = gm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = response.text
    if not text:
        raise RuntimeError("Empty Gemini response")
    return text


def _openai_complete(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty OpenAI response")
    return content
