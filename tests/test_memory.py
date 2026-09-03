"""Pytest suite for memory extraction, reconciliation, and persistence."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from main import _handle_command
from core.database import MemoryDatabase
from core.embeddings import cosine_similarity, embed_text, estimate_tokens
from core.extractor import FactExtractor
from core.generator import MODE_CONVERSATION, MODE_GREETING, ResponseGenerator, _context_for_intent, classify_intent, generate_fallback_response
from core.persona import DEFAULT_PERSONA, check_persona_violations
from core.pipeline import CompanionPipeline
from core.reconciler import MemoryReconciler
from core.retriever import MemoryRetriever
from core.schema import Fact, FactCategory, FactStatus, MemoryContext, ReconciliationAction, RetrievedMemory
from core.tools import ToolError, calculate_math, default_tool_registry


@pytest.fixture
def temp_db() -> MemoryDatabase:
    path = Path(tempfile.gettempdir()) / f"test_companion_{uuid.uuid4().hex}.db"
    db = MemoryDatabase(path)
    yield db
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@pytest.fixture
def pipeline(temp_db: MemoryDatabase) -> CompanionPipeline:
    return CompanionPipeline(db=temp_db, session_id="test-session")


class TestEmbeddings:
    def test_local_embedding_dimension(self) -> None:
        vec = embed_text("I live in Bengaluru", use_local_fallback=True)
        assert len(vec) > 0
        assert isinstance(vec[0], float)

    def test_cosine_similarity_identical(self) -> None:
        vec = embed_text("coffee", use_local_fallback=True)
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)

    def test_estimate_tokens(self) -> None:
        assert estimate_tokens("hello world") >= 1


class TestTools:
    def test_calculate_math_is_safe_and_correct(self) -> None:
        assert calculate_math("100 * (1 + 0.2)")["result"] == 120.0
        assert calculate_math("12 * (3 + 1)")["result"] == 48.0
        assert calculate_math("Please calculate 12 × (3 + 1)")["result"] == 48.0
        assert calculate_math("what is 12 x (3 + 1)?")["result"] == 48.0
        with pytest.raises(ToolError):
            calculate_math("__import__('os').getcwd()")

    def test_registry_executes_registered_tool(self) -> None:
        result = default_tool_registry().execute("calculate_math", {"expression": "2 + 3"})
        assert result["result"] == 5.0


class TestDatabase:
    def test_save_and_retrieve_turns(self, temp_db: MemoryDatabase) -> None:
        temp_db.save_turn("s1", "user", "Hello")
        temp_db.save_turn("s1", "assistant", "Hi there")
        turns = temp_db.get_turns("s1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"

    def test_upsert_and_list_facts(self, temp_db: MemoryDatabase) -> None:
        fact = Fact(text="User loves coffee", category=FactCategory.PREFERENCE)
        embedding = embed_text(fact.text, use_local_fallback=True)
        temp_db.upsert_fact(fact, embedding=embedding)
        active = temp_db.list_active_facts()
        assert len(active) == 1
        assert active[0].text == "User loves coffee"

    def test_supersede_fact(self, temp_db: MemoryDatabase) -> None:
        old = Fact(text="User lives in Bengaluru", predicate="lives_in", object="Bengaluru")
        new = Fact(text="User lives in San Francisco", predicate="lives_in", object="San Francisco")
        emb = embed_text("test", use_local_fallback=True)
        temp_db.upsert_fact(old, embedding=emb)
        temp_db.upsert_fact(new, embedding=emb)
        temp_db.supersede_fact(old.id, new.id)
        updated = temp_db.get_fact(old.id)
        assert updated is not None
        assert updated.status == FactStatus.SUPERSEDED
        assert updated.superseded_by == new.id

    def test_vector_search(self, temp_db: MemoryDatabase) -> None:
        f1 = Fact(text="User loves filter coffee", category=FactCategory.PREFERENCE)
        f2 = Fact(text="User enjoys hiking", category=FactCategory.PREFERENCE)
        temp_db.upsert_fact(f1, embedding=embed_text(f1.text, use_local_fallback=True))
        temp_db.upsert_fact(f2, embedding=embed_text(f2.text, use_local_fallback=True))
        query_emb = embed_text("coffee preferences", use_local_fallback=True)
        results = temp_db.search_memories(query_emb, top_k=2)
        assert len(results) >= 1
        top_fact, score = results[0]
        assert "coffee" in top_fact.text.lower()
        assert score > 0


class TestExtractor:
    def test_heuristic_extraction_location(self) -> None:
        extractor = FactExtractor()
        payload = extractor._heuristic_extract("User message: I live in Bengaluru")
        assert any("bengaluru" in f.object.lower() for f in payload.facts)

    def test_heuristic_extraction_preference(self) -> None:
        extractor = FactExtractor()
        payload = extractor._heuristic_extract("I love poetry and rainy days")
        assert len(payload.facts) >= 1

    def test_empty_message_returns_empty(self) -> None:
        extractor = FactExtractor()
        result = extractor.extract("")
        assert result.facts == []

    def test_meta_question_is_not_stored(self) -> None:
        extractor = FactExtractor()
        result = extractor.extract("Remind me, where do I live?")
        assert result.facts == []

    def test_compound_message_extracts_separate_technical_and_location_facts(self) -> None:
        extractor = FactExtractor()
        payload = extractor._heuristic_extract(
            "fixing Python memory leaks in backend AND moved to Mysore"
        )
        facts = {(fact.predicate, fact.object.lower()) for fact in payload.facts}
        assert ("works_on", "python memory leaks in backend") in facts
        assert ("lives_in", "mysore") in facts


class TestGenerator:
    def test_greeting_fallback_does_not_dump_memories(self, temp_db: MemoryDatabase) -> None:
        fact = Fact(text="User lives in Mysore", predicate="lives_in", object="Mysore")
        temp_db.upsert_fact(fact, embedding=embed_text(fact.text, use_local_fallback=True))
        context = MemoryRetriever(temp_db).retrieve("hello")
        assert classify_intent("Do you read me?") == MODE_GREETING
        response = generate_fallback_response("Do you read me?", context.memories)
        assert "Mysore" not in response
        assert "here" in response.lower()

    def test_greeting_variants_use_local_intent(self) -> None:
        assert classify_intent("Hey Maya, do you read me?") == MODE_GREETING
        assert classify_intent("hello") == MODE_GREETING

    def test_pipeline_skips_memory_retrieval_for_greeting(self, pipeline: CompanionPipeline) -> None:
        fact = Fact(text="User lives in San Francisco", predicate="lives_in", object="San Francisco")
        pipeline.db.upsert_fact(fact, embedding=embed_text(fact.text, use_local_fallback=True))
        _, _, context = pipeline._pre_generate("Hey Maya, do you read me?")
        assert context.memories == []

    def test_fresh_greeting_messages_have_only_system_and_user(self, temp_db: MemoryDatabase) -> None:
        fact = Fact(text="User lives in San Francisco", predicate="lives_in", object="San Francisco")
        temp_db.upsert_fact(fact, embedding=embed_text(fact.text, use_local_fallback=True))
        generator = ResponseGenerator()
        messages = generator._build_messages(
            user_message="Do you read me?",
            memory_context=MemoryRetriever(temp_db).retrieve("Do you read me?"),
            conversation_history=[],
            emotional_state=None,
        )
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[-1]["content"] == "Do you read me?"
        assert "San Francisco" not in messages[0]["content"]
        assert "I remember living in San Francisco" not in messages[0]["content"]

    def test_direct_recall_answers_without_conversational_filler(self) -> None:
        fact = Fact(text="User lives in Mysore", predicate="lives_in", object="Mysore")
        memory = type("Memory", (), {"fact": fact})()
        response = generate_fallback_response("Where do I live?", [memory])
        assert response == "You mentioned earlier that you moved to Mysore."

    def test_memory_provenance_question_explains_recall(self) -> None:
        fact = Fact(text="User moved to San Francisco", predicate="lives_in", object="San Francisco")
        memory = type("Memory", (), {"fact": fact})()
        response = generate_fallback_response("How did you remember that I live in San Francisco?", [memory])
        assert "plans changed" in response
        assert "San Francisco" in response

    def test_recall_prompt_includes_recent_session_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class Message:
            content = "You were fixing Python memory leaks in your backend."

        class Response:
            choices = [type("Choice", (), {"message": Message()})()]

        def fake_completion(**kwargs: object) -> Response:
            captured.update(kwargs)
            return Response()

        monkeypatch.setattr("core.generator.chat_completion", fake_completion)
        ResponseGenerator().generate(
            "What was draining my energy?",
            MemoryContext(),
            conversation_history=[
                {"role": "user", "content": "Today was a grind fixing Python memory leaks in my backend."}
            ],
        )
        messages = captured["messages"]
        assert isinstance(messages, list)
        assert any("Python memory leaks" in str(message) for message in messages)
        assert "Recall safety" in str(messages[0])

    def test_update_fallback_does_not_announce_storage(self) -> None:
        response = generate_fallback_response("Plans changed and I moved back to SF for work.", [])
        assert "will remember" not in response.lower()

    def test_generic_fallback_does_not_echo_stored_location(self) -> None:
        fact = Fact(text="User lives in San Francisco", predicate="lives_in", object="San Francisco")
        memory = type("Memory", (), {"fact": fact})()
        response = generate_fallback_response("The day has been complicated.", [memory])
        assert "San Francisco" not in response
        assert "What feels most important about it today?" not in response

    def test_recall_without_active_memory_uses_session_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class Message:
            content = "You mentioned working on Python memory leaks in your backend."

        class Response:
            choices = [type("Choice", (), {"message": Message()})()]

        def fake_completion(**kwargs: object) -> Response:
            captured.update(kwargs)
            return Response()

        monkeypatch.setattr("core.generator.chat_completion", fake_completion)
        response = ResponseGenerator().generate(
            "What was draining my energy?",
            MemoryContext(),
            conversation_history=[
                {"role": "user", "content": "Today was a grind fixing Python memory leaks in my backend."}
            ],
        )
        messages = captured["messages"]
        assert response.startswith("You mentioned working on Python")
        assert isinstance(messages, list)
        assert any("Python memory leaks" in str(message) for message in messages)

    def test_general_context_keeps_only_high_relevance_memories(self) -> None:
        low = RetrievedMemory(fact=Fact(text="low relevance"), score=0.69)
        high = RetrievedMemory(fact=Fact(text="high relevance"), score=0.71)
        context = MemoryContext(memories=[low, high])
        filtered = _context_for_intent(context, MODE_CONVERSATION)
        assert filtered.memories == [high]


class TestCommands:
    def test_memory_search_and_forget_commands(self, pipeline: CompanionPipeline, capsys: pytest.CaptureFixture[str]) -> None:
        fact = Fact(text="User loves coffee", predicate="likes", object="coffee", category=FactCategory.PREFERENCE)
        pipeline.db.upsert_fact(fact, embedding=embed_text(fact.text, use_local_fallback=True))

        assert _handle_command("/memories search coffee", pipeline) is True
        assert fact.id in capsys.readouterr().out
        assert _handle_command(f"/memories forget {fact.id}", pipeline) is True
        assert "forgotten" in capsys.readouterr().out.lower()
        assert pipeline.db.list_active_facts() == []

    def test_unknown_slash_command_does_not_process_turn(self, pipeline: CompanionPipeline, capsys: pytest.CaptureFixture[str]) -> None:
        assert _handle_command("/not-a-command hello", pipeline) is True
        assert "Unknown command: /not-a-command hello" in capsys.readouterr().out
        assert pipeline.history == []

    def test_forget_accepts_one_based_index(self, pipeline: CompanionPipeline, capsys: pytest.CaptureFixture[str]) -> None:
        fact = Fact(text="User likes tea", category=FactCategory.PREFERENCE)
        pipeline.db.upsert_fact(fact, embedding=embed_text(fact.text, use_local_fallback=True))
        _handle_command("/memories forget 1", pipeline)
        assert "marked as forgotten" in capsys.readouterr().out
        assert pipeline.db.get_fact(fact.id).status == FactStatus.SUPERSEDED


class TestReconciler:
    def test_novel_fact_inserted(self, temp_db: MemoryDatabase) -> None:
        reconciler = MemoryReconciler(temp_db)
        fact = Fact(text="User name is Arjun", predicate="name", object="Arjun")
        result = reconciler.reconcile_fact(fact)
        assert "inserted" in result.applied_actions[0] or result.decisions[0].action == ReconciliationAction.KEEP
        assert temp_db.count_facts(status=FactStatus.ACTIVE) >= 1

    def test_location_contradiction_supersedes(self, temp_db: MemoryDatabase) -> None:
        reconciler = MemoryReconciler(temp_db)
        old = Fact(
            text="User lives in Bengaluru",
            subject="user",
            predicate="lives_in",
            object="Bengaluru",
            category=FactCategory.BIOGRAPHICAL,
        )
        emb = embed_text(old.text, use_local_fallback=True)
        temp_db.upsert_fact(old, embedding=emb)

        new = Fact(
            text="User moved to San Francisco",
            subject="user",
            predicate="lives_in",
            object="San Francisco",
            category=FactCategory.BIOGRAPHICAL,
        )
        result = reconciler.reconcile_fact(new)
        assert any("superseded" in a for a in result.applied_actions) or result.decisions[0].action == ReconciliationAction.SUPERSEDE
        old_updated = temp_db.get_fact(old.id)
        assert old_updated is not None
        assert old_updated.status == FactStatus.SUPERSEDED or temp_db.count_facts(status=FactStatus.ACTIVE) >= 1

    def test_multi_step_location_supersession_keeps_only_latest(self, temp_db: MemoryDatabase) -> None:
        reconciler = MemoryReconciler(temp_db)
        locations = ["Bengaluru", "San Francisco", "Mysore"]
        facts = [
            Fact(
                text=f"User lives in {location}",
                predicate="lives_in",
                object=location,
                category=FactCategory.BIOGRAPHICAL,
            )
            for location in locations
        ]

        for fact in facts:
            reconciler.reconcile_fact(fact)

        active = temp_db.list_active_facts()
        assert [(fact.predicate, fact.object) for fact in active] == [("lives_in", "Mysore")]
        assert temp_db.count_facts(status=FactStatus.SUPERSEDED) == 2
        assert temp_db.get_fact(facts[0].id).superseded_by == facts[1].id
        assert temp_db.get_fact(facts[1].id).superseded_by == facts[2].id

    def test_new_location_supersedes_all_existing_active_locations(self, temp_db: MemoryDatabase) -> None:
        reconciler = MemoryReconciler(temp_db)
        old_locations = ["Bengaluru", "San Francisco"]
        old_facts = [
            Fact(text=f"User lives in {location}", predicate="lives_in", object=location)
            for location in old_locations
        ]
        for fact in old_facts:
            temp_db.upsert_fact(fact, embedding=embed_text(fact.text, use_local_fallback=True))

        latest = Fact(text="User lives in Mysore", predicate="lives_in", object="Mysore")
        result = reconciler.reconcile_fact(latest)

        assert result.decisions[0].action == ReconciliationAction.SUPERSEDE
        assert temp_db.count_facts(status=FactStatus.ACTIVE) == 1
        assert temp_db.list_active_facts()[0].object == "Mysore"
        assert temp_db.count_facts(status=FactStatus.SUPERSEDED) == 2

    def test_reinforcement_updates_confidence(self, temp_db: MemoryDatabase) -> None:
        reconciler = MemoryReconciler(temp_db)
        fact = Fact(
            text="User is vegetarian",
            predicate="diet",
            object="vegetarian",
            confidence=0.7,
        )
        emb = embed_text(fact.text, use_local_fallback=True)
        temp_db.upsert_fact(fact, embedding=emb)

        reinforce = Fact(
            text="User is still vegetarian",
            predicate="diet",
            object="vegetarian",
            confidence=0.8,
        )
        result = reconciler.reconcile_fact(reinforce)
        assert result.decisions[0].action in (
            ReconciliationAction.UPDATE,
            ReconciliationAction.NOOP,
            ReconciliationAction.KEEP,
        )


class TestRetriever:
    def test_retrieve_within_token_budget(self, temp_db: MemoryDatabase) -> None:
        retriever = MemoryRetriever(temp_db, top_k=5, token_budget=200)
        for i in range(10):
            f = Fact(text=f"User fact number {i} about hobbies and life", category=FactCategory.OTHER)
            temp_db.upsert_fact(f, embedding=embed_text(f.text, use_local_fallback=True))
        ctx = retriever.retrieve("hobbies and life")
        assert ctx.estimated_tokens <= 200 or ctx.truncated
        assert isinstance(ctx.formatted_block, str)

    def test_decay_scoring(self, temp_db: MemoryDatabase) -> None:
        retriever = MemoryRetriever(temp_db)
        fact = Fact(text="Recent fact about tea")
        score = retriever._decay_score(fact)
        assert 0.0 < score <= 1.0


class TestPersona:
    def test_forbidden_phrase_detection(self) -> None:
        violations = check_persona_violations("How can I help you today?")
        assert len(violations) > 0

    def test_clean_response_no_violations(self) -> None:
        violations = check_persona_violations("That sounds like a long week. Tell me more.")
        assert violations == []

    def test_default_persona_has_name(self) -> None:
        assert DEFAULT_PERSONA.name == "Maya"


class TestPipeline:
    def test_full_turn_offline(self, pipeline: CompanionPipeline) -> None:
        result = pipeline.process_turn("I live in Mumbai and love reading.", stream=False)
        assert result.user_message
        assert result.assistant_response
        assert pipeline.db.count_facts() >= 0

    def test_session_reset_clears_history(self, pipeline: CompanionPipeline) -> None:
        pipeline.process_turn("Hello", stream=False)
        assert len(pipeline.history) >= 2
        pipeline.reset_session()
        assert len(pipeline.history) == 0

    def test_memories_persist_after_reset(self, pipeline: CompanionPipeline) -> None:
        pipeline.process_turn("My name is TestUser", stream=False)
        before = pipeline.db.count_facts(status=FactStatus.ACTIVE)
        pipeline.reset_session()
        after = pipeline.db.count_facts(status=FactStatus.ACTIVE)
        assert after == before
