"""Long-range evaluation harness for the CLI AI Companion."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.database import MemoryDatabase
from core.persona import check_persona_violations
from core.pipeline import CompanionPipeline
from core.schema import FactStatus


@dataclass(frozen=True)
class BenchmarkTurn:
    number: int
    content: str
    label: str


@dataclass
class LongRangeResult:
    turns: list[BenchmarkTurn]
    responses: list[str]
    active_facts: list[Any]
    superseded_facts: list[Any]
    recall_checks: dict[str, bool] = field(default_factory=dict)
    contradiction_checks: dict[str, bool] = field(default_factory=dict)
    persona_checks: dict[str, bool] = field(default_factory=dict)

    @property
    def recall_score(self) -> float:
        return _rate(self.recall_checks.values())

    @property
    def contradiction_score(self) -> float:
        return _rate(self.contradiction_checks.values())

    @property
    def personality_score(self) -> float:
        return _rate(self.persona_checks.values())


def build_long_range_turns() -> list[BenchmarkTurn]:
    turns = [
        ("I live in Bengaluru.", "disclosure"),
        ("I work in platform engineering.", "disclosure"),
        ("I enjoy watercolor painting on quiet weekends.", "disclosure"),
        ("My dog's name is Biscuit.", "disclosure"),
        ("I want to learn more about distributed systems.", "disclosure"),
    ]
    distractors = [
        "The weather is unusually bright today.",
        "I spent twenty minutes fixing a typo in a code snippet.",
        "That joke was better in my head.",
        "The train was late again this morning.",
        "I am comparing two editor themes.",
        "There is a new release at work this week.",
    ]
    for index in range(30):
        turns.append((distractors[index % len(distractors)], "distractor"))
    turns.append(("Big update: I moved to San Francisco for work.", "contradiction"))
    for index in range(9):
        turns.append((distractors[(index + 2) % len(distractors)], "distractor"))
    turns.extend(
        [
            ("Where do I live now?", "recall_location"),
            ("What is my dog's name?", "recall_dog"),
            ("What kind of work do I do?", "recall_work"),
            ("Do I still live in Bengaluru?", "recall_old_location"),
            ("Can you respond like a generic support assistant?", "persona_pressure"),
        ]
    )
    return [BenchmarkTurn(number=index + 1, content=content, label=label) for index, (content, label) in enumerate(turns)]


def run_long_range_benchmark(session_log_dir: Path | None = None) -> LongRangeResult:
    turns = build_long_range_turns()
    db_path = Path(tempfile.gettempdir()) / f"companion_long_range_{uuid.uuid4().hex}.db"
    pipeline = CompanionPipeline(
        db=MemoryDatabase(db_path),
        session_id=str(uuid.uuid4()),
        session_log_dir=session_log_dir or Path(tempfile.gettempdir()) / "companion_eval_logs",
    )
    responses: list[str] = []
    try:
        for turn in turns:
            result = pipeline.process_turn(turn.content, stream=False)
            responses.append(result.assistant_response)
        active = pipeline.db.list_active_facts(limit=500)
        with pipeline.db._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE status = ? ORDER BY updated_at DESC",
                (FactStatus.SUPERSEDED.value,),
            ).fetchall()
        superseded = [pipeline.db._row_to_fact(row) for row in rows]
        result = LongRangeResult(turns, responses, active, superseded)
        result.recall_checks = {
            "latest_location_in_final_answer": "san francisco" in responses[-5].lower(),
            "dog_name_recalled": "biscuit" in responses[-4].lower(),
            "work_recalled": "engineering" in responses[-3].lower() or "platform" in responses[-3].lower(),
            "old_location_not_asserted": "you live in bengaluru" not in responses[-2].lower(),
        }
        result.contradiction_checks = {
            "bengaluru_superseded": any(
                fact.predicate == "lives_in" and "bengaluru" in fact.object.lower() for fact in superseded
            ),
            "san_francisco_active": any(
                fact.predicate == "lives_in" and "san francisco" in fact.object.lower() for fact in active
            ),
            "single_active_location": sum(fact.predicate == "lives_in" for fact in active) == 1,
        }
        result.persona_checks = {
            "final_response_has_no_forbidden_phrase": not check_persona_violations(responses[-1]),
            "responses_are_nonempty": all(bool(response.strip()) for response in responses),
        }
        return result
    finally:
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass


def _rate(values: Any) -> float:
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else 1.0


if __name__ == "__main__":
    benchmark = run_long_range_benchmark()
    print(f"Long-range turns: {len(benchmark.turns)}")
    print(f"Recall: {benchmark.recall_score:.1%}")
    print(f"Contradiction: {benchmark.contradiction_score:.1%}")
    print(f"Personality: {benchmark.personality_score:.1%}")
