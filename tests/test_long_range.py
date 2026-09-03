"""Executable 50+ turn long-range memory benchmark."""

from __future__ import annotations

from tests.eval_harness import build_long_range_turns, run_long_range_benchmark


def test_benchmark_contains_required_long_range_phases() -> None:
    turns = build_long_range_turns()
    assert len(turns) == 50
    assert [turn.label for turn in turns[:5]] == ["disclosure"] * 5
    assert turns[35].label == "contradiction"
    assert all(turn.label == "distractor" for turn in turns[5:35])
    assert all(turn.label == "distractor" for turn in turns[36:45])
    assert [turn.label for turn in turns[45:]] == [
        "recall_location", "recall_dog", "recall_work", "recall_old_location", "persona_pressure",
    ]


def test_long_range_memory_and_persona_benchmark() -> None:
    result = run_long_range_benchmark()
    assert result.recall_score == 1.0
    assert result.contradiction_score == 1.0
    assert result.personality_score == 1.0
