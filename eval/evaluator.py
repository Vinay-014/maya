"""LLM-as-a-judge evaluation harness for memory and persona stress tests."""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.database import MemoryDatabase
from core.persona import check_persona_violations
from core.pipeline import CompanionPipeline
from core.schema import FactStatus

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
TEST_SUITE_PATH = EVAL_DIR / "test_suite.json"


@dataclass
class AssertionResult:
    scenario_id: str
    assertion: dict[str, Any]
    passed: bool
    detail: str


@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    total_turns: int
    assertion_results: list[AssertionResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.assertion_results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.assertion_results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.assertion_results:
            return 1.0
        return self.passed / len(self.assertion_results)


@dataclass
class EvalReport:
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    recall_score: float = 0.0
    contradiction_score: float = 0.0
    persona_score: float = 0.0
    overall_score: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CLI AI Companion — Evaluation Report",
            "=" * 60,
            f"Overall score:     {self.overall_score:.1%}",
            f"Recall score:      {self.recall_score:.1%}",
            f"Contradiction:     {self.contradiction_score:.1%}",
            f"Persona score:     {self.persona_score:.1%}",
            "-" * 60,
        ]
        for sr in self.scenario_results:
            status = "PASS" if sr.failed == 0 else "FAIL"
            lines.append(f"[{status}] {sr.name} ({sr.passed}/{len(sr.assertion_results)} assertions)")
            for ar in sr.assertion_results:
                mark = "OK" if ar.passed else "FAIL"
                lines.append(f"    [{mark}] {ar.assertion.get('type', '?')}: {ar.detail}")
        lines.append("=" * 60)
        return "\n".join(lines)


class Evaluator:
    """Runs automated multi-turn stress tests from test_suite.json."""

    def __init__(self, suite_path: Path | None = None) -> None:
        self.suite_path = suite_path or TEST_SUITE_PATH
        with self.suite_path.open(encoding="utf-8") as fh:
            self.suite = json.load(fh)
        self.weights = self.suite.get("metrics", {})

    def run_all(self) -> EvalReport:
        report = EvalReport()
        recall_pass = recall_total = 0
        contradiction_pass = contradiction_total = 0
        persona_pass = persona_total = 0

        for scenario in self.suite["scenarios"]:
            sr = self.run_scenario(scenario)
            report.scenario_results.append(sr)
            for ar in sr.assertion_results:
                atype = ar.assertion.get("type", "")
                if atype in ("recall", "recall_negation", "memory_active", "memory_count_min"):
                    recall_total += 1
                    recall_pass += int(ar.passed)
                elif atype in ("memory_superseded", "confidence_min"):
                    contradiction_total += 1
                    contradiction_pass += int(ar.passed)
                elif atype == "persona":
                    persona_total += 1
                    persona_pass += int(ar.passed)

        report.recall_score = recall_pass / recall_total if recall_total else 1.0
        report.contradiction_score = contradiction_pass / contradiction_total if contradiction_total else 1.0
        report.persona_score = persona_pass / persona_total if persona_total else 1.0

        rw = self.weights.get("recall_weight", 0.4)
        cw = self.weights.get("contradiction_weight", 0.35)
        pw = self.weights.get("persona_weight", 0.25)
        report.overall_score = (
            rw * report.recall_score
            + cw * report.contradiction_score
            + pw * report.persona_score
        )
        return report

    def run_scenario(self, scenario: dict[str, Any]) -> ScenarioResult:
        db_path = Path(tempfile.gettempdir()) / f"companion_eval_{uuid.uuid4().hex}.db"
        pipeline = CompanionPipeline(db=MemoryDatabase(db_path), session_id=str(uuid.uuid4()))

        responses: list[str] = []
        turns = scenario.get("turns", [])

        for turn in turns:
            if turn["role"] != "user":
                continue
            turn_result = pipeline.process_turn(turn["content"], stream=False)
            assert not hasattr(turn_result, "__next__")  # sync result, not generator
            responses.append(turn_result.assistant_response)

        sr = ScenarioResult(
            scenario_id=scenario["id"],
            name=scenario["name"],
            total_turns=len(turns),
        )

        for assertion in scenario.get("assertions", []):
            ar = self._evaluate_assertion(
                scenario_id=scenario["id"],
                assertion=assertion,
                pipeline=pipeline,
                responses=responses,
            )
            sr.assertion_results.append(ar)

        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass

        return sr

    def _evaluate_assertion(
        self,
        scenario_id: str,
        assertion: dict[str, Any],
        pipeline: CompanionPipeline,
        responses: list[str],
    ) -> AssertionResult:
        atype = assertion.get("type", "")
        after_turn = assertion.get("after_turn", len(responses))
        response_idx = min(max(after_turn - 1, 0), len(responses) - 1) if responses else 0
        response_text = responses[response_idx].lower() if responses else ""

        try:
            if atype == "recall":
                keywords = [k.lower() for k in assertion.get("must_contain_any", [])]
                passed = any(k in response_text for k in keywords)
                detail = f"Response must contain one of {keywords}; got excerpt: {response_text[:80]!r}"
                return AssertionResult(scenario_id, assertion, passed, detail)

            if atype == "recall_negation":
                forbidden = [k.lower() for k in assertion.get("must_not_contain", [])]
                passed = not any(k in response_text for k in forbidden)
                detail = f"Response must not contain {forbidden}"
                return AssertionResult(scenario_id, assertion, passed, detail)

            if atype == "memory_active":
                facts = pipeline.db.list_active_facts(limit=200)
                passed = self._match_fact(facts, assertion, status=FactStatus.ACTIVE)
                detail = f"Looking for active memory matching {assertion}"
                return AssertionResult(scenario_id, assertion, passed, detail)

            if atype == "memory_superseded":
                with pipeline.db._connect() as conn:
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE status = ?",
                        (FactStatus.SUPERSEDED.value,),
                    ).fetchall()
                facts = [pipeline.db._row_to_fact(row) for row in rows]
                passed = self._match_fact(facts, assertion, status=FactStatus.SUPERSEDED)
                detail = f"Looking for superseded memory matching {assertion}"
                return AssertionResult(scenario_id, assertion, passed, detail)

            if atype == "memory_count_min":
                count = pipeline.db.count_facts(status=FactStatus.ACTIVE)
                minimum = assertion.get("min_active", 1)
                passed = count >= minimum
                detail = f"Active memories={count}, required>={minimum}"
                return AssertionResult(scenario_id, assertion, passed, detail)

            if atype == "confidence_min":
                facts = pipeline.db.list_active_facts(limit=200)
                text_part = assertion.get("text_contains", "").lower()
                min_conf = assertion.get("min_confidence", 0.7)
                matched = [f for f in facts if text_part in f.text.lower()]
                passed = any(f.confidence >= min_conf for f in matched)
                detail = f"Confidence check for '{text_part}': found {len(matched)} facts"
                return AssertionResult(scenario_id, assertion, passed, detail)

            if atype == "persona":
                resp = responses[response_idx] if responses else ""
                forbidden = assertion.get("forbidden_phrases", [])
                violations = check_persona_violations(resp)
                extra = [p for p in forbidden if p.lower() in resp.lower()]
                passed = not violations and not extra
                detail = f"Violations: {violations + extra}" if not passed else "No persona violations"
                return AssertionResult(scenario_id, assertion, passed, detail)

            return AssertionResult(scenario_id, assertion, False, f"Unknown assertion type: {atype}")
        except Exception as exc:
            return AssertionResult(scenario_id, assertion, False, f"Error: {exc}")

    def _match_fact(self, facts: list, assertion: dict[str, Any], status: FactStatus) -> bool:
        for fact in facts:
            if fact.status != status:
                continue
            if "predicate" in assertion and assertion["predicate"]:
                if fact.predicate.lower() != assertion["predicate"].lower():
                    continue
            if "expected_object" in assertion:
                if assertion["expected_object"].lower() not in (fact.object or "").lower():
                    if assertion["expected_object"].lower() not in fact.text.lower():
                        continue
            if "category" in assertion:
                cat = assertion["category"]
                if str(fact.category.value) != cat and str(fact.category) != cat:
                    continue
            if "text_contains" in assertion:
                if assertion["text_contains"].lower() not in fact.text.lower():
                    continue
            return True
        return False


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    evaluator = Evaluator()
    report = evaluator.run_all()
    print(report.summary())


if __name__ == "__main__":
    main()
