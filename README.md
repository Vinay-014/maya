# MAYA - a CLI AI Companion

A terminal AI companion prototype with **long-term memory persistence**, **semantic retrieval**, **fact reconciliation** (contradiction handling), and **immutable persona guardrails**.

!(https://github.com/Vinay-014/maya/blob/main/Screen%20Recording%202026-09-03%20223151.gif)

## Features

- **Structured fact extraction** — Pydantic schemas + LLM structured outputs extract subject/predicate/object facts from each turn
- **Memory reconciliation** — Detects contradictions (e.g. city moves, preference reversals) and marks old facts as `superseded`
- **Hybrid retrieval** — Vector similarity + keyword boost + exponential decay scoring, with strict token budgeting
- **Persona lock-in** — Maya, a warm literary companion with hard anti-assistant guardrails
- **Persistent storage** — Single-file SQLite database (`companion_memory.db`) with embeddings stored as float32 blobs
- **Evaluation harness** — 80+ synthetic turns across 10 scenarios with precision/recall/persona metrics

## Architecture

Every user turn flows through a strict pipeline:

```
User Input
    → Extraction Engine (structured facts)
    → Memory Reconciliation (contradiction resolution)
    → Context-Aware Retrieval (hybrid search, token budget)
    → Persona-Constrained Generation
    → Response + Persistence (SQLite + JSONL session logs)
```

### Design Tradeoffs

| Decision | Rationale | What Was Tried & Abandoned |
|----------|-----------|----------------------------|
| **SQLite + in-process vector search** | Single portable file; no external vector DB daemon. Embeddings stored as BLOB; cosine similarity computed in NumPy. Avoids `sqlite-vec` compilation issues on various OS environments while keeping hybrid storage in one file. | Tried running an external vector store (Chroma/Qdrant) — abandoned due to heavy daemon dependencies and process lifecycle management issues for a lightweight CLI. |
| **LiteLLM unified gateway** | Model-agnostic Gemini/Groq/OpenAI routing with one API surface and Pydantic validation. | Tried vendor-specific SDK branching in every module — abandoned due to fragile error-handling divergence between SDKs. |
| **Heuristic fallbacks** | Extraction, reconciliation, and generation degrade gracefully without API keys — useful for tests and offline demos. | Tried pure prompt-only fallback — abandoned because zero-network test suites require deterministic regex/rule baseline guarantees. |
| **Supersession model** | Facts are never hard-deleted; contradictions mark old rows `superseded` with `superseded_by` FK for auditability and lineage tracking. | Tried overwriting rows in place (`UPDATE memories SET text=...`) — abandoned because history/provenance was lost, preventing the companion from knowing *when* or *why* a fact changed. |
| **Token-budgeted memory block** | Prevents context window stuffing; only top-k active facts within budget are injected. | Tried dumping all active memories into context — abandoned because prompt bloat degraded response adherence and violated model limits. |

### Memory Lifecycle

1. **Extract** — New facts get `status=active`, confidence score, embedding
2. **Reconcile** — Compare against top-k semantic candidates:
   - `SUPERSEDE` → old fact retired (`status=superseded`), new fact inserted
   - `UPDATE` → confidence boosted, `last_accessed` touched
   - `KEEP` → novel fact inserted
   - `NOOP` → duplicate skipped
3. **Retrieve** — Only `active` facts; scored by vector + decay + confidence
4. **Decay** — Exponential half-life (default 30 days since last access)

## Quick Start

```bash
# Clone / enter project
cd "CLI AI companion"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
# Or: pip install -e .

# Configure Gemini or Groq API key
cp .env.example .env          # macOS/Linux
# copy .env.example .env      # Windows
# Edit .env and set GEMINI_API_KEY or GROQ_API_KEY

# Run interactive companion
python main.py
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/memories` | Inspect active memory store |
| `/memories search <query>` | Search memories via vector + keyword ranking |
| `/memories forget <id>` | Retire an active memory manually |
| `/history` | View current session conversation log |
| `/tools` | List registered agent tools |
| `/reset` | Clear session history (memories persist in SQLite) |
| `/exit` | Quit the application |

## Running Tests

```bash
pytest tests/ -v
```

Tests use temporary databases and local hash embeddings — **no external API key required**.

## Running Evaluation

```bash
# Run the 10-scenario synthetic test suite
python -m eval.evaluator

# Run the 50-turn long-range benchmark, LLM judge, and oracle baseline
python -m tests.eval_judge
```

The evaluator loads `eval/test_suite.json` (10 scenarios, 80+ turns) and reports:
- **Recall score** — Long-range fact recall, memory presence checks
- **Contradiction score** — Supersession and confidence update accuracy
- **Persona score** — Forbidden phrase / assistant drift detection
- **Overall weighted score**

## Known Limitations

1. **Embedding Semantic Entailment**: Vector cosine similarity measures topic proximity rather than strict logical contradiction. Contradiction detection relies on structured predicate matching (`subject`, `predicate`, `object`) supplemented by LLM reconciliation.
2. **Decay Dynamics**: Time decay is calculated based on system clock `last_accessed_at`. In simulated or rapid-fire testing, real-world temporal decay of 30 days cannot be exercised without synthetic clock spoofing.
3. **Paraphrase Sensitivity in Deterministic Eval**: Exact keyword matching in deterministic assertions can occasionally flag valid conversational paraphrases as false negatives if not broad enough.

## License

MIT — prototype for research and development.
