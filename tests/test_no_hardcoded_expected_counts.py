"""P0.3 guard: the benchmark's manual expectation list must never be hardcoded
into the production pipeline.

AGENTS.md §0: the manual benchmark counts (weighing + pipetting×3, per-object
interaction counts) are POST-RUN evaluation only. They must not appear as
detection logic, forced window counts, forced action counts, or pre-labeled
materials anywhere in the production source tree.

This is a static source scan — no GPU, no pipeline execution.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "key_action_indexer"

# Files that legitimately discuss expected-vs-detected for the *evaluation*
# layer. They read expected files; they never inject counts into detection.
ALLOWED_FILES = {
    "benchmark_accuracy.py",   # post-run scoring (reads expected_*.json)
}

# Tokens that would indicate the benchmark crib sheet leaked into code.
# These are specific enough to avoid matching ordinary thresholds.
FORBIDDEN_PATTERNS = [
    # forcing the number of experiments to 4 (weighing + pipetting x3)
    re.compile(r"expected_experiment_count\s*=\s*4\b"),
    re.compile(r"force.*window.*count", re.IGNORECASE),
    re.compile(r"hardcoded?_expected", re.IGNORECASE),
    # pre-baking the weighing crib counts (e.g. balance interaction: 3 times)
    re.compile(r"weighing_paper.*=\s*2\b", re.IGNORECASE),
    re.compile(r"balance_interaction(s)?\s*=\s*3\b", re.IGNORECASE),
]


def _python_sources():
    for path in SRC.rglob("*.py"):
        if path.name in ALLOWED_FILES:
            continue
        yield path


def test_no_forbidden_expected_count_patterns():
    offenders: list[str] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN_PATTERNS:
            for m in pat.finditer(text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}: {m.group(0)!r}")
    assert not offenders, (
        "Benchmark expectation values appear hardcoded in production source:\n"
        + "\n".join(offenders)
    )


def test_expected_experiment_count_only_a_config_field():
    """expected_experiment_count may exist as an optional CONFIG field, but must
    never be assigned a literal benchmark value (4) in source logic."""
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Allow the dataclass default of None; forbid a literal numeric default.
        for m in re.finditer(r"expected_experiment_count\s*[:=].*", text):
            snippet = m.group(0)
            assert "= 4" not in snippet and ": 4" not in snippet, (
                f"{path.name}: expected_experiment_count pinned to a literal: {snippet!r}"
            )
