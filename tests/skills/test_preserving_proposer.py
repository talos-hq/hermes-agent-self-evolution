"""Zero-cost tests for the preservation-aware GEPA proposer.

No LLM calls: the proposer takes an injectable `generate` callable, so its
prompt-routing and preservation guard are tested deterministically.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

import dspy

from evolution.skills.preserving_proposer import (
    SkillPreservingProposer,
    render_reflective_dataset,
    _strip_code_fences,
)

SKILL = "---\nname: x\n---\n\n# Doc\n\n## Procedure\n\n1. step one\n2. step two\n" + ("detail line\n" * 40)


def test_strip_code_fences():
    assert _strip_code_fences("```markdown\nhello\n```") == "hello"
    assert _strip_code_fences("no fences here") == "no fences here"


def test_render_reflective_dataset():
    entries = [{"Inputs": "task A", "Feedback": "missed verification"}]
    out = render_reflective_dataset(entries)
    assert "task A" in out and "missed verification" in out
    assert render_reflective_dataset([]) == "(no feedback provided)"


def test_guard_rejects_destructive_proposal():
    p = SkillPreservingProposer()
    gutted = "short summary"
    assert p._guard(SKILL, gutted) == SKILL          # rejected -> unchanged
    assert p._guard(SKILL, "") == SKILL               # empty -> unchanged


def test_guard_accepts_preserving_proposal():
    p = SkillPreservingProposer()
    good = SKILL + "\n## Pitfall\n\n- check the log first\n"
    assert p._guard(SKILL, good) == good.strip()


def test_call_routes_components_and_applies_guard():
    calls = {}

    def fake_generate(current, feedback):
        calls["feedback"] = feedback
        return current + "\n## Added\n\n- something\n"

    p = SkillPreservingProposer(generate=fake_generate)
    out = p(
        candidate={"predictor": SKILL},
        reflective_dataset={"predictor": [{"Feedback": "add a pitfall"}]},
        components_to_update=["predictor", "missing_component"],
    )
    assert set(out) == {"predictor"}                  # missing component skipped
    assert out["predictor"].startswith(SKILL)
    assert "add a pitfall" in calls["feedback"]


def test_call_guard_blocks_gutting_generate():
    p = SkillPreservingProposer(generate=lambda current, feedback: "tiny")
    out = p({"predictor": SKILL}, {"predictor": []}, ["predictor"])
    assert out["predictor"] == SKILL


def test_gepa_accepts_custom_instruction_proposer():
    """Construction only — proves the ProposalFn is accepted, no API call."""
    opt = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 1.0,
        max_full_evals=1,
        reflection_lm=dspy.LM("openai/gpt-4o-mini"),
        instruction_proposer=SkillPreservingProposer(),
    )
    assert opt is not None
