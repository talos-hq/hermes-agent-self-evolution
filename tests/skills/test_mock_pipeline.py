"""Zero-cost tests for the evolution pipeline plumbing.

These run with NO API calls: a mock dataset + mock optimizer exercise the same
code path a real run uses (skill → module → optimizer write-back → constraint
gates → holdout scoring → metrics → output files). Run before spending money.

    uv run pytest tests/skills/test_mock_pipeline.py -q
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

import dspy

from evolution.core.mock import MockDatasetBuilder, MockOptimizer, mock_forward_text
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig
from evolution.core.fitness import skill_fitness_metric
from evolution.skills import evolve_skill
from evolution.skills.evolve_skill import _gepa_metric, evolve
from evolution.skills.skill_module import SkillModule, load_skill, reassemble_skill

SAMPLE_SKILL = """---
name: sample-skill
description: "A tiny skill used for tests."
version: 1.0.0
---

# Sample skill

## Procedure

1. Do the first thing carefully.
2. Then do the second thing.
3. Verify the result.
"""


def _make_repo(tmp_path: Path) -> Path:
    """Create a fake hermes-agent repo with one skill under skills/."""
    skill_dir = tmp_path / "repo" / "skills" / "testing" / "sample-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL)
    return tmp_path / "repo"


# ── The core defect: skill text must be the OPTIMIZABLE INSTRUCTION ──────────

def test_skill_text_is_the_predictor_instruction():
    mod = SkillModule("PROCEDURE TEXT HERE")
    assert mod.skill_text == "PROCEDURE TEXT HERE"
    # It is literally the predictor's instruction (what GEPA mutates):
    assert mod.predictor.signature.instructions == "PROCEDURE TEXT HERE"
    # ...and NOT an input field (the upstream bug):
    assert "skill_instructions" not in mod.predictor.signature.input_fields


def test_mock_optimizer_changes_the_read_back_text():
    mod = SkillModule("original body")
    evolved = MockOptimizer().compile(mod)
    assert evolved.skill_text != mod.skill_text
    assert evolved.skill_text.startswith("original body")
    assert "mock-evolved" in evolved.skill_text


# ── The real optimizer must still accept our kwargs (no API call) ───────────

def test_real_gepa_constructor_accepts_our_kwargs():
    """Guards against dspy drift: GEPA must take max_full_evals + reflection_lm.

    Constructing an optimizer does not call the API, so this is free.
    """
    reflection_lm = dspy.LM("openai/gpt-4o-mini")
    opt = dspy.GEPA(metric=_gepa_metric, max_full_evals=2, reflection_lm=reflection_lm)
    assert opt is not None


def test_gepa_metric_accepts_gepa_arity():
    ex = dspy.Example(task_input="t", expected_behavior="do the thing").with_inputs("task_input")
    pred = dspy.Prediction(output="do the thing")
    # GEPA calls metric(gold, pred, trace, pred_name, pred_trace)
    score = _gepa_metric(ex, pred, None, "predictor", None)
    assert 0.0 <= score <= 1.0


def test_optuna_available_for_miprov2_fallback():
    import optuna  # noqa: F401  — required by dspy.MIPROv2


# ── Full pipeline, end to end, mock only ────────────────────────────────────

def test_full_mock_pipeline_writes_outputs(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    evolve(
        skill_name="sample-skill",
        iterations=2,
        hermes_repo=str(repo),
        mock=True,
        dataset_size=8,
    )

    out_root = tmp_path / "output" / "sample-skill"
    runs = sorted(out_root.iterdir())
    assert runs, "no output run directory was created"
    run = runs[-1]

    baseline = (run / "baseline_skill.md").read_text()
    evolved = (run / "evolved_skill.md").read_text()
    metrics = json.loads((run / "metrics.json").read_text())

    # The whole point: the evolved skill actually differs from the baseline.
    assert evolved != baseline
    assert metrics["changed"] is True
    assert metrics["mode"] == "mock"
    assert metrics["constraints_passed"] is True
    assert metrics["holdout_examples"] >= 1
    assert (run / "evolved_skill.md").read_text().startswith("---")  # frontmatter preserved


def test_baseline_size_over_limit_is_not_gated(tmp_path, monkeypatch):
    """A skill already over the size limit must not be blocked by the size gate."""
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    evolve(skill_name="sample-skill", iterations=1, hermes_repo=str(repo), mock=True,
           max_skill_size=10, dataset_size=6)
    run = sorted((tmp_path / "output" / "sample-skill").iterdir())[-1]
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["constraints_passed"] is True, metrics["constraint_failures"]


def test_destructive_shrink_is_blocked(tmp_path, monkeypatch):
    """An optimizer that guts the skill must be rejected, not deployed."""
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    class ShrinkingOptimizer:
        def compile(self, module, trainset=None, valset=None, **kwargs):
            return SkillModule("tiny")

    monkeypatch.setattr(evolve_skill, "MockOptimizer", ShrinkingOptimizer)
    evolve(skill_name="sample-skill", iterations=1, hermes_repo=str(repo),
           mock=True, dataset_size=6)

    run = sorted((tmp_path / "output" / "sample-skill").iterdir())[-1]
    m = json.loads((run / "metrics.json").read_text())
    assert "content_shrink" in m["constraint_failures"]
    assert m["constraints_passed"] is False
    assert m["deployed"] is False


def test_deployed_requires_positive_improvement(tmp_path, monkeypatch):
    """No score improvement => not recommended for deployment."""
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    evolve(skill_name="sample-skill", iterations=1, hermes_repo=str(repo),
           mock=True, dataset_size=6)
    run = sorted((tmp_path / "output" / "sample-skill").iterdir())[-1]
    m = json.loads((run / "metrics.json").read_text())
    assert m["changed"] is True
    assert m["deployed"] is False  # mock produces no real improvement


# ── Unit checks on the pieces ──────────────────────────────────────────────

def test_mock_dataset_split_sizes():
    cfg = EvolutionConfig(eval_dataset_size=12)
    ds = MockDatasetBuilder(cfg).generate("some artifact text " * 20)
    assert len(ds.train) >= 1 and len(ds.val) >= 1
    assert len(ds.all_examples) == 12


def test_skill_structure_constraint_on_full_text(tmp_path):
    repo = _make_repo(tmp_path)
    skill = load_skill(repo / "skills" / "testing" / "sample-skill" / "SKILL.md")
    full = reassemble_skill(skill["frontmatter"], skill["body"])
    results = {c.constraint_name: c.passed
               for c in ConstraintValidator(EvolutionConfig()).validate_all(full, "skill")}
    assert results["skill_structure"] is True


def test_cli_mock_flag_exists():
    params = {p.name for p in evolve_skill.main.params}
    assert {"mock", "max_skill_size", "dataset_size"} <= params
