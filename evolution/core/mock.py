"""Zero-cost mock components for exercising the evolution pipeline.

These let `evolve_skill --mock` run the FULL pipeline — dataset → optimize →
constraint gates → holdout scoring → metrics → output files — with **no LLM
calls and no spend**, so the plumbing can be tested in CI / before a paid run.

They exist because the upstream Phase-1 code passed the skill text as an
*input field* (which instruction optimizers never mutate). `MockOptimizer`
mutates `module.skill_text` (the predictor's instructions) exactly the way
dspy.GEPA does, so a mock run proves the write-back path is wired correctly.
"""

from __future__ import annotations

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample


class MockDatasetBuilder:
    """Deterministic stand-in for SyntheticDatasetBuilder (no LLM calls)."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: int | None = None,
    ) -> EvalDataset:
        n = num_cases or self.config.eval_dataset_size
        words = [w.strip(".,:#*-`()[]\"'") for w in artifact_text.split() if len(w) > 4]

        examples = [
            EvalExample(
                task_input=f"[mock] Apply the {artifact_type} to case #{i}.",
                expected_behavior=(
                    "[mock rubric] A good response should reference "
                    + (" ".join(words[i: i + 6]) or "the documented procedure")
                    + "."
                ),
                difficulty=["easy", "medium", "hard"][i % 3],
                category="mock",
                source="mock",
            )
            for i in range(n)
        ]

        n_train = max(1, int(n * self.config.train_ratio))
        n_val = max(1, int(n * self.config.val_ratio))
        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train: n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


class MockOptimizer:
    """Stand-in optimizer that mutates the module's instructions.

    Mirrors what dspy.GEPA / MIPROv2 do to a component (they rewrite
    ``predictor.signature.instructions``), so the downstream read-back in
    ``evolve_skill.evolve`` is genuinely exercised — with zero API calls.
    """

    def __init__(self, suffix: str | None = None):
        # Kept tiny on purpose: growth_limit (max +20% over baseline) is a real,
        # enforced gate, so the mock mutation must stay small on a small skill.
        self.suffix = suffix or "\n\n(mock-evolved)\n"

    def compile(self, module, trainset=None, valset=None, **kwargs) -> dspy.Module:
        from evolution.skills.skill_module import SkillModule

        evolved_text = module.skill_text.rstrip() + self.suffix
        return SkillModule(evolved_text)


def mock_forward_text(module) -> str:
    """Deterministic 'agent output' for a module in mock mode (no LM call).

    Uses the module's current instruction as the stand-in response so holdout
    scoring, constraint checks, and metrics all run for real — just locally.
    """
    return module.skill_text
