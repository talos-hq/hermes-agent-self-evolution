"""A preservation-aware instruction proposer for dspy.GEPA.

Why this exists
---------------
dspy's default instruction proposer is built to write *short* task
instructions. Pointed at a Hermes skill — a long procedural document — it
rewrites the whole thing into a terse generic instruction and destroys the
skill (measured: a 15,155-char skill came back at 4,390 chars, judge score
−0.078).

This proposer instead asks the reflection LM for the **complete document with
minimal, targeted edits**, and then enforces preservation in code: if the model
drops too much of the original, the proposal is rejected and the current text is
returned unchanged (so GEPA simply records "no improvement" rather than
producing a gutted skill).

It implements GEPA's ``ProposalFn`` protocol::

    __call__(candidate, reflective_dataset, components_to_update) -> dict[name, str]
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

import dspy


_SYSTEM_PREAMBLE = """You are editing a Hermes Agent skill document.

The skill is a long, procedural reference document that an AI agent follows.
Your job is to produce an IMPROVED version of the FULL document that addresses
the feedback below.

HARD RULES — a violation makes your output unusable:
1. Return the ENTIRE document, from its first line to its last. Never
   summarise, never abbreviate, never return only the changed part.
2. PRESERVE everything already in it: every heading, numbered step, command,
   code block, path, flag, table row and caveat. Do not delete or "tidy"
   existing content.
3. Make MINIMAL, TARGETED edits — clarify a step, fix an ordering problem, add
   a missing pitfall or verification step directly implied by the feedback.
4. Do not add filler, restated generalities, or marketing prose. Never grow the
   document by more than ~20%.
5. Do not invent commands, flags, paths or APIs that are not already in the
   document or clearly implied by the feedback.
6. Output markdown only — no preamble like "Here is the improved skill", and no
   surrounding code fences.
"""


class _ReviseSkill(dspy.Signature):
    """Revise a Hermes skill document, preserving all existing content."""

    current_skill: str = dspy.InputField(desc="The complete current skill document")
    feedback_examples: str = dspy.InputField(
        desc="Observed failures and reviewer feedback to address"
    )
    revised_skill: str = dspy.OutputField(
        desc="The complete revised skill document, preserving all existing content"
    )


def _strip_code_fences(text: str) -> str:
    """Remove a single wrapping ```...``` fence if the model added one."""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\n(.*)\n```$", t, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    return t


def render_reflective_dataset(entries: Sequence[Mapping[str, Any]], limit_each: int = 1200) -> str:
    """Render GEPA's reflective dataset entries into readable feedback text."""
    if not entries:
        return "(no feedback provided)"
    blocks: list[str] = []
    for i, ex in enumerate(entries, 1):
        lines = [f"--- example {i} ---"]
        for key, value in ex.items():
            text = str(value)
            if len(text) > limit_each:
                text = text[:limit_each] + " …[truncated]"
            lines.append(f"{key}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class SkillPreservingProposer:
    """GEPA ProposalFn that edits a skill without gutting it.

    Args:
        min_keep: reject a proposal whose length is below this fraction of the
            current document (default 0.95 = allow at most a 5% trim).
        generate: optional ``callable(current_skill, feedback) -> str`` used
            instead of a dspy predictor. Injected by tests to run with no LLM.
    """

    def __init__(self, min_keep: float = 0.95, generate: Callable[[str, str], str] | None = None):
        self.min_keep = min_keep
        self._generate = generate
        self._predictor: dspy.Predict | None = None

    # -- internals ---------------------------------------------------------
    def _propose(self, current: str, feedback: str) -> str:
        if self._generate is not None:
            return self._generate(current, feedback)
        if self._predictor is None:
            sig = _ReviseSkill.with_instructions(_SYSTEM_PREAMBLE)
            self._predictor = dspy.Predict(sig)
        # Runs inside GEPA's reflection_lm context (it sets the LM before
        # calling the proposer), so no LM needs to be passed here.
        pred = self._predictor(current_skill=current, feedback_examples=feedback)
        return pred.revised_skill

    def _guard(self, current: str, proposal: str) -> str:
        """Return the proposal only if it preserves the document; else current."""
        candidate = _strip_code_fences(proposal or "")
        if not candidate:
            return current
        if len(candidate) < self.min_keep * len(current):
            # Model summarised/dropped content — refuse; keeps GEPA honest.
            return current
        return candidate

    # -- ProposalFn protocol ----------------------------------------------
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        for name in components_to_update:
            if name not in candidate:
                continue
            current = candidate[name]
            feedback = render_reflective_dataset(reflective_dataset.get(name, []))
            results[name] = self._guard(current, self._propose(current, feedback))
        return results
