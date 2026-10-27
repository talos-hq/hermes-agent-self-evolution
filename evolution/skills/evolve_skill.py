"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/

Zero-cost smoke test (no LLM calls, no spend) — mock dataset + mock optimizer,
runs the full plumbing end to end:
    python -m evolution.skills.evolve_skill --skill <name> --mock
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.core.mock import MockDatasetBuilder, MockOptimizer, mock_forward_text
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)
from evolution.skills.preserving_proposer import SkillPreservingProposer

console = Console()


def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Adapter matching dspy's GEPAFeedbackMetric call signature.

    GEPA calls the metric as metric(gold, pred, trace, pred_name, pred_trace);
    skill_fitness_metric only takes (example, prediction, trace).
    """
    return skill_fitness_metric(gold, pred, trace)


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    mock: bool = False,
    max_skill_size: int = 15_000,
    dataset_size: Optional[int] = None,
    preserving_proposer: bool = True,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
        max_skill_size=max_skill_size,
    )
    if dataset_size:
        config.eval_dataset_size = dataset_size

    # ── 1. Find and load the skill ──────────────────────────────────────
    mode = "MOCK (no API calls)" if mock else "real"
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]  [dim]({mode})[/dim]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print("  Would validate constraints and write output")
        return

    # Full skill text (frontmatter + body) — the artifacts we actually gate.
    baseline_full = reassemble_skill(skill["frontmatter"], skill["body"])

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {'mock' if mock else eval_source})")

    if mock:
        dataset = MockDatasetBuilder(config).generate(
            artifact_text=skill["raw"], artifact_type="skill", num_cases=dataset_size
        )
        console.print(f"  [mock] Generated {len(dataset.all_examples)} mock examples (no LLM)")
    elif eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(baseline_full, "skill")
    baseline_pass = {c.constraint_name: c.passed for c in baseline_constraints}
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")

    # ── 4. Set up DSPy + optimizer ──────────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    baseline_module = SkillModule(skill["body"])
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    start_time = time.time()
    optimizer_name = "mock"
    proposer_name = "n/a"
    lm = None

    if mock:
        console.print("  Optimizer: MockOptimizer (deterministic, no API)")
        optimizer = MockOptimizer()
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    else:
        lm = dspy.LM(eval_model)
        dspy.configure(lm=lm)
        reflection_lm = dspy.LM(optimizer_model)
        console.print(f"  Optimizer: GEPA (max_full_evals={iterations})")
        console.print(f"  Optimizer model: {optimizer_model}")
        console.print(f"  Eval model: {eval_model}")
        proposer_name = "preserving" if preserving_proposer else "generic"
        console.print(f"  Instruction proposer: {proposer_name}")

        # ── 5. Run optimization ─────────────────────────────────────────
        console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
        # Unique per-run dir: GEPA RESUMES from a log_dir it has seen before,
        # which would silently replay an earlier run's candidate instead of
        # optimizing (observed: 0 rollouts, stale evolved text).
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        gepa_kwargs: dict = dict(
            metric=_gepa_metric,
            max_full_evals=iterations,
            reflection_lm=reflection_lm,
            track_stats=True,
            log_dir=str(Path("logs") / "gepa" / f"{skill_name}-{run_id}"),
        )
        if preserving_proposer:
            # dspy's default proposer writes a short generic instruction and
            # destroys long skill documents; this one edits in place.
            gepa_kwargs["instruction_proposer"] = SkillPreservingProposer()
        try:
            optimizer = dspy.GEPA(**gepa_kwargs)
            optimizer_name = "gepa"
            optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
        except Exception as e:
            console.print(f"[yellow]GEPA unavailable ({type(e).__name__}: {e}) — falling back to MIPROv2[/yellow]")
            optimizer = dspy.MIPROv2(metric=_gepa_metric, auto="light")
            optimizer_name = "miprov2"
            optimized_module = optimizer.compile(baseline_module, trainset=trainset)

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)
    changed = evolved_body.strip() != skill["body"].strip()
    if not changed:
        console.print("[yellow]⚠ Optimizer produced no change to the skill text (baseline == evolved)[/yellow]")

    # ── 7. Validate evolved skill ───────────────────────────────────────
    # A constraint the BASELINE already failed is not held against the evolved
    # version (e.g. a skill that was already over the size limit, or a body
    # checked without frontmatter). Otherwise the gate is a hard regression gate.
    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=baseline_full)
    hard_failures = []
    for c in evolved_constraints:
        enforced = baseline_pass.get(c.constraint_name, True)  # only gate what baseline passed
        ok = c.passed or not enforced
        icon = "✓" if c.passed else ("✗" if enforced else "⚠")
        color = "green" if c.passed else ("red" if enforced else "yellow")
        note = "" if enforced else "  [dim](baseline already failed — not gated)[/dim]"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}{note}")
        if not ok:
            hard_failures.append(c.constraint_name)
    # Destructive-shrink guard: GEPA's instruction proposer tends to rewrite
    # toward a short generic instruction, which can gut a long skill document.
    # A skill that loses most of its content is a regression even if the judge
    # score moves, so treat heavy shrinkage as a hard failure.
    shrink_floor = 0.70
    shrink_ratio = len(evolved_body) / max(1, len(skill["body"]))
    destructive = shrink_ratio < shrink_floor
    if destructive:
        console.print(
            f"  [red]✗ content_shrink[/red]: {shrink_ratio:.0%} of baseline "
            f"({len(evolved_body):,}/{len(skill['body']):,} chars) — below {shrink_floor:.0%} floor"
        )
        hard_failures.append("content_shrink")

    all_pass = not hard_failures

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    scoring_errors = 0
    for ex in holdout_examples:
        try:
            if mock:
                baseline_out = mock_forward_text(baseline_module)
                evolved_out = mock_forward_text(optimized_module)
            else:
                with dspy.context(lm=lm):
                    baseline_out = baseline_module(task_input=ex.task_input).output
                    evolved_out = optimized_module(task_input=ex.task_input).output
            baseline_scores.append(skill_fitness_metric(ex, dspy.Prediction(output=baseline_out)))
            evolved_scores.append(skill_fitness_metric(ex, dspy.Prediction(output=evolved_out)))
        except Exception as e:
            # One malformed example must not kill the run — metrics still get written.
            scoring_errors += 1
            console.print(f"  [yellow]⚠ holdout example failed ({type(e).__name__}: {e})[/yellow]")
            baseline_scores.append(0.0)
            evolved_scores.append(0.0)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report ───────────────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    change_color = "green" if improvement > 0 else "red"
    table.add_row("Holdout Score", f"{avg_baseline:.3f}", f"{avg_evolved:.3f}",
                  f"[{change_color}]{improvement:+.3f}[/{change_color}]")
    table.add_row("Skill Size", f"{len(skill['body']):,} chars", f"{len(evolved_body):,} chars",
                  f"{len(evolved_body) - len(skill['body']):+,} chars")
    table.add_row("Optimizer", optimizer_name, "", "")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    console.print()
    console.print(table)

    # ── 10. Save output (ALWAYS — even on constraint failure) ───────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evolved_skill.md").write_text(evolved_full)
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "mode": "mock" if mock else "real",
        "optimizer": optimizer_name,
        "proposer": proposer_name,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "changed": changed,
        "shrink_ratio": round(shrink_ratio, 3),
        # "deployed" is only a RECOMMENDATION: constraints pass, content changed
        # non-destructively, and the holdout score actually improved.
        "deployed": bool(all_pass and changed and improvement > 0),
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "scoring_errors": scoring_errors,
        "constraints_passed": all_pass,
        "constraint_failures": hard_failures,
        "constraints": [
            {"name": c.constraint_name, "passed": c.passed, "message": c.message}
            for c in evolved_constraints
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n  Output saved to {output_dir}/")

    if not all_pass:
        console.print(f"[red]✗ Evolved skill failed hard constraints: {', '.join(hard_failures)} — NOT deployable[/red]")
    elif improvement > 0 and changed:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} "
                      f"({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ No usable improvement (change: {improvement:+.3f}, changed: {changed})[/yellow]")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="GEPA max_full_evals (optimization budget)")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--mock", is_flag=True, help="Zero-cost run: mock dataset + mock optimizer, no API calls")
@click.option("--max-skill-size", default=15_000, help="Hard size limit for a skill body (chars)")
@click.option("--dataset-size", default=None, type=int, help="Number of eval examples to build")
@click.option("--generic-proposer", is_flag=True,
              help="Use dspy's default (short-instruction) proposer instead of the "
                   "preservation-aware one. Default is preservation-aware.")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo,
         run_tests, dry_run, mock, max_skill_size, dataset_size, generic_proposer):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        mock=mock,
        max_skill_size=max_skill_size,
        dataset_size=dataset_size,
        preserving_proposer=not generic_proposer,
    )


if __name__ == "__main__":
    main()
