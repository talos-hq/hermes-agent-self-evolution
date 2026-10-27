#!/usr/bin/env bash
# Weekly Hermes skill evolution: pick the next skill from rotation.txt, run the
# DSPy+GEPA optimizer on it, and print a concise result block for delivery.
#
# Extra args are forwarded to ./evolve (e.g. --dry-run for a no-spend test).
#   SE_WEEKLY_ITERATIONS   GEPA iterations (default 10)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITERS="${SE_WEEKLY_ITERATIONS:-10}"

SKILL="$("$DIR/next-skill.sh")"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$DIR/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/weekly-$TS-$SKILL.log"

echo "== Hermes weekly skill evolution =="
echo "skill:      $SKILL"
echo "iterations: $ITERS"
echo "log:        $LOG"

rc=0
"$DIR/evolve" --skill "$SKILL" --iterations "$ITERS" "$@" >"$LOG" 2>&1 || rc=$?
echo "exit:       $rc"

M="$(ls -td "$DIR/output/$SKILL"/*/metrics.json 2>/dev/null | head -1 || true)"
if [ -n "$M" ]; then
  OUT="$(dirname "$M")"
  echo "outdir:     $OUT"
  echo "--- metrics ---"
  uv run --project "$DIR" python - "$M" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
keys = ("skill_name","baseline_score","evolved_score","improvement",
        "baseline_size","evolved_size","constraints_passed","elapsed_seconds",
        "train_examples","val_examples","holdout_examples")
for k in keys:
    if k in d:
        print(f"{k}: {d[k]}")
PY
  echo "--- diffstat ---"
  diff "$OUT/baseline_skill.md" "$OUT/evolved_skill.md" | sed -n '1,40p' || true
else
  echo "no metrics produced (exit $rc)"
  echo "--- last 25 log lines ---"
  tail -25 "$LOG" || true
fi
