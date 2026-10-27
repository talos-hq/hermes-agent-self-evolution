#!/usr/bin/env bash
# Weekly Hermes skill evolution: pick the next skill from rotation.txt, run the
# DSPy+GEPA optimizer on it, and print a concise result block for delivery.
#
# Extra args are forwarded to ./evolve (e.g. --dry-run for a no-spend test).
#   SE_WEEKLY_ITERATIONS   GEPA iterations (default 6; a 10-iteration run can
#                          exceed a 15-minute cron terminal timeout)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITERS="${SE_WEEKLY_ITERATIONS:-6}"

SKILL="$("$DIR/next-skill.sh")"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$DIR/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/weekly-$TS-$SKILL.log"

echo "== Hermes weekly skill evolution =="
echo "skill:      $SKILL"
echo "iterations: $ITERS"
echo "log:        $LOG"

# Snapshot the latest existing run dir so we can prove THIS run produced a new
# one (a stale metrics.json from an earlier run must not count as success).
BEFORE_DIR="$(ls -1 "$DIR/output/$SKILL" 2>/dev/null | sort | tail -1 || true)"

rc=0
"$DIR/evolve" --skill "$SKILL" --iterations "$ITERS" "$@" >"$LOG" 2>&1 || rc=$?
echo "exit:       $rc"

NEW_DIR="$(ls -1t "$DIR/output/$SKILL" 2>/dev/null | head -1 || true)"
M=""
if [ -n "$NEW_DIR" ] && [ "$NEW_DIR" != "$BEFORE_DIR" ] && [ -f "$DIR/output/$SKILL/$NEW_DIR/metrics.json" ]; then
  M="$DIR/output/$SKILL/$NEW_DIR/metrics.json"
fi
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

# Advance the rotation ONLY when the run produced results, so a failed week is
# retried next time instead of being silently skipped.
if [ -n "$M" ] && [ "${SE_WEEKLY_NO_COMMIT:-0}" != "1" ]; then
  "$DIR/next-skill.sh" --commit >/dev/null
  echo "rotation:   advanced"
fi
