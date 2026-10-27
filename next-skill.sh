#!/usr/bin/env bash
# Rotation pointer for rotation.txt.
#   next-skill.sh            -> print the CURRENT skill (peek; does NOT advance)
#   next-skill.sh --commit   -> advance to the next skill (no output)
# State lives in .rotation_state (an integer); wraps around automatically.
#
# Peek-then-commit lets the caller advance the rotation ONLY after a run
# actually succeeded, so a failed week is retried instead of skipped.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="$DIR/rotation.txt"
STATE="$DIR/.rotation_state"

mapfile -t skills < <(grep -vE '^[[:space:]]*(#|$)' "$LIST" || true)
n=${#skills[@]}
if [ "$n" -eq 0 ]; then
  echo "rotation.txt has no skills" >&2
  exit 1
fi

idx=0
if [ -f "$STATE" ]; then
  read -r idx < "$STATE" || idx=0
  case "$idx" in ''|*[!0-9]*) idx=0;; esac
fi
idx=$(( idx % n ))

case "${1:-}" in
  ""|--peek)
    echo "${skills[$idx]}"
    ;;
  --commit)
    echo $(( (idx + 1) % n )) > "$STATE"
    ;;
  *)
    echo "usage: next-skill.sh [--peek|--commit]" >&2
    exit 2
    ;;
esac
