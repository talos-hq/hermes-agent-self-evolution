#!/usr/bin/env bash
# Print the next skill in rotation.txt and advance the persistent index.
# State lives in .rotation_state (an integer); wraps around automatically.
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

echo "${skills[$idx]}"
echo $(( (idx + 1) % n )) > "$STATE"
