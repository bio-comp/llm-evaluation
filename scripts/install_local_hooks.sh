#!/usr/bin/env bash
# Install only on a fresh standalone clone; never overwrite shared user hooks.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git_dir="$(git rev-parse --absolute-git-dir)"
common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
if [[ "$git_dir" != "$common_dir" ]]; then
  echo "REFUSED: linked worktree shares hooks with other worktrees" >&2
  exit 1
fi
if git config --get core.hooksPath >/dev/null; then
  echo "REFUSED: existing core.hooksPath must be reviewed before hook installation" >&2
  exit 1
fi
hooks="$(git rev-parse --git-path hooks)"
if [[ -e "$hooks/pre-commit" || -e "$hooks/pre-push" ]]; then
  echo "REFUSED: existing hooks require explicit coordination; no overwrite" >&2
  exit 1
fi
uv run --locked prek validate-config .pre-commit-config.yaml
uv run --locked prek install --hook-type pre-commit
install -m 755 .githooks/pre-push "$hooks/pre-push"
echo "Installed pinned prek pre-commit and native ref-aware pre-push hooks."
