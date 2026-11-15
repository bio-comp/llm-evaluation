#!/usr/bin/env bash
# Install hooks for exactly this checkout; never write the shared hooks directory.
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo "REFUSED: no installer flags are supported; inspect existing hooks first" >&2
  exit 1
fi
cd "$(git rev-parse --show-toplevel)"
git_dir="$(cd "$(git rev-parse --absolute-git-dir)" && pwd -P)"
common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
destination="$git_dir/llm-evaluation-hooks"
worktree_count="$(git worktree list --porcelain | grep -c '^worktree ' || true)"

# An existing manager must be explicitly reconciled; never hide its hooks.
effective="$(git config --get core.hooksPath || true)"
if [[ -n "$effective" && "$effective" != "$destination" ]]; then
  echo "REFUSED: existing core.hooksPath requires explicit coordination" >&2
  exit 1
fi
for hook in pre-commit pre-push; do
  if [[ -e "$common_dir/hooks/$hook" ]]; then
    echo "REFUSED: existing shared $hook hook requires explicit coordination" >&2
    exit 1
  fi
done

# Git reads config.worktree only with the shared worktreeConfig extension.
# With multiple worktrees, changing that shared extension requires an explicit
# operator decision, NOT a side effect of activating hooks in this checkout.
enabled="$(git config --local --bool --get extensions.worktreeConfig || true)"
if [[ "$enabled" != "true" && "$worktree_count" -gt 1 ]]; then
  echo "REFUSED: multiple worktrees; explicitly inspect and enable" >&2
  echo "  git config --local extensions.worktreeConfig true" >&2
  echo "Then rerun this installer from ONLY the checkout to be protected." >&2
  exit 1
fi
if [[ "$enabled" != "true" ]] && git config --local --get core.worktree >/dev/null; then
  echo "REFUSED: relocate core.worktree before enabling worktreeConfig" >&2
  exit 1
fi

# Validate all prerequisites BEFORE adding active hooks.
uv run --locked prek validate-config .pre-commit-config.yaml
test -f .githooks/pre-push
test -f scripts/pre_push_gate.py
if [[ -e "$destination" ]]; then
  if [[ "$effective" == "$destination" ]] \
    && cmp -s "$destination/pre-push" .githooks/pre-push \
    && cmp -s "$destination/pre-commit" .githooks/pre-commit; then
    echo "Already installed in this worktree; no changes made."
    exit 0
  fi
  echo "REFUSED: existing hook installation differs; review upgrade manually" >&2
  exit 1
fi

# Prepare privately within this checkout's gitdir. Activate config LAST.
staging="$(mktemp -d "$git_dir/.llm-evaluation-hooks.XXXXXXXX")"
created_destination=false
cleanup() {
  if [[ -d "$staging" ]]; then rm -rf "$staging"; fi
  if [[ "$created_destination" == true ]]; then rm -rf "$destination"; fi
}
trap cleanup EXIT
install -m 755 .githooks/pre-commit "$staging/pre-commit"
install -m 755 .githooks/pre-push "$staging/pre-push"
if [[ "$enabled" != "true" ]]; then
  git config --local extensions.worktreeConfig true
fi
mv "$staging" "$destination"
created_destination=true
git config --worktree core.hooksPath "$destination"
test "$(git config --worktree --get core.hooksPath)" = "$destination"
created_destination=false
trap - EXIT
echo "Installed isolated hooks for this worktree only: $destination"
