"""Fast disposable-Git controls for per-worktree hooks and outgoing history.

The fake uv binary only replaces the expensive full quality checks: it invokes
the REAL ref parser and committed-whitespace validator during ordinary Git pushes.
Actual pinned uv/prek/full-hook qualification is separately required before push.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2]
STAGED = (
    ".githooks/pre-commit",
    ".githooks/pre-push",
    "scripts/install_local_hooks.sh",
    "scripts/pre_push_gate.py",
    "scripts/__init__.py",
    ".pre-commit-config.yaml",
)


def git(
    root: Path, *args: str, check: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def sha(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def git_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, str]]:
    primary = tmp_path / "primary"
    primary.mkdir()
    git(primary, "init", "-q", "-b", "main")
    git(primary, "config", "user.name", "Gate fixture")
    git(primary, "config", "user.email", "fixture@example.invalid")
    (primary / "README.md").write_text("base\n")
    git(primary, "add", "README.md")
    git(primary, "commit", "-qm", "base")
    remote = tmp_path / "remote.git"
    subprocess.run(("git", "init", "--bare", "-q", str(remote)), check=True)
    git(primary, "remote", "add", "origin", str(remote))

    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    shim = bin_dir / "uv"
    shim.write_text(
        "#!/bin/sh\n"
        'echo "$PWD: $*" >> "$GATE_HOOK_LOG"\n'
        'case "$*" in\n'
        '  *"prek validate-config"*) exit 0 ;;\n'
        '  *"prek run --hook-stage pre-commit"*) exit 0 ;;\n'
        '  *"python scripts/pre_push_gate.py"*)\n'
        "    test -f scripts/pre_push_gate.py || exit 49\n"
        "    exec python3 -c \"import sys; from scripts.pre_push_gate import git, parse_single_update, check_outgoing_commits; u=parse_single_update(sys.stdin, head=git('rev-parse', 'HEAD'), branch_ref=git('symbolic-ref', 'HEAD')); check_outgoing_commits(u)\" ;;\n"
        "  *) exit 39 ;;\n"
        "esac\n"
    )
    shim.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["GATE_HOOK_LOG"] = str(tmp_path / "hook.log")
    monkeypatch.setenv("GATE_HOOK_LOG", env["GATE_HOOK_LOG"])
    monkeypatch.setenv("PATH", env["PATH"])
    return primary, remote, tmp_path, env


def add_gate(root: Path) -> None:
    for filename in STAGED:
        dest = root / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / filename, dest)
    git(root, "add", *STAGED)
    git(root, "commit", "-qm", "gate fixture")


def install(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", "scripts/install_local_hooks.sh"),
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def hooks_path(root: Path) -> str:
    return git(root, "config", "--worktree", "--get", "core.hooksPath").stdout.strip()


def test_standalone_install_push_repeat_and_failure(
    git_case: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    primary, remote, _, env = git_case
    git(primary, "checkout", "-qb", "qualified")
    add_gate(primary)
    original = sha(primary)
    result = install(primary, env)
    assert result.returncode == 0, result.stderr
    path = Path(hooks_path(primary))
    assert path.is_dir()
    assert (path / "pre-commit").stat().st_mode & 0o111
    assert (path / "pre-push").stat().st_mode & 0o111
    assert install(primary, env).returncode == 0
    assert "Already installed" in install(primary, env).stdout

    good = git(primary, "push", "origin", "qualified", env=env)
    assert good.returncode == 0, good.stderr
    assert git(
        primary, "ls-remote", "origin", "refs/heads/qualified"
    ).stdout.startswith(original)

    (primary / "README.md").write_text("bad whitespace    \n")
    git(primary, "add", "README.md")
    git(primary, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "bad")
    bad = git(primary, "push", "origin", "qualified", check=False, env=env)
    assert bad.returncode != 0
    assert "committed whitespace" in bad.stderr
    assert git(
        primary, "ls-remote", "origin", "refs/heads/qualified"
    ).stdout.startswith(original)

    # Existing hook manager remains protected from accidental upgrade/replacement.
    assert git(
        primary, "config", "--worktree", "--get", "core.hooksPath"
    ).stdout.strip() == str(path)
    assert remote.exists()


def test_primary_with_linked_old_checkout_requires_explicit_opt_in(
    git_case: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    primary, remote, root, env = git_case
    git(primary, "checkout", "-qb", "qualified")
    add_gate(primary)
    older = root / "older"
    git(primary, "worktree", "add", "-q", "-b", "older", str(older), "main")
    assert not (older / "scripts/pre_push_gate.py").exists()

    blocked = install(primary, env)
    assert blocked.returncode != 0
    assert "multiple worktrees" in blocked.stderr
    assert not (primary / ".git/hooks/pre-push").exists()

    git(primary, "config", "--local", "extensions.worktreeConfig", "true")
    assert install(primary, env).returncode == 0
    assert hooks_path(primary)
    assert git(older, "config", "--get", "core.hooksPath", check=False).returncode != 0

    # Older worktree's push must not invoke the new hook from the primary.
    old_push = git(older, "push", "origin", "HEAD:refs/heads/older", env=env)
    assert old_push.returncode == 0, old_push.stderr
    assert git(older, "ls-remote", "origin", "refs/heads/older").stdout.startswith(
        sha(older)
    )
    new_push = git(primary, "push", "origin", "qualified", env=env)
    assert new_push.returncode == 0, new_push.stderr
    assert remote.exists()


def test_linked_checkout_can_be_activated_without_changing_primary(
    git_case: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    primary, _, root, env = git_case
    linked = root / "linked"
    git(primary, "worktree", "add", "-q", "-b", "qualified", str(linked), "main")
    add_gate(linked)
    git(primary, "config", "--local", "extensions.worktreeConfig", "true")
    assert install(linked, env).returncode == 0
    assert hooks_path(linked)
    assert (
        git(primary, "config", "--get", "core.hooksPath", check=False).returncode != 0
    )
    assert git(primary, "push", "origin", "main", env=env).returncode == 0
    assert git(linked, "push", "origin", "qualified", env=env).returncode == 0


@pytest.mark.parametrize("conflict", ("shared-hook", "core-hook-path"))
def test_existing_hook_managers_are_not_overwritten(
    git_case: tuple[Path, Path, Path, dict[str, str]],
    conflict: str,
) -> None:
    primary, _, root, env = git_case
    git(primary, "checkout", "-qb", "qualified")
    add_gate(primary)
    if conflict == "shared-hook":
        (primary / ".git/hooks/pre-push").write_text("#!/bin/sh\nexit 0\n")
    else:
        original = root / "existing-hooks"
        original.mkdir()
        git(primary, "config", "core.hooksPath", str(original))
    refusal = install(primary, env)
    assert refusal.returncode != 0
    assert "REFUSED" in refusal.stderr
    assert (
        git(
            primary, "config", "--get", "extensions.worktreeConfig", check=False
        ).returncode
        != 0
    )


def test_changed_installed_hook_refuses_silent_upgrade(
    git_case: tuple[Path, Path, Path, dict[str, str]],
) -> None:
    primary, _, _, env = git_case
    git(primary, "checkout", "-qb", "qualified")
    add_gate(primary)
    assert install(primary, env).returncode == 0
    prepush = Path(hooks_path(primary)) / "pre-push"
    prepush.write_text("# changed by another manager\n")
    refusal = install(primary, env)
    assert refusal.returncode != 0
    assert "REFUSED" in refusal.stderr
    assert prepush.read_text() == "# changed by another manager\n"
