"""Fail-closed local push qualification for one checked-out outgoing Git ref.

This is invoked by the native Git pre-push hook, which supplies ALL ref updates on
stdin. Unlike prek 0.5.10's pre-push stage, it detects and rejects multi-ref and
non-HEAD pushes rather than falsely claiming those revisions were checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PushUpdate:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


def parse_single_update(
    lines: Iterable[str], *, head: str, branch_ref: str
) -> PushUpdate:
    """Only the exact checked-out HEAD branch can pass this single-ref gate."""
    updates = [line.split() for line in lines if line.strip()]
    if len(updates) != 1:
        raise ValueError("push must contain exactly one outgoing ref")
    fields = updates[0]
    if len(fields) != 4:
        raise ValueError("Git pre-push record must contain four fields")
    local_ref, local_sha, remote_ref, remote_sha = fields
    if local_sha == "0" * 40:
        raise ValueError("ref delete requires separately qualified authorization")
    if local_sha != head:
        raise ValueError("outgoing SHA differs from checked-out HEAD")
    if local_ref not in (branch_ref, "HEAD"):
        raise ValueError("outgoing ref differs from checked-out branch")
    if not remote_ref.startswith("refs/heads/"):
        raise ValueError("only a single branch update is supported")
    return PushUpdate(local_ref, local_sha, remote_ref, remote_sha)


def git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), text=True, stderr=subprocess.STDOUT
    ).strip()


def run(*argv: str) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def require_exact_clean_head(root: Path, sha: str) -> None:
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise RuntimeError("local gate must run from repository root")
    if git("rev-parse", "HEAD") != sha:
        raise RuntimeError("HEAD changed during push qualification")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("dirty worktree: qualify a clean exact HEAD")


def changed_python_files(sha: str) -> tuple[str, ...]:
    # Explicit transitional baseline: whole-repository formatter debt belongs to #37.
    for ref in ("refs/remotes/origin/main", "refs/heads/main"):
        try:
            base = git("merge-base", sha, ref)
            break
        except subprocess.CalledProcessError:
            continue
    else:
        raise RuntimeError(
            "cannot establish main merge-base for changed-file format gate"
        )
    raw = subprocess.check_output(
        (
            "git",
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMR",
            base,
            sha,
            "--",
            "*.py",
        )
    )
    paths = tuple(item.decode("utf-8") for item in raw.split(b"\0") if item)
    for path in paths:
        if not Path(path).is_file():
            raise RuntimeError(f"changed Python file unavailable: {path}")
    return paths


def python_floor(root: Path) -> str:
    content = (root / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'^requires-python\s*=\s*">=(3\.(?:11|12))"\s*$', content, re.M)
    if found is None:
        raise RuntimeError("unsupported Python minimum; review mypy policy before push")
    return found.group(1)


def pytest_report(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise RuntimeError("pytest produced no JUnit XML evidence")
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    failures = sum(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    )
    executed = len(cases) - skipped
    if executed == 0 or failures:
        raise RuntimeError(
            f"vacuous/failed tests: total={len(cases)}, executed={executed}, "
            f"skipped={skipped}, failures={failures}"
        )
    return executed, skipped


def run_installed_package_smoke(temp: Path) -> None:
    dist = temp / "dist"
    run("uv", "build", "--no-sources", "--out-dir", str(dist))
    wheels = tuple(dist.glob("llm_evaluation-*.whl"))
    sdists = tuple(dist.glob("llm_evaluation-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("expected exactly one wheel and one sdist")
    venv = temp / "wheelvenv"
    run(
        "uv",
        "venv",
        "--python",
        Path(".python-version").read_text(encoding="utf-8").strip(),
        str(venv),
    )
    interpreter = str(venv / "bin" / "python")
    run("uv", "pip", "install", "--python", interpreter, str(wheels[0]))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    preview = subprocess.run(
        (
            str(venv / "bin" / "llm-evaluation"),
            "fixtures/e002_synthetic.json",
            "--k",
            "2",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    if json.loads(preview.stdout)["macro"]["marginal"] != 0.5:
        raise RuntimeError("installed wheel CLI preview failed E-002 oracle")
    forbidden = temp / "wheel-finalized.json"
    refused = subprocess.run(
        (
            str(venv / "bin" / "llm-evaluation"),
            "fixtures/e002_synthetic.json",
            "--k",
            "2",
            "--output",
            str(forbidden),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if (
        refused.returncode == 0
        or forbidden.exists()
        or "finalization requires a source checkout" not in refused.stderr
    ):
        raise RuntimeError("installed wheel failed E-002 no-finalization safeguard")


def run_source_immutable_check(temp: Path) -> None:
    record = temp / "source-e002.json"
    command = (
        "uv",
        "run",
        "--locked",
        "--extra",
        "drive",
        "--extra",
        "local-auth",
        "python",
        "-m",
        "llm_evaluation.cli",
        "fixtures/e002_synthetic.json",
        "--k",
        "2",
        "--output",
        str(record),
    )
    run(*command)
    first_digest = hashlib.sha256(record.read_bytes()).hexdigest()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    if (
        second.returncode == 0
        or "FileExistsError" not in second.stderr
        or hashlib.sha256(record.read_bytes()).hexdigest() != first_digest
    ):
        raise RuntimeError("E-002 no-overwrite negative control did not discriminate")


def qualify(root: Path, sha: str) -> None:
    require_exact_clean_head(root, sha)
    print(f"LOCAL_GATE_TARGET_SHA={sha}", flush=True)
    print(f"LOCAL_GATE_TARGET_TREE={git('rev-parse', 'HEAD^{tree}')}", flush=True)
    print(f"LOCAL_GATE_PYTHON_FLOOR={python_floor(root)}", flush=True)
    run("uv", "--version")
    run("uv", "lock", "--check")
    run(
        "uv",
        "sync",
        "--locked",
        "--group",
        "dev",
        "--extra",
        "drive",
        "--extra",
        "local-auth",
    )
    run("uv", "run", "--locked", "prek", "validate-config", ".pre-commit-config.yaml")
    run("git", "diff", "--check")
    run("uv", "run", "--locked", "ruff", "check", "--no-fix", "src", "tests", "scripts")
    changed = changed_python_files(sha)
    print(
        f"FORMAT_TRANSITION_CHANGED_FILES={len(changed)}; full baseline pending #37",
        flush=True,
    )
    if changed:
        run("uv", "run", "--locked", "ruff", "format", "--check", *changed)
    run("uv", "run", "--locked", "mypy", "--python-version", python_floor(root), "src")
    # Newly added gate script is held to strict typing even while #38 is open.
    run("uv", "run", "--locked", "mypy", "--strict", "scripts/pre_push_gate.py")
    with tempfile.TemporaryDirectory(prefix="llm-eval-local-gate-") as folder:
        temp = Path(folder).resolve()
        report = temp / "pytest.xml"
        run(
            "uv",
            "run",
            "--locked",
            "coverage",
            "run",
            "--branch",
            "-m",
            "pytest",
            "-m",
            "not live_drive",
            "-q",
            "--randomly-seed=400040",
            f"--junitxml={report}",
        )
        executed, skipped = pytest_report(report)
        print(f"PYTEST_EXECUTED={executed} SKIPPED={skipped}", flush=True)
        run(
            "uv",
            "run",
            "--locked",
            "coverage",
            "report",
            "--include=src/llm_evaluation/*",
            "--fail-under=80",
        )
        run_installed_package_smoke(temp)
        run_source_immutable_check(temp)
    require_exact_clean_head(root, sha)
    print(f"LOCAL_GATE_PASS_SHA={sha}", flush=True)


def main() -> int:
    try:
        root = Path(git("rev-parse", "--show-toplevel")).resolve()
        os.chdir(root)
        head = git("rev-parse", "HEAD")
        branch = git("symbolic-ref", "HEAD")
        update = parse_single_update(sys.stdin, head=head, branch_ref=branch)
        print(f"OUTGOING_REF={update.local_ref} TO={update.remote_ref}", flush=True)
        qualify(root, update.local_sha)
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        ET.ParseError,
    ) as exc:
        print(f"LOCAL_GATE_FAIL: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
