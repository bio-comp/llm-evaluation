"""Pure negative controls for the outgoing-ref identity gate."""

from pathlib import Path

import pytest

from scripts.pre_push_gate import parse_single_update

HEAD = "1" * 40
OTHER = "2" * 40
ZERO = "0" * 40
BRANCH = "refs/heads/feature"


def update(local_ref: str = BRANCH, sha: str = HEAD) -> str:
    return f"{local_ref} {sha} refs/heads/feature {ZERO}\n"


def test_single_checked_out_branch_is_accepted() -> None:
    ref = parse_single_update([update()], head=HEAD, branch_ref=BRANCH)
    assert ref.local_sha == HEAD
    assert ref.local_ref == BRANCH


def test_explicit_head_push_is_accepted() -> None:
    ref = parse_single_update([update("HEAD")], head=HEAD, branch_ref=BRANCH)
    assert ref.local_ref == "HEAD"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "exactly one"),
        ([update(), update()], "exactly one"),
        (["not enough fields\n"], "four fields"),
        ([update(sha=OTHER)], "HEAD"),
        ([update(local_ref="refs/heads/other")], "checked-out"),
        ([update(sha=ZERO)], "delete"),
    ],
)
def test_unsafe_or_ambiguous_pushes_fail_closed(lines: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_single_update(lines, head=HEAD, branch_ref=BRANCH)


def test_junit_requires_at_least_one_executed_test(tmp_path: Path) -> None:
    from scripts.pre_push_gate import pytest_report

    xml = tmp_path / "report.xml"
    xml.write_text('<testsuite tests="1"><testcase><skipped/></testcase></testsuite>')
    with pytest.raises(RuntimeError, match="vacuous"):
        pytest_report(xml)

    xml.write_text('<testsuite tests="1"><testcase/></testsuite>')
    assert pytest_report(xml) == (1, 0)


@pytest.mark.parametrize("minimum", ("3.11", "3.12"))
def test_python_floor_matches_project_declared_minimum(
    tmp_path: Path, minimum: str
) -> None:
    from scripts.pre_push_gate import python_floor

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python = ">=' + minimum + '"\n'
    )
    assert python_floor(tmp_path) == minimum


def test_unknown_python_minimum_requires_gate_review(tmp_path: Path) -> None:
    from scripts.pre_push_gate import python_floor

    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.15"\n')
    with pytest.raises(RuntimeError, match="unsupported Python minimum"):
        python_floor(tmp_path)


def _git(root: Path, *args: str) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _commit(root: Path, text: str) -> str:
    (root / "document.md").write_text(text)
    _git(root, "add", "document.md")
    _git(root, "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD")


def test_committed_whitespace_in_intermediate_outgoing_commit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _commit(tmp_path, "clean\n")
    _git(tmp_path, "checkout", "-qb", "feature")
    _commit(tmp_path, "bad trailing   \n")
    head = _commit(tmp_path, "clean again\n")
    monkeypatch.chdir(tmp_path)
    update = PushUpdate("refs/heads/feature", head, "refs/heads/feature", "0" * 40)
    with pytest.raises(RuntimeError, match="committed whitespace"):
        check_outgoing_commits(update)


def test_committed_whitespace_on_existing_remote_ref_update_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    base = _commit(tmp_path, "clean\n")
    _git(tmp_path, "checkout", "-qb", "feature")
    head = _commit(tmp_path, "bad trailing   \n")
    monkeypatch.chdir(tmp_path)
    update = PushUpdate("refs/heads/feature", head, "refs/heads/feature", base)
    with pytest.raises(RuntimeError, match="committed whitespace"):
        check_outgoing_commits(update)


def test_clean_multiple_outgoing_commits_are_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _commit(tmp_path, "clean\n")
    _git(tmp_path, "checkout", "-qb", "feature")
    first = _commit(tmp_path, "clean again\n")
    second = _commit(tmp_path, "clean third\n")
    monkeypatch.chdir(tmp_path)
    ref = PushUpdate("refs/heads/feature", second, "refs/heads/feature", "0" * 40)
    assert check_outgoing_commits(ref) == (first, second)


@pytest.mark.parametrize("invalid", ("f" * 40, "not-a-sha"))
def test_unavailable_remote_reference_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    head = _commit(tmp_path, "clean\n")
    monkeypatch.chdir(tmp_path)
    ref = PushUpdate("refs/heads/main", head, "refs/heads/main", invalid)
    with pytest.raises(ValueError, match="remote SHA|remote reference SHA"):
        check_outgoing_commits(ref)


def test_non_fast_forward_remote_reference_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _commit(tmp_path, "base\n")
    old_remote = _commit(tmp_path, "remote later\n")
    _git(tmp_path, "checkout", "-qb", "feature", "HEAD^")
    outgoing = _commit(tmp_path, "different branch\n")
    monkeypatch.chdir(tmp_path)
    ref = PushUpdate("refs/heads/feature", outgoing, "refs/heads/feature", old_remote)
    with pytest.raises(ValueError, match="not an ancestor"):
        check_outgoing_commits(ref)
