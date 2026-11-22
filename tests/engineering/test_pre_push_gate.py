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


def test_empty_push_stream_is_noop_without_git_or_qualification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import sys

    import scripts.pre_push_gate as gate

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    def unexpected_git(*args: str) -> str:
        raise AssertionError(f"no-op must not query Git or qualify: {args}")

    monkeypatch.setattr(gate, "git", unexpected_git)
    assert gate.main() == 0
    output = capsys.readouterr()
    assert "LOCAL_GATE_NOOP" in output.out
    assert "no revision published or qualified" in output.out
    assert "LOCAL_GATE_PASS_SHA" not in output.out
    assert "LOCAL_GATE_TARGET_SHA" not in output.out
    assert output.err == ""


def test_blank_record_is_malformed_not_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io
    import sys

    import scripts.pre_push_gate as gate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))

    def fake_git(*args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("rev-parse", "HEAD"):
            return HEAD
        if args == ("symbolic-ref", "HEAD"):
            return BRANCH
        raise AssertionError(f"unexpected Git call: {args}")

    monkeypatch.setattr(gate, "git", fake_git)
    assert gate.main() == 1
    output = capsys.readouterr()
    assert "push must contain exactly one outgoing ref" in output.err
    assert "LOCAL_GATE_NOOP" not in output.out


def test_initial_remote_branch_at_trusted_origin_main_head_has_no_new_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Actual ref update at a trusted equal baseline is NOT a no-op push."""
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    head = _commit(tmp_path, "clean baseline\n")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", head)
    _git(tmp_path, "checkout", "-qb", "new-branch")
    monkeypatch.chdir(tmp_path)
    ref = PushUpdate("refs/heads/new-branch", head, "refs/heads/new-branch", ZERO)
    assert check_outgoing_commits(ref) == ()
    output = capsys.readouterr().out
    assert f"OUTGOING_RANGE_BASE={head} SOURCE=main COMMITS=0" in output


def test_initial_remote_branch_without_trusted_main_baseline_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "feature")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    head = _commit(tmp_path, "isolated root\n")
    monkeypatch.chdir(tmp_path)
    ref = PushUpdate("refs/heads/feature", head, "refs/heads/feature", ZERO)
    with pytest.raises(ValueError, match="trustworthy ancestral main base"):
        check_outgoing_commits(ref)


def test_initial_remote_branch_with_divergent_main_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.pre_push_gate import PushUpdate, check_outgoing_commits

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _commit(tmp_path, "shared ancestor\n")
    main_head = _commit(tmp_path, "only on main\n")
    _git(tmp_path, "checkout", "-qb", "feature", "HEAD^")
    feature_head = _commit(tmp_path, "only on feature\n")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", main_head)
    monkeypatch.chdir(tmp_path)
    ref = PushUpdate("refs/heads/feature", feature_head, "refs/heads/feature", ZERO)
    with pytest.raises(ValueError, match="trustworthy ancestral main base"):
        check_outgoing_commits(ref)


def test_equal_main_new_ref_dispatches_full_qualification_not_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lightweight dispatch assertion; real expensive gate is checked externally."""
    import io
    import sys

    import scripts.pre_push_gate as gate

    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    head = _commit(tmp_path, "clean baseline\n")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", head)
    _git(tmp_path, "checkout", "-qb", "new-branch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(f"refs/heads/new-branch {head} refs/heads/new-branch {ZERO}\n"),
    )
    calls: list[tuple[Path, str, gate.PushUpdate]] = []

    def qualified(root: Path, sha: str, update: gate.PushUpdate) -> None:
        calls.append((root, sha, update))

    monkeypatch.setattr(gate, "qualify", qualified)
    assert gate.main() == 0
    assert len(calls) == 1
    assert calls[0][0] == tmp_path.resolve()
    assert calls[0][1] == head
    assert calls[0][2].remote_sha == ZERO
    output = capsys.readouterr()
    assert "OUTGOING_REF=refs/heads/new-branch" in output.out
    assert "LOCAL_GATE_NOOP" not in output.out


def test_push_gate_isolates_nested_git_repositories_from_hook_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-push Git context must not redirect fixture Git writes to its parent."""
    import io
    import os
    import subprocess
    import sys

    import scripts.pre_push_gate as gate

    primary = tmp_path / "primary"
    primary.mkdir()
    _git(primary, "init", "-q", "-b", "main")
    _git(primary, "config", "user.name", "host")
    _git(primary, "config", "user.email", "host@example.invalid")
    head = _commit(primary, "base\n")
    original_config = (primary / ".git" / "config").read_bytes()
    child = tmp_path / "child"
    child.mkdir()

    # Git exports these repository pointers when executing pre-push hooks.
    monkeypatch.setenv("GIT_DIR", str(primary / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(primary))
    monkeypatch.setenv("GIT_INDEX_FILE", str(primary / ".git" / "index"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "preserved-author")
    monkeypatch.chdir(primary)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(f"refs/heads/main {head} refs/heads/test {ZERO}\n"),
    )

    qualified: list[str] = []

    def verify_nested(root: Path, sha: str, update: gate.PushUpdate) -> None:
        assert root == primary.resolve()
        assert sha == head
        assert update.remote_ref == "refs/heads/test"
        assert "GIT_DIR" not in os.environ
        assert "GIT_WORK_TREE" not in os.environ
        assert "GIT_INDEX_FILE" not in os.environ
        assert os.environ["GIT_AUTHOR_NAME"] == "preserved-author"
        subprocess.run(("git", "init", "-q", "-b", "main"), cwd=child, check=True)
        subprocess.run(("git", "config", "user.name", "child"), cwd=child, check=True)
        assert (child / ".git").is_dir()
        assert _git(child, "config", "user.name") == "child"
        assert (primary / ".git" / "config").read_bytes() == original_config
        qualified.append(sha)

    monkeypatch.setattr(gate, "qualify", verify_nested)
    assert gate.main() == 0
    assert qualified == [head]
    assert (primary / ".git" / "config").read_bytes() == original_config
