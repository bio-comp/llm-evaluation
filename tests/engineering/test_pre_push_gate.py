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
