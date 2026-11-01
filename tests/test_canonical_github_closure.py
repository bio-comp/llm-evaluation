from __future__ import annotations

from typing import Any, Final
import pytest
from pydantic import ValidationError

from canonical_sync.codec import parse_canonical_record
from llm_evaluation.canonical_reconciliation import (
    GitHubPullRequestMergeRecord,
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReviewedCanonicalImpact,
    current_executable_policy_contract,
    reconcile_reviewed_snapshot,
)
from llm_evaluation.canonical_github import (
    VerifiedGitHubClosureFacts,
    acquire_and_verify_github_closure,
    complete_reconciliation_with_github,
)

MERGED_SHA = "a" * 40
REVIEWED_HEAD_SHA = "c" * 40
TREE_SHA = "2" * 64
H1 = "1" * 64
H2 = "2" * 64
REPO: Final = "bio-comp/llm-evaluation"


def valid_pr_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "number": 9,
        "state": "closed",
        "merged": True,
        "merged_at": "2026-10-28T21:00:00Z",
        "merge_commit_sha": MERGED_SHA,
        "head": {
            "ref": "feat/task2-drive-reconciliation",
            "sha": REVIEWED_HEAD_SHA,
            "repo": {"full_name": REPO},
        },
        "base": {
            "ref": "main",
            "sha": "0" * 40,
            "repo": {"full_name": REPO},
        },
    }
    payload.update(overrides)
    return payload


def valid_run_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": 99999,
        "name": "Public CI",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        "head_sha": MERGED_SHA,
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": REPO},
    }
    payload.update(overrides)
    return payload


def valid_jobs_payload(**overrides: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": 88888,
        "run_id": 99999,
        "name": "verify",
        "head_sha": MERGED_SHA,
        "status": "completed",
        "conclusion": "success",
    }
    job.update(overrides)
    return {"jobs": [job]}


class FakeGitHubProvider:
    def __init__(
        self,
        pr_payload: dict[str, Any],
        run_payload: dict[str, Any],
        jobs_payload: dict[str, Any],
    ) -> None:
        self.pr_payload = pr_payload
        self.run_payload = run_payload
        self.jobs_payload = jobs_payload

    def get_pull_request(self, repo: str, pull_number: int) -> dict[str, Any]:
        return self.pr_payload

    def get_workflow_run(self, repo: str, run_id: int) -> dict[str, Any]:
        return self.run_payload

    def get_workflow_run_jobs(self, repo: str, run_id: int) -> dict[str, Any]:
        return self.jobs_payload


def test_authoritative_closure_happy_path() -> None:
    facts = acquire_and_verify_github_closure(
        pr_payload=valid_pr_payload(),
        run_payload=valid_run_payload(),
        jobs_payload=valid_jobs_payload(),
        expected_repo=REPO,
        expected_pr_number=9,
        expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
        expected_merged_commit=MERGED_SHA,
    )
    assert isinstance(facts, VerifiedGitHubClosureFacts)
    assert facts.pull_request.pull_request_number == 9
    assert facts.pull_request.merged_commit_sha == MERGED_SHA
    assert facts.pull_request.head_ref == "refs/heads/feat/task2-drive-reconciliation"
    assert facts.ci.run_id == 99999
    assert facts.ci.workflow_path == ".github/workflows/ci.yml"
    assert facts.ci.event == "push"
    assert facts.ci.job_name == "verify"
    assert facts.ci.job_conclusion == "success"


def test_authoritative_closure_rejects_synthetic_merge_sha_when_pr_not_merged() -> None:
    # Open or unmerged PR with non-null merge_commit_sha (synthetic test merge)
    unmerged_pr = valid_pr_payload(merged=False, merged_at=None, state="open")
    with pytest.raises(ValueError, match="not merged"):
        acquire_and_verify_github_closure(
            pr_payload=unmerged_pr,
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_pull_request_event_ci_run() -> None:
    pr_ci_run = valid_run_payload(event="pull_request")
    with pytest.raises(ValueError, match="event"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=pr_ci_run,
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_wrong_workflow_path() -> None:
    wrong_path = valid_run_payload(path=".github/workflows/other.yml")
    with pytest.raises(ValueError, match="workflow"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=wrong_path,
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_push_to_wrong_branch() -> None:
    wrong_branch = valid_run_payload(head_branch="feature-x")
    with pytest.raises(ValueError, match="branch"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=wrong_branch,
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_ci_with_wrong_head_sha() -> None:
    wrong_sha = valid_run_payload(head_sha="b" * 40)
    with pytest.raises(ValueError, match="commit"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=wrong_sha,
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_absent_verify_job() -> None:
    no_verify = {"jobs": [{"id": 1, "run_id": 99999, "name": "lint", "head_sha": MERGED_SHA, "status": "completed", "conclusion": "success"}]}
    with pytest.raises(ValueError, match="verify"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(),
            jobs_payload=no_verify,
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped"])
def test_authoritative_closure_rejects_unsuccessful_verify_job(conclusion: str) -> None:
    bad_job = valid_jobs_payload(conclusion=conclusion)
    with pytest.raises(ValueError, match="conclusion"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(),
            jobs_payload=bad_job,
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_wrong_repository() -> None:
    wrong_repo_pr = valid_pr_payload(
        head={"ref": "feat", "sha": REVIEWED_HEAD_SHA, "repo": {"full_name": "other/repo"}},
        base={"ref": "main", "sha": "0" * 40, "repo": {"full_name": "other/repo"}},
    )
    with pytest.raises(ValueError, match="repository"):
        acquire_and_verify_github_closure(
            pr_payload=wrong_repo_pr,
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_wrong_pr_head_sha() -> None:
    wrong_head = valid_pr_payload(
        head={"ref": "feat", "sha": "e" * 40, "repo": {"full_name": REPO}}
    )
    with pytest.raises(ValueError, match="head"):
        acquire_and_verify_github_closure(
            pr_payload=wrong_head,
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_authoritative_closure_rejects_commit_disagreement() -> None:
    # Run and PR disagree on commit
    mismatched_run = valid_run_payload(head_sha="f" * 40)
    with pytest.raises(ValueError, match="commit"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=mismatched_run,
            jobs_payload=valid_jobs_payload(head_sha="f" * 40),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_complete_reconciliation_with_github_operational_path(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from canonical_sync.errors import RemoteVerificationError
    from canonical_sync.git_publication import GitPublisher, PublicationFile, PublicationTree

    reconciled = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(
            impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
            public_contract=current_executable_policy_contract(),
        ),
    )
    required = reconciled.disposition
    assert required.disposition is ReconciliationDisposition.REQUIRED
    assert required.publication_tree_sha256 is not None
    assert required.public_contract_sha256 is not None
    assert reconciled.publication_tree is not None

    # Seed a disposable bare Git remote with the approved projection merged on refs/heads/main
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")
    base_pub = publisher.publish(
        PublicationTree(
            files=(
                PublicationFile("README.md", b"# llm-evaluation\n"),
                *reconciled.publication_tree.files,
            )
        ),
        expected_base=None,
        commit_message="Merge pull request #9",
    )
    merged_sha = base_pub.resulting_commit
    assert merged_sha != REVIEWED_HEAD_SHA  # Reviewed head H != Merge commit M

    provider = FakeGitHubProvider(
        pr_payload=valid_pr_payload(merge_commit_sha=merged_sha),
        run_payload=valid_run_payload(head_sha=merged_sha),
        jobs_payload=valid_jobs_payload(head_sha=merged_sha),
    )

    # Track git commands during closure to prove it is strictly read-only,
    # while routing the public GitHub remote to our disposable bare repository in tests.
    git_commands: list[str] = []
    original_git = GitPublisher._git
    active_authoritative_remote = remote

    def track_git(self_pub: GitPublisher, repo: Any, *args: str, **kwargs: Any) -> bytes:
        if args:
            git_commands.append(args[0])
        if (
            len(args) == 4
            and args[:3] == ("remote", "add", "origin")
            and args[3] == "https://github.com/bio-comp/llm-evaluation.git"
        ):
            args = ("remote", "add", "origin", str(active_authoritative_remote))
        return original_git(self_pub, repo, *args, **kwargs)

    monkeypatch.setattr(GitPublisher, "_git", track_git)

    completed = complete_reconciliation_with_github(
        required=required,
        publication_tree=reconciled.publication_tree,
        pr_number=9,
        reviewed_head_sha=REVIEWED_HEAD_SHA,
        github_client=provider,
        run_id=99999,
    )

    assert not {"push", "commit-tree", "hash-object", "mktree"} & set(git_commands)
    assert "fetch" in git_commands
    assert completed.disposition is ReconciliationDisposition.COMPLETE
    assert completed.verified_publication is not None
    assert completed.verified_publication.pull_request.pull_request_number == 9
    assert completed.verified_publication.pull_request.head_commit_sha == REVIEWED_HEAD_SHA
    assert completed.verified_publication.pull_request.merged_commit_sha == merged_sha
    assert completed.verified_publication.git.resulting_commit == merged_sha
    assert completed.verified_publication.git.remote == "https://github.com/bio-comp/llm-evaluation.git"
    assert completed.verified_publication.ci.run_id == 99999

    # Caller-constructed receipt parameter must not be accepted
    with pytest.raises(TypeError):
        complete_reconciliation_with_github(  # type: ignore[call-arg]
            required=required,
            git_readback=completed.verified_publication.git,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # Reproduction of PR #8 (2436e562ee6aa9f29e5c6fd9235cf0eab258eeff) + run 40183314427:
    # Genuine PR/CI facts at a commit lacking fixtures/reconciliation_policy_contract.json must fail!
    pr8_remote = tmp_path / "pr8_remote.git"
    subprocess.run(["git", "init", "--bare", str(pr8_remote)], check=True, capture_output=True)
    pr8_pub = GitPublisher(str(pr8_remote), ref="refs/heads/main").publish(
        PublicationTree(files=(PublicationFile("README.md", b"# pre-D016 main without fixture\n"),)),
        expected_base=None,
        commit_message="Merge pull request #8",
    )
    pr8_merge_sha = pr8_pub.resulting_commit
    pr8_head_sha = "9117ce1f39a1b6476eb7f976847d94a91df27821"
    pr8_provider = FakeGitHubProvider(
        pr_payload=valid_pr_payload(
            number=8,
            merge_commit_sha=pr8_merge_sha,
            head={
                "ref": "feat/pr8",
                "sha": pr8_head_sha,
                "repo": {"full_name": REPO},
            },
        ),
        run_payload=valid_run_payload(id=40183314427, head_sha=pr8_merge_sha),
        jobs_payload=valid_jobs_payload(run_id=40183314427, head_sha=pr8_merge_sha),
    )
    active_authoritative_remote = pr8_remote
    with pytest.raises(RemoteVerificationError, match="remote projection differs"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=8,
            reviewed_head_sha=pr8_head_sha,
            github_client=pr8_provider,
            run_id=40183314427,
        )

    # Wrong bytes in fixtures/reconciliation_policy_contract.json on main must also fail
    wrong_bytes_pub = GitPublisher(str(pr8_remote), ref="refs/heads/main").publish(
        PublicationTree(
            files=(
                PublicationFile(
                    "fixtures/reconciliation_policy_contract.json",
                    b'{"stale":true}\n',
                ),
            )
        ),
        expected_base=pr8_merge_sha,
        commit_message="wrong fixture bytes",
    )
    wrong_bytes_sha = wrong_bytes_pub.resulting_commit
    wrong_bytes_provider = FakeGitHubProvider(
        pr_payload=valid_pr_payload(merge_commit_sha=wrong_bytes_sha),
        run_payload=valid_run_payload(head_sha=wrong_bytes_sha),
        jobs_payload=valid_jobs_payload(head_sha=wrong_bytes_sha),
    )
    with pytest.raises(RemoteVerificationError, match="remote projection differs"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=wrong_bytes_provider,
            run_id=99999,
        )

    # If remote main tip moved away from merged_sha, verification must reject
    with pytest.raises(RemoteVerificationError, match="remote commit differs"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,  # expects merged_sha, but pr8_remote is at wrong_bytes_sha
            run_id=99999,
        )

    # Two-repository counterfactual: a local mirror at M must never be accepted or relabeled
    # as authoritative public-repository evidence when public main has advanced to N.
    for local_input in (remote, str(remote)):
        with pytest.raises(ValueError, match="authorized"):
            complete_reconciliation_with_github(
                required=required,
                publication_tree=reconciled.publication_tree,
                pr_number=9,
                reviewed_head_sha=REVIEWED_HEAD_SHA,
                github_client=provider,
                run_id=99999,
                git_remote=local_input,  # type: ignore[arg-type]
            )

    # Mismatched publication_tree vs required obligation must fail before completion
    tampered_tree = PublicationTree(
        files=(
            PublicationFile(
                "fixtures/reconciliation_policy_contract.json",
                b'{"tampered":true}\n',
            ),
        )
    )
    with pytest.raises(ValueError, match="REQUIRED"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=tampered_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # Unauthorized remote URL must fail closed
    with pytest.raises(ValueError, match="authorized"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
            git_remote="https://github.com/other/repo.git",
        )



def test_canonical_record_rejects_bare_branch_payload() -> None:
    import json

    # A durable payload with bare branch "feat/branch" instead of canonical "refs/heads/feat/branch"
    bare_head_payload = {
        "repository": REPO,
        "pull_request_number": 9,
        "head_ref": "feat/branch",
        "head_commit_sha": REVIEWED_HEAD_SHA,
        "base_ref": "refs/heads/main",
        "merged_commit_sha": MERGED_SHA,
        "merged_at": "2026-10-28T21:00:00Z",
    }
    raw_head_bytes = json.dumps(bare_head_payload).encode("utf-8")
    with pytest.raises(ValidationError):
        parse_canonical_record(raw_head_bytes, GitHubPullRequestMergeRecord)

    # A durable payload with bare "main" instead of canonical "refs/heads/main"
    bare_base_payload = {
        "repository": REPO,
        "pull_request_number": 9,
        "head_ref": "refs/heads/feat/branch",
        "head_commit_sha": REVIEWED_HEAD_SHA,
        "base_ref": "main",
        "merged_commit_sha": MERGED_SHA,
        "merged_at": "2026-10-28T21:00:00Z",
    }
    raw_base_bytes = json.dumps(bare_base_payload).encode("utf-8")
    with pytest.raises(ValidationError):
        parse_canonical_record(raw_base_bytes, GitHubPullRequestMergeRecord)


def test_closure_rejects_pr_head_ref_being_main() -> None:
    with pytest.raises(ValueError, match="head ref cannot be the main branch"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(
                head={"ref": "refs/heads/main", "sha": REVIEWED_HEAD_SHA, "repo": {"full_name": REPO}},
            ),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_pr_wrong_base_ref() -> None:
    with pytest.raises(ValueError, match="base ref"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(
                base={"ref": "refs/heads/develop", "sha": "0" * 40, "repo": {"full_name": REPO}},
            ),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_pr_wrong_repo_in_base() -> None:
    with pytest.raises(ValueError, match="repository"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(
                base={"ref": "refs/heads/main", "sha": "0" * 40, "repo": {"full_name": "other/repo"}},
            ),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_workflow_run_wrong_repo() -> None:
    with pytest.raises(ValueError, match="repository"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(repository={"full_name": "other/repo"}),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_workflow_run_non_success_conclusion() -> None:
    with pytest.raises(ValueError, match="conclusion"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(conclusion="failure"),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_job_wrong_head_sha() -> None:
    with pytest.raises(ValueError, match="does not match expected merged commit"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(head_sha="0" * 40),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_default_github_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    from io import BytesIO
    from llm_evaluation.canonical_github import DefaultGitHubTransport
    import urllib.request

    calls: list[str] = []

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._bio = BytesIO(data)

        def read(self) -> bytes:
            return self._bio.read()

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def fake_urlopen(req: urllib.request.Request, *args: Any, **kwargs: Any) -> FakeResponse:
        calls.append(req.full_url)
        assert req.get_header("Authorization") == "Bearer test-token"
        assert req.get_header("Accept") == "application/vnd.github+json"
        if "pulls" in req.full_url:
            return FakeResponse(b'{"number": 9, "state": "closed"}')
        if "jobs" in req.full_url:
            return FakeResponse(b'{"total_count": 0, "jobs": []}')
        return FakeResponse(b'{"id": 12345}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    transport = DefaultGitHubTransport(token="test-token")
    pr = transport.get_pull_request(REPO, 9)
    assert pr["number"] == 9
    run = transport.get_workflow_run(REPO, 12345)
    assert run["id"] == 12345
    jobs = transport.get_workflow_run_jobs(REPO, 12345)
    assert jobs["jobs"] == []


def test_default_github_transport_rejects_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_evaluation.canonical_github import DefaultGitHubTransport
    import urllib.request

    class FakeResponse:
        def read(self) -> bytes:
            return b'["not", "a", "dict"]'

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse())

    transport = DefaultGitHubTransport()
    with pytest.raises(ValueError, match="non-dictionary"):
        transport.get_pull_request(REPO, 9)


def test_default_github_transport_passes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_evaluation.canonical_github import DefaultGitHubTransport
    import urllib.request

    observed_timeouts: list[float] = []

    class FakeResponse:
        def read(self) -> bytes:
            return b'{"number": 9}'

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> FakeResponse:
        observed_timeouts.append(timeout)
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    transport = DefaultGitHubTransport(timeout=15.0)
    transport.get_pull_request(REPO, 9)
    assert observed_timeouts == [15.0]


def test_closure_rejects_multiple_verify_jobs() -> None:
    jobs = [
        {
            "id": 88881,
            "run_id": 99999,
            "name": "verify",
            "head_sha": MERGED_SHA,
            "status": "completed",
            "conclusion": "success",
        },
        {
            "id": 88882,
            "run_id": 99999,
            "name": "verify",
            "head_sha": MERGED_SHA,
            "status": "completed",
            "conclusion": "success",
        },
    ]
    with pytest.raises(ValueError, match="multiple jobs with required name 'verify'"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(),
            jobs_payload={"jobs": jobs},
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_accepts_pr_from_contributor_fork() -> None:
    facts = acquire_and_verify_github_closure(
        pr_payload=valid_pr_payload(
            head={
                "ref": "feat/my-feature",
                "sha": REVIEWED_HEAD_SHA,
                "repo": {"full_name": "contributor/llm-evaluation"},
            },
            base={
                "ref": "refs/heads/main",
                "sha": "0" * 40,
                "repo": {"full_name": REPO},
            },
        ),
        run_payload=valid_run_payload(),
        jobs_payload=valid_jobs_payload(),
        expected_repo=REPO,
        expected_pr_number=9,
        expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
        expected_merged_commit=MERGED_SHA,
    )
    assert facts.pull_request.head_ref == "refs/heads/feat/my-feature"
    assert facts.pull_request.repository == REPO


@pytest.mark.parametrize(
    "bad_ts",
    [
        "",
        "   ",
        " 2026-10-28T21:00:00Z",
        "2026-10-28T21:00:00Z ",
        "not-a-timestamp",
        "2026-10-28T21:00:00",
        "2026-10-28T21:00:00+00:00",
        "2026-10-28T21:00:00+05:00",
        "2026-10-28T21:00:00-08:00",
        "2026-10-28T21:00:00.000000Z",
        "2026-10-28T21:00:00.000Z",
        "2026-10-28t21:00:00z",
        "2026-10-28 21:00:00Z",
        "2026-10-28T21:00Z",
        "2026-02-30T21:00:00Z",
        "2026-13-01T21:00:00Z",
        "2026-10-28T25:00:00Z",
    ],
)
def test_pull_request_record_rejects_non_canonical_utc_timestamp(bad_ts: str) -> None:
    with pytest.raises(ValidationError):
        GitHubPullRequestMergeRecord(
            repository=REPO,
            pull_request_number=9,
            head_ref="refs/heads/feat/task2",
            head_commit_sha=REVIEWED_HEAD_SHA,
            base_ref="refs/heads/main",
            merged_commit_sha=MERGED_SHA,
            merged_at=bad_ts,
        )


def test_wire_model_rejects_lax_type_coercion() -> None:
    # Stringified number should fail
    with pytest.raises(ValueError, match="failed wire validation"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(number="9"),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )

    # Stringified boolean merged should fail
    with pytest.raises(ValueError, match="failed wire validation"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(merged="true"),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )

    # Stringified run id should fail
    with pytest.raises(ValueError, match="failed wire validation"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(id="99999"),
            jobs_payload=valid_jobs_payload(),
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )

    # Stringified job id should fail
    with pytest.raises(ValueError, match="failed wire validation"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(),
            jobs_payload={"jobs": [dict(valid_jobs_payload()["jobs"][0], id="88888")]},
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_unauthorized_expected_repo() -> None:
    with pytest.raises(ValueError, match="must match authorized project repository"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(),
            jobs_payload=valid_jobs_payload(),
            expected_repo="other/repo",
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_closure_rejects_verify_job_with_mismatched_run_id() -> None:
    job = dict(valid_jobs_payload()["jobs"][0], run_id=77777)
    with pytest.raises(ValueError, match="does not match expected run ID"):
        acquire_and_verify_github_closure(
            pr_payload=valid_pr_payload(),
            run_payload=valid_run_payload(id=99999),
            jobs_payload={"jobs": [job]},
            expected_repo=REPO,
            expected_pr_number=9,
            expected_reviewed_head_sha=REVIEWED_HEAD_SHA,
            expected_merged_commit=MERGED_SHA,
        )


def test_default_github_transport_paginates_workflow_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_evaluation.canonical_github import DefaultGitHubTransport
    import json
    import urllib.request

    requested_urls: list[str] = []

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def fake_urlopen(req: urllib.request.Request, *args: Any, **kwargs: Any) -> FakeResponse:
        requested_urls.append(req.full_url)
        if "&page=1" in req.full_url:
            payload = {"total_count": 2, "jobs": [{"id": 1, "name": "setup"}]}
        elif "&page=2" in req.full_url:
            payload = {"total_count": 2, "jobs": [{"id": 2, "name": "verify"}]}
        else:
            payload = {"total_count": 2, "jobs": []}
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    transport = DefaultGitHubTransport()
    resp = transport.get_workflow_run_jobs(REPO, 12345)
    assert resp["total_count"] == 2
    assert len(resp["jobs"]) == 2
    assert [j["id"] for j in resp["jobs"]] == [1, 2]
    assert len(requested_urls) == 2
    assert "per_page=100&page=1" in requested_urls[0]
    assert "per_page=100&page=2" in requested_urls[1]


def test_default_github_transport_workflow_jobs_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_evaluation.canonical_github import DefaultGitHubTransport
    import urllib.request

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    # 1. Empty response
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"total_count": 0, "jobs": []}')
    )
    transport = DefaultGitHubTransport()
    assert transport.get_workflow_run_jobs(REPO, 12345) == {"total_count": 0, "jobs": []}

    # 2. Non-list jobs field raises ValueError
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"total_count": 1, "jobs": "invalid"}')
    )
    with pytest.raises(ValueError, match="unexpected non-list 'jobs' field"):
        transport.get_workflow_run_jobs(REPO, 12345)

    # 3. Missing or non-integer total_count raises ValueError
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"jobs": []}')
    )
    with pytest.raises(ValueError, match="missing valid integer 'total_count'"):
        transport.get_workflow_run_jobs(REPO, 12345)

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"total_count": "1", "jobs": []}')
    )
    with pytest.raises(ValueError, match="missing valid integer 'total_count'"):
        transport.get_workflow_run_jobs(REPO, 12345)

    # Boolean total_count (subclass of int in Python) must also be rejected
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"total_count": true, "jobs": []}')
    )
    with pytest.raises(ValueError, match="missing valid integer 'total_count'"):
        transport.get_workflow_run_jobs(REPO, 12345)

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"total_count": false, "jobs": []}')
    )
    with pytest.raises(ValueError, match="missing valid integer 'total_count'"):
        transport.get_workflow_run_jobs(REPO, 12345)

    # 4. Premature truncation (fewer jobs returned than total_count before empty page)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, *a, **kw: FakeResponse(b'{"total_count": 5, "jobs": []}')
    )
    with pytest.raises(ValueError, match="truncated prematurely"):
        transport.get_workflow_run_jobs(REPO, 12345)

    # 5. Safety cap exceeded before complete inventory fails closed
    def fake_capped_urlopen(req: urllib.request.Request, *args: Any, **kwargs: Any) -> FakeResponse:
        import json
        payload = {"total_count": 5, "jobs": [{"id": 1, "name": "job"}]}
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_capped_urlopen)
    capped_transport = DefaultGitHubTransport(max_pages=2)
    with pytest.raises(ValueError, match="pagination exceeded safety bound"):
        capped_transport.get_workflow_run_jobs(REPO, 12345)

    # 6. total_count drift during pagination fails closed
    drift_responses = [
        FakeResponse(b'{"total_count": 2, "jobs": [{"id": 1, "name": "verify"}]}'),
        FakeResponse(b'{"total_count": 3, "jobs": [{"id": 2, "name": "other"}]}'),
    ]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, *a, **kw: drift_responses.pop(0))
    drift_transport = DefaultGitHubTransport()
    with pytest.raises(ValueError, match="total_count changed during pagination"):
        drift_transport.get_workflow_run_jobs(REPO, 12345)

    # 7. Total count mismatch (received more jobs than declared total_count)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, *a, **kw: FakeResponse(
            b'{"total_count": 1, "jobs": [{"id": 1, "name": "job1"}, {"id": 2, "name": "job2"}]}'
        ),
    )
    with pytest.raises(ValueError, match="response size mismatch"):
        transport.get_workflow_run_jobs(REPO, 12345)

    # 8. max_pages parameter validation
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        DefaultGitHubTransport(max_pages=0)
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        DefaultGitHubTransport(max_pages=-1)
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        DefaultGitHubTransport(max_pages="10")  # type: ignore[arg-type]


def test_complete_reconciliation_with_github_projection_and_remote_binding_edges(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from canonical_sync.errors import RemoteVerificationError
    from canonical_sync.git_publication import GitPublisher, PublicationFile, PublicationTree

    contract = current_executable_policy_contract()
    reconciled = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(
            impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
            public_contract=contract,
        ),
    )
    required = reconciled.disposition
    assert reconciled.publication_tree is not None

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")
    pub = publisher.publish(
        reconciled.publication_tree,
        expected_base=None,
        commit_message="Merge pull request #9",
    )
    merged_sha = pub.resulting_commit
    provider = FakeGitHubProvider(
        pr_payload=valid_pr_payload(merge_commit_sha=merged_sha),
        run_payload=valid_run_payload(head_sha=merged_sha),
        jobs_payload=valid_jobs_payload(head_sha=merged_sha),
    )

    original_git = GitPublisher._git

    def route_public_remote(
        self_pub: GitPublisher, repo: Any, *args: str, **kwargs: Any
    ) -> bytes:
        if (
            len(args) == 4
            and args[:3] == ("remote", "add", "origin")
            and args[3] == "https://github.com/bio-comp/llm-evaluation.git"
        ):
            args = ("remote", "add", "origin", str(remote))
        return original_git(self_pub, repo, *args, **kwargs)

    monkeypatch.setattr(GitPublisher, "_git", route_public_remote)

    # 1. Supplying public_contract alone derives and verifies the approved PublicationTree
    completed_by_contract = complete_reconciliation_with_github(
        required=required,
        public_contract=contract,
        pr_number=9,
        reviewed_head_sha=REVIEWED_HEAD_SHA,
        github_client=provider,
        run_id=99999,
    )
    assert completed_by_contract.disposition is ReconciliationDisposition.COMPLETE

    # 2. Supplying both public_contract and matching publication_tree succeeds
    completed_both = complete_reconciliation_with_github(
        required=required,
        publication_tree=reconciled.publication_tree,
        public_contract=contract,
        pr_number=9,
        reviewed_head_sha=REVIEWED_HEAD_SHA,
        github_client=provider,
        run_id=99999,
    )
    assert completed_both.disposition is ReconciliationDisposition.COMPLETE

    # 3. Non-REQUIRED disposition is rejected
    none_res = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(impact=ProjectImpact.NO_PUBLIC_IMPACT),
    )
    with pytest.raises(ValueError, match="only a REQUIRED reconciliation"):
        complete_reconciliation_with_github(
            required=none_res.disposition,
            public_contract=contract,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # 4. Missing both publication_tree and public_contract is rejected
    with pytest.raises(ValueError, match="requires an approved publication_tree or public_contract"):
        complete_reconciliation_with_github(
            required=required,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # 5. Stale/mismatched public_contract vs REQUIRED is rejected
    stale_contract = contract.model_copy(update={"complete_requires_readback": False})
    with pytest.raises(ValueError, match="public_contract_sha256"):
        complete_reconciliation_with_github(
            required=required,
            public_contract=stale_contract,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # 6. Disagreement between supplied public_contract and supplied publication_tree is rejected
    other_tree = PublicationTree(
        files=(PublicationFile("fixtures/other.json", b"{}\n"),)
    )
    with pytest.raises(ValueError, match="does not match supplied public_contract"):
        complete_reconciliation_with_github(
            required=required,
            public_contract=contract,
            publication_tree=other_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # 7. publication_tree without fixtures/reconciliation_policy_contract.json is rejected
    with pytest.raises(ValueError, match="fixtures/reconciliation_policy_contract.json"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=other_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )

    # 8. Invalid git_remote values (local Path, local directory string, untrimmed string) are rejected
    with pytest.raises(ValueError, match="authorized"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
            git_remote=tmp_path / "does-not-exist.git",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="authorized"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
            git_remote="  ",
        )

    # 9. Remote movement during readback in complete_reconciliation_with_github fails closed
    second_pub = publisher.publish(
        PublicationTree(
            files=(
                PublicationFile("EXTRA.txt", b"concurrent\n"),
                *reconciled.publication_tree.files,
            )
        ),
        expected_base=merged_sha,
        commit_message="advance main",
    )
    advanced_sha = second_pub.resulting_commit
    subprocess.run(
        ["git", "-C", str(remote), "update-ref", "refs/heads/main", merged_sha],
        check=True,
        capture_output=True,
    )

    def move_remote_after_fetch(
        self_pub: GitPublisher, repo: Any, *args: str, **kwargs: Any
    ) -> bytes:
        res = route_public_remote(self_pub, repo, *args, **kwargs)
        if args and args[0] == "fetch":
            subprocess.run(
                ["git", "-C", str(remote), "update-ref", "refs/heads/main", advanced_sha],
                check=True,
                capture_output=True,
            )
        return res

    monkeypatch.setattr(GitPublisher, "_git", move_remote_after_fetch)
    with pytest.raises(RemoteVerificationError, match="remote moved during readback"):
        complete_reconciliation_with_github(
            required=required,
            publication_tree=reconciled.publication_tree,
            pr_number=9,
            reviewed_head_sha=REVIEWED_HEAD_SHA,
            github_client=provider,
            run_id=99999,
        )
