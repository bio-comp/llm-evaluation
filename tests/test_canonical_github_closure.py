from __future__ import annotations

from typing import Any, Final
import pytest
from pydantic import ValidationError

from canonical_sync.codec import parse_canonical_record
from canonical_sync.records import GitPublicationReceiptRecord
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


def test_complete_reconciliation_with_github_operational_path() -> None:
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

    git_receipt = GitPublicationReceiptRecord(
        status="unchanged",
        remote="https://github.com/bio-comp/llm-evaluation.git",
        remote_ref="refs/heads/main",
        expected_base_commit=MERGED_SHA,
        resulting_commit=MERGED_SHA,
        git_tree_oid="b" * 40,
        publication_tree_sha256=required.publication_tree_sha256,
        verified_remote_commit=MERGED_SHA,
        readback_verified=True,
        fast_forward_only=True,
    )

    provider = FakeGitHubProvider(
        pr_payload=valid_pr_payload(),
        run_payload=valid_run_payload(),
        jobs_payload=valid_jobs_payload(),
    )

    completed = complete_reconciliation_with_github(
        required=required,
        git_readback=git_receipt,
        pr_number=9,
        reviewed_head_sha=REVIEWED_HEAD_SHA,
        github_client=provider,
        run_id=99999,
    )

    assert completed.disposition is ReconciliationDisposition.COMPLETE
    assert completed.verified_publication is not None
    assert completed.verified_publication.pull_request.pull_request_number == 9
    assert completed.verified_publication.ci.run_id == 99999


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
            return FakeResponse(b'{"jobs": []}')
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
        "not-a-timestamp",
        "2026-10-28T21:00:00",
        "2026-10-28T21:00:00+00:00",
        "2026-10-28T21:00:00+05:00",
        "2026-10-28T21:00:00-08:00",
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
