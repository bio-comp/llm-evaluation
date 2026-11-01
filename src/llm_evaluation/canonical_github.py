from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from canonical_sync.codec import (
    canonical_record_bytes,
    canonical_record_hash,
    parse_canonical_record,
)
from canonical_sync.git_publication import (
    DEFAULT_GIT_COMMAND_TIMEOUT,
    GitPublisher,
    PublicationTree,
)
from canonical_sync.records import safe_remote_identity
from llm_evaluation.canonical_reconciliation import (
    _PUBLIC_GIT_REMOTES,
    GitHubCIVerificationRecord,
    GitHubPullRequestMergeRecord,
    ReconciliationDisposition,
    ReconciliationDispositionRecord,
    ReconciliationPolicyContract,
    VerifiedPublicationReceiptRecord,
    complete_reconciliation,
)

_EXPECTED_REPO: Final = "bio-comp/llm-evaluation"
_EXPECTED_BASE_REF: Final = "refs/heads/main"
_EXPECTED_WORKFLOW_NAME: Final = "Public CI"
_EXPECTED_WORKFLOW_PATH: Final = ".github/workflows/ci.yml"
_EXPECTED_JOB_NAME: Final = "verify"
_EXPECTED_EVENT: Final = "push"
_DEFAULT_PUBLIC_GIT_REMOTE: Final = "https://github.com/bio-comp/llm-evaluation.git"
_POLICY_FIXTURE_PATH: Final = "fixtures/reconciliation_policy_contract.json"


class _UntrustedWireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _WireRepoRef(_UntrustedWireModel):
    ref: str
    sha: str
    repo: dict[str, Any] = Field(default_factory=dict)


class _WirePullRequest(_UntrustedWireModel):
    number: int
    state: str
    merged: bool = False
    merged_at: str | None = None
    merge_commit_sha: str | None = None
    head: _WireRepoRef
    base: _WireRepoRef


class _WireWorkflowRun(_UntrustedWireModel):
    id: int
    name: str
    path: str
    event: str
    head_branch: str
    head_sha: str
    status: str
    conclusion: str | None = None
    repository: dict[str, Any] = Field(default_factory=dict)


class _WireWorkflowJob(_UntrustedWireModel):
    id: int
    run_id: int
    name: str
    head_sha: str
    status: str
    conclusion: str | None = None


class _WireJobsResponse(_UntrustedWireModel):
    jobs: list[_WireWorkflowJob]


class GitHubClosureProvider(Protocol):
    def get_pull_request(self, repo: str, pull_number: int) -> dict[str, Any]: ...
    def get_workflow_run(self, repo: str, run_id: int) -> dict[str, Any]: ...
    def get_workflow_run_jobs(self, repo: str, run_id: int) -> dict[str, Any]: ...


class DefaultGitHubTransport:
    def __init__(
        self,
        token: str | None = None,
        timeout: float = 30.0,
        *,
        max_pages: int = 100,
    ) -> None:
        if type(max_pages) is not int or max_pages < 1:
            raise ValueError("max_pages must be a positive integer")
        self.token = token
        self.timeout = timeout
        self.max_pages = max_pages

    def _get(self, path: str) -> dict[str, Any]:
        url = f"https://api.github.com/{path.lstrip('/')}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "llm-evaluation-canonical-sync",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read().decode("utf-8")
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("unexpected non-dictionary response from GitHub API")
            return result

    def get_pull_request(self, repo: str, pull_number: int) -> dict[str, Any]:
        return self._get(f"repos/{repo}/pulls/{pull_number}")

    def get_workflow_run(self, repo: str, run_id: int) -> dict[str, Any]:
        return self._get(f"repos/{repo}/actions/runs/{run_id}")

    def get_workflow_run_jobs(self, repo: str, run_id: int) -> dict[str, Any]:
        all_jobs: list[dict[str, Any]] = []
        page = 1
        total_count: int | None = None
        while True:
            resp = self._get(
                f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}"
            )
            jobs = resp.get("jobs")
            if not isinstance(jobs, list):
                raise ValueError(
                    "unexpected non-list 'jobs' field in GitHub workflow jobs response"
                )
            raw_total = resp.get("total_count")
            if type(raw_total) is not int or raw_total < 0:
                raise ValueError(
                    "GitHub workflow jobs response missing valid integer 'total_count'"
                )
            if total_count is None:
                total_count = raw_total
            elif raw_total != total_count:
                raise ValueError(
                    f"GitHub workflow jobs total_count changed during pagination (was {total_count}, now {raw_total})"
                )
            all_jobs.extend(jobs)
            if len(all_jobs) >= total_count:
                break
            if not jobs:
                raise ValueError(
                    f"GitHub workflow jobs response truncated prematurely: "
                    f"received {len(all_jobs)} of {total_count} jobs"
                )
            if page >= self.max_pages:
                raise ValueError(
                    "workflow job pagination exceeded safety bound before complete inventory"
                )
            page += 1
        if len(all_jobs) != total_count:
            raise ValueError(
                f"GitHub workflow jobs response size mismatch: received {len(all_jobs)} jobs for total_count {total_count}"
            )
        return {"total_count": total_count, "jobs": all_jobs}


@dataclass(frozen=True, slots=True)
class VerifiedGitHubClosureFacts:
    pull_request: GitHubPullRequestMergeRecord
    ci: GitHubCIVerificationRecord


def _normalize_branch_ref(ref: str) -> str:
    if ref.startswith("refs/heads/"):
        return ref
    return f"refs/heads/{ref}"


def acquire_and_verify_github_closure(
    *,
    pr_payload: dict[str, Any],
    run_payload: dict[str, Any],
    jobs_payload: dict[str, Any],
    expected_repo: str = _EXPECTED_REPO,
    expected_pr_number: int,
    expected_reviewed_head_sha: str,
    expected_merged_commit: str,
) -> VerifiedGitHubClosureFacts:
    if expected_repo != _EXPECTED_REPO:
        raise ValueError(
            f"expected_repo '{expected_repo}' must match authorized project repository '{_EXPECTED_REPO}'"
        )

    try:
        wire_pr = _WirePullRequest.model_validate(pr_payload)
    except Exception as exc:
        raise ValueError(f"GitHub PR response failed wire validation: {exc}") from exc

    if wire_pr.number != expected_pr_number:
        raise ValueError(
            f"GitHub PR number {wire_pr.number} differs from expected PR #{expected_pr_number}"
        )

    pr_repo_name = wire_pr.base.repo.get("full_name")
    if pr_repo_name != expected_repo:
        raise ValueError(
            f"GitHub PR repository '{pr_repo_name}' differs from expected repository '{expected_repo}'"
        )

    if not wire_pr.merged or wire_pr.merged_at is None:
        raise ValueError(f"GitHub PR #{expected_pr_number} is not merged")

    if (
        wire_pr.merge_commit_sha is None
        or wire_pr.merge_commit_sha != expected_merged_commit
    ):
        raise ValueError(
            f"GitHub PR #{expected_pr_number} merged commit '{wire_pr.merge_commit_sha}' "
            f"does not match expected commit '{expected_merged_commit}'"
        )

    if wire_pr.head.sha != expected_reviewed_head_sha:
        raise ValueError(
            f"GitHub PR #{expected_pr_number} head SHA '{wire_pr.head.sha}' "
            f"does not match expected reviewed head SHA '{expected_reviewed_head_sha}'"
        )

    canonical_base_ref = _normalize_branch_ref(wire_pr.base.ref)
    if canonical_base_ref != _EXPECTED_BASE_REF:
        raise ValueError(
            f"GitHub PR #{expected_pr_number} base ref '{canonical_base_ref}' is not '{_EXPECTED_BASE_REF}'"
        )

    canonical_head_ref = _normalize_branch_ref(wire_pr.head.ref)
    if canonical_head_ref == _EXPECTED_BASE_REF:
        raise ValueError("GitHub PR head ref cannot be the main branch")

    try:
        pr_record = GitHubPullRequestMergeRecord(
            repository=_EXPECTED_REPO,
            pull_request_number=wire_pr.number,
            head_ref=canonical_head_ref,
            head_commit_sha=wire_pr.head.sha,
            base_ref=_EXPECTED_BASE_REF,
            merged_commit_sha=wire_pr.merge_commit_sha,
            merged_at=wire_pr.merged_at,
        )
    except Exception as exc:
        raise ValueError(
            f"constructed PR merge record failed validation: {exc}"
        ) from exc

    try:
        wire_run = _WireWorkflowRun.model_validate(run_payload)
    except Exception as exc:
        raise ValueError(
            f"GitHub workflow run response failed wire validation: {exc}"
        ) from exc

    run_repo_name = wire_run.repository.get("full_name")
    if run_repo_name != expected_repo:
        raise ValueError(
            f"GitHub CI run repository '{run_repo_name}' differs from expected repository '{expected_repo}'"
        )

    if wire_run.event != _EXPECTED_EVENT:
        raise ValueError(
            f"GitHub CI run event was '{wire_run.event}', required post-merge event is '{_EXPECTED_EVENT}'"
        )

    canonical_run_branch = _normalize_branch_ref(wire_run.head_branch)
    if canonical_run_branch != _EXPECTED_BASE_REF:
        raise ValueError(
            f"GitHub CI run branch was '{canonical_run_branch}', required branch is '{_EXPECTED_BASE_REF}'"
        )

    if wire_run.path != _EXPECTED_WORKFLOW_PATH:
        raise ValueError(
            f"GitHub CI run workflow path was '{wire_run.path}', required is '{_EXPECTED_WORKFLOW_PATH}'"
        )

    if wire_run.name != _EXPECTED_WORKFLOW_NAME:
        raise ValueError(
            f"GitHub CI run workflow name was '{wire_run.name}', required is '{_EXPECTED_WORKFLOW_NAME}'"
        )

    if wire_run.head_sha != expected_merged_commit:
        raise ValueError(
            f"GitHub CI run commit '{wire_run.head_sha}' does not match expected merged commit '{expected_merged_commit}'"
        )

    if wire_run.status != "completed" or wire_run.conclusion != "success":
        raise ValueError(
            f"GitHub CI run {wire_run.id} conclusion was '{wire_run.conclusion}' (status: '{wire_run.status}'), expected 'success'"
        )

    try:
        wire_jobs = _WireJobsResponse.model_validate(jobs_payload)
    except Exception as exc:
        raise ValueError(
            f"GitHub workflow jobs response failed wire validation: {exc}"
        ) from exc

    matching_jobs = [job for job in wire_jobs.jobs if job.name == _EXPECTED_JOB_NAME]
    if not matching_jobs:
        raise ValueError(
            f"GitHub CI run {wire_run.id} does not contain the required '{_EXPECTED_JOB_NAME}' job"
        )
    if len(matching_jobs) > 1:
        raise ValueError(
            f"GitHub CI run {wire_run.id} contains multiple jobs with required name '{_EXPECTED_JOB_NAME}'"
        )
    verify_job = matching_jobs[0]

    if verify_job.run_id != wire_run.id:
        raise ValueError(
            f"GitHub CI '{_EXPECTED_JOB_NAME}' job run ID {verify_job.run_id} does not match expected run ID {wire_run.id}"
        )

    if verify_job.head_sha != expected_merged_commit:
        raise ValueError(
            f"GitHub CI '{_EXPECTED_JOB_NAME}' job commit '{verify_job.head_sha}' does not match expected merged commit '{expected_merged_commit}'"
        )

    if verify_job.status != "completed" or verify_job.conclusion != "success":
        raise ValueError(
            f"GitHub CI '{_EXPECTED_JOB_NAME}' job conclusion was '{verify_job.conclusion}', expected 'success'"
        )

    try:
        ci_record = GitHubCIVerificationRecord(
            repository=_EXPECTED_REPO,
            run_id=wire_run.id,
            workflow_name=_EXPECTED_WORKFLOW_NAME,
            workflow_path=_EXPECTED_WORKFLOW_PATH,
            event=_EXPECTED_EVENT,
            head_branch=_EXPECTED_BASE_REF,
            commit_sha=wire_run.head_sha,
            run_conclusion="success",
            job_id=verify_job.id,
            job_name=_EXPECTED_JOB_NAME,
            job_conclusion="success",
        )
    except Exception as exc:
        raise ValueError(
            f"constructed CI verification record failed validation: {exc}"
        ) from exc

    return VerifiedGitHubClosureFacts(pull_request=pr_record, ci=ci_record)


def _resolve_verified_projection_tree(
    *,
    required: ReconciliationDispositionRecord,
    publication_tree: PublicationTree | None,
    public_contract: ReconciliationPolicyContract | None,
) -> tuple[PublicationTree, str, str]:
    if required.disposition is not ReconciliationDisposition.REQUIRED:
        raise ValueError("only a REQUIRED reconciliation can become COMPLETE")
    if (
        required.public_contract_sha256 is None
        or required.publication_tree_sha256 is None
    ):
        raise ValueError(
            "REQUIRED reconciliation disposition missing contract or tree hash"
        )
    if publication_tree is None:
        raise ValueError(
            "operational closure requires the exact approved publication_tree; "
            "a public_contract alone cannot synthesize implementation evidence"
        )
    if not isinstance(publication_tree, PublicationTree):
        raise TypeError("publication_tree must be a PublicationTree")
    files_by_path = publication_tree.as_dict()
    contract_bytes = files_by_path.get(_POLICY_FIXTURE_PATH)
    if contract_bytes is None:
        raise ValueError(
            f"approved publication_tree must contain '{_POLICY_FIXTURE_PATH}'"
        )
    try:
        parsed_contract = parse_canonical_record(
            contract_bytes, ReconciliationPolicyContract
        )
    except Exception as exc:
        raise ValueError(
            "supplied publication_tree does not contain the REQUIRED public contract"
        ) from exc
    if canonical_record_hash(parsed_contract) != required.public_contract_sha256:
        raise ValueError(
            "supplied publication_tree does not match REQUIRED public_contract_sha256"
        )
    if public_contract is not None:
        if not isinstance(public_contract, ReconciliationPolicyContract):
            raise TypeError("public_contract must be a ReconciliationPolicyContract")
        if canonical_record_hash(public_contract) != required.public_contract_sha256:
            raise ValueError(
                "supplied public_contract does not match REQUIRED public_contract_sha256"
            )
        if contract_bytes != canonical_record_bytes(public_contract):
            raise ValueError(
                "supplied publication_tree policy fixture does not exactly match supplied public_contract"
            )

    if publication_tree.sha256 != required.publication_tree_sha256:
        raise ValueError(
            "supplied publication_tree does not match REQUIRED publication_tree_sha256"
        )
    return (
        publication_tree,
        required.public_contract_sha256,
        required.publication_tree_sha256,
    )


def _resolve_closure_git_remote(git_remote: str) -> str:
    if (
        not isinstance(git_remote, str)
        or not git_remote
        or git_remote != git_remote.strip()
    ):
        raise ValueError(
            "git_remote must identify the authorized bio-comp/llm-evaluation repository"
        )
    if (
        git_remote in _PUBLIC_GIT_REMOTES
        or safe_remote_identity(git_remote) in _PUBLIC_GIT_REMOTES
    ):
        return git_remote
    raise ValueError(
        "git_remote must identify the authorized bio-comp/llm-evaluation repository"
    )


def complete_reconciliation_with_github(
    *,
    required: ReconciliationDispositionRecord,
    publication_tree: PublicationTree | None = None,
    public_contract: ReconciliationPolicyContract | None = None,
    pr_number: int,
    reviewed_head_sha: str,
    github_client: GitHubClosureProvider,
    run_id: int,
    git_remote: str = _DEFAULT_PUBLIC_GIT_REMOTE,
    git_command_timeout: int = DEFAULT_GIT_COMMAND_TIMEOUT,
) -> ReconciliationDispositionRecord:
    verified_tree, contract_sha256, tree_sha256 = _resolve_verified_projection_tree(
        required=required,
        publication_tree=publication_tree,
        public_contract=public_contract,
    )
    remote_locator = _resolve_closure_git_remote(git_remote)

    pr_raw = github_client.get_pull_request(_EXPECTED_REPO, pr_number)
    run_raw = github_client.get_workflow_run(_EXPECTED_REPO, run_id)
    jobs_raw = github_client.get_workflow_run_jobs(_EXPECTED_REPO, run_id)

    raw_merged_commit = (
        pr_raw.get("merge_commit_sha") if isinstance(pr_raw, dict) else None
    )
    merged_commit = raw_merged_commit if isinstance(raw_merged_commit, str) else ""

    facts = acquire_and_verify_github_closure(
        pr_payload=pr_raw,
        run_payload=run_raw,
        jobs_payload=jobs_raw,
        expected_repo=_EXPECTED_REPO,
        expected_pr_number=pr_number,
        expected_reviewed_head_sha=reviewed_head_sha,
        expected_merged_commit=merged_commit,
    )

    publisher = GitPublisher(
        remote_locator,
        ref=_EXPECTED_BASE_REF,
        command_timeout=git_command_timeout,
    )
    git_readback = publisher.verify(
        verified_tree,
        expected_commit=facts.pull_request.merged_commit_sha,
    )

    receipt = VerifiedPublicationReceiptRecord(
        snapshot_state_sha256=required.current_snapshot_sha256,
        public_contract_sha256=contract_sha256,
        publication_tree_sha256=tree_sha256,
        obligation_ids=required.obligation_ids,
        git=git_readback,
        pull_request=facts.pull_request,
        ci=facts.ci,
    )

    return complete_reconciliation(required, receipt)
