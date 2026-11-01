from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from canonical_sync.codec import (
    canonical_record_bytes,
    canonical_record_hash,
)
from canonical_sync.records import GitPublicationReceiptRecord
from llm_evaluation.canonical_authority import D016_REQUIRED_ROLES, load_d016_authority_profile
from llm_evaluation.canonical_reconciliation import (
    DEFAULT_RECONCILIATION_STATE_PATH,
    RECONCILIATION_STATE_ENV_VAR,
    DurableReconciliationStateRecord,
    GitHubCIVerificationRecord,
    GitHubPullRequestMergeRecord,
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReconciliationDispositionRecord,
    ReconciliationPolicyContract,
    ReconciliationStateError,
    ReviewedCanonicalImpact,
    VerifiedPublicationReceiptRecord,
    approved_policy_tree,
    current_executable_policy_contract,
    load_durable_reconciliation_state,
    load_last_reconciled_state,
    persist_completed_reconciliation,
    reconcile_reviewed_snapshot,
    resolve_reconciliation_state_path,
    save_durable_reconciliation_state,
    verify_executable_policy_contract,
)

H1 = "1" * 64
H2 = "2" * 64


def _write_profile(path: Path, *, roles: tuple[str, ...] | None = None) -> None:
    selected = roles or tuple(sorted(D016_REQUIRED_ROLES))
    lines = ['[profile]', 'profile_name = "llm-evaluation-d016"']
    for index, role in enumerate(selected, start=1):
        lines.extend(
            [
                "",
                "[[profile.objects]]",
                f'logical_path = "{role}"',
                'provider = "google_drive"',
                f'object_id = "private-object-{index}"',
                f'expected_name = "{role}"',
                'expected_mime_type = "application/vnd.google-apps.document"',
                'expected_parent_id = "private-parent"',
                'representation_schema = "google-drive-export-markdown-v1"',
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def test_private_d016_profile_loads_exact_roles_without_public_defaults(tmp_path: Path) -> None:
    profile_path = tmp_path / "authority.toml"
    _write_profile(profile_path)
    profile = load_d016_authority_profile(profile_path)

    assert {item.logical_path for item in profile.objects} == D016_REQUIRED_ROLES
    assert {item.object_id for item in profile.objects} == {
        "private-object-1",
        "private-object-2",
        "private-object-3",
        "private-object-4",
    }
    public_bytes = canonical_record_bytes(current_executable_policy_contract())
    assert all(item.object_id.encode() not in public_bytes for item in profile.objects)
    assert b"private-parent" not in public_bytes


def test_private_profile_rejects_missing_extra_and_unknown_top_level_state(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    _write_profile(missing, roles=tuple(sorted(D016_REQUIRED_ROLES))[:-1])
    with pytest.raises(ValueError, match="exact D-016 roles"):
        load_d016_authority_profile(missing)

    extra = tmp_path / "extra.toml"
    _write_profile(extra, roles=tuple(sorted(D016_REQUIRED_ROLES)) + ("SOURCE_INDEX.md",))
    with pytest.raises(ValueError, match="exact D-016 roles"):
        load_d016_authority_profile(extra)

    unknown = tmp_path / "unknown.toml"
    _write_profile(unknown)
    unknown.write_text(unknown.read_text() + '\nunexpected = "value"\n')
    with pytest.raises(ValueError):
        load_d016_authority_profile(unknown)


def test_private_profile_requires_expected_parent_id(tmp_path: Path) -> None:
    no_parent = tmp_path / "no_parent.toml"
    _write_profile(no_parent)
    content = no_parent.read_text().replace('expected_parent_id = "private-parent"\n', "", 1)
    no_parent.write_text(content)
    with pytest.raises(ValueError, match="parent"):
        load_d016_authority_profile(no_parent)


def test_private_profile_requires_shared_parent_id_across_required_root_roles(tmp_path: Path) -> None:
    diff_parent = tmp_path / "diff_parent.toml"
    _write_profile(diff_parent)
    content = diff_parent.read_text().replace(
        'expected_parent_id = "private-parent"', 'expected_parent_id = "different-parent"', 1
    )
    diff_parent.write_text(content)
    with pytest.raises(ValueError, match="parent"):
        load_d016_authority_profile(diff_parent)


def test_private_profile_accepts_valid_memberships(tmp_path: Path) -> None:
    with_memberships = tmp_path / "memberships.toml"
    _write_profile(with_memberships)
    extra_content = (
        '\n[[profile.memberships]]\n'
        'parent_id = "private-parent"\n'
        'expected_child_ids = ["private-object-1", "private-object-2", "private-object-3", "private-object-4"]\n'
    )
    with_memberships.write_text(with_memberships.read_text() + extra_content)
    profile = load_d016_authority_profile(with_memberships)
    assert len(profile.memberships) == 1
    assert profile.memberships[0].parent_id == "private-parent"



def test_counterfactual_authoritative_contract_fails_stale_executable_until_reconciled() -> None:
    reconciled = current_executable_policy_contract()
    stale_flags = [
        {"complete_requires_authoritative_github_evidence": False},
        {"complete_requires_merged_pull_request": False},
        {"complete_requires_post_merge_main_push_ci": False},
        {"complete_requires_exact_workflow_identity": False},
        {"complete_requires_exact_required_job_identity": False},
        {"complete_requires_readback": False},
    ]
    for flag in stale_flags:
        stale_authoritative = reconciled.model_copy(update=flag)
        with pytest.raises(ValueError, match="stale"):
            verify_executable_policy_contract(stale_authoritative)

    verify_executable_policy_contract(reconciled)


def test_public_change_yields_required_contract_and_matching_identity() -> None:
    contract = current_executable_policy_contract()
    result = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(
            impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
            public_contract=contract,
        ),
    )
    assert result.disposition.disposition is ReconciliationDisposition.REQUIRED
    assert result.public_projection == contract
    assert result.publication_tree == approved_policy_tree(contract)
    assert result.disposition.public_contract_sha256 == canonical_record_hash(contract)
    assert result.disposition.publication_tree_sha256 == result.publication_tree.sha256


def test_private_only_change_produces_none_and_no_public_artifact() -> None:
    result = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(impact=ProjectImpact.NO_PUBLIC_IMPACT),
    )
    assert result.disposition.disposition is ReconciliationDisposition.NONE
    assert result.public_projection is None
    assert result.publication_tree is None


def test_unknown_or_unsafe_change_blocks_without_public_artifact() -> None:
    result = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(impact=ProjectImpact.UNKNOWN_OR_UNSAFE),
    )
    assert result.disposition.disposition is ReconciliationDisposition.BLOCKED
    assert result.public_projection is None
    assert result.publication_tree is None


def test_public_contract_is_exactly_the_four_authorized_reconciliation_states() -> None:
    contract = current_executable_policy_contract()
    assert contract.dispositions == ("NONE", "REQUIRED", "BLOCKED", "COMPLETE")
    assert contract.complete_requires_snapshot_match is True
    assert contract.complete_requires_contract_match is True
    assert contract.complete_requires_readback is True
    assert contract.complete_requires_authoritative_github_evidence is True
    assert contract.complete_requires_merged_pull_request is True
    assert contract.complete_requires_post_merge_main_push_ci is True
    assert contract.complete_requires_exact_workflow_identity is True
    assert contract.complete_requires_exact_required_job_identity is True


def test_public_policy_fixture_is_canonical_and_matches_executable_contract() -> None:
    payload = Path("fixtures/reconciliation_policy_contract.json").read_bytes()
    contract = ReconciliationPolicyContract.model_validate_json(payload, strict=True)
    assert canonical_record_bytes(contract) == payload
    verify_executable_policy_contract(contract)


@pytest.mark.parametrize(
    "disposition,impact",
    [
        (ReconciliationDisposition.REQUIRED, ProjectImpact.PUBLIC_CHANGE_REQUIRED),
        (ReconciliationDisposition.BLOCKED, ProjectImpact.UNKNOWN_OR_UNSAFE),
    ],
)
def test_disposition_rejects_unchanged_snapshots(
    disposition: ReconciliationDisposition, impact: ProjectImpact
) -> None:
    extra = {}
    if disposition is ReconciliationDisposition.REQUIRED:
        extra = {"public_contract_sha256": H1, "publication_tree_sha256": H2}
    with pytest.raises(ValidationError, match="requires a changed snapshot"):
        ReconciliationDispositionRecord(
            current_snapshot_sha256=H1,
            previous_snapshot_sha256=H1,
            impact=impact,
            disposition=disposition,
            **extra,
        )


@pytest.mark.parametrize(
    "disposition,impact",
    [
        (ReconciliationDisposition.NONE, ProjectImpact.NO_PUBLIC_IMPACT),
        (ReconciliationDisposition.BLOCKED, ProjectImpact.UNKNOWN_OR_UNSAFE),
    ],
)
def test_disposition_rejects_stray_publication_hashes(
    disposition: ReconciliationDisposition, impact: ProjectImpact
) -> None:
    with pytest.raises(ValidationError, match="must not declare publication hashes"):
        ReconciliationDispositionRecord(
            current_snapshot_sha256=H2,
            previous_snapshot_sha256=H1,
            impact=impact,
            disposition=disposition,
            public_contract_sha256=H1,
            publication_tree_sha256=H2,
        )


def test_repository_git_tree_contains_only_supported_blob_modes() -> None:
    import subprocess
    from pathlib import Path

    if not Path(".git").exists():
        pytest.skip("Not inside a git repository")

    proc = subprocess.run(
        ["git", "ls-files", "-s"],
        check=True,
        capture_output=True,
        text=True,
    )
    unsupported_entries: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        mode = line.split(" ", 1)[0]
        if mode != "100644":
            unsupported_entries.append(line)
    assert not unsupported_entries, (
        f"Git index contains unsupported entry modes for GitPublisher (must be 100644): {unsupported_entries}"
    )

    tree_proc = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    unsupported_tree_entries: list[str] = []
    for line in tree_proc.stdout.splitlines():
        if not line.strip():
            continue
        mode = line.split(" ", 1)[0]
        if mode != "100644":
            unsupported_tree_entries.append(line)
    assert not unsupported_tree_entries, (
        f"Git tree at HEAD contains unsupported entry modes for GitPublisher (must be 100644): {unsupported_tree_entries}"
    )


MERGED_SHA = "c" * 40
HEAD_SHA = "d" * 40


def _make_valid_complete_disposition(
    *,
    snapshot_sha: str = H1,
    contract_sha: str = H2,
    tree_sha: str = H2,
    merged_sha: str = MERGED_SHA,
) -> ReconciliationDispositionRecord:
    git_receipt = GitPublicationReceiptRecord(
        remote="https://github.com/bio-comp/llm-evaluation",
        remote_ref="refs/heads/main",
        expected_base_commit=merged_sha,
        resulting_commit=merged_sha,
        git_tree_oid="e" * 40,
        publication_tree_sha256=tree_sha,
        verified_remote_commit=merged_sha,
        status="unchanged",
        fast_forward_only=True,
        readback_verified=True,
    )
    pr_record = GitHubPullRequestMergeRecord(
        repository="bio-comp/llm-evaluation",
        pull_request_number=9,
        head_ref="refs/heads/feat/d016-doc",
        head_commit_sha=HEAD_SHA,
        base_ref="refs/heads/main",
        merged_commit_sha=merged_sha,
        merged_at="2026-11-01T03:00:00Z",
    )
    ci_record = GitHubCIVerificationRecord(
        repository="bio-comp/llm-evaluation",
        run_id=12345,
        workflow_name="Public CI",
        workflow_path=".github/workflows/ci.yml",
        event="push",
        head_branch="refs/heads/main",
        commit_sha=merged_sha,
        run_conclusion="success",
        job_id=67890,
        job_name="verify",
        job_conclusion="success",
    )
    receipt = VerifiedPublicationReceiptRecord(
        snapshot_state_sha256=snapshot_sha,
        public_contract_sha256=contract_sha,
        publication_tree_sha256=tree_sha,
        git=git_receipt,
        pull_request=pr_record,
        ci=ci_record,
    )
    return ReconciliationDispositionRecord(
        current_snapshot_sha256=snapshot_sha,
        previous_snapshot_sha256="0" * 64,
        impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
        disposition=ReconciliationDisposition.COMPLETE,
        public_contract_sha256=contract_sha,
        publication_tree_sha256=tree_sha,
        verified_publication=receipt,
    )


def test_durable_reconciliation_state_record_valid_construction() -> None:
    comp = _make_valid_complete_disposition()
    state = LastReconciledStateRecord(snapshot_state_sha256=comp.current_snapshot_sha256)
    record = DurableReconciliationStateRecord(state=state, disposition=comp)
    assert record.state.snapshot_state_sha256 == H1
    assert record.disposition.disposition is ReconciliationDisposition.COMPLETE
    assert record.disposition.verified_publication is not None


def test_durable_reconciliation_state_record_rejects_non_complete_disposition() -> None:
    req = ReconciliationDispositionRecord(
        current_snapshot_sha256=H1,
        previous_snapshot_sha256="0" * 64,
        impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
        disposition=ReconciliationDisposition.REQUIRED,
        public_contract_sha256=H2,
        publication_tree_sha256=H2,
    )
    state = LastReconciledStateRecord(snapshot_state_sha256=H1)
    with pytest.raises(ValidationError, match="requires disposition COMPLETE"):
        DurableReconciliationStateRecord(state=state, disposition=req)


def test_durable_reconciliation_state_record_rejects_mismatched_snapshot_hash() -> None:
    comp = _make_valid_complete_disposition(snapshot_sha=H1)
    state = LastReconciledStateRecord(snapshot_state_sha256=H2)
    with pytest.raises(ValidationError, match="snapshot hash must equal disposition"):
        DurableReconciliationStateRecord(state=state, disposition=comp)


def test_resolve_reconciliation_state_path_default_and_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(RECONCILIATION_STATE_ENV_VAR, raising=False)
    default_path = resolve_reconciliation_state_path()
    assert default_path == DEFAULT_RECONCILIATION_STATE_PATH.expanduser()

    custom = tmp_path / "custom.json"
    assert resolve_reconciliation_state_path(custom) == custom

    env_target = tmp_path / "env-target.json"
    monkeypatch.setenv(RECONCILIATION_STATE_ENV_VAR, str(env_target))
    assert resolve_reconciliation_state_path() == env_target
    # explicit argument takes precedence over env var
    assert resolve_reconciliation_state_path(custom) == custom


def test_save_and_load_durable_reconciliation_state_roundtrip(tmp_path: Path) -> None:
    store_path = tmp_path / "state" / "reconciled.json"
    comp = _make_valid_complete_disposition()
    state = LastReconciledStateRecord(snapshot_state_sha256=comp.current_snapshot_sha256)
    record = DurableReconciliationStateRecord(state=state, disposition=comp)

    saved_path = save_durable_reconciliation_state(record, path=store_path)
    assert saved_path == store_path
    assert store_path.exists()

    loaded = load_durable_reconciliation_state(store_path)
    assert loaded is not None
    assert loaded == record
    assert canonical_record_hash(loaded) == canonical_record_hash(record)

    loaded_last = load_last_reconciled_state(store_path)
    assert loaded_last == state


def test_save_durable_reconciliation_state_permissions_and_atomicity(tmp_path: Path) -> None:
    store_path = tmp_path / "secure_dir" / "last-reconciled-state.json"
    comp = _make_valid_complete_disposition()
    state = LastReconciledStateRecord(snapshot_state_sha256=comp.current_snapshot_sha256)
    record = DurableReconciliationStateRecord(state=state, disposition=comp)

    save_durable_reconciliation_state(record, path=store_path)
    # Check permissions (owner-only)
    file_mode = store_path.stat().st_mode & 0o777
    assert file_mode == 0o600
    dir_mode = store_path.parent.stat().st_mode & 0o777
    assert dir_mode == 0o700


def test_load_durable_reconciliation_state_missing_file_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.json"
    assert load_durable_reconciliation_state(missing) is None
    assert load_last_reconciled_state(missing) is None


def test_load_durable_reconciliation_state_empty_or_malformed_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(ReconciliationStateError, match="empty"):
        load_durable_reconciliation_state(empty)

    malformed = tmp_path / "bad.json"
    malformed.write_text('{"record_type": "llm_evaluation_durable_reconciliation_state"')
    with pytest.raises(ReconciliationStateError, match="parse"):
        load_durable_reconciliation_state(malformed)

    wrong_schema = tmp_path / "wrong.json"
    wrong_schema.write_text('{"record_type": "wrong_type", "schema_version": 1}')
    with pytest.raises(ReconciliationStateError, match="parse"):
        load_durable_reconciliation_state(wrong_schema)


def test_durable_reconciliation_state_rejects_symlink_target_and_parent(tmp_path: Path) -> None:
    real_file = tmp_path / "real.json"
    comp = _make_valid_complete_disposition()
    state = LastReconciledStateRecord(snapshot_state_sha256=comp.current_snapshot_sha256)
    record = DurableReconciliationStateRecord(state=state, disposition=comp)
    save_durable_reconciliation_state(record, path=real_file)

    symlink_file = tmp_path / "symlink.json"
    symlink_file.symlink_to(real_file)

    with pytest.raises(ReconciliationStateError, match="symlink"):
        load_durable_reconciliation_state(symlink_file)

    with pytest.raises(ReconciliationStateError, match="symlink"):
        save_durable_reconciliation_state(record, path=symlink_file)

    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    symlink_dir = tmp_path / "symlink_dir"
    symlink_dir.symlink_to(real_dir)

    target_under_symlink_dir = symlink_dir / "state.json"
    with pytest.raises(ReconciliationStateError, match="symlink"):
        save_durable_reconciliation_state(record, path=target_under_symlink_dir)


def test_save_durable_reconciliation_state_readback_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "tampered.json"
    comp = _make_valid_complete_disposition()
    state = LastReconciledStateRecord(snapshot_state_sha256=comp.current_snapshot_sha256)
    record = DurableReconciliationStateRecord(state=state, disposition=comp)

    # Simulate readback byte corruption
    original_read_bytes = Path.read_bytes
    tampered_bytes = b'{"corrupted": true}'

    def tampered_read_bytes(self: Path) -> bytes:
        if self == target.resolve():
            return tampered_bytes
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", tampered_read_bytes)

    with pytest.raises(ReconciliationStateError, match="readback verification failed"):
        save_durable_reconciliation_state(record, path=target)


def test_persist_completed_reconciliation_success_and_rejections(tmp_path: Path) -> None:
    target = tmp_path / "persisted.json"
    comp = _make_valid_complete_disposition()

    durable = persist_completed_reconciliation(comp, path=target)
    assert durable.state.snapshot_state_sha256 == comp.current_snapshot_sha256
    assert durable.disposition == comp

    reloaded = load_last_reconciled_state(target)
    assert reloaded == durable.state

    # Rejection of non-COMPLETE disposition
    req = ReconciliationDispositionRecord(
        current_snapshot_sha256=H1,
        previous_snapshot_sha256="0" * 64,
        impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
        disposition=ReconciliationDisposition.REQUIRED,
        public_contract_sha256=H2,
        publication_tree_sha256=H2,
    )
    with pytest.raises(ValueError, match="only a COMPLETE disposition can be durably persisted"):
        persist_completed_reconciliation(req, path=target)

