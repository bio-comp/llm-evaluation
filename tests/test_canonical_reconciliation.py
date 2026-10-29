from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from canonical_sync.codec import canonical_record_bytes, canonical_record_hash
from llm_evaluation.canonical_authority import D016_REQUIRED_ROLES, load_d016_authority_profile
from llm_evaluation.canonical_reconciliation import (
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReconciliationDispositionRecord,
    ReconciliationPolicyContract,
    ReviewedCanonicalImpact,
    approved_policy_tree,
    current_executable_policy_contract,
    reconcile_reviewed_snapshot,
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
