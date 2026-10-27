from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ConfigDict

from canonical_sync.errors import PublicationBoundaryError
from canonical_sync.publication import PublicProjectionRecord, publish_public_projection
from canonical_sync.records import SemanticEntryRecord, SnapshotStateRecord

H1 = "1" * 64


class DemoProjection(PublicProjectionRecord):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    projection_type: Literal["demo"] = "demo"
    schema_version: Literal[1] = 1
    dispositions: tuple[str, ...]
    note: str = ""


def projection(*, note: str = "") -> DemoProjection:
    return DemoProjection(dispositions=("NONE", "REQUIRED", "BLOCKED", "COMPLETE"), note=note)


def test_narrow_public_projection_is_atomically_written_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "public"
    root.mkdir()
    first = publish_public_projection(
        projection(), public_root=root, relative_path="contracts/policy.json",
        source_snapshot_sha256=H1,
    )
    second = publish_public_projection(
        projection(), public_root=root, relative_path="contracts/policy.json",
        source_snapshot_sha256=H1,
    )
    payload = (root / "contracts" / "policy.json").read_bytes()
    assert first.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert first == second


def test_private_markers_raw_body_and_credentials_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "public"
    root.mkdir()
    cases = (
        (projection(note="private-object-id-123"), ("private-object-id-123",), ()),
        (projection(note="PRIVATE CANONICAL BODY"), (), (b"PRIVATE CANONICAL BODY",)),
        (projection(note="ya29.secret_token_value"), (), ()),
        (projection(note="-----BEGIN PRIVATE KEY-----"), (), ()),
        (projection(note='{\"client_secret\":\"secret\"}'), (), ()),
    )
    for index, (value, markers, bodies) in enumerate(cases):
        with pytest.raises(PublicationBoundaryError):
            publish_public_projection(
                value, public_root=root, relative_path=f"contracts/{index}.json",
                source_snapshot_sha256=H1, private_markers=markers, private_bodies=bodies,
            )


def test_raw_snapshot_record_cannot_cross_public_projection_type_boundary(tmp_path: Path) -> None:
    raw = SnapshotStateRecord(
        authority_profile_sha256=H1,
        entries=(SemanticEntryRecord(logical_path="A.md", representation_schema="x", content_sha256="2" * 64),),
    )
    root = tmp_path / "public"
    root.mkdir()
    with pytest.raises(TypeError, match="PublicProjectionRecord"):
        publish_public_projection(
            raw, public_root=root, relative_path="contracts/raw.json", source_snapshot_sha256=H1,
        )


def test_traversal_and_symlink_targets_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "public"
    root.mkdir()
    with pytest.raises(PublicationBoundaryError):
        publish_public_projection(
            projection(), public_root=root, relative_path="../escape.json", source_snapshot_sha256=H1,
        )
    outside = tmp_path / "outside.json"
    outside.write_text("outside")
    target = root / "policy.json"
    target.symlink_to(outside)
    with pytest.raises(PublicationBoundaryError, match="symlink"):
        publish_public_projection(
            projection(), public_root=root, relative_path="policy.json", source_snapshot_sha256=H1,
        )
    assert outside.read_text() == "outside"


def test_existing_different_target_requires_compare_and_swap_hash(tmp_path: Path) -> None:
    root = tmp_path / "public"
    root.mkdir()
    target = root / "policy.json"
    target.write_bytes(b"old")
    with pytest.raises(PublicationBoundaryError, match="existing"):
        publish_public_projection(
            projection(), public_root=root, relative_path="policy.json", source_snapshot_sha256=H1,
        )
    old_hash = hashlib.sha256(b"old").hexdigest()
    receipt = publish_public_projection(
        projection(), public_root=root, relative_path="policy.json", source_snapshot_sha256=H1,
        expected_existing_sha256=old_hash,
    )
    assert hashlib.sha256(target.read_bytes()).hexdigest() == receipt.artifact_sha256
