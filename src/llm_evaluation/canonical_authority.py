from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import ValidationError

from canonical_sync.domain import AuthorityProfile, profile_from_record
from canonical_sync.records import (
    AuthorityObjectSpecRecord,
    AuthorityProfileRecord,
    ExactMembershipSpecRecord,
)

D016_REQUIRED_ROLES = frozenset(
    {"README.md", "DECISIONS.md", "EXPERIMENTS.md", "REVIEW_LEDGER.md"}
)
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_REPRESENTATION = "google-drive-export-markdown-v1"


def load_d016_authority_profile(path: Path) -> AuthorityProfile:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("D-016 authority profile could not be read as TOML") from exc
    if set(raw) != {"profile"} or not isinstance(raw["profile"], dict):
        raise ValueError("D-016 authority profile must contain only a [profile] table")
    profile_raw = raw["profile"]
    if not (
        {"profile_name", "objects"}.issubset(set(profile_raw))
        and set(profile_raw).issubset({"profile_name", "objects", "memberships"})
    ):
        raise ValueError("D-016 authority profile contains unexpected fields")
    profile_name = profile_raw.get("profile_name")
    objects_raw = profile_raw.get("objects")
    memberships_raw = profile_raw.get("memberships", [])
    if (
        not isinstance(profile_name, str)
        or not isinstance(objects_raw, list)
        or not isinstance(memberships_raw, list)
    ):
        raise ValueError("D-016 authority profile has invalid field types")
    try:
        objects = tuple(
            AuthorityObjectSpecRecord.model_validate(item, strict=True) for item in objects_raw
        )
        memberships = tuple(
            ExactMembershipSpecRecord.model_validate(
                dict(item, expected_child_ids=tuple(item["expected_child_ids"]))
                if isinstance(item, dict) and isinstance(item.get("expected_child_ids"), list)
                else item,
                strict=True,
            )
            for item in memberships_raw
        )
        record = AuthorityProfileRecord(
            profile_name=profile_name,
            objects=objects,
            memberships=memberships,
        )
    except ValidationError as exc:
        raise ValueError("D-016 authority profile failed strict validation") from exc
    roles = {item.logical_path for item in record.objects}
    if roles != D016_REQUIRED_ROLES:
        raise ValueError("D-016 authority profile must contain the exact D-016 roles")
    parents = {item.expected_parent_id for item in record.objects}
    if len(parents) != 1 or None in parents:
        raise ValueError("D-016 authority profile root roles must share one non-null parent identity")
    for item in record.objects:
        if (
            item.provider != "google_drive"
            or item.expected_name != item.logical_path
            or item.expected_mime_type != _GOOGLE_DOC_MIME
            or item.representation_schema != _REPRESENTATION
            or item.expected_parent_id is None
        ):
            raise ValueError("D-016 authority profile contains an unsupported source declaration")
    return profile_from_record(record)
