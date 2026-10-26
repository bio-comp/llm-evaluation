from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .contracts import (
    LastReconciledState,
    PublicImpact,
    ReconciliationObligation,
    ReconciliationStatus,
    SemanticState,
    VerifiedPublicationReceipt,
)


class PublicationBoundaryError(ValueError):
    pass


class RemoteRaceError(RuntimeError):
    pass


_SECRET_PATTERNS = (
    re.compile(rb"ya29\.[0-9A-Za-z._-]+"),
    re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(
        rb'(?i)["\'](?:refresh_token|client_secret|access_token)["\']\s*:\s*["\'][^"\']+["\']'
    ),
)


def classify_reconciliation(
    semantic: SemanticState,
    last: LastReconciledState | None,
    public_impact: PublicImpact,
) -> ReconciliationObligation:
    current_identity = (
        semantic.logical_role,
        semantic.projection_schema,
        semantic.sha256,
    )
    previous_identity = (
        None
        if last is None
        else (last.logical_role, last.projection_schema, last.semantic_sha256)
    )
    if previous_identity == current_identity:
        status = ReconciliationStatus.NONE
    elif public_impact is PublicImpact.NONE:
        status = ReconciliationStatus.NONE
    elif public_impact is PublicImpact.REQUIRED:
        status = ReconciliationStatus.REQUIRED
    else:
        status = ReconciliationStatus.BLOCKED
    return ReconciliationObligation(
        status=status,
        logical_role=semantic.logical_role,
        projection_schema=semantic.projection_schema,
        semantic_sha256=semantic.sha256,
        previous_logical_role=None if last is None else last.logical_role,
        previous_projection_schema=None if last is None else last.projection_schema,
        previous_semantic_sha256=None if last is None else last.semantic_sha256,
        public_impact=public_impact,
    )


def complete_reconciliation(
    obligation: ReconciliationObligation,
    receipt: VerifiedPublicationReceipt,
) -> ReconciliationObligation:
    if obligation.status is not ReconciliationStatus.REQUIRED:
        raise ValueError("only a REQUIRED obligation can become COMPLETE")
    obligation_identity = (
        obligation.logical_role,
        obligation.projection_schema,
        obligation.semantic_sha256,
    )
    receipt_identity = (
        receipt.logical_role,
        receipt.projection_schema,
        receipt.semantic_sha256,
    )
    if receipt_identity != obligation_identity:
        raise ValueError(
            "publication receipt does not bind the required semantic identity"
        )
    if receipt.resulting_commit != receipt.readback_commit:
        raise ValueError("publication commit was not verified by exact readback")
    return obligation.model_copy(update={"status": ReconciliationStatus.COMPLETE})


def filter_public_payload(payload: bytes, *, private_markers: tuple[str, ...]) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("public payload must be bytes")
    for marker in private_markers:
        if marker and marker.encode("utf-8") in payload:
            raise PublicationBoundaryError(
                "derived public payload contains a private acquisition marker"
            )
    if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
        raise PublicationBoundaryError(
            "derived public payload resembles credential material"
        )
    return payload


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_replace_public(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
        _fsync_directory(path.parent)
    finally:
        if staging.exists():
            staging.unlink()


def verify_remote_base(expected_oid: str, observed_oid: str) -> None:
    if expected_oid != observed_oid:
        raise RemoteRaceError("remote Git base moved during reconciliation")


def _checkpoint_bytes(state: LastReconciledState) -> bytes:
    return (
        json.dumps(
            {"schema": "llm-evaluation-last-reconciled-v2", **state.model_dump()},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_last_reconciled_state(path: Path, state: LastReconciledState) -> None:
    if ".private-drive-cache" not in path.parts:
        raise ValueError("last-reconciled state must remain under .private-drive-cache")
    atomic_replace_public(path, _checkpoint_bytes(state))
    if read_last_reconciled_state(path) != state:
        raise RuntimeError("last-reconciled state readback failed")


def read_last_reconciled_state(path: Path) -> LastReconciledState:
    raw = json.loads(path.read_bytes())
    if (
        not isinstance(raw, dict)
        or raw.pop("schema", None) != "llm-evaluation-last-reconciled-v2"
    ):
        raise ValueError("invalid last-reconciled checkpoint schema")
    return LastReconciledState.model_validate(raw)


def public_payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
