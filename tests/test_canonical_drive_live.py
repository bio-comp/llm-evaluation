from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from canonical_sync.acquisition import acquire_stable_candidate
from canonical_sync.errors import AuthorityMismatchError, CanonicalSyncError
from canonical_sync.google_drive import GoogleDriveTransport
from canonical_sync.store import SnapshotStore
from canonical_sync.workflow import acquire_and_publish
from llm_evaluation.canonical_authority import load_d016_authority_profile
from llm_evaluation.canonical_live import live_drive_config_from_environment


def test_live_drive_configuration_is_opt_in_and_requires_private_profile() -> None:
    assert live_drive_config_from_environment({}) is None
    assert live_drive_config_from_environment({"LLM_EVALUATION_LIVE_DRIVE": "1"}) is None
    config = live_drive_config_from_environment(
        {
            "LLM_EVALUATION_LIVE_DRIVE": "1",
            "LLM_EVALUATION_DRIVE_PROFILE": "/private/config/authority.toml",
        }
    )
    assert config is not None
    assert config.profile_path == Path("/private/config/authority.toml")


@pytest.mark.live_drive
def test_registered_d016_authority_live(tmp_path: Path) -> None:
    import os

    config = live_drive_config_from_environment(os.environ)
    if config is None:
        pytest.skip("live Drive acquisition requires explicit opt-in and a private profile path")

    failed_boundary = False
    try:
        profile = load_d016_authority_profile(config.profile_path)
        transport = GoogleDriveTransport.from_application_default()
        store = SnapshotStore(tmp_path / "private-snapshot-store")
        run = acquire_and_publish(transport, profile, store, max_movement_attempts=3)
        verified = store.verify_completed(run.publication.observation_sha256)
        assert verified.state == run.candidate.state
        assert verified.observation == run.candidate.observation
        assert store.verify_receipt(run.receipt_sha256) == run.receipt
        assert run.receipt.status == "success"

        first = profile.objects[0]
        mismatched = replace(first, expected_name=f"{first.expected_name}.mismatch")
        bad_profile = replace(profile, objects=(mismatched, *profile.objects[1:]))
        with pytest.raises(AuthorityMismatchError):
            acquire_stable_candidate(transport, bad_profile, max_movement_attempts=1)
    except (CanonicalSyncError, ValueError):
        failed_boundary = True

    if failed_boundary:
        pytest.fail("live Drive acquisition failed at a sanitized canonical-sync boundary")


def test_live_drive_adc_failure_sanitizes_exception_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import traceback
    import unittest.mock as mock
    from google.auth.exceptions import DefaultCredentialsError
    from canonical_sync.errors import TransportAuthError

    sentinel = "PRIVATE_SENTINEL_PATH_/Users/secret/creds.json"
    with mock.patch("google.auth.default", side_effect=DefaultCredentialsError(sentinel)):
        with pytest.raises(TransportAuthError) as exc_info:
            GoogleDriveTransport.from_application_default()
        assert exc_info.value.__cause__ is None
        tb = "".join(traceback.format_exception(exc_info.value))
        assert sentinel not in tb

    # Also assert failure in test_registered_d016_authority_live does not chain sentinel
    monkeypatch.setenv("LLM_EVALUATION_LIVE_DRIVE", "1")
    monkeypatch.setenv("LLM_EVALUATION_DRIVE_PROFILE", "/private/config/authority.toml")
    monkeypatch.setattr(
        "llm_evaluation.canonical_authority.load_d016_authority_profile",
        lambda _path: mock.MagicMock(),
    )
    with mock.patch("google.auth.default", side_effect=DefaultCredentialsError(sentinel)):
        try:
            test_registered_d016_authority_live(tmp_path)
        except pytest.fail.Exception as fail_exc:
            fail_tb = "".join(traceback.format_exception(fail_exc))
            assert sentinel not in fail_tb
            assert fail_exc.__cause__ is None
            assert fail_exc.__context__ is None
        else:
            pytest.fail("expected live test to fail at sanitized boundary")


def test_live_drive_profile_loading_failure_sanitizes_exception_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import traceback

    sentinel_path = "/private/secret/path/to/profile.toml"
    monkeypatch.setenv("LLM_EVALUATION_LIVE_DRIVE", "1")
    monkeypatch.setenv("LLM_EVALUATION_DRIVE_PROFILE", sentinel_path)
    try:
        test_registered_d016_authority_live(tmp_path)
    except pytest.fail.Exception as fail_exc:
        fail_tb = "".join(traceback.format_exception(fail_exc))
        assert sentinel_path not in fail_tb
        assert fail_exc.__cause__ is None
        assert fail_exc.__context__ is None
    else:
        pytest.fail("expected live test to fail at sanitized boundary")
