from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from canonical_sync.acquisition import acquire_stable_candidate
from canonical_sync.errors import AuthorityMismatchError, CanonicalSyncError
from canonical_sync.google_drive import GoogleDriveDocsTransport
from canonical_sync.store import SnapshotStore
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

    try:
        profile = load_d016_authority_profile(config.profile_path)
        transport = GoogleDriveDocsTransport.from_application_default()
        candidate = acquire_stable_candidate(transport, profile, max_movement_attempts=3)
        store = SnapshotStore(tmp_path / "private-snapshot-store")
        publication = store.publish(store.stage(candidate))
        verified = store.verify_completed(publication.observation_sha256)
        assert verified.state == candidate.state
        assert verified.observation == candidate.observation

        first = profile.objects[0]
        mismatched = replace(first, expected_name=f"{first.expected_name}.mismatch")
        bad_profile = replace(profile, objects=(mismatched, *profile.objects[1:]))
        with pytest.raises(AuthorityMismatchError):
            acquire_stable_candidate(transport, bad_profile, max_movement_attempts=1)
    except CanonicalSyncError:
        pytest.fail("live Drive acquisition failed at a sanitized canonical-sync boundary")
