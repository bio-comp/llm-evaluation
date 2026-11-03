from __future__ import annotations

import tomllib
from pathlib import Path

EXPECTED_COMMIT = "a157898a4c2fd56984afb410d426f913ae7b2f09"
EXPECTED_URL = "git+https://github.com/bio-comp/canonical-sync.git@" + EXPECTED_COMMIT


def test_canonical_sync_is_external_commit_pinned_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert f"canonical-sync @ {EXPECTED_URL}" in dependencies
    assert "tool" not in project or "uv" not in project["tool"] or "workspace" not in project["tool"]["uv"]
    assert not (Path("packages") / "canonical-sync").exists()


def test_consumer_no_longer_owns_adc_helper() -> None:
    assert not Path("scripts/refresh_google_drive_adc.py").exists()
    assert not Path("tests/test_refresh_google_drive_adc.py").exists()
