from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class LiveDriveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    profile_path: Path

    @field_validator("profile_path")
    @classmethod
    def validate_profile_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("live Drive profile path must be absolute")
        return value


def live_drive_config_from_environment(environ: Mapping[str, str]) -> LiveDriveConfig | None:
    if environ.get("LLM_EVALUATION_LIVE_DRIVE") != "1":
        return None
    raw_path = environ.get("LLM_EVALUATION_DRIVE_PROFILE")
    if raw_path is None or not raw_path.strip():
        return None
    return LiveDriveConfig(profile_path=Path(raw_path))
