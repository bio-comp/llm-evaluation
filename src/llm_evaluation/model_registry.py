"""Machine-local authority for user-approved model artifacts.

The registry identifies exact bytes. It never downloads, converts, requantizes, searches
for substitutes, or treats a runtime's model name as artifact identity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .runtime.codec import parse_json_unique
from .runtime.contracts import ClosedRecord


DEFAULT_MODEL_REGISTRY = Path.home() / ".config" / "llm-runtime" / "model-registry.json"

RuntimeName = Literal["llamacpp", "lm_studio"]
CompatibilityStatus = Literal["qualified", "unqualified", "unsupported"]


class RuntimeCompatibility(ClosedRecord):
    status: CompatibilityStatus
    served_model_id: str | None = None
    runtime_identity: str | None = None
    qualified_on: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def qualification_is_complete(self) -> Self:
        if self.status == "qualified" and not all(
            value is not None and value.strip()
            for value in (
                self.served_model_id,
                self.runtime_identity,
                self.qualified_on,
            )
        ):
            raise ValueError(
                "qualified runtime requires served model, runtime identity and date"
            )
        return self


class ApprovedModel(ClosedRecord):
    logical_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    name: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    format: Literal["gguf"]
    quantization: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    context_capability: int = Field(ge=1)
    date_verified: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    llamacpp: RuntimeCompatibility
    lm_studio: RuntimeCompatibility

    @field_validator("artifact_path")
    @classmethod
    def absolute_artifact_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("approved model artifact_path must be absolute")
        return str(path)


class ApprovedModelRegistry(ClosedRecord):
    schema_version: Literal[1] = 1
    models: tuple[ApprovedModel, ...]

    @field_validator("models", mode="before")
    @classmethod
    def freeze_json_models(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_model_ids(self) -> Self:
        ids = [model.logical_id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate approved model logical_id")
        return self


def load_model_registry(path: Path | str) -> ApprovedModelRegistry:
    registry_path = Path(path).expanduser()
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ValueError("approved model registry must be an ordinary file")
    try:
        payload = parse_json_unique(registry_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("invalid approved model registry") from exc
    return ApprovedModelRegistry.model_validate(payload, strict=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_approved_model(
    registry_path: Path | str,
    logical_id: str,
    *,
    runtime: RuntimeName,
) -> ApprovedModel:
    """Resolve one approved model and independently prove its current local bytes."""

    registry = load_model_registry(registry_path)
    matches = [model for model in registry.models if model.logical_id == logical_id]
    if len(matches) != 1:
        raise KeyError(logical_id)
    model = matches[0]
    compatibility = model.llamacpp if runtime == "llamacpp" else model.lm_studio
    if compatibility.status != "qualified":
        raise ValueError(f"approved model is not qualified for {runtime}")

    artifact = Path(model.artifact_path).expanduser()
    try:
        resolved = artifact.resolve(strict=True)
    except OSError as exc:
        raise ValueError("approved model artifact is unavailable") from exc
    if artifact.is_symlink() or not resolved.is_file():
        raise ValueError("approved model artifact must be an ordinary file")
    if resolved.suffix.lower() != ".gguf" or model.format != "gguf":
        raise ValueError("approved model artifact format mismatch")
    if resolved.stat().st_size != model.size_bytes:
        raise ValueError("approved model artifact size mismatch")
    if _hash_file(resolved) != model.sha256:
        raise ValueError("approved model artifact digest mismatch")
    return model
