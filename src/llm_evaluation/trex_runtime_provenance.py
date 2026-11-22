"""Qualify installed T-Rex runtime bytes against the pinned research wheel.

The expected wheel identity is an *intended release*, not evidence that its
archive was installed. Its package-file fingerprint was independently derived
from the PyPI macOS arm64 CPython 3.11 wheel after verifying the wheel's
SHA-256 against the publisher's release metadata (2026-11-16).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXPECTED_VERSION = "1.0.0"
EXPECTED_WHEEL_SHA256 = (
    "82b645e5bb8b67547fb35f652a83d0a9ca3d801839961f8d07c407e26b161f2e"
)
EXPECTED_SDIST_SHA256 = (
    "61f0a57b275cf4fc2a2f8c690f75a5ce7364951f3f27caa2729a7128991ab538"
)
# SHA-256 over sorted <wheel package path> NUL <SHA-256 bytes> LF, 40 files.
EXPECTED_PACKAGE_PAYLOAD_SHA256 = (
    "bee76cb51af7ef631507cec8bc1556f43f56fe4fe46744558b5a8bc0870874bb"
)
EXPECTED_PACKAGE_FILES = 40
EXPECTED_NATIVE_SHA256 = (
    "d5358ee2750d0cd4ec35a0cd77987df42bba5cd2d3a9252db906a9a3f44e8605"
)


class StrictEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class IntendedRelease(StrictEvidence):
    version: Literal["1.0.0"] = "1.0.0"
    upstream_tag_commit: str = "aa7e5cade4b7129e83417f19c437a811e828e8ac"
    macos_arm64_cp311_wheel_sha256: str = EXPECTED_WHEEL_SHA256
    sdist_sha256: str = EXPECTED_SDIST_SHA256
    wheel_package_payload_sha256: str = EXPECTED_PACKAGE_PAYLOAD_SHA256


class InstalledArtifact(StrictEvidence):
    status: Literal["qualified", "unavailable", "unqualified"]
    installed_version: str | None = None
    observed_package_payload_sha256: str | None = None
    observed_native_sha256: str | None = None
    verified_file_count: int = Field(default=0, ge=0)
    reason: str | None = None


def _payload_digest(files: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _supported_runtime() -> bool:
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and sys.implementation.cache_tag == "cpython-311"
    )


def inspect_installed_trex() -> InstalledArtifact:
    """Only the exact verified CPython 3.11 macOS-arm64 package is supported.

    Do not infer an installed wheel archive's SHA from distribution metadata:
    installed files do not contain the original ZIP bytes. Compare the exact
    40 executable/source package files against a verified wheel payload instead.
    """
    try:
        distribution = importlib.metadata.distribution("trexselector")
    except importlib.metadata.PackageNotFoundError:
        return InstalledArtifact(
            status="unavailable", reason="trexselector not installed"
        )
    version = distribution.version
    if version != EXPECTED_VERSION:
        return InstalledArtifact(
            status="unqualified",
            installed_version=version,
            reason="installed trexselector version differs from pinned 1.0.0",
        )
    if not _supported_runtime():
        return InstalledArtifact(
            status="unqualified",
            installed_version=version,
            reason="installed runtime is not the qualified macOS arm64 CPython 3.11 wheel",
        )
    try:
        entries = distribution.files
        if entries is None:
            raise ValueError("installed distribution has no file inventory")
        paths = sorted(
            (str(entry), Path(str(distribution.locate_file(entry))))
            for entry in entries
            if str(entry).startswith("trexselector/")
        )
        if len(paths) != EXPECTED_PACKAGE_FILES or len(
            {name for name, _ in paths}
        ) != len(paths):
            raise ValueError(
                "installed package file inventory differs from qualified wheel"
            )
        package_root = Path(str(distribution.locate_file("trexselector"))).resolve()
        init_path = package_root / "__init__.py"
        spec = importlib.util.find_spec("trexselector")
        if (
            spec is None
            or spec.origin is None
            or Path(spec.origin).resolve() != init_path
        ):
            raise ValueError("import resolution differs from installed package")
        present = {
            "trexselector/" + str(path.relative_to(package_root))
            for path in package_root.rglob("*")
            if path.is_file()
            and path.suffix not in (".pyc", ".pyo")
            and "__pycache__" not in path.parts
        }
        if present != {name for name, _ in paths}:
            raise ValueError("installed package contains unexpected or missing files")
        observed = _payload_digest(paths)
        native = next(
            (
                hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in paths
                if name == "trexselector/_core.cpython-311-darwin.so"
            ),
            None,
        )
        if (
            native != EXPECTED_NATIVE_SHA256
            or observed != EXPECTED_PACKAGE_PAYLOAD_SHA256
        ):
            return InstalledArtifact(
                status="unqualified",
                installed_version=version,
                observed_package_payload_sha256=observed,
                observed_native_sha256=native,
                verified_file_count=len(paths),
                reason="installed package contents differ from verified release wheel payload",
            )
        return InstalledArtifact(
            status="qualified",
            installed_version=version,
            observed_package_payload_sha256=observed,
            observed_native_sha256=native,
            verified_file_count=len(paths),
        )
    except (OSError, ValueError):
        return InstalledArtifact(
            status="unqualified",
            installed_version=version,
            reason="installed package identity could not be verified",
        )
