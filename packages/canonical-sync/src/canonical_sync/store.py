from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .acquisition import AcquiredRepresentation, AcquisitionCandidate
from .codec import canonical_record_bytes, canonical_record_hash, parse_canonical_record
from .errors import SnapshotIntegrityError
from .records import AcquisitionReceiptRecord, SnapshotObservationRecord, SnapshotStateRecord


@dataclass(frozen=True, slots=True)
class StagedAcquisition:
    path: Path
    state_sha256: str
    observation_sha256: str


@dataclass(frozen=True, slots=True)
class PublicationResult:
    state_sha256: str
    observation_sha256: str
    published_state: bool
    published_observation: bool


@dataclass(frozen=True, slots=True)
class VerifiedSnapshot:
    state: SnapshotStateRecord
    observation: SnapshotObservationRecord
    representations: tuple[AcquiredRepresentation, ...]


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(mode):
                raise SnapshotIntegrityError("snapshot store path contains a symlink")
        if current.parent == current:
            break
        current = current.parent


def _require_directory(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise SnapshotIntegrityError("required snapshot-store directory is missing") from exc
    if stat.S_ISLNK(mode):
        raise SnapshotIntegrityError("snapshot-store directory is a symlink")
    if not stat.S_ISDIR(mode):
        raise SnapshotIntegrityError("snapshot-store operational path is not a directory")


def _require_regular(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as exc:
        raise SnapshotIntegrityError("required snapshot file is missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise SnapshotIntegrityError("snapshot entries must be regular files; symlink rejected")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_fsync(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        _reject_symlink_chain(root)
        root.mkdir(parents=True, exist_ok=True)
        _require_directory(root)
        self._staging = root / "staging"
        self._snapshots = root / "snapshots"
        self._observations = root / "observations"
        self._receipts = root / "receipts"
        for directory in (self._staging, self._snapshots, self._observations, self._receipts):
            directory.mkdir(exist_ok=True)
            _require_directory(directory)
        _fsync_dir(root)

    def _validate_operational_dirs(self) -> None:
        _reject_symlink_chain(self.root)
        _require_directory(self.root)
        for directory in (self._staging, self._snapshots, self._observations, self._receipts):
            _require_directory(directory)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._validate_operational_dirs()
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.root / ".lock", flags, 0o600)
        except OSError as exc:
            raise SnapshotIntegrityError("snapshot lock could not be opened safely") from exc
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                raise SnapshotIntegrityError("snapshot lock must be a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _validate_candidate(candidate: AcquisitionCandidate) -> None:
        state_hash = canonical_record_hash(candidate.state)
        if state_hash != candidate.observation.snapshot_state_sha256:
            raise SnapshotIntegrityError("snapshot state hash does not match observation")
        if candidate.state.authority_profile_sha256 != candidate.observation.authority_profile_sha256:
            raise SnapshotIntegrityError("authority profile hash does not match observation")
        state_entries = {entry.logical_path: entry for entry in candidate.state.entries}
        observation_entries = {entry.logical_path: entry for entry in candidate.observation.sources}
        representations = {entry.logical_path: entry for entry in candidate.representations}
        if set(state_entries) != set(observation_entries) or set(state_entries) != set(representations):
            raise SnapshotIntegrityError("snapshot candidate inventories do not match")
        for logical_path, representation in representations.items():
            digest = hashlib.sha256(representation.data).hexdigest()
            if digest != representation.content_sha256:
                raise SnapshotIntegrityError("snapshot representation hash mismatch")
            state_entry = state_entries[logical_path]
            observation_entry = observation_entries[logical_path]
            if (
                digest != state_entry.content_sha256
                or digest != observation_entry.content_sha256
                or representation.representation_schema != state_entry.representation_schema
                or representation.representation_schema != observation_entry.representation_schema
            ):
                raise SnapshotIntegrityError("snapshot representation metadata mismatch")

    def stage(self, candidate: AcquisitionCandidate) -> StagedAcquisition:
        self._validate_operational_dirs()
        self._validate_candidate(candidate)
        stage_root = Path(tempfile.mkdtemp(prefix="acquisition-", dir=self._staging))
        snapshot_root = stage_root / "snapshot"
        observation_root = stage_root / "observation"
        (snapshot_root / "files").mkdir(parents=True)
        observation_root.mkdir()
        _write_fsync(snapshot_root / "state.json", canonical_record_bytes(candidate.state))
        for representation in candidate.representations:
            _write_fsync(snapshot_root / "files" / representation.logical_path, representation.data)
        _write_fsync(
            observation_root / "observation.json",
            canonical_record_bytes(candidate.observation),
        )
        for directory in sorted(
            {path.parent for path in snapshot_root.rglob("*") if path.is_file()},
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_dir(directory)
        _fsync_dir(snapshot_root)
        _fsync_dir(observation_root)
        _fsync_dir(stage_root)
        _fsync_dir(self._staging)
        return StagedAcquisition(
            path=stage_root,
            state_sha256=canonical_record_hash(candidate.state),
            observation_sha256=canonical_record_hash(candidate.observation),
        )

    def _verify_snapshot_dir(
        self, path: Path, expected_hash: str
    ) -> tuple[SnapshotStateRecord, tuple[AcquiredRepresentation, ...]]:
        _require_directory(path)
        state_path = path / "state.json"
        _require_regular(state_path)
        try:
            state = parse_canonical_record(state_path.read_bytes(), SnapshotStateRecord)
        except (OSError, ValueError) as exc:
            raise SnapshotIntegrityError("snapshot state record failed canonical validation") from exc
        if canonical_record_hash(state) != expected_hash:
            raise SnapshotIntegrityError("snapshot state hash mismatch")
        expected_files = {Path("state.json")}
        expected_dirs = {Path("files")}
        representations: list[AcquiredRepresentation] = []
        for entry in state.entries:
            relative = Path("files") / entry.logical_path
            expected_files.add(relative)
            parent = relative.parent
            while parent != Path("."):
                expected_dirs.add(parent)
                if parent == Path("files"):
                    break
                parent = parent.parent
            file_path = path / relative
            _require_regular(file_path)
            data = file_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry.content_sha256:
                raise SnapshotIntegrityError("snapshot content hash mismatch")
            representations.append(
                AcquiredRepresentation(
                    logical_path=entry.logical_path,
                    representation_schema=entry.representation_schema,
                    data=data,
                    content_sha256=digest,
                )
            )
        actual_files: set[Path] = set()
        actual_dirs: set[Path] = set()
        for item in path.rglob("*"):
            relative = item.relative_to(path)
            mode = os.lstat(item).st_mode
            if stat.S_ISLNK(mode):
                raise SnapshotIntegrityError("snapshot entries must be regular files/directories; symlink rejected")
            if stat.S_ISREG(mode):
                actual_files.add(relative)
            elif stat.S_ISDIR(mode):
                actual_dirs.add(relative)
            else:
                raise SnapshotIntegrityError("snapshot inventory contains a non-regular entry")
        if actual_files != expected_files or actual_dirs != expected_dirs:
            raise SnapshotIntegrityError("snapshot inventory is not exact")
        return state, tuple(sorted(representations, key=lambda item: item.logical_path))

    def _verify_observation_dir(self, path: Path, expected_hash: str) -> SnapshotObservationRecord:
        _require_directory(path)
        items = list(path.iterdir())
        if len(items) != 1 or items[0].name != "observation.json":
            raise SnapshotIntegrityError("observation inventory is not exact")
        observation_path = path / "observation.json"
        _require_regular(observation_path)
        try:
            observation = parse_canonical_record(
                observation_path.read_bytes(), SnapshotObservationRecord
            )
        except (OSError, ValueError) as exc:
            raise SnapshotIntegrityError("observation record failed canonical validation") from exc
        if canonical_record_hash(observation) != expected_hash:
            raise SnapshotIntegrityError("observation hash mismatch")
        return observation

    def publish(self, staged: StagedAcquisition) -> PublicationResult:
        self._validate_operational_dirs()
        if staged.path.parent != self._staging:
            raise SnapshotIntegrityError("staged acquisition is outside this store")
        _require_directory(staged.path)
        staged_snapshot = staged.path / "snapshot"
        staged_observation = staged.path / "observation"
        state, _ = self._verify_snapshot_dir(staged_snapshot, staged.state_sha256)
        observation = self._verify_observation_dir(staged_observation, staged.observation_sha256)
        if observation.snapshot_state_sha256 != staged.state_sha256:
            raise SnapshotIntegrityError("staged observation points at the wrong semantic state")
        if observation.authority_profile_sha256 != state.authority_profile_sha256:
            raise SnapshotIntegrityError("staged authority identity is inconsistent")

        published_state = False
        published_observation = False
        with self._locked():
            state_target = self._snapshots / staged.state_sha256
            if state_target.exists():
                self._verify_snapshot_dir(state_target, staged.state_sha256)
                shutil.rmtree(staged_snapshot)
            else:
                os.rename(staged_snapshot, state_target)
                _fsync_dir(self._snapshots)
                published_state = True

            observation_target = self._observations / staged.observation_sha256
            if observation_target.exists():
                self._verify_observation_dir(observation_target, staged.observation_sha256)
                shutil.rmtree(staged_observation)
            else:
                os.rename(staged_observation, observation_target)
                _fsync_dir(self._observations)
                published_observation = True
            staged.path.rmdir()
            _fsync_dir(self._staging)

        self.verify_completed(staged.observation_sha256)
        return PublicationResult(
            state_sha256=staged.state_sha256,
            observation_sha256=staged.observation_sha256,
            published_state=published_state,
            published_observation=published_observation,
        )

    def _verify_receipt_file(
        self, path: Path, expected_hash: str
    ) -> AcquisitionReceiptRecord:
        _require_regular(path)
        try:
            receipt = parse_canonical_record(path.read_bytes(), AcquisitionReceiptRecord)
        except (OSError, ValueError) as exc:
            raise SnapshotIntegrityError("acquisition receipt failed canonical validation") from exc
        if canonical_record_hash(receipt) != expected_hash:
            raise SnapshotIntegrityError("acquisition receipt hash mismatch")
        return receipt

    def persist_receipt(self, receipt: AcquisitionReceiptRecord) -> str:
        self._validate_operational_dirs()
        receipt_hash = canonical_record_hash(receipt)
        target = self._receipts / f"{receipt_hash}.json"
        payload = canonical_record_bytes(receipt)
        with self._locked():
            try:
                os.lstat(target)
            except FileNotFoundError:
                pass
            else:
                self._verify_receipt_file(target, receipt_hash)
                return receipt_hash
            fd, temp_name = tempfile.mkstemp(prefix=".receipt-", suffix=".tmp", dir=self._receipts)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.rename(temp_name, target)
                _fsync_dir(self._receipts)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
        self._verify_receipt_file(target, receipt_hash)
        return receipt_hash

    def verify_receipt(self, receipt_sha256: str) -> AcquisitionReceiptRecord:
        self._validate_operational_dirs()
        if len(receipt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in receipt_sha256
        ):
            raise SnapshotIntegrityError("acquisition receipt identity is malformed")
        return self._verify_receipt_file(
            self._receipts / f"{receipt_sha256}.json", receipt_sha256
        )

    def list_receipt_hashes(self) -> tuple[str, ...]:
        self._validate_operational_dirs()
        hashes: list[str] = []
        for item in self._receipts.iterdir():
            _require_regular(item)
            if not item.name.endswith(".json"):
                raise SnapshotIntegrityError("receipt inventory contains an unexpected file")
            receipt_hash = item.name[:-5]
            self.verify_receipt(receipt_hash)
            hashes.append(receipt_hash)
        return tuple(sorted(hashes))

    def verify_completed(self, observation_sha256: str) -> VerifiedSnapshot:
        self._validate_operational_dirs()
        observation_path = self._observations / observation_sha256
        if not observation_path.exists():
            raise SnapshotIntegrityError("completed observation is missing")
        observation = self._verify_observation_dir(observation_path, observation_sha256)
        snapshot_path = self._snapshots / observation.snapshot_state_sha256
        if not snapshot_path.exists():
            raise SnapshotIntegrityError("semantic snapshot referenced by observation is missing")
        state, representations = self._verify_snapshot_dir(
            snapshot_path, observation.snapshot_state_sha256
        )
        if state.authority_profile_sha256 != observation.authority_profile_sha256:
            raise SnapshotIntegrityError("persisted state and observation authority identities differ")
        state_entries = {entry.logical_path: entry for entry in state.entries}
        observation_entries = {entry.logical_path: entry for entry in observation.sources}
        if set(state_entries) != set(observation_entries):
            raise SnapshotIntegrityError("persisted state and observation inventories differ")
        for logical_path, entry in state_entries.items():
            observed = observation_entries[logical_path]
            if (
                entry.content_sha256 != observed.content_sha256
                or entry.representation_schema != observed.representation_schema
            ):
                raise SnapshotIntegrityError("persisted state and observation content identities differ")
        return VerifiedSnapshot(
            state=state,
            observation=observation,
            representations=representations,
        )
