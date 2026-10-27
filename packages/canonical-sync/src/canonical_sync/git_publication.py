from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath, Path

from .errors import GitPublicationError, RemoteRaceError, RemoteVerificationError
from .records import GitPublicationReceiptRecord


def _safe_git_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("publication path must be a non-empty POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value:
        raise ValueError("publication path must be canonical and relative")
    if any(part in {"", ".", ".."} or part.casefold() == ".git" for part in pure.parts):
        raise ValueError("publication path contains traversal or reserved components")
    return value


@dataclass(frozen=True, slots=True)
class PublicationFile:
    path: str
    data: bytes

    def __post_init__(self) -> None:
        _safe_git_path(self.path)
        if type(self.data) is not bytes:
            raise TypeError("publication file data must be bytes")


@dataclass(frozen=True, slots=True)
class PublicationTree:
    """Caller-approved file projection applied mechanically to the verified base tree.

    Paths omitted from this projection are preserved from the expected base; omission is
    never implicit authorization to delete an unrelated repository path.
    """

    files: tuple[PublicationFile, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("publication tree requires at least one file")
        ordered = tuple(sorted(self.files, key=lambda item: item.path))
        if len({item.path for item in ordered}) != len(ordered):
            raise ValueError("publication tree paths must be unique")
        object.__setattr__(self, "files", ordered)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256(b"canonical-sync-publication-tree-v1\0")
        for item in self.files:
            path_bytes = item.path.encode("utf-8")
            digest.update(len(path_bytes).to_bytes(8, "big"))
            digest.update(path_bytes)
            digest.update(len(item.data).to_bytes(8, "big"))
            digest.update(item.data)
        return digest.hexdigest()

    def as_dict(self) -> dict[str, bytes]:
        return {item.path: item.data for item in self.files}


class GitPublisher:
    def __init__(self, remote: str, *, ref: str) -> None:
        if not remote or remote != remote.strip():
            raise ValueError("Git remote must be non-empty and trimmed")
        if not ref.startswith("refs/heads/"):
            raise ValueError("Git publication ref must be a branch ref")
        self.remote = remote
        self.ref = ref

    @staticmethod
    def _env() -> dict[str, str]:
        env = dict(os.environ)
        for key in tuple(env):
            if key.startswith("GIT_CONFIG_"):
                env.pop(key, None)
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
            "GIT_SHALLOW_FILE",
            "GIT_REPLACE_REF_BASE",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_PROXY_COMMAND",
            "GIT_TEMPLATE_DIR",
        ):
            env.pop(key, None)
        env.update(
            GIT_TERMINAL_PROMPT="0",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_AUTHOR_NAME="canonical-sync",
            GIT_AUTHOR_EMAIL="canonical-sync@example.invalid",
            GIT_COMMITTER_NAME="canonical-sync",
            GIT_COMMITTER_EMAIL="canonical-sync@example.invalid",
        )
        return env

    def _git(self, repo: Path, *args: str, data: bytes | None = None) -> bytes:
        try:
            return subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
                input=data,
                check=True,
                capture_output=True,
                timeout=60,
                env=self._env(),
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitPublicationError("Git publication command failed") from exc

    def _tip(self, repo: Path) -> str | None:
        output = self._git(repo, "ls-remote", "--refs", "origin", self.ref).decode().splitlines()
        if not output:
            return None
        if len(output) != 1:
            raise RemoteVerificationError("remote ref lookup was ambiguous")
        oid, ref = output[0].split("\t", 1)
        if ref != self.ref or len(oid) not in {40, 64} or any(c not in "0123456789abcdef" for c in oid):
            raise RemoteVerificationError("remote ref lookup returned an invalid object ID")
        return oid

    def _write_tree(self, repo: Path, files: dict[str, bytes], prefix: str = "") -> str:
        entries: list[bytes] = []
        directories: set[str] = set()
        for path in sorted(files):
            if not path.startswith(prefix):
                continue
            tail = path[len(prefix) :]
            if "/" in tail:
                directories.add(tail.split("/", 1)[0])
            else:
                blob_oid = self._git(repo, "hash-object", "-w", "--stdin", data=files[path]).strip()
                entries.append(
                    b"100644 blob " + blob_oid + b"\t" + tail.encode("utf-8") + b"\0"
                )
        for name in sorted(directories):
            tree_oid = self._write_tree(repo, files, prefix + name + "/")
            entries.append(
                b"040000 tree "
                + tree_oid.encode("ascii")
                + b"\t"
                + name.encode("utf-8")
                + b"\0"
            )
        return self._git(repo, "mktree", "-z", data=b"".join(entries)).decode().strip()

    def _read_tree(self, repo: Path, commit: str) -> dict[str, bytes]:
        files: dict[str, bytes] = {}
        for raw in self._git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not raw:
                continue
            metadata, name = raw.split(b"\t", 1)
            mode, kind, oid = metadata.split(b" ")
            path = name.decode("utf-8")
            _safe_git_path(path)
            if mode != b"100644" or kind != b"blob" or path in files:
                raise RemoteVerificationError("remote tree contains an unsupported Git entry")
            files[path] = self._git(repo, "cat-file", "blob", oid.decode("ascii"))
        return files

    def publish(
        self,
        tree: PublicationTree,
        *,
        expected_base: str | None,
        commit_message: str,
    ) -> GitPublicationReceiptRecord:
        if not isinstance(tree, PublicationTree):
            raise TypeError("Git publication requires a PublicationTree")
        if not commit_message or commit_message != commit_message.strip() or "\x00" in commit_message:
            raise ValueError("commit message must be non-empty, trimmed, and contain no NUL")
        if expected_base is not None and (
            len(expected_base) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in expected_base)
        ):
            raise ValueError("expected base must be a lowercase Git object ID")

        with tempfile.TemporaryDirectory(prefix="canonical-sync-git-") as temporary:
            repo = Path(temporary)
            self._git(repo, "init", "--bare", "--template=", ".")
            self._git(repo, "remote", "add", "origin", self.remote)
            observed = self._tip(repo)
            if observed != expected_base:
                raise RemoteRaceError("remote ref moved relative to expected publication base")

            approved_projection = tree.as_dict()
            desired_tree = approved_projection
            if observed is not None:
                self._git(repo, "fetch", "--no-tags", "origin", self.ref)
                fetched = self._git(repo, "rev-parse", "FETCH_HEAD").decode().strip()
                if fetched != observed:
                    raise RemoteRaceError("remote ref moved during publication preflight")
                base_tree = self._read_tree(repo, observed)
                desired_tree = base_tree | approved_projection
                if base_tree == desired_tree:
                    if self._tip(repo) != observed:
                        raise RemoteRaceError("remote ref moved during unchanged verification")
                    git_tree = self._git(repo, "rev-parse", observed + "^{tree}").decode().strip()
                    return GitPublicationReceiptRecord(
                        status="unchanged",
                        remote=self.remote,
                        remote_ref=self.ref,
                        expected_base_commit=expected_base,
                        resulting_commit=observed,
                        git_tree_oid=git_tree,
                        publication_tree_sha256=tree.sha256,
                        verified_remote_commit=observed,
                        readback_verified=True,
                        fast_forward_only=True,
                    )

            git_tree = self._write_tree(repo, desired_tree)
            parents: tuple[str, ...] = () if expected_base is None else ("-p", expected_base)
            commit = self._git(
                repo,
                "commit-tree",
                git_tree,
                *parents,
                data=(commit_message + "\n").encode("utf-8"),
            ).decode().strip()
            if self._tip(repo) != expected_base:
                raise RemoteRaceError("remote ref moved before publication push")
            try:
                self._git(repo, "push", "origin", f"{commit}:{self.ref}")
            except GitPublicationError as exc:
                if self._tip(repo) != expected_base:
                    raise RemoteRaceError("remote ref moved during publication push") from exc
                raise
            if self._tip(repo) != commit:
                raise RemoteVerificationError("remote ref does not contain the published commit")
            self._git(repo, "fetch", "--no-tags", "origin", self.ref)
            fetched = self._git(repo, "rev-parse", "FETCH_HEAD").decode().strip()
            if (
                fetched != commit
                or self._read_tree(repo, commit) != desired_tree
                or self._tip(repo) != commit
            ):
                raise RemoteVerificationError("remote publication readback does not match approved tree")
            return GitPublicationReceiptRecord(
                status="published",
                remote=self.remote,
                remote_ref=self.ref,
                expected_base_commit=expected_base,
                resulting_commit=commit,
                git_tree_oid=git_tree,
                publication_tree_sha256=tree.sha256,
                verified_remote_commit=commit,
                readback_verified=True,
                fast_forward_only=True,
            )
