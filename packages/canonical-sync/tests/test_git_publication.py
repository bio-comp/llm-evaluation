from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from canonical_sync.git_publication import (
    GitPublisher,
    PublicationFile,
    PublicationTree,
)
from canonical_sync.records import GitPublicationReceiptRecord


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def tree() -> PublicationTree:
    return PublicationTree(
        files=(
            PublicationFile(path="contracts/policy.json", data=b'{"policy":"approved"}\n'),
            PublicationFile(path="docs/note.txt", data=b"approved bytes only\n"),
        )
    )


def test_git_publication_accepts_only_caller_supplied_tree_and_verifies_remote(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/canonical-sync-test")

    first = publisher.publish(tree(), expected_base=None, commit_message="publish approved tree")
    assert isinstance(first, GitPublicationReceiptRecord)
    assert first.status == "published"
    assert first.expected_base_commit is None
    assert first.resulting_commit == first.verified_remote_commit
    assert first.fast_forward_only is True
    assert first.readback_verified is True

    second = publisher.publish(
        tree(), expected_base=first.resulting_commit, commit_message="same approved tree"
    )
    assert second.status == "unchanged"
    assert second.resulting_commit == first.resulting_commit


def test_remote_movement_blocks_publication_and_never_forces(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/canonical-sync-test")
    first = publisher.publish(tree(), expected_base=None, commit_message="first")

    with pytest.raises(RuntimeError, match="remote"):
        publisher.publish(
            PublicationTree(files=(PublicationFile(path="contracts/policy.json", data=b"changed\n"),)),
            expected_base="0" * 40,
            commit_message="must not publish",
        )
    assert git(remote, "rev-parse", "refs/heads/canonical-sync-test") == first.resulting_commit


def test_publication_tree_rejects_unsafe_or_duplicate_paths() -> None:
    with pytest.raises(ValueError):
        PublicationTree(files=(PublicationFile(path="../escape", data=b"x"),))
    with pytest.raises(ValueError):
        PublicationTree(
            files=(
                PublicationFile(path="A.txt", data=b"one"),
                PublicationFile(path="A.txt", data=b"two"),
            )
        )


def test_generic_git_layer_does_not_accept_verified_snapshot_or_project_policy(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/canonical-sync-test")
    with pytest.raises(TypeError, match="PublicationTree"):
        publisher.publish(object(), expected_base=None, commit_message="not a tree")  # type: ignore[arg-type]
