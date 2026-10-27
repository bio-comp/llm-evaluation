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


def git_bytes(path: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
    ).stdout


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


def test_publication_projection_preserves_unrelated_base_files(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/canonical-sync-test")
    base = publisher.publish(
        PublicationTree(files=(PublicationFile(path="KEEP.txt", data=b"keep byte-for-byte\n"),)),
        expected_base=None,
        commit_message="seed unrelated base file",
    )

    approved = PublicationTree(
        files=(
            PublicationFile(
                path="fixtures/reconciliation_policy_contract.json",
                data=b'{"policy":"approved"}\n',
            ),
        )
    )
    result = publisher.publish(
        approved,
        expected_base=base.resulting_commit,
        commit_message="apply narrow approved projection",
    )

    assert git_bytes(remote, "show", f"{result.resulting_commit}:KEEP.txt") == b"keep byte-for-byte\n"
    assert (
        git_bytes(
            remote,
            "show",
            f"{result.resulting_commit}:fixtures/reconciliation_policy_contract.json",
        )
        == b'{"policy":"approved"}\n'
    )


def test_projection_noop_on_nonempty_base_preserves_commit(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/canonical-sync-test")
    approved_data = b'{"policy":"approved"}\n'
    base = publisher.publish(
        PublicationTree(
            files=(
                PublicationFile(path="KEEP.txt", data=b"unchanged\n"),
                PublicationFile(path="contracts/policy.json", data=approved_data),
            )
        ),
        expected_base=None,
        commit_message="seed nonempty base",
    )

    result = publisher.publish(
        PublicationTree(files=(PublicationFile(path="contracts/policy.json", data=approved_data),)),
        expected_base=base.resulting_commit,
        commit_message="same narrow projection",
    )

    assert result.status == "unchanged"
    assert result.resulting_commit == base.resulting_commit
    assert git_bytes(remote, "show", f"{result.resulting_commit}:KEEP.txt") == b"unchanged\n"


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


def test_environment_url_rewrite_cannot_redirect_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intended = tmp_path / "intended.git"
    victim = tmp_path / "victim.git"
    subprocess.run(["git", "init", "--bare", str(intended)], check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", str(victim)], check=True, capture_output=True)

    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{victim}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(intended))
    publisher = GitPublisher(str(intended), ref="refs/heads/canonical-sync-test")
    publisher.publish(tree(), expected_base=None, commit_message="publish intended remote")

    monkeypatch.delenv("GIT_CONFIG_COUNT")
    monkeypatch.delenv("GIT_CONFIG_KEY_0")
    monkeypatch.delenv("GIT_CONFIG_VALUE_0")
    assert git(intended, "rev-parse", "refs/heads/canonical-sync-test")
    assert git(victim, "for-each-ref") == ""


@pytest.mark.parametrize(('base_path', 'projected'), [('dir/KEEP.txt', 'dir'), ('a', 'a/b')])
def test_path_hierarchy_collision_fails_before_remote_mutation(tmp_path, base_path, projected):
    from canonical_sync.errors import GitPublicationError
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref='refs/heads/main')
    base = publisher.publish(PublicationTree((PublicationFile(base_path, b'keep'),)),
                             expected_base=None, commit_message='base')
    with pytest.raises(GitPublicationError):
        publisher.publish(PublicationTree((PublicationFile(projected, b'replace'),)),
                          expected_base=base.resulting_commit, commit_message='collision')
    assert git(remote, 'rev-parse', 'refs/heads/main') == base.resulting_commit
    assert git_bytes(remote, 'show', f'{base.resulting_commit}:{base_path}') == b'keep'


def test_readback_mismatch_never_commits_or_pushes(tmp_path, monkeypatch):
    from canonical_sync.errors import RemoteVerificationError
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref='refs/heads/main')
    base = publisher.publish(tree(), expected_base=None, commit_message='base')
    commands = []
    original = publisher._git
    def track(repo, *args, **kwargs):
        commands.append(args[0])
        return original(repo, *args, **kwargs)
    monkeypatch.setattr(publisher, '_git', track)
    with pytest.raises(RemoteVerificationError):
        publisher.verify(PublicationTree((PublicationFile('docs/note.txt', b'wrong'),)),
                         expected_commit=base.resulting_commit)
    assert not {'push', 'commit-tree', 'hash-object', 'mktree'} & set(commands)
    assert git(remote, 'rev-parse', 'refs/heads/main') == base.resulting_commit
    assert publisher.verify(tree(), expected_commit=base.resulting_commit).status == 'unchanged'


def test_receipt_remote_drops_transport_secrets(tmp_path):
    from canonical_sync.codec import canonical_record_bytes
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    receipt = GitPublisher(str(remote), ref='refs/heads/main').publish(
        tree(), expected_base=None, commit_message='base')
    raw = receipt.model_dump()
    raw['remote'] = 'https://user:FAKE_TOKEN@github.com/bio-comp/llm-evaluation.git?key=FAKE_QUERY#FAKE_FRAGMENT'
    safe = GitPublicationReceiptRecord.model_validate(raw)
    payload = canonical_record_bytes(safe)
    assert all(x not in payload for x in (b'FAKE_TOKEN', b'FAKE_QUERY', b'FAKE_FRAGMENT'))
    assert str(tmp_path).encode() not in canonical_record_bytes(receipt)


@pytest.mark.parametrize('mode', ['100755', '120000'])
def test_unsupported_base_mode_rejected_before_remote_mutation(tmp_path, mode):
    from canonical_sync.errors import RemoteVerificationError
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    blob = subprocess.run(['git', '-C', str(remote), 'hash-object', '-w', '--stdin'],
                          input=b'preserve', capture_output=True, check=True).stdout.strip()
    oid = subprocess.run(['git', '-C', str(remote), 'mktree'],
                         input=mode.encode() + b' blob ' + blob + b'\tKEEP\n',
                         capture_output=True, check=True).stdout.strip().decode()
    commit = git(remote, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit-tree', oid, '-m', 'mode base')
    git(remote, 'update-ref', 'refs/heads/main', commit)
    before = git_bytes(remote, 'ls-tree', '-r', commit)
    with pytest.raises(RemoteVerificationError):
        GitPublisher(str(remote), ref='refs/heads/main').publish(
            tree(), expected_base=commit, commit_message='must not change mode')
    assert git(remote, 'rev-parse', 'refs/heads/main') == commit
    assert git_bytes(remote, 'ls-tree', '-r', commit) == before
