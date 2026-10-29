from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from canonical_sync.errors import (
    GitPublicationError,
    RemoteRaceError,
)
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


def test_publish_race_remote_rewound_to_ancestor_immediately_before_push_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from canonical_sync.errors import RemoteRaceError

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")

    # Commit A
    base_a = publisher.publish(
        PublicationTree((PublicationFile("A.txt", b"a\n"),)),
        expected_base=None,
        commit_message="commit A",
    )
    commit_a = base_a.resulting_commit

    # Commit B (descendant of A)
    base_b = publisher.publish(
        PublicationTree((PublicationFile("A.txt", b"a\n"), PublicationFile("B.txt", b"b\n"))),
        expected_base=commit_a,
        commit_message="commit B",
    )
    commit_b = base_b.resulting_commit

    # Candidate C will be based on B.
    candidate_tree = PublicationTree(
        (PublicationFile("A.txt", b"a\n"), PublicationFile("B.txt", b"b\n"), PublicationFile("C.txt", b"c\n"))
    )

    # Hook publisher._git immediately before push to rewind remote from B back to A
    original_git = publisher._git
    push_intercepted = False

    def hook_git(repo: Path, *args: str, **kwargs: object) -> bytes:
        nonlocal push_intercepted
        if args and args[0] == "push":
            push_intercepted = True
            # Rewind remote ref to ancestor A
            git(remote, "update-ref", "refs/heads/main", commit_a)
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(publisher, "_git", hook_git)

    with pytest.raises(RemoteRaceError):
        publisher.publish(candidate_tree, expected_base=commit_b, commit_message="commit C")

    assert push_intercepted is True
    # Verify remote state remained at commit A and was not overwritten by candidate C
    assert git(remote, "rev-parse", "refs/heads/main") == commit_a


def test_publish_race_remote_deleted_immediately_before_push_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from canonical_sync.errors import RemoteRaceError

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")

    base = publisher.publish(
        PublicationTree((PublicationFile("A.txt", b"a\n"),)),
        expected_base=None,
        commit_message="commit A",
    )
    commit_a = base.resulting_commit

    candidate_tree = PublicationTree(
        (PublicationFile("A.txt", b"a\n"), PublicationFile("B.txt", b"b\n"))
    )

    original_git = publisher._git
    push_intercepted = False

    def hook_git(repo: Path, *args: str, **kwargs: object) -> bytes:
        nonlocal push_intercepted
        if args and args[0] == "push":
            push_intercepted = True
            # Delete remote ref
            git(remote, "update-ref", "-d", "refs/heads/main")
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(publisher, "_git", hook_git)

    with pytest.raises(RemoteRaceError):
        publisher.publish(candidate_tree, expected_base=commit_a, commit_message="commit B")

    assert push_intercepted is True
    # Remote ref must still be deleted, not recreated by candidate push
    assert "refs/heads/main" not in git(remote, "for-each-ref")


def test_publish_race_branch_creation_target_created_immediately_before_push_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from canonical_sync.errors import RemoteRaceError

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")

    # Candidate root commit with expected_base=None
    candidate_tree = PublicationTree((PublicationFile("A.txt", b"a\n"),))

    # Another commit created concurrently on the remote
    competitor_oid = subprocess.run(
        ["git", "-C", str(remote), "hash-object", "-w", "--stdin"],
        input=b"competitor content",
        capture_output=True,
        check=True,
    ).stdout.strip()
    competitor_tree = subprocess.run(
        ["git", "-C", str(remote), "mktree"],
        input=b"100644 blob " + competitor_oid + b"\tCOMPETITOR.txt\n",
        capture_output=True,
        check=True,
    ).stdout.strip().decode()
    competitor_commit = git(
        remote,
        "-c", "user.name=Competitor",
        "-c", "user.email=competitor@example.invalid",
        "commit-tree", competitor_tree, "-m", "competitor commit",
    )

    original_git = publisher._git
    push_intercepted = False

    def hook_git(repo: Path, *args: str, **kwargs: object) -> bytes:
        nonlocal push_intercepted
        if args and args[0] == "push":
            push_intercepted = True
            # Competitor creates the target ref immediately before push
            git(remote, "update-ref", "refs/heads/main", competitor_commit)
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(publisher, "_git", hook_git)

    with pytest.raises(RemoteRaceError):
        publisher.publish(candidate_tree, expected_base=None, commit_message="candidate root")

    assert push_intercepted is True
    # Remote ref must still point to competitor commit, not candidate
    assert git(remote, "rev-parse", "refs/heads/main") == competitor_commit


def test_publish_race_remote_advanced_immediately_before_push_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from canonical_sync.errors import RemoteRaceError

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")

    base = publisher.publish(
        PublicationTree((PublicationFile("A.txt", b"a\n"),)),
        expected_base=None,
        commit_message="commit A",
    )
    commit_a = base.resulting_commit

    # Competitor commit on top of A
    competitor_tree = PublicationTree(
        (PublicationFile("A.txt", b"a\n"), PublicationFile("COMPETITOR.txt", b"concurrent\n"))
    )
    competitor_res = publisher.publish(
        competitor_tree, expected_base=commit_a, commit_message="competitor commit"
    )
    commit_competitor = competitor_res.resulting_commit

    # Candidate tree based on commit A
    candidate_tree = PublicationTree(
        (PublicationFile("A.txt", b"a\n"), PublicationFile("B.txt", b"b\n"))
    )

    # Reset remote back to commit A for our publisher run with expected_base=commit_a
    git(remote, "update-ref", "refs/heads/main", commit_a)

    original_git = publisher._git
    push_intercepted = False

    def hook_git(repo: Path, *args: str, **kwargs: object) -> bytes:
        nonlocal push_intercepted
        if args and args[0] == "push":
            push_intercepted = True
            # Competitor advances remote to commit_competitor immediately before push
            git(remote, "update-ref", "refs/heads/main", commit_competitor)
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(publisher, "_git", hook_git)

    with pytest.raises(RemoteRaceError):
        publisher.publish(candidate_tree, expected_base=commit_a, commit_message="candidate B")

    assert push_intercepted is True
    assert git(remote, "rev-parse", "refs/heads/main") == commit_competitor


def test_publish_race_lease_stale_info_in_stderr_raises_remote_race_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(remote=str(remote), ref="refs/heads/main")
    base = publisher.publish(
        PublicationTree((PublicationFile("A.txt", b"a\n"),)),
        expected_base=None,
        commit_message="base",
    )
    original_git = publisher._git

    def hook_git(repo: Path, *args: str, **kwargs: object) -> bytes:
        if args and args[0] == "push":
            err = subprocess.CalledProcessError(
                1, ["git", "push"], stderr=b" ! [rejected] refs/heads/main -> refs/heads/main (stale info)\n"
            )
            raise GitPublicationError("Git publication command failed") from err
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(publisher, "_git", hook_git)
    with pytest.raises(RemoteRaceError, match="remote ref lease rejected publication push"):
        publisher.publish(
            PublicationTree((PublicationFile("A.txt", b"a\n"), PublicationFile("B.txt", b"b\n"))),
            expected_base=base.resulting_commit,
            commit_message="candidate",
        )


def test_base_tree_case_fold_collision_rejected_before_remote_mutation(tmp_path: Path) -> None:
    from canonical_sync.errors import RemoteVerificationError

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    # Mechanically construct a base tree with case-fold collision: README.md and readme.md
    blob1 = subprocess.run(
        ["git", "-C", str(remote), "hash-object", "-w", "--stdin"],
        input=b"content 1", capture_output=True, check=True
    ).stdout.strip()
    blob2 = subprocess.run(
        ["git", "-C", str(remote), "hash-object", "-w", "--stdin"],
        input=b"content 2", capture_output=True, check=True
    ).stdout.strip()
    mktree_input = (
        b"100644 blob " + blob1 + b"\tREADME.md\0"
        b"100644 blob " + blob2 + b"\treadme.md\0"
    )
    tree_oid = subprocess.run(
        ["git", "-C", str(remote), "mktree", "-z"],
        input=mktree_input, capture_output=True, check=True
    ).stdout.strip().decode()
    base_commit = git(
        remote,
        "-c", "user.name=Seed", "-c", "user.email=seed@example.invalid",
        "commit-tree", tree_oid, "-m", "base with case collision"
    )
    git(remote, "update-ref", "refs/heads/main", base_commit)

    publisher = GitPublisher(str(remote), ref="refs/heads/main")
    with pytest.raises(RemoteVerificationError, match="remote tree contains"):
        publisher.publish(tree(), expected_base=base_commit, commit_message="must reject base collision")
    assert git(remote, "rev-parse", "refs/heads/main") == base_commit


def test_base_tree_nfc_nfd_collision_rejected_before_remote_mutation(tmp_path: Path) -> None:
    import unicodedata
    from canonical_sync.errors import RemoteVerificationError

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    nfc_name = unicodedata.normalize("NFC", "Café.md").encode("utf-8")
    nfd_name = unicodedata.normalize("NFD", "Café.md").encode("utf-8")
    assert nfc_name != nfd_name

    blob1 = subprocess.run(
        ["git", "-C", str(remote), "hash-object", "-w", "--stdin"],
        input=b"nfc", capture_output=True, check=True
    ).stdout.strip()
    blob2 = subprocess.run(
        ["git", "-C", str(remote), "hash-object", "-w", "--stdin"],
        input=b"nfd", capture_output=True, check=True
    ).stdout.strip()
    mktree_input = (
        b"100644 blob " + blob1 + b"\t" + nfc_name + b"\0"
        b"100644 blob " + blob2 + b"\t" + nfd_name + b"\0"
    )
    tree_oid = subprocess.run(
        ["git", "-C", str(remote), "mktree", "-z"],
        input=mktree_input, capture_output=True, check=True
    ).stdout.strip().decode()
    base_commit = git(
        remote,
        "-c", "user.name=Seed", "-c", "user.email=seed@example.invalid",
        "commit-tree", tree_oid, "-m", "base with unicode collision"
    )
    git(remote, "update-ref", "refs/heads/main", base_commit)

    publisher = GitPublisher(str(remote), ref="refs/heads/main")
    with pytest.raises(RemoteVerificationError, match="remote tree contains"):
        publisher.publish(tree(), expected_base=base_commit, commit_message="must reject unicode collision")
    assert git(remote, "rev-parse", "refs/heads/main") == base_commit


def test_base_tree_valid_portable_base_remains_accepted(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    publisher = GitPublisher(str(remote), ref="refs/heads/main")
    base = publisher.publish(
        PublicationTree((
            PublicationFile("src/alpha.txt", b"alpha\n"),
            PublicationFile("src/beta.txt", b"beta\n"),
        )),
        expected_base=None,
        commit_message="valid portable base",
    )
    result = publisher.publish(
        PublicationTree((PublicationFile("src/gamma.txt", b"gamma\n"),)),
        expected_base=base.resulting_commit,
        commit_message="add gamma",
    )
    assert result.status == "published"
    assert git_bytes(remote, "show", f"{result.resulting_commit}:src/alpha.txt") == b"alpha\n"
    assert git_bytes(remote, "show", f"{result.resulting_commit}:src/gamma.txt") == b"gamma\n"
