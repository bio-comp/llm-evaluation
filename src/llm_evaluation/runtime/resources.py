"""Kernel leases for cooperating processes in one configured per-user namespace."""

import errno
import fcntl
import json
import os
import stat
from pathlib import Path
from .codec import sha256_hex


class ResourceBusyError(RuntimeError):
    pass


def canonical_resource(resource: str) -> str:
    if (
        not isinstance(resource, str)
        or not resource.startswith("llama-device:")
        or not resource[13:].strip()
    ):
        raise ValueError("explicit llama physical-device identity required")
    return "llama-device:" + resource[13:].strip().upper()


def _root_fd(root: Path) -> int:
    """Walk using dirfds and no-follow; never mutate through a symlink parent."""
    if not root.is_absolute():
        root = Path.cwd() / root
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            if part in (".", ".."):
                raise ValueError("canonical lease root required")
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise PermissionError("lease root must be owned and not writable by others")
        return fd
    except BaseException:
        os.close(fd)
        raise


class ResourceLease:
    def __init__(self, fd: int, path: Path):
        self.fd = fd
        self.path = path

    def __enter__(self) -> "ResourceLease":
        return self

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def acquire_resource_lease(root: Path | str, resource: str) -> ResourceLease:
    resource = canonical_resource(resource)
    root = Path(root).absolute()
    directory = _root_fd(root)
    name = sha256_hex(resource.encode()) + ".lock"
    fd = -1
    try:
        fd = os.open(
            name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            raise PermissionError("owned regular persistent lease file required")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise ResourceBusyError("resource busy") from None
            raise
        metadata = json.dumps(
            {
                "resource": resource,
                "pid": os.getpid(),
                "metadata_role": "diagnostic_only",
            }
        ).encode()
        os.ftruncate(fd, 0)
        os.write(fd, metadata)
        os.fsync(fd)
        lease = ResourceLease(fd, root / name)
        fd = -1
        return lease
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(directory)
