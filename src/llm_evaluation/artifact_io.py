"""Small immutable file publication primitive; trusted local filesystem assumed."""
import os
import tempfile
from pathlib import Path


def reject_symlink_components(path: Path) -> None:
    for part in (path.absolute(), *path.absolute().parents):
        if part.is_symlink():
            raise ValueError("symbolic links are forbidden in artifact paths")


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_immutable(path: Path, data: bytes) -> None:
    """Fsync staged bytes, link exclusively, fsync directory; no hostile-race claim."""
    reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path)
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".e002-",
                                         delete=False) as handle:
            staging = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path, follow_symlinks=False)
        fsync_directory(path.parent)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
            fsync_directory(path.parent)
