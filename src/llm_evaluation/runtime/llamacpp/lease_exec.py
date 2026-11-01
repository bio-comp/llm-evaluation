"""Actual exec target inherits the kernel-locked open file description."""

import os
import sys
from pathlib import Path
from ..resources import acquire_resource_lease, ResourceBusyError


def lease_exec(
    root: Path | str,
    resource: str,
    executable: str,
    argv: list[str],
    env: dict[str, str],
) -> None:
    with acquire_resource_lease(root, resource) as lease:
        # Explicit on every launch: supported Python defaults are not the invariant.
        os.set_inheritable(lease.fd, True)
        if not os.get_inheritable(lease.fd):
            raise RuntimeError("lease descriptor not inheritable")
        os.execve(executable, argv, env)


def main() -> None:
    try:
        root, resource, exe, *args = sys.argv[1:]
        lease_exec(root, resource, exe, [exe, *args], dict(os.environ))
    except ResourceBusyError:
        sys.exit(73)
    except (OSError, ValueError):
        # Never print exec exception details or environment.
        sys.exit(74)


if __name__ == "__main__":
    main()
