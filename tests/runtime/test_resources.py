import subprocess
import sys
import pytest
from llm_evaluation.runtime.resources import acquire_resource_lease


def contender(root, key="llama-device:MTL0"):
    code = """from llm_evaluation.runtime.resources import acquire_resource_lease,ResourceBusyError
import sys
try:
 with acquire_resource_lease(sys.argv[1],sys.argv[2]): print("acquired")
except ResourceBusyError: print("busy")
"""
    return subprocess.check_output(
        [sys.executable, "-c", code, str(root), key], text=True
    ).strip()


def test_two_processes_same_resource_second_busy_and_other_independent(tmp_path):
    with acquire_resource_lease(tmp_path, "llama-device:MTL0") as lease:
        assert contender(tmp_path) == "busy"
        assert contender(tmp_path, "llama-device:MTL1") == "acquired"
        path = lease.path
    assert path.exists()
    assert contender(tmp_path) == "acquired"


def test_stale_and_malformed_metadata_do_not_override_kernel_lock(tmp_path):
    with acquire_resource_lease(tmp_path, "llama-device:MTL0") as lease:
        lease.path.write_text("not json")
        assert contender(tmp_path) == "busy"
    assert contender(tmp_path) == "acquired"


def test_symlink_root_and_file_fail_without_writes(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        acquire_resource_lease(link, "llama-device:MTL0")
    with acquire_resource_lease(root, "llama-device:MTL0") as lease:
        path = lease.path
    path.unlink()
    target = tmp_path / "target"
    target.write_text("keep")
    path.symlink_to(target)
    with pytest.raises(OSError):
        acquire_resource_lease(root, "llama-device:MTL0")
    assert target.read_text() == "keep"


def test_exec_target_retains_lock_and_exit_releases(tmp_path):
    target = tmp_path / "target.py"
    target.write_text('import time\nprint("ready",flush=True)\ntime.sleep(30)\n')
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_evaluation.runtime.llamacpp.lease_exec",
            str(tmp_path / "leases"),
            "llama-device:MTL0",
            sys.executable,
            str(target),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert p.stdout.readline().strip() == "ready"
        assert contender(tmp_path / "leases") == "busy"
    finally:
        p.terminate()
        p.wait(timeout=5)
    assert contender(tmp_path / "leases") == "acquired"


def test_exec_failure_releases_lock(tmp_path):
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_evaluation.runtime.llamacpp.lease_exec",
            str(tmp_path),
            "llama-device:MTL0",
            "/no/such/executable",
        ],
        capture_output=True,
    )
    assert p.returncode != 0
    assert contender(tmp_path) == "acquired"



def test_launcher_explicitly_sets_inheritance_before_exec(tmp_path, monkeypatch):
    import os
    import llm_evaluation.runtime.llamacpp.lease_exec as launcher

    with acquire_resource_lease(tmp_path / "default", "llama-device:MTL0") as lease:
        assert os.get_inheritable(lease.fd) is False

    original = os.set_inheritable
    captured = []

    def set_inheritable(fd, enabled):
        captured.append(fd)
        return original(fd, enabled)

    class StopExec(Exception):
        pass

    def inspect_exec(executable, argv, env):
        assert captured
        assert os.get_inheritable(captured[0]) is True
        raise StopExec

    monkeypatch.setattr(launcher.os, "set_inheritable", set_inheritable)
    monkeypatch.setattr(launcher.os, "execve", inspect_exec)
    with pytest.raises(StopExec):
        launcher.lease_exec(
            tmp_path / "exec", "llama-device:MTL0",
            sys.executable, [sys.executable], {},
        )
