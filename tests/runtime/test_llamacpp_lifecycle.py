import asyncio
import sys
import pytest
from llm_evaluation.runtime.llamacpp.lifecycle import (
    ManagedServerDeclaration,
    ManagedServerMode,
    ReasoningMode,
    build_launch,
    start_managed_server,
)
from llm_evaluation.runtime.resources import ResourceBusyError, acquire_resource_lease
from llm_evaluation.runtime.llamacpp.qualification import hash_model_artifact


def declaration(tmp_path, mode=ManagedServerMode.FORMAL):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return ManagedServerDeclaration(
        executable=sys.executable,
        artifact=str(model),
        alias="test",
        mode=mode,
        reasoning_mode=ReasoningMode.AUTO,
        lease_root=str(tmp_path / "leases"),
        resource="llama-device:MTL0",
    )


def test_formal_profile_secret_and_constructed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "all")
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "999999")
    monkeypatch.setenv("LLAMA_ARG_SLOT_SAVE_PATH", "/bad/path")
    d = declaration(tmp_path)
    args, env, authorization = build_launch(d, 12345)
    assert authorization is not None
    key = authorization.reveal()
    assert key not in repr(args) and key not in repr(authorization)
    assert env["LLAMA_API_KEY"] == key
    assert "LLAMA_ARG_TOOLS" not in env
    assert "LLAMA_ARG_CTX_SIZE" not in env
    assert "LLAMA_ARG_SLOT_SAVE_PATH" not in env
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--ctx-size") + 1] == "2048"
    assert args[args.index("--parallel") + 1] == "1"
    assert "--no-webui" in args and "--no-agent" in args
    assert "--log-disable" in args and "--cache-ram" in args
    assert "--no-cache-prompt" in args
    assert build_launch(d, 12346)[2].reveal() != key


def test_prompt_cache_is_disabled_only_for_formal_profile(tmp_path):
    formal_args, _, _ = build_launch(declaration(tmp_path), 12345)
    shared_args, _, _ = build_launch(
        declaration(tmp_path, ManagedServerMode.SHARED), 12346
    )
    assert "--no-cache-prompt" in formal_args
    assert "--no-cache-prompt" not in shared_args


@pytest.mark.parametrize("mode", list(ManagedServerMode))
def test_shared_formal_same_resource_never_take_over(tmp_path, mode):
    d = declaration(tmp_path, mode)
    with acquire_resource_lease(d.lease_root, d.resource):
        with pytest.raises(ResourceBusyError):
            asyncio.run(start_managed_server(d))


def test_failed_start_releases_own_lease(tmp_path):
    d = declaration(tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(start_managed_server(d, startup_seconds=0.5))
    with acquire_resource_lease(d.lease_root, d.resource):
        pass


def test_artifact_hash_is_measured_not_provider_attested(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"a")
    one = hash_model_artifact(path)
    path.write_bytes(b"b")
    assert hash_model_artifact(path).sha256 != one.sha256


def test_startup_cancellation_reaps_owned_child(tmp_path):
    d = declaration(tmp_path)
    slow = tmp_path / "slow-server"
    slow.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(100)\n")
    slow.chmod(0o700)
    d = d.model_copy(update={"executable": str(slow)})

    async def run():
        task = asyncio.create_task(start_managed_server(d))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    with acquire_resource_lease(d.lease_root, d.resource):
        pass



def test_reasoning_profile_is_explicit_and_changes_launch(tmp_path):
    base = declaration(tmp_path)
    with pytest.raises(ValueError, match="reasoning"):
        build_launch(base.model_copy(update={"reasoning_mode": None}), 12345)
    auto_args, _, _ = build_launch(base, 12345)
    off_args, _, _ = build_launch(
        base.model_copy(update={"reasoning_mode": ReasoningMode.OFF}), 12345
    )
    assert auto_args[auto_args.index("--reasoning") + 1] == "auto"
    assert off_args[off_args.index("--reasoning") + 1] == "off"