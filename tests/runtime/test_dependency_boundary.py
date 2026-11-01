"""Structural guardrails plus a focused no-write behavioral control."""

import ast
import builtins
from pathlib import Path
from unittest.mock import patch
from test_preparation import native_binding
from llm_evaluation.runtime.preparation import prepare_generation

ROOT = Path(__file__).resolve().parents[2] / "src/llm_evaluation/runtime"
BANNED = {
    "artifact_io",
    "records",
    "metrics",
    "stopping",
    "stopping_calibration",
    "stopping_records",
    "model_runner",
    "authority_store",
    "evidiome",
    "prlab",
}


def prohibited_imports(source):
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        for name in names:
            if set(name.split(".")) & BANNED:
                yield name


def test_dependency_direction_and_no_transport_persistence():
    for path in ROOT.rglob("*.py"):
        source = path.read_text()
        assert not list(prohibited_imports(source)), str(path)
        if path.name in {"resources.py", "lease_exec.py"}:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "write_text",
                    "write_bytes",
                    "mkdir",
                    "unlink",
                    "ftruncate",
                }, str(path)


def test_known_bad_import_control_is_detected():
    assert list(prohibited_imports("from llm_evaluation import artifact_io")) == [
        "artifact_io"
    ]
    assert list(prohibited_imports("from ..model_runner import ModelCondition")) == [
        "model_runner"
    ]


def test_preparation_performs_no_filesystem_writes():
    binding, request = native_binding()
    with patch.object(
        builtins, "open", side_effect=AssertionError("unexpected file access")
    ):
        assert prepare_generation(binding, request).wire_bytes


def test_no_obsolete_urllib_transmission():
    runner = (ROOT.parent / "model_runner.py").read_text()
    assert "urllib.request" not in runner
    assert "urllib.error" not in runner
