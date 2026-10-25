"""Strict v2 synthetic records and explicit source/finalization verification."""

import copy
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .metrics import all_k, best_of_k, marginal

CONDITION_FIELDS = {"subject", "prompt", "harness", "tools", "runtime", "sampling",
                    "retry_policy", "budget", "benchmark", "oracle", "evaluator", "software"}
METRICS = ("marginal", "best_of_k", "all_k")
INTERPRETATION = "observed finite-sample subsets; future iid probability is conditional on assumptions"


def canonical_bytes(value: dict[str, Any]) -> bytes:
    """Sorted, compact UTF-8 JSON with a terminal LF; no float NaN/Infinity."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: dict[str, Any]) -> str:
    """SHA-256 of canonical_bytes, not the input file's literal bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(data: bytes) -> str:
    """SHA-256 of the exact input artifact bytes."""
    return hashlib.sha256(data).hexdigest()


# Compatibility for callers of the old helper. Its meaning was always canonical JSON.
digest = canonical_sha256


def _fields(value: object, required: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{label}: expected exactly {sorted(required)}")
    return value


def _hex(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(ch in "0123456789abcdef" for ch in value)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_fixture(data: bytes) -> dict[str, Any]:
    return validate_fixture(json.loads(data, object_pairs_hook=_object_pairs,
                                       parse_constant=_bad_constant))


def validate_fixture(value: object) -> dict[str, Any]:
    record = _fields(value, {"schema_version", "experiment", "condition", "tasks"}, "fixture")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1 or record["experiment"] != "E-002":
        raise ValueError("unsupported fixture schema or experiment")
    condition = _fields(record["condition"], CONDITION_FIELDS, "condition")
    if any(type(field) is not str or not field for field in condition.values()):
        raise ValueError("condition fields must be nonempty public identifiers")
    tasks = record["tasks"]
    if type(tasks) is not list or not tasks:
        raise ValueError("tasks must be nonempty")
    seen: set[str] = set()
    for task in tasks:
        item = _fields(task, {"id", "outcomes", "role"}, "task")
        if type(item["id"]) is not str or not item["id"] or item["id"] in seen:
            raise ValueError("task identifiers must be unique and nonempty")
        seen.add(item["id"])
        if item["role"] != "synthetic":
            raise ValueError("only public synthetic tasks in this slice")
        if type(item["outcomes"]) is not list or not item["outcomes"] or any(type(x) is not bool for x in item["outcomes"]):
            raise ValueError("outcomes must be nonempty binary oracle results")
    return record


def analyze(value: object, k: int, *, source_bytes: bytes | None = None) -> dict[str, Any]:
    """Return a preview; finalization requires source bytes and a clean Git identity."""
    fixture = validate_fixture(value)
    if type(k) is not int or k < 1 or any(k > len(t["outcomes"]) for t in fixture["tasks"]):
        raise ValueError("k must be positive and <= every task's trial count")
    if source_bytes is not None and parse_fixture(source_bytes) != fixture:
        raise ValueError("source bytes do not represent the supplied fixture")
    rows = []
    for task in fixture["tasks"]:
        outcomes = task["outcomes"]
        n, c = len(outcomes), sum(outcomes)
        rows.append({"task": task["id"], "n": n, "c": c, "marginal": marginal(n, c),
                     "best_of_k": best_of_k(n, c, k), "all_k": all_k(n, c, k)})
    return {"schema_version": 2, "record_type": "e002_diagnostic", "experiment": "E-002",
            "fixture_canonical_sha256": canonical_sha256(fixture),
            "fixture_raw_sha256": raw_sha256(source_bytes) if source_bytes is not None else None,
            "condition": copy.deepcopy(fixture["condition"]), "software_commit": None,
            "python_version": platform.python_version(), "k": k, "task_weighting": "equal",
            "per_task": rows,
            "macro": {key: sum(row[key] for row in rows) / len(rows) for key in METRICS},
            "interpretation": INTERPRETATION}


def validate_result(value: object, *, finalized: bool = True) -> dict[str, Any]:
    result = _fields(value, {"schema_version", "record_type", "experiment",
                             "fixture_canonical_sha256", "fixture_raw_sha256", "condition",
                             "software_commit", "python_version", "k", "task_weighting",
                             "per_task", "macro", "interpretation"}, "result")
    if type(result["schema_version"]) is not int or result["schema_version"] != 2 or result["record_type"] != "e002_diagnostic" or result["experiment"] != "E-002":
        raise ValueError("unsupported result schema")
    if not _hex(result["fixture_canonical_sha256"], 64):
        raise ValueError("invalid canonical source digest")
    raw = result["fixture_raw_sha256"]
    if raw is not None and not _hex(raw, 64):
        raise ValueError("invalid raw source digest")
    commit = result["software_commit"]
    if commit is not None and not _hex(commit, 40):
        raise ValueError("invalid software commit")
    if finalized and (raw is None or commit is None):
        raise ValueError("finalized record requires raw source and software commit")
    if type(result["python_version"]) is not str or not result["python_version"]:
        raise ValueError("missing Python version")
    condition = _fields(result["condition"], CONDITION_FIELDS, "condition")
    if any(type(item) is not str or not item for item in condition.values()):
        raise ValueError("invalid condition")
    if type(result["k"]) is not int or result["k"] < 1 or result["task_weighting"] != "equal":
        raise ValueError("invalid k or task weighting")
    if result["interpretation"] != INTERPRETATION:
        raise ValueError("unknown interpretation")
    rows = result["per_task"]
    if type(rows) is not list or not rows:
        raise ValueError("missing task rows")
    names: set[str] = set()
    for row in rows:
        _fields(row, {"task", "n", "c"} | set(METRICS), "task row")
        if type(row["task"]) is not str or not row["task"] or row["task"] in names or any(type(row[x]) is not int for x in ("n", "c")):
            raise ValueError("invalid task row")
        names.add(row["task"])
        if any(type(row[x]) is not float or not 0 <= row[x] <= 1 for x in METRICS):
            raise ValueError("invalid task metric")
        expected = (marginal(row["n"], row["c"]), best_of_k(row["n"], row["c"], result["k"]),
                    all_k(row["n"], row["c"], result["k"]))
        if any(abs(row[key] - correct) >= 1e-12 for key, correct in zip(METRICS, expected)):
            raise ValueError("inconsistent task metric")
    macro = _fields(result["macro"], set(METRICS), "macro")
    for key in METRICS:
        if type(macro[key]) is not float or abs(macro[key] - sum(row[key] for row in rows) / len(rows)) >= 1e-12:
            raise ValueError("inconsistent macro metric")
    return result


def verify_result_source(value: object, source_bytes: bytes) -> dict[str, Any]:
    """Verify a finalized result against the exact supplied file, not a digest alone."""
    result = validate_result(value)
    fixture = parse_fixture(source_bytes)
    if result["fixture_raw_sha256"] != raw_sha256(source_bytes):
        raise ValueError("raw source digest mismatch")
    if result["fixture_canonical_sha256"] != canonical_sha256(fixture):
        raise ValueError("canonical source digest mismatch")
    expected = analyze(fixture, result["k"], source_bytes=source_bytes)
    for field in ("condition", "per_task", "macro", "interpretation", "task_weighting"):
        if result[field] != expected[field]:
            raise ValueError(f"source/result mismatch: {field}")
    return result


def load_fixture(path: Path) -> dict[str, Any]:
    return parse_fixture(path.read_bytes())


def current_git_commit() -> str:
    """Identity of the clean checkout that actually executes this installed source."""
    root = Path(__file__).resolve().parents[2]
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("implementation is not executing from its repository root")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("refusing to finalize against a dirty checkout")
    commit = git("rev-parse", "--verify", "HEAD")
    if not _hex(commit, 40):
        raise ValueError("invalid checkout commit")
    return commit


def write_immutable(path: Path, value: dict[str, Any], source_bytes: bytes) -> None:
    """Bind to source and exact clean HEAD, then atomically publish without overwrite."""
    result = verify_result_source(value, source_bytes)
    if result["software_commit"] != current_git_commit():
        raise ValueError("software commit does not match executing clean checkout")
    if result["python_version"] != platform.python_version():
        raise ValueError("Python runtime differs from record")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".e002-",
                                         delete=False) as handle:
            staging = Path(handle.name)
            handle.write(canonical_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path)  # existing path/symlink fails; final name appears complete
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def read_finalized(path: Path, source_bytes: bytes) -> dict[str, Any]:
    data = path.read_bytes()
    result = json.loads(data, object_pairs_hook=_object_pairs, parse_constant=_bad_constant)
    verify_result_source(result, source_bytes)
    if canonical_bytes(result) != data:
        raise ValueError("finalized record is not canonical JSON")
    return result
