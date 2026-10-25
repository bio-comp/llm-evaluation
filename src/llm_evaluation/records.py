"""Strict v1 boundary for public synthetic input and immutable analysis records."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .metrics import all_k, best_of_k, marginal


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _fields(value: object, required: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{label}: expected exactly {sorted(required)}")
    return value


def validate_fixture(value: object) -> dict[str, Any]:
    record = _fields(value, {"schema_version", "experiment", "condition", "tasks"}, "fixture")
    if record["schema_version"] != 1 or type(record["schema_version"]) is not int or record["experiment"] != "E-002":
        raise ValueError("unsupported fixture schema or experiment")
    condition = _fields(record["condition"], {"subject", "prompt", "harness", "tools", "runtime", "sampling", "retry_policy", "budget", "benchmark", "oracle", "evaluator", "software"}, "condition")
    for key, field in condition.items():
        if type(field) is not str or not field:
            raise ValueError(f"condition.{key} must be a nonempty public identifier")
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


def analyze(value: object, k: int) -> dict[str, Any]:
    fixture = validate_fixture(value)
    if type(k) is not int or k < 1 or any(k > len(t["outcomes"]) for t in fixture["tasks"]):
        raise ValueError("k must be positive and <= every task's trial count")
    rows = []
    for task in fixture["tasks"]:
        outcomes = task["outcomes"]
        n, c = len(outcomes), sum(outcomes)
        rows.append({"task": task["id"], "n": n, "c": c, "marginal": marginal(n, c),
                     "best_of_k": best_of_k(n, c, k), "all_k": all_k(n, c, k)})
    return {"schema_version": 1, "record_type": "e002_diagnostic", "experiment": "E-002",
            "fixture_sha256": digest(fixture), "condition": fixture["condition"], "software_commit": None, "k": k,
            "task_weighting": "equal", "per_task": rows,
            "macro": {key: sum(row[key] for row in rows) / len(rows)
                      for key in ("marginal", "best_of_k", "all_k")},
            "interpretation": "observed finite-sample subsets; future iid probability is conditional on assumptions"}


def validate_result(value: object) -> dict[str, Any]:
    result = _fields(value, {"schema_version", "record_type", "experiment", "fixture_sha256", "condition", "software_commit", "k", "task_weighting", "per_task", "macro", "interpretation"}, "result")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1 or result["record_type"] != "e002_diagnostic" or result["experiment"] != "E-002":
        raise ValueError("unsupported result schema")
    sha = result["fixture_sha256"]
    if type(sha) is not str or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("invalid source digest")
    commit = result["software_commit"]
    if commit is not None and (type(commit) is not str or len(commit) not in (40, 64) or any(ch not in "0123456789abcdef" for ch in commit)):
        raise ValueError("invalid software commit")
    condition = _fields(result["condition"], {"subject", "prompt", "harness", "tools", "runtime", "sampling", "retry_policy", "budget", "benchmark", "oracle", "evaluator", "software"}, "condition")
    if any(type(item) is not str or not item for item in condition.values()):
        raise ValueError("invalid condition")
    if type(result["k"]) is not int or result["k"] < 1 or result["task_weighting"] != "equal":
        raise ValueError("invalid k or task weighting")
    if result["interpretation"] != "observed finite-sample subsets; future iid probability is conditional on assumptions":
        raise ValueError("unknown interpretation")
    keys = {"marginal", "best_of_k", "all_k"}
    rows = result["per_task"]
    if type(rows) is not list or not rows:
        raise ValueError("missing task rows")
    for row in rows:
        _fields(row, {"task", "n", "c"} | keys, "task row")
        if type(row["task"]) is not str or not row["task"] or any(type(row[x]) is not int for x in ("n", "c")):
            raise ValueError("invalid task row")
        if any(type(row[x]) is not float or not 0 <= row[x] <= 1 for x in keys):
            raise ValueError("invalid task metric")
        if any(not abs(row[x] - fn(row["n"], row["c"], result["k"])) < 1e-12
               for x, fn in (("marginal", lambda n, c, k: marginal(n, c)),
                             ("best_of_k", best_of_k), ("all_k", all_k))):
            raise ValueError("inconsistent task metric")
    macro = _fields(result["macro"], keys, "macro")
    for key in keys:
        if type(macro[key]) is not float or not abs(macro[key] - sum(row[key] for row in rows) / len(rows)) < 1e-12:
            raise ValueError("inconsistent macro metric")
    return result


def load_fixture(path: Path) -> dict[str, Any]:
    return validate_fixture(json.loads(path.read_text(encoding="utf-8")))


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    """Create a run record once; never silently replace a finalized record."""
    validate_result(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))
