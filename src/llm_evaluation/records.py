"""Strict v3 synthetic records and distinct source/software/finalization checks."""

import copy
import hashlib
import json
import os
import platform
import stat
import subprocess
from pathlib import Path
from typing import Any

from .artifact_io import publish_immutable, reject_symlink_components
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
    return {"schema_version": 3, "record_type": "e002_diagnostic", "experiment": "E-002",
            "fixture_canonical_sha256": canonical_sha256(fixture),
            "fixture_raw_sha256": raw_sha256(source_bytes) if source_bytes is not None else None,
            "condition": copy.deepcopy(fixture["condition"]), "software_commit": None,
            "implementation_sha256": None,
            "python_version": platform.python_version(), "k": k, "task_weighting": "equal",
            "per_task": rows,
            "macro": {key: sum(row[key] for row in rows) / len(rows) for key in METRICS},
            "interpretation": INTERPRETATION}


def validate_result(value: object, *, finalized: bool = True) -> dict[str, Any]:
    result = _fields(value, {"schema_version", "record_type", "experiment",
                             "fixture_canonical_sha256", "fixture_raw_sha256", "condition",
                             "software_commit", "implementation_sha256", "python_version", "k", "task_weighting",
                             "per_task", "macro", "interpretation"}, "result")
    if type(result["schema_version"]) is not int or result["schema_version"] != 3 or result["record_type"] != "e002_diagnostic" or result["experiment"] != "E-002":
        raise ValueError("unsupported result schema")
    if not _hex(result["fixture_canonical_sha256"], 64):
        raise ValueError("invalid canonical source digest")
    raw = result["fixture_raw_sha256"]
    if raw is not None and not _hex(raw, 64):
        raise ValueError("invalid raw source digest")
    commit = result["software_commit"]
    if commit is not None and (type(commit) is not str or len(commit) not in (40, 64) or not _hex(commit, len(commit))):
        raise ValueError("invalid software commit")
    if finalized and (raw is None or commit is None):
        raise ValueError("finalized record requires raw source and software commit")
    implementation = result["implementation_sha256"]
    if implementation is not None and not _hex(implementation, 64):
        raise ValueError("invalid implementation digest")
    if finalized and implementation is None:
        raise ValueError("finalized record requires implementation digest")
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


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          check=True).stdout


def _package_files(root: Path) -> dict[str, bytes]:
    """Actual regular Python source under the package; reject linked or special files."""
    package = root / "src" / "llm_evaluation"
    if (root / "src").is_symlink() or package.is_symlink() or not package.is_dir():
        raise ValueError("implementation package is missing or linked")
    files: dict[str, bytes] = {}
    for path in package.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("implementation symlink is forbidden")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("implementation special file is forbidden")
        if path.suffix == ".py":
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _package_digest(files: dict[str, bytes]) -> str:
    """SHA-256 over sorted UTF-8 relative paths and lengths plus exact file bytes."""
    digest = hashlib.sha256(b"llm-evaluation-python-package-v1\0")
    for name, data in sorted(files.items()):
        path_bytes = name.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def software_identity() -> tuple[str, str]:
    """Commit and measured package digest, checked against HEAD's committed blobs.

    .py files under src/llm_evaluation are identity-bearing; paths and bytes
    enter the digest. Untracked additions and removals change the inventory.
    Python caches are excluded. The Git object format determines commit length.
    """
    root = Path(__file__).resolve().parents[2]
    if Path(os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root:
        raise ValueError("implementation is not executing from its repository root")
    commit = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    fmt = _git(root, "rev-parse", "--show-object-format").decode("ascii").strip()
    length = {"sha1": 40, "sha256": 64}.get(fmt)
    if length is None or not _hex(commit, length):
        raise ValueError("invalid checkout commit")
    prefix = b"src/llm_evaluation/"
    names = _git(root, "ls-tree", "-r", "-z", "--name-only", commit, "--", "src/llm_evaluation")
    committed: dict[str, bytes] = {}
    for raw_name in names.split(b"\0"):
        if not raw_name:
            continue
        if not raw_name.startswith(prefix):
            raise ValueError("unexpected Git tree path")
        name = raw_name.decode("utf-8")
        if name.endswith(".py"):
            committed[name] = _git(root, "show", f"{commit}:{name}")
    actual = _package_files(root)
    if set(actual) != set(committed):
        raise ValueError("implementation source inventory differs from Git HEAD")
    measured = _package_digest(actual)
    if measured != _package_digest(committed):
        raise ValueError("executing implementation bytes differ from Git HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("refusing to finalize against a dirty checkout")
    return commit, measured


def current_git_commit() -> str:
    """Compatibility helper; also verifies executing package bytes against HEAD."""
    return software_identity()[0]


def verify_result_software(value: object) -> dict[str, Any]:
    """Check a v3 record against this local checkout and Python runtime."""
    result = validate_result(value)
    commit, implementation = software_identity()
    if result["software_commit"] != commit or result["implementation_sha256"] != implementation:
        raise ValueError("software identity does not match executing Git checkout")
    if result["python_version"] != platform.python_version():
        raise ValueError("Python runtime differs from record")
    return result


def write_immutable(path: Path, value: dict[str, Any], source_bytes: bytes) -> None:
    """Bind source and measured code to HEAD, then publish complete bytes once."""
    result = verify_result_source(value, source_bytes)
    verify_result_software(result)
    publish_immutable(path, canonical_bytes(result))


def read_finalized(path: Path, source_bytes: bytes) -> dict[str, Any]:
    reject_symlink_components(path)
    if not path.is_file():
        raise ValueError("finalized artifact is not an ordinary file")
    data = path.read_bytes()
    result = json.loads(data, object_pairs_hook=_object_pairs, parse_constant=_bad_constant)
    verify_result_source(result, source_bytes)
    if canonical_bytes(result) != data:
        raise ValueError("finalized record is not canonical JSON")
    return result
