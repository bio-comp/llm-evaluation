"""Versioned, exact-rational E-002 simulation records; no model observations."""
import json
import platform
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .artifact_io import publish_immutable, reject_symlink_components
from .records import software_identity
from .stopping import (Policy, Scenario, operating_characteristics, ratio,
                       rational, reference_error, reference_comparison, task_variance)

LOOKS = (2, 4, 8, 16)
CONFIG = {
    "threshold": "1/2", "alpha": "1/10", "low": "3/10", "high": "7/10",
    "looks": list(LOOKS), "fixed_counts": [1, 2, 4, 8, 16],
    "development_p": ["3/10", "1/2", "7/10"],
    "holdout_p": ["1/50", "1/10", "1/4", "7/20", "9/20",
                  "11/20", "13/20", "3/4", "9/10", "49/50"],
    "reference_counts": [16, 64, 256],
    "sensitivity_thresholds": ["2/5", "1/2", "3/5"],
    "sensitivity_halfwidths": ["1/10", "1/5"],
    "loss_weights": [[1, 1, 0], [5, 1, 0], [1, 5, 0], [1, 1, 1]],
    "cost_per_trial": "1/1",  # abstract units; never a provider price
}


def policies(config: dict) -> tuple[tuple[str, Policy], ...]:
    threshold, alpha = rational(config["threshold"]), rational(config["alpha"])
    low, high = rational(config["low"]), rational(config["high"])
    shared = {"threshold": threshold, "alpha": alpha, "low": low, "high": high}
    return tuple((f"fixed_{n}", Policy("fixed", (n,), **shared))
                 for n in config["fixed_counts"]) + tuple(
        (kind, Policy(kind, tuple(config["looks"]), **shared))
        for kind in ("naive", "group", "sequence", "sprt")
    )


def _row(name: str, scenario_name: str, scenario: Scenario, policy: Policy,
         cost: Fraction) -> dict:
    result = operating_characteristics(policy, scenario)
    def quantile(q: Fraction) -> int:
        cumulative = Fraction()
        for n, mass in sorted(result["stopping"].items()):
            cumulative += mass
            if cumulative >= q:
                return n
        raise AssertionError("stopping mass did not sum to one")
    return {
        "policy": name, "scenario": scenario_name, "p": ratio(scenario.p),
        "accept": ratio(result["accept"]), "reject": ratio(result["reject"]),
        "inconclusive": ratio(result["inconclusive"]),
        "expected_n": ratio(result["expected_n"]),
        "expected_cost": ratio(result["expected_n"] * cost),
        "interval_coverage": (ratio(result["interval_coverage"])
                              if result["interval_coverage"] is not None else None),
        "stopping": {str(n): ratio(mass) for n, mass in result["stopping"].items()},
        "stopping_q50": quantile(Fraction(1, 2)),
        "stopping_q90": quantile(Fraction(9, 10)),
    }


def compute(config: dict) -> dict:
    """Exact development and held-out grids; no constants tuned on holdout."""
    if config != CONFIG:
        raise ValueError("unsupported simulation design")
    candidates = policies(config)
    cost = rational(config["cost_per_trial"])
    development = [_row(name, "iid", Scenario("iid", rational(p)), policy, cost)
                   for p in config["development_p"] for name, policy in candidates]
    holdout = [_row(name, "iid", Scenario("iid", rational(p)), policy, cost)
               for p in config["holdout_p"] for name, policy in candidates]
    stresses = [
        ("beta_mixture", Scenario("beta_mixture", Fraction(3, 10),
                                  concentration=Fraction(2))),
        ("change_point", Scenario("change_point", Fraction(3, 10),
                                  after=Fraction(7, 10), change_at=8)),
    ]
    stress = [_row(name, label, scenario, policy, cost)
              for label, scenario in stresses for name, policy in candidates]
    reference = [
        {"p": p, "n": n, **{key: ratio(value) for key, value in
                            reference_error(rational(p), Fraction(1, 2), n).items()}}
        for p in ("9/20", "11/20", "7/10")
        for n in config["reference_counts"]
    ]
    sensitivity = []
    loss = []
    for theta in config["sensitivity_thresholds"]:
        for width in config["sensitivity_halfwidths"]:
            threshold, gap = rational(theta), rational(width)
            low, high = threshold-gap, threshold+gap
            arms = tuple((name, Policy(policy.kind, policy.looks, threshold,
                                       rational(config["alpha"]), low, high))
                         for name, policy in candidates)
            lower = Scenario("iid", low)
            upper = Scenario("iid", high)
            for name, policy in arms:
                a, b = operating_characteristics(policy, lower), operating_characteristics(policy, upper)
                sensitivity.extend((
                    {"threshold": theta, "halfwidth": width,
                     **_row(name, "iid", lower, policy, cost)},
                    {"threshold": theta, "halfwidth": width,
                     **_row(name, "iid", upper, policy, cost)},
                ))
                for weights in config["loss_weights"]:
                    false_accept, false_reject, cap = (Fraction(w) for w in weights)
                    # Equally weighted endpoint scenarios; abstract one-unit trial
                    # cost is kept separate from decision loss.
                    risk = (false_accept*a["accept"] + false_reject*b["reject"]
                            + cap*(a["inconclusive"] + b["inconclusive"])) / 2
                    loss.append({"threshold": theta, "halfwidth": width,
                                 "policy": name, "weights": weights,
                                 "decision_loss": ratio(risk),
                                 "mean_expected_n": ratio((a["expected_n"]+b["expected_n"])/2)})
    comparison = [
        {"p": p, "reference_n": n, "policy": name,
         **{key: ratio(value) for key, value in
            reference_comparison(policy, rational(p), n).items()}}
        for p in ("9/20", "11/20", "7/10")
        for n in config["reference_counts"]
        for name, policy in (candidates[4], candidates[-1])  # fixed_16, sprt
    ]
    variance = [
        {"task_ps": ["1/5", "1/2", "4/5"], "repetitions": k,
         **{key: ratio(value) for key, value in
            task_variance((Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)), k).items()}}
        for k in config["fixed_counts"]
    ]
    return {"development": development, "holdout": holdout, "stress": stress,
            "sensitivity": sensitivity, "loss": loss, "reference": reference,
            "reference_comparison": comparison, "task_variance": variance}


def analyze() -> dict:
    return {
        "schema_version": 1, "record_type": "e002_stopping_simulation",
        "experiment": "E-002", "method": "exact_finite_state_rational_v1",
        "config": json.loads(json.dumps(CONFIG)), "results": compute(CONFIG),
        "software_commit": None, "package_source_sha256": None,
        "python_version": platform.python_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate(value: object, *, finalized: bool = True) -> dict:
    if type(value) is not dict or set(value) != {
        "schema_version", "record_type", "experiment", "method", "config",
        "results", "software_commit", "package_source_sha256", "python_version",
        "generated_at_utc",
    }:
        raise ValueError("simulation record fields differ")
    record = value
    if (type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["record_type"] != "e002_stopping_simulation"
            or record["experiment"] != "E-002"
            or record["method"] != "exact_finite_state_rational_v1"):
        raise ValueError("unknown simulation record contract")
    if (type(record["config"]) is not dict or
            canonical_bytes(record["config"]) != canonical_bytes(CONFIG)):
        raise ValueError("unknown simulation configuration")
    # Exact recomputation also rejects binary floats, missing rows, reordered
    # scenarios, changed rational counts, and fabricated conclusions.
    if (type(record["results"]) is not dict or
            canonical_bytes(record["results"]) != canonical_bytes(compute(CONFIG))):
        raise ValueError("simulation results do not match exact calculation")
    for field, lengths in (("software_commit", (40, 64)),
                           ("package_source_sha256", (64,))):
        item = record[field]
        if item is None and not finalized:
            continue
        if type(item) is not str or len(item) not in lengths or any(c not in "0123456789abcdef" for c in item):
            raise ValueError(f"missing/invalid {field}")
    if type(record["python_version"]) is not str or not record["python_version"]:
        raise ValueError("missing Python version")
    stamp = record["generated_at_utc"]
    if type(stamp) is not str or not stamp.endswith("Z"):
        raise ValueError("invalid UTC timestamp")
    try:
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid UTC timestamp") from exc
    return record


def finalize(path: Path, value: dict) -> None:
    record = validate(value)
    commit, package = software_identity()
    if (record["software_commit"], record["package_source_sha256"]) != (commit, package):
        raise ValueError("simulation software differs from checked-out Git package")
    if record["python_version"] != platform.python_version():
        raise ValueError("simulation runtime differs")
    publish_immutable(path, canonical_bytes(record))


def read_finalized(path: Path) -> dict:
    reject_symlink_components(path)
    if not path.is_file():
        raise ValueError("simulation record is not an ordinary file")
    data = path.read_bytes()
    value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    result = validate(value)
    if canonical_bytes(result) != data:
        raise ValueError("simulation record is not canonical JSON")
    return result
