"""Exact calibration helpers for the finite Bernoulli E-002 diagnostic.

The composite guarantees in this module are iid Bernoulli guarantees.  They rely on
finite-path monotonicity plus the usual common-uniform Bernoulli coupling; they are
not claims about dependent, drifting, or heterogeneous latent-item processes.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import TypedDict, cast

from .stopping import Policy, Scenario, decision, operating_characteristics, reference_error


MAX_EXHAUSTIVE_VERIFICATION_CAP = 16


@dataclass(frozen=True)
class MonotonicityProof:
    """Mechanical finite-path check supporting the Bernoulli coupling argument."""

    accept_is_increasing: bool
    reject_is_decreasing: bool
    paths_checked: int


def _path_outcome(policy: Policy, mask: int, decisions: dict[tuple[int, int], str]) -> str:
    successes = 0
    for n in range(1, policy.cap + 1):
        successes += (mask >> (n - 1)) & 1
        if n in policy.looks:
            verdict = decisions[(n, successes)]
            if verdict != "inconclusive" or n == policy.cap:
                return verdict
    raise AssertionError("cap is a planned look")


def finite_path_monotonicity(policy: Policy) -> MonotonicityProof:
    """Check monotonic terminal events over every edge of the finite Boolean cube.

    If changing any failure to a success can never destroy an accept event, accept
    is coordinatewise increasing.  Dually, if changing any success to a failure can
    never destroy a reject event, reject is coordinatewise decreasing.  Together
    with X_i(p)=1{U_i<=p}, this puts iid Bernoulli composite error suprema at the
    nearest boundary point of each one-sided region.
    """
    if policy.cap > MAX_EXHAUSTIVE_VERIFICATION_CAP:
        raise ValueError(
            f"exhaustive verification cap is {MAX_EXHAUSTIVE_VERIFICATION_CAP}; "
            f"got {policy.cap}"
        )
    decisions = {
        (n, c): decision(policy, n, c)
        for n in policy.looks
        for c in range(n + 1)
    }
    count = 1 << policy.cap
    outcomes = [_path_outcome(policy, mask, decisions) for mask in range(count)]
    accept_ok = True
    reject_ok = True
    for mask, outcome in enumerate(outcomes):
        if outcome == "accept":
            for bit in range(policy.cap):
                if not mask & (1 << bit) and outcomes[mask | (1 << bit)] != "accept":
                    accept_ok = False
                    break
        elif outcome == "reject":
            for bit in range(policy.cap):
                if mask & (1 << bit) and outcomes[mask & ~(1 << bit)] != "reject":
                    reject_ok = False
                    break
        if not accept_ok or not reject_ok:
            break
    return MonotonicityProof(accept_ok, reject_ok, count)


def finite_count_path_monotonicity(
    *, looks: tuple[int, ...], accept_min: tuple[int, ...], reject_max: tuple[int, ...]
) -> MonotonicityProof:
    """Independently enumerate terminal events for a finite count-boundary rule."""
    if not looks or len(looks) != len(accept_min) or len(looks) != len(reject_max):
        raise ValueError("looks and count boundaries must have equal nonzero length")
    if tuple(sorted(set(looks))) != looks or looks[0] < 1:
        raise ValueError("looks must be positive and strictly increasing")
    if any(reject >= accept for reject, accept in zip(reject_max, accept_min, strict=True)):
        raise ValueError("reject boundary must be strictly below accept boundary")
    if looks[-1] > MAX_EXHAUSTIVE_VERIFICATION_CAP:
        raise ValueError(
            f"exhaustive verification cap is {MAX_EXHAUSTIVE_VERIFICATION_CAP}; "
            f"got {looks[-1]}"
        )

    def outcome(mask: int) -> str:
        successes = 0
        for n in range(1, looks[-1] + 1):
            successes += (mask >> (n - 1)) & 1
            if n not in looks:
                continue
            idx = looks.index(n)
            if successes >= accept_min[idx]:
                return "accept"
            if successes <= reject_max[idx]:
                return "reject"
            if n == looks[-1]:
                return "inconclusive"
        raise AssertionError("cap is a planned look")

    count = 1 << looks[-1]
    outcomes = [outcome(mask) for mask in range(count)]
    accept_ok = True
    reject_ok = True
    for mask, terminal in enumerate(outcomes):
        if terminal == "accept":
            for bit in range(looks[-1]):
                if not mask & (1 << bit) and outcomes[mask | (1 << bit)] != "accept":
                    accept_ok = False
                    break
        elif terminal == "reject":
            for bit in range(looks[-1]):
                if mask & (1 << bit) and outcomes[mask & ~(1 << bit)] != "reject":
                    reject_ok = False
                    break
        if not accept_ok or not reject_ok:
            break
    return MonotonicityProof(accept_ok, reject_ok, count)


def composite_endpoint_guarantees(
    policy: Policy, *, p0: Fraction, p1: Fraction
) -> dict[str, Fraction]:
    """Exact iid composite-region errors, after mechanically checking monotonicity."""
    if not 0 <= p0 < policy.threshold < p1 <= 1:
        raise ValueError("require p0 < threshold < p1")
    proof = finite_path_monotonicity(policy)
    if not proof.accept_is_increasing or not proof.reject_is_decreasing:
        raise ValueError("policy terminal events are not monotone")
    lower = operating_characteristics(policy, Scenario("iid", p0))
    upper = operating_characteristics(policy, Scenario("iid", p1))
    return {
        "false_accept": lower["accept"],
        "false_reject": upper["reject"],
        "false_accept_supremum_at": p0,
        "false_reject_supremum_at": p1,
    }


def _wrong_probability(
    p: Fraction, threshold: Fraction, n: int, tie_handling: str
) -> Fraction:
    result = reference_error(p, threshold, n)
    wrong = result["accept"] if p < threshold else result["reject"]
    if tie_handling == "wrong":
        wrong += result["inconclusive"]
    elif tie_handling == "half":
        wrong += result["inconclusive"] / 2
    elif tie_handling != "inconclusive":
        raise ValueError("tie_handling must be inconclusive, half, or wrong")
    return wrong


def reference_sample_size(
    *, threshold: Fraction, delta: Fraction, epsilon: Fraction,
    tie_handling: str = "inconclusive", max_n: int = 10000,
    epsilon_inconclusive: Fraction | None = None,
) -> dict[str, Fraction | int | str | bool | None]:
    """Smallest exact fixed n meeting explicit wrong and optional tie constraints.

    The nearest points ``threshold-delta`` and ``threshold+delta`` are worst case
    for the one-sided wrong-decision events by Bernoulli coupling.  Raw tie and
    correct-decision probabilities are reported separately so abstention cannot be
    hidden inside the wrong-decision target.
    """
    if not 0 < threshold < 1 or not 0 < delta or threshold - delta < 0 or threshold + delta > 1:
        raise ValueError("delta must stay inside the Bernoulli parameter space")
    if not 0 < epsilon < 1 or max_n < 1:
        raise ValueError("invalid epsilon or max_n")
    if epsilon_inconclusive is not None and not 0 <= epsilon_inconclusive < 1:
        raise ValueError("epsilon_inconclusive must be in [0, 1)")
    if tie_handling not in {"inconclusive", "half", "wrong"}:
        raise ValueError("tie_handling must be inconclusive, half, or wrong")
    lower_p, upper_p = threshold - delta, threshold + delta
    previous_wrong = Fraction(1)
    previous_inconclusive = Fraction(1)
    previous_satisfied = False
    for n in range(1, max_n + 1):
        lower = reference_error(lower_p, threshold, n)
        upper = reference_error(upper_p, threshold, n)
        lower_raw_wrong = lower["accept"]
        upper_raw_wrong = upper["reject"]
        lower_inconclusive = lower["inconclusive"]
        upper_inconclusive = upper["inconclusive"]
        lower_correct = lower["reject"]
        upper_correct = upper["accept"]
        lower_wrong = _wrong_probability(lower_p, threshold, n, tie_handling)
        upper_wrong = _wrong_probability(upper_p, threshold, n, tie_handling)
        worst_wrong = max(lower_wrong, upper_wrong)
        worst_inconclusive = max(lower_inconclusive, upper_inconclusive)
        satisfied = worst_wrong <= epsilon and (
            epsilon_inconclusive is None or worst_inconclusive <= epsilon_inconclusive
        )
        if satisfied:
            return {
                "n": n,
                "threshold": threshold,
                "delta": delta,
                "epsilon": epsilon,
                "epsilon_inconclusive": epsilon_inconclusive,
                "tie_handling": tie_handling,
                "lower_wrong": lower_wrong,
                "upper_wrong": upper_wrong,
                "worst_wrong": worst_wrong,
                "lower_raw_wrong": lower_raw_wrong,
                "upper_raw_wrong": upper_raw_wrong,
                "lower_inconclusive": lower_inconclusive,
                "upper_inconclusive": upper_inconclusive,
                "worst_inconclusive": worst_inconclusive,
                "lower_correct": lower_correct,
                "upper_correct": upper_correct,
                "min_correct": min(lower_correct, upper_correct),
                "previous_worst_wrong": previous_wrong,
                "previous_worst_inconclusive": previous_inconclusive,
                "previous_satisfied": previous_satisfied,
            }
        previous_wrong = worst_wrong
        previous_inconclusive = worst_inconclusive
        previous_satisfied = satisfied
    raise ValueError("no reference size met requested constraints within max_n")


def classify_candidate(
    row: dict[str, object], epsilon_accept: Fraction, epsilon_reject: Fraction
) -> str:
    """Classify a policy using realized endpoint errors, not nominal alpha."""
    false_accept = cast(Fraction, row["false_accept"])
    false_reject = cast(Fraction, row["false_reject"])
    if false_accept <= epsilon_accept and false_reject <= epsilon_reject:
        return "feasible"
    return "infeasible"


def _dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    keys: tuple[str, ...] = ("false_accept", "false_reject")
    endpoint_keys = (
        "inconclusive_p0", "inconclusive_p1", "expected_n_p0", "expected_n_p1"
    )
    if all(key in left and key in right for key in endpoint_keys):
        keys += endpoint_keys
    else:
        # Small caller-supplied summaries may not have endpoint-specific fields.
        keys += ("inconclusive", "expected_n")
    left_values = tuple(cast(Fraction, left[key]) for key in keys)
    right_values = tuple(cast(Fraction, right[key]) for key in keys)
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def pareto_efficient(
    rows: list[dict[str, object]], epsilon_accept: Fraction, epsilon_reject: Fraction
) -> list[dict[str, object]]:
    """Return feasible nondominated rows without collapsing abstention and cost."""
    feasible = [
        row for row in rows
        if classify_candidate(row, epsilon_accept, epsilon_reject) == "feasible"
    ]
    return [row for row in feasible if not any(_dominates(other, row) for other in feasible if other is not row)]


def calibrate_lr_grid(
    *, looks: tuple[int, ...], threshold: Fraction, p0: Fraction, p1: Fraction,
    reject_boundaries: tuple[Fraction, ...], accept_boundaries: tuple[Fraction, ...],
    epsilon_accept: Fraction, epsilon_reject: Fraction,
) -> list[dict[str, object]]:
    """Evaluate a deliberately small finite LR-boundary grid at composite endpoints."""
    if not reject_boundaries or not accept_boundaries:
        raise ValueError("calibration grid must be nonempty")
    rows: list[dict[str, object]] = []
    for reject_boundary in reject_boundaries:
        for accept_boundary in accept_boundaries:
            policy = Policy(
                "sprt", looks, threshold=threshold, low=p0, high=p1,
                lr_reject=reject_boundary, lr_accept=accept_boundary,
            )
            guarantee = composite_endpoint_guarantees(policy, p0=p0, p1=p1)
            lower = operating_characteristics(policy, Scenario("iid", p0))
            upper = operating_characteristics(policy, Scenario("iid", p1))
            row: dict[str, object] = {
                "name": f"lr_{reject_boundary}_{accept_boundary}",
                "lr_reject": reject_boundary,
                "lr_accept": accept_boundary,
                "false_accept": guarantee["false_accept"],
                "false_reject": guarantee["false_reject"],
                "inconclusive_p0": lower["inconclusive"],
                "inconclusive_p1": upper["inconclusive"],
                "inconclusive": (lower["inconclusive"] + upper["inconclusive"]) / 2,
                "expected_n_p0": lower["expected_n"],
                "expected_n_p1": upper["expected_n"],
                "expected_n": (lower["expected_n"] + upper["expected_n"]) / 2,
                "stopping_p0": lower["stopping"],
                "stopping_p1": upper["stopping"],
            }
            row["classification"] = classify_candidate(row, epsilon_accept, epsilon_reject)
            rows.append(row)
    return rows


class FiniteCountOC(TypedDict):
    accept: Fraction
    reject: Fraction
    inconclusive: Fraction
    expected_n: Fraction
    stopping: dict[int, Fraction]


def _finite_count_oc(
    *, looks: tuple[int, ...], accept_min: tuple[int, ...], reject_max: tuple[int, ...], p: Fraction
) -> FiniteCountOC:
    states: dict[int, Fraction] = {0: Fraction(1)}
    accept = reject = inconclusive = Fraction()
    stopping: dict[int, Fraction] = {}
    for n in range(1, looks[-1] + 1):
        advanced: dict[int, Fraction] = {}
        for successes, mass in states.items():
            advanced[successes] = advanced.get(successes, Fraction()) + mass * (1 - p)
            advanced[successes + 1] = advanced.get(successes + 1, Fraction()) + mass * p
        states = advanced
        if n not in looks:
            continue
        idx = looks.index(n)
        continuing: dict[int, Fraction] = {}
        stopped = Fraction()
        for successes, mass in states.items():
            if successes >= accept_min[idx]:
                accept += mass
                stopped += mass
            elif successes <= reject_max[idx]:
                reject += mass
                stopped += mass
            elif n == looks[-1]:
                inconclusive += mass
                stopped += mass
            else:
                continuing[successes] = continuing.get(successes, Fraction()) + mass
        if stopped:
            stopping[n] = stopped
        states = continuing
    if states or accept + reject + inconclusive != 1:
        raise AssertionError("finite count boundary mass did not resolve")
    expected_n = sum((Fraction(n) * mass for n, mass in stopping.items()), Fraction())
    return {
        "accept": accept,
        "reject": reject,
        "inconclusive": inconclusive,
        "expected_n": expected_n,
        "stopping": stopping,
    }


def calibrate_finite_count_boundaries(
    *, looks: tuple[int, ...], threshold: Fraction, p0: Fraction, p1: Fraction,
    epsilon_accept: Fraction, epsilon_reject: Fraction, symmetric: bool = True,
) -> list[dict[str, object]]:
    """Enumerate a small exact finite-horizon count-boundary family.

    For the current diagnostic, symmetry means that at look n an acceptance
    threshold ``a`` is paired with rejection threshold ``n-a``.  ``a=n+1``
    disables both boundaries at that look.  This is intentionally a finite
    calibration laboratory, not a general sequential-optimization framework.
    """
    if not symmetric:
        raise ValueError("only the prespecified symmetric calibration family is implemented")
    if threshold != Fraction(1, 2):
        raise ValueError("symmetric count calibration currently requires threshold=1/2")
    if not 0 <= p0 < threshold < p1 <= 1:
        raise ValueError("require p0 < threshold < p1")
    from itertools import product

    choices = [tuple(range(n // 2 + 1, n + 2)) for n in looks]
    rows: list[dict[str, object]] = []
    for accept_min in product(*choices):
        reject_max = tuple(n - a for n, a in zip(looks, accept_min, strict=True))
        lower = _finite_count_oc(
            looks=looks, accept_min=accept_min, reject_max=reject_max, p=p0
        )
        upper = _finite_count_oc(
            looks=looks, accept_min=accept_min, reject_max=reject_max, p=p1
        )
        row: dict[str, object] = {
            "name": "count_" + "_".join(str(value) for value in accept_min),
            "accept_min": accept_min,
            "reject_max": reject_max,
            "false_accept": lower["accept"],
            "false_reject": upper["reject"],
            "inconclusive_p0": lower["inconclusive"],
            "inconclusive_p1": upper["inconclusive"],
            "inconclusive": (lower["inconclusive"] + upper["inconclusive"]) / 2,
            "expected_n_p0": lower["expected_n"],
            "expected_n_p1": upper["expected_n"],
            "expected_n": (lower["expected_n"] + upper["expected_n"]) / 2,
            "stopping_p0": lower["stopping"],
            "stopping_p1": upper["stopping"],
        }
        row["classification"] = classify_candidate(row, epsilon_accept, epsilon_reject)
        rows.append(row)
    return rows
