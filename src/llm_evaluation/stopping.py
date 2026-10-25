"""Exact finite-horizon operating characteristics for synthetic E-002 binary decisions.

This is a fixed-item Bernoulli diagnostic, not a model runner. Rational arithmetic
makes the probability propagation and output identities independent of binary floats.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import TypedDict


def rational(value: str) -> Fraction:
    """Parse a public decimal/fraction without binary floating point."""
    result = Fraction(value)
    if result.denominator == 0:
        raise ValueError("invalid rational")
    return result


def ratio(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Policy:
    kind: str  # fixed, naive, group, sequence, sprt
    looks: tuple[int, ...]
    threshold: Fraction = Fraction(1, 2)
    alpha: Fraction = Fraction(1, 10)
    low: Fraction = Fraction(3, 10)
    high: Fraction = Fraction(7, 10)
    lr_reject: Fraction | None = None
    lr_accept: Fraction | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"fixed", "naive", "group", "sequence", "sprt"}:
            raise ValueError("unknown policy")
        if not self.looks or tuple(sorted(set(self.looks))) != self.looks or self.looks[0] < 1:
            raise ValueError("looks must be unique increasing positive integers")
        if not 0 < self.threshold < 1 or not 0 < self.alpha < 1:
            raise ValueError("invalid threshold or error allocation")
        if not 0 < self.low < self.high < 1:
            raise ValueError("invalid SPRT endpoints")
        if self.kind == "sprt" and not self.low < self.threshold < self.high:
            raise ValueError("SPRT requires low < threshold < high")
        if self.kind == "sprt":
            if not 0 < self.effective_lr_reject < 1 < self.effective_lr_accept:
                raise ValueError("SPRT LR boundaries require 0 < reject < 1 < accept")
        elif self.lr_reject is not None or self.lr_accept is not None:
            raise ValueError("LR boundaries apply only to SPRT policies")
        if self.kind == "fixed" and len(self.looks) != 1:
            raise ValueError("fixed policy has exactly one look")

    @property
    def effective_lr_reject(self) -> Fraction:
        return self.alpha if self.lr_reject is None else self.lr_reject

    @property
    def effective_lr_accept(self) -> Fraction:
        return 1 / self.alpha if self.lr_accept is None else self.lr_accept

    @property
    def cap(self) -> int:
        return self.looks[-1]


def binomial_tail(n: int, c: int, p: Fraction, *, upper: bool) -> Fraction:
    """Exact P_p(X>=c) or P_p(X<=c) for X~Binomial(n,p)."""
    indices = range(c, n + 1) if upper else range(c + 1)
    return sum((comb(n, j) * p**j * (1-p)**(n-j) for j in indices), Fraction())


def decision(policy: Policy, n: int, c: int) -> str:
    """accept means success probability above threshold; reject means below."""
    if n not in policy.looks or not 0 <= c <= n:
        raise ValueError("not a planned look or invalid count")
    if policy.kind == "fixed":
        return "accept" if Fraction(c, n) > policy.threshold else (
            "reject" if Fraction(c, n) < policy.threshold else "inconclusive"
        )
    if policy.kind == "sprt":
        likelihood = (policy.high / policy.low)**c * (
            (1-policy.high) / (1-policy.low)
        )**(n-c)
        if likelihood >= policy.effective_lr_accept:
            return "accept"
        if likelihood <= policy.effective_lr_reject:
            return "reject"
        return "inconclusive"
    if policy.kind == "naive":
        tail_budget = policy.alpha / 2
    elif policy.kind == "group":
        tail_budget = policy.alpha / (2 * len(policy.looks))
    else:
        # Sum over n>=1 of 1/[n(n+1)] = 1, even if looks change.
        tail_budget = policy.alpha / (2 * n * (n+1))
    if binomial_tail(n, c, policy.threshold, upper=True) <= tail_budget:
        return "accept"
    if binomial_tail(n, c, policy.threshold, upper=False) <= tail_budget:
        return "reject"
    return "inconclusive"


@dataclass(frozen=True)
class Scenario:
    kind: str  # iid, beta_mixture, change_point
    p: Fraction
    concentration: Fraction = Fraction(2)
    after: Fraction = Fraction(1, 2)
    change_at: int = 8

    def __post_init__(self) -> None:
        if self.kind not in {"iid", "beta_mixture", "change_point"}:
            raise ValueError("unknown scenario")
        if not 0 <= self.p <= 1 or not 0 <= self.after <= 1:
            raise ValueError("invalid success probability")
        if self.kind == "beta_mixture" and not (0 < self.p < 1 and self.concentration > 0):
            raise ValueError("beta_mixture requires a proper Beta prior")
        if self.concentration <= 0 or self.change_at < 1:
            raise ValueError("invalid scenario parameter")

    def next_p(self, n: int, c: int) -> Fraction:
        if self.kind == "beta_mixture":
            # Draw one latent item probability from Beta(a,b), then repeat
            # conditionally iid. This predictive recursion is exact.
            return (self.p * self.concentration + c) / (self.concentration + n)
        if self.kind == "change_point" and n >= self.change_at:
            return self.after
        return self.p


class OperatingCharacteristics(TypedDict):
    accept: Fraction
    reject: Fraction
    inconclusive: Fraction
    expected_n: Fraction
    stopping: dict[int, Fraction]
    interval_coverage: Fraction | None


def operating_characteristics(policy: Policy, scenario: Scenario) -> OperatingCharacteristics:
    """Exact probability-mass DP; a capped continuation is inconclusive."""
    states = {0: Fraction(1)}
    outcomes = {"accept": Fraction(), "reject": Fraction(),
                "inconclusive": Fraction()}
    stopped: dict[int, Fraction] = {}
    covered = Fraction()
    for n in range(1, policy.cap + 1):
        following: dict[int, Fraction] = {}
        for c, mass in states.items():
            p = scenario.next_p(n - 1, c)
            following[c] = following.get(c, Fraction()) + mass * (1-p)
            following[c+1] = following.get(c+1, Fraction()) + mass * p
        states = {}
        for c, mass in following.items():
            verdict = decision(policy, n, c) if n in policy.looks else "inconclusive"
            if verdict == "inconclusive" and n < policy.cap:
                states[c] = states.get(c, Fraction()) + mass
            else:
                outcomes[verdict] += mass
                stopped[n] = stopped.get(n, Fraction()) + mass
                if policy.kind in {"naive", "group", "sequence"} and scenario.kind == "iid":
                    budget = (policy.alpha / 2 if policy.kind == "naive"
                              else policy.alpha / (2*len(policy.looks)) if policy.kind == "group"
                              else policy.alpha / (2*n*(n+1)))
                    if (binomial_tail(n, c, scenario.p, upper=True) > budget
                            and binomial_tail(n, c, scenario.p, upper=False) > budget):
                        covered += mass
    assert sum(outcomes.values()) == 1
    expected = sum((n * mass for n, mass in stopped.items()), Fraction())
    return {
        "accept": outcomes["accept"], "reject": outcomes["reject"],
        "inconclusive": outcomes["inconclusive"],
        "expected_n": expected, "stopping": stopped,
        "interval_coverage": covered if (policy.kind in {"naive", "group", "sequence"}
                                         and scenario.kind == "iid") else None,
    }


def reference_error(p: Fraction, threshold: Fraction, n: int) -> dict[str, Fraction]:
    """Finite independent reference arm; ties remain inconclusive."""
    if not 0 <= p <= 1 or not 0 < threshold < 1 or n < 1:
        raise ValueError("invalid reference")
    probs = {"accept": Fraction(), "reject": Fraction(), "inconclusive": Fraction()}
    for c in range(n+1):
        verdict = "accept" if Fraction(c, n) > threshold else (
            "reject" if Fraction(c, n) < threshold else "inconclusive"
        )
        probs[verdict] += comb(n, c) * p**c * (1-p)**(n-c)
    return probs


def task_variance(probabilities: tuple[Fraction, ...], repetitions: int) -> dict[str, Fraction]:
    """Macro variance for T iid sampled tasks from a discrete p population.

    Between-task term is Var(p_i); within-task term is mean[p_i(1-p_i)]/k.
    Macro variance is their sum divided by T. For a fixed, fully enumerated
    task set, the between-task component is descriptive heterogeneity, not
    repeated-generation uncertainty conditional on those exact tasks.
    """
    if not probabilities or repetitions < 1 or any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid task design")
    t = len(probabilities)
    mean = sum(probabilities, Fraction()) / t
    between = sum(((p-mean)**2 for p in probabilities), Fraction()) / t
    within = sum((p*(1-p) for p in probabilities), Fraction()) / (t*repetitions)
    return {"between": between, "within": within, "macro": (between+within)/t}


def reference_comparison(policy: Policy, p: Fraction, reference_n: int) -> dict[str, Fraction]:
    """Independent policy/reference disagreement split by known synthetic truth.

    The reference and policy streams are independent conditional on p.
    Ties/capped outcomes remain inconclusive, not silently scored as errors.
    """
    if p == policy.threshold:
        raise ValueError("truth is undefined exactly at the decision threshold")
    result = operating_characteristics(policy, Scenario("iid", p))
    reference = reference_error(p, policy.threshold, reference_n)
    if p > policy.threshold:
        policy_correct, policy_wrong = result["accept"], result["reject"]
        reference_correct, reference_wrong = reference["accept"], reference["reject"]
    else:
        policy_correct, policy_wrong = result["reject"], result["accept"]
        reference_correct, reference_wrong = reference["reject"], reference["accept"]
    return {
        "apparent_disagreement": (result["accept"]*reference["reject"]
                                  + result["reject"]*reference["accept"]),
        "reference_wrong_policy_correct": policy_correct * reference_wrong,
        "policy_wrong_reference_correct": policy_wrong * reference_correct,
        "policy_inconclusive": result["inconclusive"],
        "reference_inconclusive": reference["inconclusive"],
    }
