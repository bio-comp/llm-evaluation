"""Finite-sample repeated-trial estimators, independent of benchmark runners."""

from math import comb


def _validate(n: int, c: int, k: int) -> None:
    if any(type(x) is not int for x in (n, c, k)):
        raise ValueError("n, c, k must be integers")
    if not (n >= 1 and 0 <= c <= n and 1 <= k <= n):
        raise ValueError("require n>=1, 0<=c<=n, and 1<=k<=n")


def marginal(n: int, c: int) -> float:
    _validate(n, c, 1)
    return c / n


def best_of_k(n: int, c: int, k: int) -> float:
    """Fraction of observed size-k subsets with at least one success."""
    _validate(n, c, k)
    return 1 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)


def all_k(n: int, c: int, k: int) -> float:
    """Fraction of observed size-k subsets with every trial succeeding."""
    _validate(n, c, k)
    return comb(c, k) / comb(n, k) if c >= k else 0.0
