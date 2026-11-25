"""Trusted Python source references only; never execute model-provided code."""

from __future__ import annotations

from statistics import median as library_median


def median_sorted(xs: list[int]) -> float:
    ordered = sorted(xs)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def median_library(xs: list[int]) -> float:
    return float(library_median(xs))


def median_bug_even(xs: list[int]) -> float:
    return float(sorted(xs)[len(xs) // 2])


def clamp_minmax(x: int, low: int, high: int) -> int:
    return min(max(x, low), high)


def clamp_branch(x: int, low: int, high: int) -> int:
    if x < low:
        return low
    if x > high:
        return high
    return x


def clamp_bug(x: int, low: int, high: int) -> int:
    return min(max(x, low), high - 1)


def test_median_references() -> None:
    for xs, truth in [
        ([1, 2, 3], 2.0),
        ([1, 2, 3, 4], 2.5),
        ([-5, -1], -3.0),
        ([4, 4], 4.0),
    ]:
        assert median_sorted(xs) == truth
        assert median_library(xs) == truth
    assert median_bug_even([1, 2, 3, 4]) != 2.5


def test_clamp_references() -> None:
    for x, low, high, truth in [
        (10, 0, 10, 10),
        (-3, 0, 10, 0),
        (15, 0, 10, 10),
        (5, 0, 10, 5),
    ]:
        assert clamp_minmax(x, low, high) == truth
        assert clamp_branch(x, low, high) == truth
    assert clamp_bug(10, 0, 10) != 10
