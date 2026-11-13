from __future__ import annotations


import pytest

from llm_evaluation import composition as c


def test_baseline() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(800, 200),
        stratum_errors=(40, 40),
        target_weights=(0.8, 0.2),
    )
    assert result.marginal_rate == pytest.approx(0.08)
    assert result.standardized_rate == pytest.approx(0.08)
    assert result.status == "ESTIMABLE"
    assert result.missing_target_strata == ()


def test_pure_composition_shift() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(400, 600),
        stratum_errors=(20, 120),
        target_weights=(0.8, 0.2),
    )
    assert result.marginal_rate == pytest.approx(0.14)
    assert result.standardized_rate == pytest.approx(0.08)
    assert result.status == "ESTIMABLE"


def test_process_only() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(800, 200),
        stratum_errors=(80, 50),
        target_weights=(0.8, 0.2),
    )
    assert result.marginal_rate == pytest.approx(0.13)
    assert result.standardized_rate == pytest.approx(0.13)
    assert result.status == "ESTIMABLE"


def test_combined_shift_and_degradation() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(400, 600),
        stratum_errors=(40, 150),
        target_weights=(0.8, 0.2),
    )
    assert result.marginal_rate == pytest.approx(0.19)
    assert result.standardized_rate == pytest.approx(0.13)
    assert result.status == "ESTIMABLE"


def test_cancellation() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(500, 500),
        stratum_errors=(0, 80),
        target_weights=(0.8, 0.2),
    )
    assert result.marginal_rate == pytest.approx(0.08)
    assert result.standardized_rate == pytest.approx(0.032)
    assert result.status == "ESTIMABLE"


def test_absent_positive_weight_stratum() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(100, 0),
        stratum_errors=(10, 0),
        target_weights=(0.8, 0.2),
    )
    assert result.marginal_rate == pytest.approx(0.10)
    assert result.standardized_rate is None
    assert result.status == "NOT_ESTIMABLE"
    assert result.missing_target_strata == (1,)


def test_absent_zero_weight_stratum() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(100, 0),
        stratum_errors=(10, 0),
        target_weights=(1.0, 0.0),
    )
    assert result.marginal_rate == pytest.approx(0.10)
    assert result.standardized_rate == pytest.approx(0.10)
    assert result.status == "ESTIMABLE"
    assert result.missing_target_strata == ()


def test_observed_mix_exact_ratios() -> None:
    r1 = c.descriptive_composition_rates(
        stratum_counts=(800, 200),
        stratum_errors=(40, 40),
        target_weights=(0.8, 0.2),
    )
    assert r1.observed_mix == (0.8, 0.2)

    r2 = c.descriptive_composition_rates(
        stratum_counts=(400, 600),
        stratum_errors=(20, 120),
        target_weights=(0.8, 0.2),
    )
    assert r2.observed_mix == (0.4, 0.6)


def test_one_stratum_valid() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(10,),
        stratum_errors=(3,),
        target_weights=(1.0,),
    )
    assert result.marginal_rate == pytest.approx(0.3)
    assert result.standardized_rate == pytest.approx(0.3)
    assert result.status == "ESTIMABLE"
    assert result.observed_mix == (1.0,)
    assert result.missing_target_strata == ()


def test_reject_empty_inputs() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(),
            stratum_errors=(),
            target_weights=(),
        )


def test_reject_length_mismatch() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(1, 2),
            stratum_errors=(1,),
            target_weights=(0.5, 0.5),
        )


def test_reject_all_zero_counts() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(0, 0),
            stratum_errors=(0, 0),
            target_weights=(0.5, 0.5),
        )


def test_reject_negative_count() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(-1, 2),
            stratum_errors=(0, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_negative_error() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(-1, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_errors_greater_than_counts() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(3, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_bool_counts() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(True, 2),
            stratum_errors=(0, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_float_counts() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(1.0, 2),
            stratum_errors=(0, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_bool_errors() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(True, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_float_errors() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1.0, 1),
            target_weights=(0.5, 0.5),
        )


def test_reject_bool_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=(True, False),
        )


def test_reject_nan_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=(float("nan"), 0.5),
        )


def test_reject_inf_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=(float("inf"), float("-inf")),
        )


def test_reject_negative_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=(-0.1, 1.1),
        )


def test_reject_nonunit_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=(0.5, 0.6),
        )


def test_reject_non_numeric_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=("a", "b"),  # type: ignore[arg-type]
        )


def test_missing_stratum_no_divide_by_zero() -> None:
    result = c.descriptive_composition_rates(
        stratum_counts=(100, 0),
        stratum_errors=(10, 0),
        target_weights=(0.8, 0.2),
    )
    assert result.standardized_rate is None


def test_reject_string_weights() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=("0.5", "0.5"),  # type: ignore[arg-type]
        )


def test_reject_huge_int_weight() -> None:
    with pytest.raises(ValueError):
        c.descriptive_composition_rates(
            stratum_counts=(2, 2),
            stratum_errors=(1, 1),
            target_weights=((1 << 5000), 0.0),  # type: ignore[arg-type]
        )
