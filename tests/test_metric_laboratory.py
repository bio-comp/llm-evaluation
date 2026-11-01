"""Tests for known-truth metric, calibration, and reference-error laboratory."""

import math
import pytest

from llm_evaluation.metric_laboratory import (
    KNOWN_P_VALUES,
    brier_score,
    expected_brier,
    log_score,
    expected_log_score,
    fixed_bin_ece,
    equal_mass_ece,
    debiased_ece,
    debiased_squared_l2_calibration_error,
    debiased_l2_calibration_error,
    binary_decision,
    population_ece,
    mean_prefix_risk,
    empirical_aurc_harmonic,
    compute_aurc,
    noisy_target_probability,
    noisy_expected_brier,
    check_reference_ranking_reversal,
    compute_cluster_vs_naive_se,
    compute_forecast,
    SimulationCellConfig,
    generate_replicate_dataset,
    evaluate_replicate,
    run_simulation_cell,
    run_all_deterministic_microprobes,
)


class TestScoringRules:
    def test_brier_score_basic(self):
        # Perfect forecasts
        assert brier_score([1.0, 0.0], [1, 0]) == 0.0
        # Worst forecasts
        assert brier_score([0.0, 1.0], [1, 0]) == 1.0
        # Uniform 0.5
        assert brier_score([0.5, 0.5], [1, 0]) == 0.25

    def test_expected_brier_analytic(self):
        # E_p[(q-Y)^2] = (q-p)^2 + p(1-p)
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            for q in (0.0, 0.2, 0.5, 0.8, 1.0):
                expected = (q - p) ** 2 + p * (1 - p)
                assert math.isclose(expected_brier(p, q), expected, rel_tol=1e-9)

    def test_log_score_and_expected_log_score(self):
        # q=0.5 on binary: -ln(0.5) = ln(2)
        assert math.isclose(log_score([0.5, 0.5], [1, 0]), math.log(2), rel_tol=1e-9)
        assert math.isclose(expected_log_score(0.5, 0.5), math.log(2), rel_tol=1e-9)
        # Clipping prevents math domain error at 0 and 1
        assert log_score([0.0], [1], eps=1e-12) > 0.0
        assert log_score([1.0], [0], eps=1e-12) > 0.0


class TestCalibrationEstimators:
    def test_coarse_bin_cancellation_probe_2(self):
        # Microprobe 2: Ten q=0.1, Y=0 plus ten q=0.9, Y=1
        forecasts = [0.1] * 10 + [0.9] * 10
        outcomes = [0] * 10 + [1] * 10

        # One bin: mean forecast = 0.5, mean outcome = 0.5 -> ECE = 0.0
        ece_1_bin, bins_1 = fixed_bin_ece(forecasts, outcomes, num_bins=1)
        assert math.isclose(ece_1_bin, 0.0, abs_tol=1e-9)

        # Two bins [0, 0.5) and [0.5, 1.0]:
        # Bin 0: conf=0.1, acc=0.0 -> gap=0.1
        # Bin 1: conf=0.9, acc=1.0 -> gap=0.1
        # Average ECE = 0.1
        ece_2_bin, bins_2 = fixed_bin_ece(forecasts, outcomes, num_bins=2)
        assert math.isclose(ece_2_bin, 0.1, rel_tol=1e-9)

    def test_uninformative_predictor_probe_3(self):
        # Microprobe 3: Constant q=0.5 on p=0.5
        # Population calibration is 0.0, but expected Brier is 0.25 (not 0.0)
        assert population_ece([0.5], [0.5]) == 0.0
        assert expected_brier(0.5, 0.5) == 0.25
        # Perfect informative predictor q=p has same calibration 0.0, but Brier 0.0 when p in {0, 1}
        assert expected_brier(1.0, 1.0) == 0.0

    def test_equal_mass_ece(self):
        forecasts = [0.05 * i for i in range(20)]
        outcomes = [1 if i % 2 == 0 else 0 for i in range(20)]
        ece, bins = equal_mass_ece(forecasts, outcomes, num_bins=4)
        assert len(bins) == 4
        assert all(b.count == 5 for b in bins)
        assert 0.0 <= ece <= 1.0

    def test_debiased_l2_matches_kumar_unbiased_squared_definition(self):
        forecasts = [0.5] * 4
        outcomes = [1, 1, 0, 0]
        # Kumar et al. Definition 5.2 subtracts the Bernoulli variance term from
        # the squared bin gap.  With mean label == mean forecast == 0.5, the
        # unbiased squared estimate is negative rather than silently truncated.
        expected_squared = -(0.5 * 0.5) / (4 - 1)
        squared = debiased_squared_l2_calibration_error(forecasts, outcomes, num_bins=1)
        assert math.isclose(squared, expected_squared, rel_tol=1e-12)
        assert debiased_l2_calibration_error(forecasts, outcomes, num_bins=1) == 0.0

    def test_debiased_l2_positive_case_matches_exact_formula(self):
        forecasts = [0.2] * 4
        outcomes = [1, 1, 1, 1]
        assert math.isclose(
            debiased_squared_l2_calibration_error(forecasts, outcomes, num_bins=1),
            0.64,
            rel_tol=1e-12,
        )
        assert math.isclose(
            debiased_l2_calibration_error(forecasts, outcomes, num_bins=1),
            0.8,
            rel_tol=1e-12,
        )

    def test_debiased_ece_is_explicit_seeded_l1_normal_bias_correction(self):
        forecasts = [0.2, 0.2, 0.8, 0.8]
        outcomes = [1, 1, 0, 0]
        # Zero within-bin Bernoulli variance makes the normal-resampling
        # correction reduce exactly to the L1 plugin ECE here.
        assert math.isclose(
            debiased_ece(forecasts, outcomes, num_bins=2, seed=17, resamples=50),
            0.8,
            rel_tol=1e-12,
        )
        variable_forecasts = [0.2] * 4 + [0.8] * 4
        variable_outcomes = [0, 1, 0, 1, 1, 0, 1, 0]
        first = debiased_ece(
            variable_forecasts, variable_outcomes, num_bins=2, seed=19, resamples=100
        )
        second = debiased_ece(
            variable_forecasts, variable_outcomes, num_bins=2, seed=19, resamples=100
        )
        assert first == second

    def test_equal_mass_binning_does_not_split_identical_forecast_ties(self):
        forecasts = [0.2] * 4 + [0.8] * 4
        outcomes = [0, 1, 0, 1, 0, 1, 0, 1]
        _, bins = equal_mass_ece(forecasts, outcomes, num_bins=4)
        nonempty = [b for b in bins if b.count]
        # A tied score value must belong to one bin, not be split solely to hit a
        # requested equal-count target.  Two unique scores therefore imply no
        # more than two non-empty bins.
        assert len(nonempty) <= 2
        assert sorted(b.count for b in nonempty) == [4, 4]


class TestAURCAndLE103Gate:
    def test_le103_two_item_oracle_untied(self):
        # In increasing confidence [.1, .9], losses [0, 1]:
        # Highest confidence is item with loss 1 (index 1, conf 0.9).
        # Lowest confidence is item with loss 0 (index 0, conf 0.1).
        # Sorted by decreasing confidence:
        # Prefix 1: [1] -> risk 1.0
        # Prefix 2: [1, 0] -> risk 0.5
        # Mean prefix risk: (1.0 + 0.5) / 2 = 0.75
        confidences = [0.1, 0.9]
        errors = [0, 1]
        assert mean_prefix_risk(confidences, errors) == 0.75
        assert empirical_aurc_harmonic(confidences, errors) == 0.75
        assert compute_aurc(confidences, errors, tie_convention="average") == 0.75

        # Losses [1, 1]:
        # Prefix 1: [1] -> risk 1.0
        # Prefix 2: [1, 1] -> risk 1.0
        # Mean prefix risk: 1.0
        errors_both = [1, 1]
        assert mean_prefix_risk(confidences, errors_both) == 1.0
        assert empirical_aurc_harmonic(confidences, errors_both) == 1.0
        assert compute_aurc(confidences, errors_both, tie_convention="average") == 1.0

    def test_all_four_binary_two_item_patterns(self):
        conf = [0.9, 0.1]
        # [0, 0] -> 0.0
        assert compute_aurc(conf, [0, 0]) == 0.0
        # [0, 1] -> prefix 1: 0/1, prefix 2: 1/2 -> mean = 0.25
        assert compute_aurc(conf, [0, 1]) == 0.25
        # [1, 0] -> prefix 1: 1/1, prefix 2: 1/2 -> mean = 0.75
        assert compute_aurc(conf, [1, 0]) == 0.75
        # [1, 1] -> prefix 1: 1/1, prefix 2: 2/2 -> mean = 1.0
        assert compute_aurc(conf, [1, 1]) == 1.0

    def test_harmonic_weight_equality_exhaustive_small_n(self):
        import itertools

        for n in range(2, 6):
            conf = [1.0 - (i / n) for i in range(n)]  # strictly decreasing, distinct
            for err in itertools.product((0, 1), repeat=n):
                prefix_val = mean_prefix_risk(conf, list(err))
                harmonic_val = empirical_aurc_harmonic(conf, list(err))
                assert math.isclose(prefix_val, harmonic_val, rel_tol=1e-9)

    def test_tie_conventions_and_point_mass(self):
        # Two items with identical score, one error and one correct
        conf = [0.5, 0.5]
        errors = [0, 1]

        opt = compute_aurc(conf, errors, tie_convention="optimistic")
        pess = compute_aurc(conf, errors, tie_convention="pessimistic")
        avg = compute_aurc(conf, errors, tie_convention="average")

        assert opt == 0.25
        assert pess == 0.75
        assert avg == 0.50
        assert opt < avg < pess


class TestReferenceNoiseAndRankingReversals:
    def test_noisy_helpers(self):
        p_noisy = noisy_target_probability(0.8, 0.1)
        assert math.isclose(p_noisy, 0.1 + 0.8 * 0.8, rel_tol=1e-9)
        exp_b = noisy_expected_brier(0.8, 0.74, 0.1)
        assert exp_b > 0.0

    def test_probe_4_reference_noise_ranking_reversal(self):
        # Clean p=0.9, eta=0.2
        # Noisy target = 0.2 + 0.6 * 0.9 = 0.74
        # Forecasts q1=0.9, q2=0.74
        res = check_reference_ranking_reversal(
            clean_p=0.9, q_clean=0.9, q_noisy=0.74, eta=0.2
        )
        assert res["reversal_detected"] is True
        assert math.isclose(res["clean_brier_q_clean"], 0.0900, rel_tol=1e-4)
        assert math.isclose(res["clean_brier_q_noisy"], 0.1156, rel_tol=1e-4)
        assert math.isclose(res["noisy_brier_q_clean"], 0.2180, rel_tol=1e-4)
        assert math.isclose(res["noisy_brier_q_noisy"], 0.1924, rel_tol=1e-4)
        # On clean truth, q_clean (0.09) < q_noisy (0.1156)
        assert res["clean_brier_q_clean"] < res["clean_brier_q_noisy"]
        # On noisy target, q_noisy (0.1924) < q_clean (0.218) -> reversal!
        assert res["noisy_brier_q_noisy"] < res["noisy_brier_q_clean"]


class TestPseudoreplicationClusterSE:
    def test_probe_5_cluster_vs_naive_se(self):
        # 4 items [0, 0, 1, 1], each duplicated 25 times
        items = [[0] * 25, [0] * 25, [1] * 25, [1] * 25]
        res = compute_cluster_vs_naive_se(items)

        # Sample SE for 4 items: sqrt( (1/3) * (0.25*4) / 4 ) = sqrt(1/12) ~= 0.288675135
        expected_item_se = 1.0 / (2 * math.sqrt(3))
        # Naive row SE for 100 rows (50 zeros, 50 ones): sqrt( (25/99) / 100 ) ~= 0.050251891
        expected_naive_se = math.sqrt(25 / 99 / 100)

        assert math.isclose(res.item_cluster_se, expected_item_se, rel_tol=1e-6)
        assert math.isclose(res.naive_row_se, expected_naive_se, rel_tol=1e-6)
        assert res.se_ratio > 5.0  # Naive SE is severely deflated!


class TestDeterministicMicroprobesSuite:
    def test_all_deterministic_microprobes(self):
        probes = run_all_deterministic_microprobes()
        assert "probe_1_prevalence_precision" in probes
        assert "probe_2_coarse_bin_cancellation" in probes
        assert "probe_3_constant_uninformative_predictor" in probes
        assert "probe_4_reference_noise_ranking_reversal" in probes
        assert "probe_5_pseudoreplication_cluster_se" in probes

        # Probe 1 checks
        p1 = probes["probe_1_prevalence_precision"]
        assert math.isclose(p1["precision_prev_50"], 90 / 91, rel_tol=1e-6)
        assert math.isclose(p1["precision_prev_01"], 10 / 21, rel_tol=1e-6)


class TestCandidateSimulationGrid:
    def test_evaluate_replicate(self):
        config = SimulationCellConfig(
            p_values=KNOWN_P_VALUES,
            n_base_items=14,
            k_within_item=2,
            shared_within_item=True,
            reference_flip_eta=0.1,
            forecast_kind="overconfident",
        )
        rep = evaluate_replicate(config, seed=123)
        assert 0.0 <= rep.brier_score <= 1.0
        assert rep.log_score >= 0.0
        assert 0.0 <= rep.fixed_bin_ece <= 1.0
        assert 0.0 <= rep.equal_mass_ece <= 1.0
        assert 0.0 <= rep.aurc <= 1.0
        assert rep.item_cluster_se >= 0.0
        assert rep.naive_row_se >= 0.0

    def test_binary_decision_uses_explicit_positive_tie_at_half(self):
        assert binary_decision(0.499999) == 0
        assert binary_decision(0.5) == 1
        assert binary_decision(0.500001) == 1
        assert binary_decision(0.5, tie_break="negative") == 0

    def test_forecast_generation_kinds(self):
        p = 0.7
        assert compute_forecast(p, "oracle") == 0.7
        assert compute_forecast(p, "constant") == 0.5
        # overconfident: clip(0.5 + 1.5*(0.7 - 0.5), 0.001, 0.999) = 0.8
        assert math.isclose(compute_forecast(p, "overconfident"), 0.8, rel_tol=1e-9)
        # underconfident: 0.2 + 0.6 * 0.7 = 0.62
        assert math.isclose(compute_forecast(p, "underconfident"), 0.62, rel_tol=1e-9)
        with pytest.raises(ValueError):
            compute_forecast(p, "unknown_kind")

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            brier_score([], [])
        with pytest.raises(ValueError):
            brier_score([0.5], [1, 0])
        with pytest.raises(ValueError):
            expected_brier(-0.1, 0.5)
        with pytest.raises(ValueError):
            log_score([], [])
        with pytest.raises(ValueError):
            expected_log_score(0.5, 1.5)
        with pytest.raises(ValueError):
            fixed_bin_ece([], [])
        with pytest.raises(ValueError):
            fixed_bin_ece([0.5], [1], num_bins=0)
        with pytest.raises(ValueError):
            equal_mass_ece([], [])
        with pytest.raises(ValueError):
            equal_mass_ece([0.5], [1], num_bins=0)
        with pytest.raises(ValueError):
            population_ece([], [])
        with pytest.raises(ValueError):
            population_ece([0.5], [0.5], weights=[1.0, 2.0])
        with pytest.raises(ValueError):
            mean_prefix_risk([], [])
        with pytest.raises(ValueError):
            empirical_aurc_harmonic([], [])
        with pytest.raises(ValueError):
            compute_aurc([], [])
        with pytest.raises(ValueError):
            compute_aurc([0.5], [1], tie_convention="invalid")  # type: ignore
        with pytest.raises(ValueError):
            noisy_target_probability(-0.1, 0.1)
        with pytest.raises(ValueError):
            noisy_target_probability(0.5, 0.6)
        with pytest.raises(ValueError):
            compute_cluster_vs_naive_se([[1]])  # < 2 items
        with pytest.raises(ValueError):
            compute_cluster_vs_naive_se([[1], []])  # empty item

    def test_replicate_dataset_generation_deterministic(self):
        config = SimulationCellConfig(
            p_values=(0.1, 0.5, 0.9),
            n_base_items=30,
            k_within_item=4,
            shared_within_item=False,
            reference_flip_eta=0.1,
            forecast_kind="oracle",
        )
        ds1 = generate_replicate_dataset(config, seed=42)
        ds2 = generate_replicate_dataset(config, seed=42)
        assert ds1 == ds2

        ds_diff = generate_replicate_dataset(config, seed=43)
        assert ds1 != ds_diff

    def test_simulation_cell_execution(self):
        config = SimulationCellConfig(
            p_values=(0.3, 0.7),
            n_base_items=20,
            k_within_item=2,
            shared_within_item=False,
            reference_flip_eta=0.0,
            forecast_kind="oracle",
        )
        summary = run_simulation_cell(config, num_replicates=10, seed_base=100)
        assert summary.replicates_count == 10
        assert 0.0 <= summary.mean_brier <= 0.5
        assert summary.mc_se_brier >= 0.0
        assert 0.0 <= summary.mean_ece <= 1.0
        assert 0.0 <= summary.mean_aurc <= 1.0
