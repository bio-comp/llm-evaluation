import unittest
from fractions import Fraction as F

from llm_evaluation.stopping import Policy, Scenario
from llm_evaluation.stopping_calibration import (
    calibrate_finite_count_boundaries,
    calibrate_lr_grid,
    classify_candidate,
    composite_endpoint_guarantees,
    finite_count_path_monotonicity,
    finite_path_monotonicity,
    pareto_efficient,
    reference_sample_size,
)


class StoppingCalibrationTests(unittest.TestCase):
    def test_sprt_requires_threshold_between_endpoints(self):
        with self.assertRaisesRegex(ValueError, "low < threshold < high"):
            Policy("sprt", (2, 4), threshold=F(1, 2), low=F(3, 5), high=F(4, 5))
        with self.assertRaisesRegex(ValueError, "low < threshold < high"):
            Policy("sprt", (2, 4), threshold=F(1, 2), low=F(1, 5), high=F(2, 5))

    def test_beta_mixture_requires_proper_beta_prior(self):
        for mean in (F(0), F(1)):
            with self.subTest(mean=mean), self.assertRaisesRegex(ValueError, "proper Beta"):
                Scenario("beta_mixture", mean, concentration=F(2))
        with self.assertRaisesRegex(ValueError, "proper Beta"):
            Scenario("beta_mixture", F(1, 2), concentration=F(0))

    def test_finite_path_events_are_monotone_for_supported_policies(self):
        for kind in ("fixed", "naive", "group", "sequence", "sprt"):
            policy = Policy(kind, (4,) if kind == "fixed" else (2, 4, 8, 16))
            with self.subTest(kind=kind):
                proof = finite_path_monotonicity(policy)
                self.assertTrue(proof.accept_is_increasing)
                self.assertTrue(proof.reject_is_decreasing)
                self.assertEqual(proof.paths_checked, 2 ** policy.cap)

    def test_selected_finite_count_boundary_terminal_events_are_monotone(self):
        proof = finite_count_path_monotonicity(
            looks=(2, 4, 8, 16),
            accept_min=(3, 4, 6, 9),
            reject_max=(-1, 0, 2, 7),
        )
        self.assertTrue(proof.accept_is_increasing)
        self.assertTrue(proof.reject_is_decreasing)
        self.assertEqual(proof.paths_checked, 2 ** 16)

    def test_composite_iid_guarantees_reduce_to_boundary_points(self):
        policy = Policy("sprt", (2, 4, 8, 16))
        result = composite_endpoint_guarantees(policy, p0=F(3, 10), p1=F(7, 10))
        self.assertEqual(result["false_accept_supremum_at"], F(3, 10))
        self.assertEqual(result["false_reject_supremum_at"], F(7, 10))
        self.assertLessEqual(result["false_accept"], F(1, 10))
        self.assertLessEqual(result["false_reject"], F(1, 10))

    def test_reference_size_is_exact_and_tie_rule_is_explicit(self):
        inconclusive = reference_sample_size(
            threshold=F(1, 2), delta=F(1, 20), epsilon=F(1, 20), tie_handling="inconclusive"
        )
        wrong = reference_sample_size(
            threshold=F(1, 2), delta=F(1, 20), epsilon=F(1, 20), tie_handling="wrong"
        )
        half = reference_sample_size(
            threshold=F(1, 2), delta=F(1, 20), epsilon=F(1, 20), tie_handling="half"
        )
        self.assertLessEqual(inconclusive["worst_wrong"], F(1, 20))
        self.assertLessEqual(half["worst_wrong"], F(1, 20))
        self.assertLessEqual(wrong["worst_wrong"], F(1, 20))
        self.assertLessEqual(inconclusive["n"], half["n"])
        self.assertLessEqual(half["n"], wrong["n"])
        for result in (inconclusive, half, wrong):
            if result["n"] > 1:
                self.assertGreater(result["previous_worst_wrong"], F(1, 20))

    def test_matched_error_classification_and_pareto_do_not_reward_abstention_silently(self):
        feasible_fast = {
            "name": "fast", "false_accept": F(1, 100), "false_reject": F(1, 100),
            "inconclusive": F(1, 5), "expected_n": F(8),
        }
        feasible_slow = {
            "name": "slow", "false_accept": F(1, 100), "false_reject": F(1, 100),
            "inconclusive": F(1, 10), "expected_n": F(12),
        }
        infeasible = {
            "name": "bad", "false_accept": F(1, 10), "false_reject": F(1, 100),
            "inconclusive": F(0), "expected_n": F(4),
        }
        self.assertEqual(classify_candidate(feasible_fast, F(1, 20), F(1, 20)), "feasible")
        self.assertEqual(classify_candidate(infeasible, F(1, 20), F(1, 20)), "infeasible")
        frontier = pareto_efficient([feasible_fast, feasible_slow, infeasible], F(1, 20), F(1, 20))
        self.assertEqual({row["name"] for row in frontier}, {"fast", "slow"})

    def test_pareto_dominance_does_not_average_away_endpoint_tradeoffs(self):
        better_on_average_but_worse_at_p0 = {
            "name": "left", "false_accept": F(1, 100), "false_reject": F(1, 100),
            "inconclusive_p0": F(3, 10), "inconclusive_p1": F(0),
            "expected_n_p0": F(12), "expected_n_p1": F(4),
            "inconclusive": F(3, 20), "expected_n": F(8),
        }
        balanced = {
            "name": "right", "false_accept": F(1, 100), "false_reject": F(1, 100),
            "inconclusive_p0": F(1, 5), "inconclusive_p1": F(1, 5),
            "expected_n_p0": F(10), "expected_n_p1": F(10),
            "inconclusive": F(1, 5), "expected_n": F(10),
        }
        frontier = pareto_efficient(
            [better_on_average_but_worse_at_p0, balanced], F(1, 20), F(1, 20)
        )
        self.assertEqual({row["name"] for row in frontier}, {"left", "right"})

    def test_finite_lr_grid_uses_realized_endpoint_errors_and_reports_abstention(self):
        rows = calibrate_lr_grid(
            looks=(2, 4, 8, 16), threshold=F(1, 2), p0=F(3, 10), p1=F(7, 10),
            reject_boundaries=(F(1, 20), F(1, 10), F(3, 20)),
            accept_boundaries=(F(20, 3), F(10), F(20)),
            epsilon_accept=F(1, 20), epsilon_reject=F(1, 20),
        )
        self.assertEqual(len(rows), 9)
        self.assertTrue(any(row["classification"] == "feasible" for row in rows))
        for row in rows:
            self.assertIn("inconclusive_p0", row)
            self.assertIn("inconclusive_p1", row)
            self.assertIn("expected_n_p0", row)
            self.assertIn("expected_n_p1", row)
            self.assertEqual(sum(row["stopping_p0"].values()), 1)
            self.assertEqual(sum(row["stopping_p1"].values()), 1)

    def test_small_finite_count_boundary_search_can_improve_realized_frontier(self):
        rows = calibrate_finite_count_boundaries(
            looks=(2, 4, 8, 16), threshold=F(1, 2), p0=F(3, 10), p1=F(7, 10),
            epsilon_accept=F(1, 20), epsilon_reject=F(1, 20), symmetric=True,
        )
        self.assertEqual(len(rows), 270)
        feasible = [row for row in rows if row["classification"] == "feasible"]
        self.assertTrue(feasible)
        for row in feasible:
            self.assertLessEqual(row["false_accept"], F(1, 20))
            self.assertLessEqual(row["false_reject"], F(1, 20))
            self.assertEqual(sum(row["stopping_p0"].values()), 1)
            self.assertEqual(sum(row["stopping_p1"].values()), 1)
        # Exact finite calibration can spend more of the allowed 5% error budget
        # to reduce abstention without increasing expected sampling cost.
        self.assertTrue(any(row["expected_n"] <= F(32186471, 3125000)
                            and row["inconclusive"] < F(166307112610317, 1250000000000000)
                            for row in feasible))


if __name__ == "__main__":
    unittest.main()
