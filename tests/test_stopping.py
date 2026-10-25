import copy
import hashlib
import itertools
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

from llm_evaluation.stopping import (
    Policy, Scenario, decision, operating_characteristics, ratio,
    reference_error, reference_comparison, task_variance,
)
from llm_evaluation import stopping_records


def enumerate_paths(policy, scenario):
    totals = {"accept": F(), "reject": F(), "inconclusive": F(),
              "expected_n": F()}
    for bits in itertools.product((0, 1), repeat=policy.cap):
        weight = F(1)
        successes = 0
        for n, bit in enumerate(bits, start=1):
            p = scenario.next_p(n-1, successes)
            weight *= p if bit else 1-p
            successes += bit
            if n in policy.looks:
                verdict = decision(policy, n, successes)
                if verdict != "inconclusive" or n == policy.cap:
                    # Each prefix occurs 2^(cap-n) times in this full enumeration.
                    weight /= 2 ** (policy.cap - n)
                    totals[verdict] += weight
                    totals["expected_n"] += weight*n
                    break
    return totals


class ExactStoppingTests(unittest.TestCase):
    def test_fixed_analytic_and_exhaustive_independent_oracle(self):
        self.assertEqual(operating_characteristics(
            Policy("fixed", (1,)), Scenario("iid", F(1, 2)))["accept"], F(1, 2))
        for kind in ("fixed", "naive", "group", "sequence", "sprt"):
            policy = Policy(kind, (4,) if kind == "fixed" else (2, 4))
            for scenario in (Scenario("iid", F(1, 3)),
                             Scenario("beta_mixture", F(1, 3)),
                             Scenario("change_point", F(1, 3), after=F(2, 3),
                                      change_at=2)):
                with self.subTest(kind=kind, scenario=scenario.kind):
                    exact = operating_characteristics(policy, scenario)
                    enumerated = enumerate_paths(policy, scenario)
                    for key in enumerated:
                        self.assertEqual(exact[key], enumerated[key])

    def test_optional_looks_negative_and_valid_rule(self):
        p = Scenario("iid", F(1, 2))
        looks = (2, 4, 8, 16)
        naive = operating_characteristics(Policy("naive", looks), p)
        group = operating_characteristics(Policy("group", looks), p)
        sequence = operating_characteristics(Policy("sequence", looks), p)
        self.assertGreater(naive["accept"] + naive["reject"], F(1, 10))
        self.assertLess(naive["interval_coverage"], F(9, 10))
        self.assertLessEqual(group["accept"] + group["reject"], F(1, 10))
        self.assertGreaterEqual(group["interval_coverage"], F(9, 10))
        self.assertLessEqual(sequence["accept"] + sequence["reject"], F(1, 10))
        for result in (naive, group, sequence):
            self.assertEqual(sum(result[key] for key in
                                 ("accept", "reject", "inconclusive")), 1)
            self.assertEqual(sum(result["stopping"].values()), 1)
            self.assertGreaterEqual(result["expected_n"], 2)
            self.assertLessEqual(result["expected_n"], 16)

    def test_sprt_endpoint_errors_and_cost_are_exact(self):
        policy = Policy("sprt", (2, 4, 8, 16))
        lower = operating_characteristics(policy, Scenario("iid", F(3, 10)))
        upper = operating_characteristics(policy, Scenario("iid", F(7, 10)))
        self.assertLessEqual(lower["accept"], policy.alpha)
        self.assertLessEqual(upper["reject"], policy.alpha)
        self.assertEqual(lower["expected_n"], upper["expected_n"])
        self.assertLess(lower["expected_n"], 16)
        self.assertGreater(lower["inconclusive"], 0)

    def test_dependence_drift_and_finite_reference(self):
        policy = Policy("sprt", (2, 4, 8, 16))
        iid = operating_characteristics(policy, Scenario("iid", F(3, 10)))
        mixture = operating_characteristics(policy, Scenario("beta_mixture", F(3, 10)))
        drift = operating_characteristics(policy, Scenario(
            "change_point", F(3, 10), after=F(7, 10), change_at=8))
        self.assertNotEqual(iid["accept"], mixture["accept"])
        self.assertNotEqual(iid["accept"], drift["accept"])
        # A finite reference is a noisy comparator, not the known p.
        near = [reference_error(F(11, 20), F(1, 2), n)["reject"]
                for n in (16, 64, 256)]
        self.assertGreater(near[0], near[1])
        self.assertGreater(near[1], near[2])
        self.assertGreater(near[2], 0)
        joint = reference_comparison(Policy("fixed", (16,)), F(11, 20), 16)
        self.assertEqual(joint["apparent_disagreement"],
                         joint["reference_wrong_policy_correct"]
                         + joint["policy_wrong_reference_correct"])
        self.assertGreater(joint["reference_wrong_policy_correct"], 0)
        v1 = task_variance((F(1, 5), F(1, 2), F(4, 5)), 1)
        v16 = task_variance((F(1, 5), F(1, 2), F(4, 5)), 16)
        self.assertEqual(v1["between"], v16["between"])
        self.assertLess(v16["within"], v1["within"])

    def test_record_exact_reproduction_schema_and_exclusive_publication(self):
        a = stopping_records.analyze()
        b = stopping_records.analyze()
        self.assertEqual(a["results"], b["results"])
        self.assertEqual(a["config"], b["config"])
        self.assertEqual(a["results"], stopping_records.compute(a["config"]))
        self.assertEqual(stopping_records.validate(a, finalized=False), a)
        broken = copy.deepcopy(a)
        broken["results"]["development"][0]["accept"] = 0.1
        with self.assertRaises(ValueError):
            stopping_records.validate(broken, finalized=False)
        broken = copy.deepcopy(a)
        broken["config"]["fixed_counts"][0] = True
        with self.assertRaises(ValueError):
            stopping_records.validate(broken, finalized=False)
        a["software_commit"] = "a" * 40
        a["package_source_sha256"] = "b" * 64
        self.assertEqual(stopping_records.validate(a), a)
        with tempfile.TemporaryDirectory() as directory, patch(
            "llm_evaluation.stopping_records.software_identity",
            return_value=("a" * 40, "b" * 64),
        ):
            path = Path(directory).resolve() / "sim.json"
            stopping_records.finalize(path, a)
            self.assertEqual(stopping_records.read_finalized(path), a)
            self.assertEqual(path.read_bytes(), stopping_records.canonical_bytes(a))
            with self.assertRaises(FileExistsError):
                stopping_records.finalize(path, a)
            path.unlink()
            linked = Path(directory).resolve() / "alias"
            linked.symlink_to(Path(directory).resolve(), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                stopping_records.finalize(linked / "sim.json", a)


if __name__ == "__main__":
    unittest.main()
