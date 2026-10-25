import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from llm_evaluation.metrics import all_k, best_of_k, marginal
from llm_evaluation.records import analyze, canonical_bytes, digest, load_fixture, validate_fixture, validate_result, write_immutable

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/e002_synthetic.json"


class EstimatorTests(unittest.TestCase):
    def test_known_values_and_edges(self):
        self.assertEqual(best_of_k(10, 8, 3), 1.0)
        self.assertAlmostEqual(all_k(10, 8, 3), 56 / 120)
        self.assertEqual(marginal(10, 8), 0.8)
        self.assertEqual(best_of_k(4, 0, 4), 0)
        self.assertEqual(all_k(4, 4, 4), 1)
        self.assertEqual(best_of_k(4, 2, 1), all_k(4, 2, 1))

    def test_exhaustive_subset_oracle(self):
        for n in range(1, 7):
            for outcomes in itertools.product((False, True), repeat=n):
                c = sum(outcomes)
                for k in range(1, n + 1):
                    subsets = list(itertools.combinations(outcomes, k))
                    self.assertAlmostEqual(best_of_k(n, c, k),
                                           sum(any(s) for s in subsets) / len(subsets))
                    self.assertAlmostEqual(all_k(n, c, k),
                                           sum(all(s) for s in subsets) / len(subsets))

    def test_iid_exact_enumeration_and_bad_plugin(self):
        # Enumerate all equally likely n=4 Bernoulli(1/2) outcome vectors.
        vectors = list(itertools.product((False, True), repeat=4))
        self.assertAlmostEqual(sum(best_of_k(4, sum(v), 2) for v in vectors) / 16, 0.75)
        self.assertAlmostEqual(sum(all_k(4, sum(v), 2) for v in vectors) / 16, 0.25)
        plugin_best = sum(1 - (1 - sum(v) / 4) ** 2 for v in vectors) / 16
        plugin_all = sum((sum(v) / 4) ** 2 for v in vectors) / 16
        self.assertLess(plugin_best, 0.75)
        self.assertGreater(plugin_all, 0.25)

    def test_dependent_and_nonstationary_future_events(self):
        # Perfectly shared Bernoulli latent variable: P(all two)=P(any two)=1/2.
        scenarios = [(False, False), (True, True)]
        self.assertEqual(sum(all(s) for s in scenarios) / 2, 0.5)
        self.assertEqual(sum(any(s) for s in scenarios) / 2, 0.5)
        self.assertNotEqual(0.5, (0.5) ** 2)
        # Alternating deterministic trial positions: marginal 1/2, all-two 0.
        self.assertEqual(sum((True, False)) / 2, 0.5)
        self.assertEqual(all((True, False)), False)

    def test_reject_invalid_counts(self):
        for args in [(0, 0, 1), (2, -1, 1), (2, 3, 1), (2, 1, 0), (2, 1, 3),
                     (True, 1, 1), (2, 1.0, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                best_of_k(*args)


class RecordTests(unittest.TestCase):
    def test_public_fixture_result_and_macro_weighting(self):
        fixture = load_fixture(FIXTURE)
        self.assertEqual(digest(fixture), "dfa517c2c76ad0c6cebacb157e67fdb3b177d0a1bcf6526096b8c80f2a3772a2")
        result = analyze(fixture, 2)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["per_task"][0]["best_of_k"], 5 / 6)
        self.assertEqual(result["per_task"][0]["all_k"], 1 / 6)
        self.assertAlmostEqual(result["macro"]["best_of_k"], 11 / 18)
        self.assertAlmostEqual(result["macro"]["all_k"], 7 / 18)
        self.assertEqual(canonical_bytes(json.loads(canonical_bytes(result))), canonical_bytes(result))

    def test_strict_boundary_and_immutable_record(self):
        fixture = load_fixture(FIXTURE)
        for mutation in (lambda f: f.update(token="private"),
                         lambda f: f["tasks"][0]["outcomes"].append(1),
                         lambda f: f["tasks"][0].update(role="private"),
                         lambda f: f["condition"].pop("oracle"),
                         lambda f: f.update(schema_version=2)):
            broken = copy.deepcopy(fixture)
            mutation(broken)
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_fixture(broken)
        for mutation in (lambda r: r["per_task"][0].update(all_k=0.9),
                         lambda r: r["macro"].update(extra=1),
                         lambda r: r.update(software_commit="unknown")):
            broken = copy.deepcopy(analyze(fixture, 2))
            mutation(broken)
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_result(broken)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            result = analyze(fixture, 2)
            write_immutable(path, result)
            self.assertEqual(json.loads(path.read_text()), result)
            with self.assertRaises(FileExistsError):
                write_immutable(path, result)


if __name__ == "__main__":
    unittest.main()
