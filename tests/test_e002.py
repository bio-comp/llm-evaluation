import copy
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_evaluation import cli, records
from llm_evaluation.metrics import all_k, best_of_k, marginal
from llm_evaluation.records import (
    analyze, canonical_bytes, canonical_sha256, load_fixture, parse_fixture,
    raw_sha256, read_finalized, validate_fixture, validate_result,
    verify_result_source, write_immutable,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/e002_synthetic.json"
COMMIT = "a" * 40
IMPLEMENTATION = "b" * 64


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
        vectors = list(itertools.product((False, True), repeat=4))
        self.assertAlmostEqual(sum(best_of_k(4, sum(v), 2) for v in vectors) / 16, 0.75)
        self.assertAlmostEqual(sum(all_k(4, sum(v), 2) for v in vectors) / 16, 0.25)
        plugin_best = sum(1 - (1 - sum(v) / 4) ** 2 for v in vectors) / 16
        plugin_all = sum((sum(v) / 4) ** 2 for v in vectors) / 16
        self.assertLess(plugin_best, 0.75)
        self.assertGreater(plugin_all, 0.25)

    def test_dependent_and_nonstationary_future_events(self):
        scenarios = [(False, False), (True, True)]
        self.assertEqual(sum(all(s) for s in scenarios) / 2, 0.5)
        self.assertEqual(sum(any(s) for s in scenarios) / 2, 0.5)
        self.assertNotEqual(0.5, (0.5) ** 2)
        self.assertEqual(sum((True, False)) / 2, 0.5)
        self.assertEqual(all((True, False)), False)

    def test_reject_invalid_counts(self):
        for args in [(0, 0, 1), (2, -1, 1), (2, 3, 1), (2, 1, 0), (2, 1, 3),
                     (True, 1, 1), (2, 1.0, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                best_of_k(*args)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.source = FIXTURE.read_bytes()
        self.fixture = parse_fixture(self.source)

    def finalized(self):
        result = analyze(self.fixture, 2, source_bytes=self.source)
        result["software_commit"] = COMMIT
        result["implementation_sha256"] = IMPLEMENTATION
        return result

    def test_public_fixture_result_and_macro_weighting(self):
        self.assertEqual(canonical_sha256(self.fixture),
                         "dfa517c2c76ad0c6cebacb157e67fdb3b177d0a1bcf6526096b8c80f2a3772a2")
        result = self.finalized()
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(result["per_task"][0]["best_of_k"], 5 / 6)
        self.assertEqual(result["per_task"][0]["all_k"], 1 / 6)
        self.assertAlmostEqual(result["macro"]["best_of_k"], 11 / 18)
        self.assertAlmostEqual(result["macro"]["all_k"], 7 / 18)
        self.assertEqual(canonical_bytes(json.loads(canonical_bytes(result))), canonical_bytes(result))

    def test_finalized_writer_rejects_missing_software_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "record.json"
            preview = analyze(self.fixture, 2, source_bytes=self.source)
            with self.assertRaisesRegex(ValueError, "software commit"):
                write_immutable(path, preview, self.source)
            self.assertFalse(path.exists())
            preview["software_commit"] = COMMIT
            preview["implementation_sha256"] = IMPLEMENTATION
            preview["fixture_raw_sha256"] = None
            with self.assertRaisesRegex(ValueError, "raw source"):
                write_immutable(path, preview, self.source)
            self.assertFalse(path.exists())

    def test_mutating_input_does_not_mutate_output(self):
        fixture = copy.deepcopy(self.fixture)
        result = analyze(fixture, 2, source_bytes=self.source)
        before = canonical_bytes(result)
        fixture["condition"]["subject"] = "mutated-after-analysis"
        fixture["tasks"][0]["outcomes"][0] = False
        self.assertEqual(canonical_bytes(result), before)
        result["software_commit"] = COMMIT
        result["implementation_sha256"] = IMPLEMENTATION
        self.assertIs(verify_result_source(result, self.source), result)

    def test_source_binding_and_hash_contract(self):
        result = self.finalized()
        self.assertIs(verify_result_source(result, self.source), result)
        modified = self.source.replace(b"synthetic-binary-generator-v1",
                                       b"synthetic-binary-generator-v2")
        with self.assertRaisesRegex(ValueError, "raw source"):
            verify_result_source(result, modified)
        reformatted = canonical_bytes(self.fixture)
        self.assertEqual(canonical_sha256(parse_fixture(reformatted)),
                         result["fixture_canonical_sha256"])
        self.assertNotEqual(raw_sha256(reformatted), raw_sha256(self.source))
        with self.assertRaisesRegex(ValueError, "raw source"):
            verify_result_source(result, reformatted)
        copied = copy.deepcopy(result)
        copied["condition"]["subject"] = "changed-result-condition"
        with self.assertRaisesRegex(ValueError, "source/result"):
            verify_result_source(copied, self.source)
        copied = copy.deepcopy(result)
        copied["fixture_canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "canonical source"):
            verify_result_source(copied, self.source)

    def test_strict_boundary_and_immutable_record(self):
        for mutation in (lambda f: f.update(token="private"),
                         lambda f: f["tasks"][0]["outcomes"].append(1),
                         lambda f: f["tasks"][0].update(role="private"),
                         lambda f: f["condition"].pop("oracle"),
                         lambda f: f.update(schema_version=2)):
            broken = copy.deepcopy(self.fixture)
            mutation(broken)
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_fixture(broken)
        for mutation in (lambda r: r["per_task"][0].update(all_k=0.9),
                         lambda r: r["macro"].update(extra=1),
                         lambda r: r.update(software_commit="unknown")):
            broken = copy.deepcopy(self.finalized())
            mutation(broken)
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                validate_result(broken)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "run.json"
            result = self.finalized()
            with patch("llm_evaluation.records.software_identity", return_value=(COMMIT, IMPLEMENTATION)):
                write_immutable(path, result, self.source)
                self.assertEqual(read_finalized(path, self.source), result)
                with self.assertRaises(FileExistsError):
                    write_immutable(path, result, self.source)
            self.assertEqual(list(Path(directory).resolve().iterdir()), [path])

    def test_reject_wrong_executing_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "run.json"
            with patch("llm_evaluation.records.software_identity", return_value=("c" * 40, IMPLEMENTATION)):
                with self.assertRaisesRegex(ValueError, "executing Git checkout"):
                    write_immutable(path, self.finalized(), self.source)
            self.assertFalse(path.exists())

    def test_finalized_requires_measured_implementation(self):
        result = self.finalized()
        result["implementation_sha256"] = None
        with self.assertRaisesRegex(ValueError, "implementation digest"):
            validate_result(result)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "implementation digest"):
                write_immutable(Path(directory).resolve() / "out.json", result, self.source)

    def test_software_digest_mismatch_rejected_at_writer(self):
        result = self.finalized()
        result["implementation_sha256"] = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "out.json"
            with patch("llm_evaluation.records.software_identity",
                       return_value=(COMMIT, IMPLEMENTATION)):
                with self.assertRaisesRegex(ValueError, "software identity"):
                    write_immutable(path, result, self.source)
            self.assertFalse(path.exists())

    def test_fsync_parent_after_publishing_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "out.json"
            with patch("llm_evaluation.records.software_identity",
                       return_value=(COMMIT, IMPLEMENTATION)), patch(
                           "llm_evaluation.records._fsync_directory",
                           wraps=records._fsync_directory
                       ) as sync:
                write_immutable(path, self.finalized(), self.source)
            self.assertGreaterEqual(sync.call_count, 2)
            self.assertEqual(read_finalized(path, self.source), self.finalized())

    def test_symlinked_artifact_parent_and_target_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with patch("llm_evaluation.records.software_identity",
                       return_value=(COMMIT, IMPLEMENTATION)):
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    write_immutable(alias / "out.json", self.finalized(), self.source)
                target = real / "out.json"
                target.symlink_to(root / "absent")
                with self.assertRaisesRegex(ValueError, "symbolic links"):
                    write_immutable(target, self.finalized(), self.source)
            self.assertFalse((real / "absent").exists())

    def test_git_objects_reject_hidden_implementation_modifications(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory).resolve() / "repo"
            clone.mkdir()
            shutil.copytree(root / "src", clone / "src",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copytree(root / "fixtures", clone / "fixtures")
            shutil.copy2(root / ".gitignore", clone / ".gitignore")
            def git(*args):
                return subprocess.run(["git", *args], cwd=clone, check=True,
                                      capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            git("add", "src", "fixtures", ".gitignore")
            git("commit", "-qm", "fixture")
            environment = dict(os.environ, PYTHONPATH=str(clone / "src"))
            output = Path(directory).resolve() / "out.json"
            command = [sys.executable, "-m", "llm_evaluation.cli",
                       str(clone / "fixtures" / "e002_synthetic.json"),
                       "--k", "2", "--output", str(output)]
            first = subprocess.run(command, cwd=clone, env=environment,
                                   capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(git("status", "--porcelain", "--untracked-files=all"), "")
            record = read_finalized(output, self.source)
            self.assertEqual(record["software_commit"], git("rev-parse", "HEAD"))
            self.assertEqual(len(record["implementation_sha256"]), 64)
            for flag, reset in (("--assume-unchanged", "--no-assume-unchanged"),
                                ("--skip-worktree", "--no-skip-worktree")):
                with self.subTest(flag=flag):
                    git("update-index", flag, "src/llm_evaluation/metrics.py")
                    target = clone / "src/llm_evaluation/metrics.py"
                    original = target.read_text()
                    target.write_text(original.replace("return c / n",
                                                        "return 0.123456789"))
                    self.assertEqual(git("status", "--porcelain", "--untracked-files=all"), "")
                    output.unlink(missing_ok=True)
                    failed = subprocess.run(command, cwd=clone, env=environment,
                                            capture_output=True, text=True)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn("executing implementation bytes differ", failed.stderr)
                    self.assertFalse(output.exists())
                    git("update-index", reset, "src/llm_evaluation/metrics.py")
                    target.write_text(original)

    def test_real_cli_surface_with_controlled_checkout_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "run.json"
            argv = ["llm-evaluation", str(FIXTURE), "--k", "2", "--output", str(path)]
            with patch.object(sys, "argv", argv), patch("llm_evaluation.cli.software_identity", return_value=(COMMIT, IMPLEMENTATION)), patch("llm_evaluation.records.software_identity", return_value=(COMMIT, IMPLEMENTATION)):
                cli.main()
                self.assertEqual(read_finalized(path, self.source)["software_commit"], COMMIT)
                with self.assertRaises(FileExistsError):
                    cli.main()


if __name__ == "__main__":
    unittest.main()
