"""Independent Q0 multi-domain reference and negative-control qualification."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from llm_evaluation.q0_multidomain_dev import (
    Q0ControlSet,
    Q0Packet,
    evaluate_all_controls,
    evaluate_response,
    load_controls,
    load_packet,
    readiness_report,
)
from reference_q0_multidomain import (
    clamp_branch,
    clamp_bug,
    clamp_minmax,
    median_bug_even,
    median_library,
    median_sorted,
    test_clamp_references,
    test_median_references,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "fixtures/q0_multidomain_packet_dev_v1.json"
CONTROLS = ROOT / "fixtures/q0_multidomain_controls_dev_v1.json"
PREDECESSOR = ROOT / "fixtures/local_runtime_preflight_v1.json"
E002 = ROOT / "fixtures/e002_objective_packet_dev_v1.json"


def test_trusted_reference_solutions_have_independent_truth_vectors() -> None:
    test_median_references()
    test_clamp_references()
    # Expected values are independently supplied, not computed with the scorer.
    median_vectors = (
        ([1, 2, 3], 2.0),
        ([1, 2, 3, 4], 2.5),
        ([-5, -1], -3.0),
        ([4, 4], 4.0),
    )
    for xs, expected in median_vectors:
        assert median_sorted(xs) == expected
        assert median_library(xs) == expected
    assert median_bug_even([1, 2, 3, 4]) != 2.5
    clamp_vectors = (
        (10, 0, 10, 10),
        (-3, 0, 10, 0),
        (15, 0, 10, 10),
        (5, 0, 10, 5),
    )
    for x, low, high, expected in clamp_vectors:
        assert clamp_minmax(x, low, high) == expected
        assert clamp_branch(x, low, high) == expected
    assert clamp_bug(10, 0, 10) != 10


def test_independent_objective_reference_derivations() -> None:
    packet = json.loads(PACKET.read_bytes())
    cases = {x["case_id"]: x for x in packet["cases"]}
    assert cases["sanity-arithmetic"]["reference"]["expected"] == str(17 * 23 - 19)
    source = "Beni=7; Ada=12; Cora=5"
    parsed = dict(item.split("=") for item in source.split("; "))
    assert json.loads(cases["sanity-record-map"]["reference"]["expected"]) == {
        key: int(value) for key, value in parsed.items()
    }
    lines = ["line A tag GX-17", "line B has no tag", "line C tag QZ-04"]
    tags = [line.split("tag ")[1] for line in lines if " tag " in line]
    assert json.loads(cases["sanity-tag-extraction"]["reference"]["expected"]) == tags
    assert json.loads(cases["coding-tool-call"]["reference"]["expected"]) == {
        "tool": "sort_numbers",
        "arguments": {"numbers": [3, 1, 2], "descending": False},
    }
    assert Fraction(6 + 12, 20 + 30) == Fraction(9, 25)
    assert Fraction(Decimal("0.36")) == Fraction(9, 25)
    assert cases["science-pooled-rate"]["reference"]["expected"] == "9/25"
    assert -0.1 <= 0 <= 0.4
    assert cases["science-ci-decision"]["reference"]["expected"] == "DO_NOT_REJECT"
    assert cases["general-exact-steps"]["reference"][
        "expected"
    ] == "STEP-1\nSTEP-2\nDONE".replace("\\n", "\n")


def test_packet_inventory_provenance_and_unmodified_history() -> None:
    p = load_packet(PACKET)
    r = readiness_report(p)
    assert p.packet.phase == "DEVELOPMENT"
    assert p.packet.schema_version == 1
    assert p.packet.packet_id == "e002-q0-multidomain-dev-v1"
    assert p.packet_sha256 == hashlib.sha256(PACKET.read_bytes()).hexdigest()
    assert p.packet_size_bytes == len(PACKET.read_bytes())
    assert [c.category for c in p.packet.cases].count("sanity") == 3
    assert [c.category for c in p.packet.cases].count("coding") == 3
    assert [c.category for c in p.packet.cases].count("scientific_reasoning") == 3
    assert [c.category for c in p.packet.cases].count(
        "general_instruction_following"
    ) == 3
    assert len(r.objective_case_ids) == 7
    assert len(r.sandbox_case_ids) == 2
    assert len(r.expert_case_ids) == 3
    assert "D-017" in " ".join(r.q1_dependencies)
    assert hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() == (
        "6f879e8ee1bca440306af91c6a6169c97be2c1473acffc0c1882cc20bf9a7bbc"
    )
    assert hashlib.sha256(E002.read_bytes()).hexdigest() == (
        "16547692cd5c4350790fb57cc7cf1936f706dc82af7d770d441bcc28d5a2ae26"
    )


def test_all_controls_match_independently_labeled_expectations() -> None:
    p = load_packet(PACKET)
    c = load_controls(CONTROLS, p)
    outcomes = evaluate_all_controls(p, c)
    assert len(outcomes) == 39
    assert {x.disposition for x in outcomes} == {
        "PASS",
        "FAIL",
        "NOT_AUTOMATICALLY_SCORABLE",
    }
    assert all(x.packet_sha256 == p.packet_sha256 for x in outcomes)
    assert all(x.reference_sha256 and x.answer_sha256 for x in outcomes)


@pytest.mark.parametrize(
    "case_id,answer",
    [
        ("sanity-arithmetic", "391"),
        ("sanity-record-map", '{"Ada":12,"Beni":7,"Cora":6}'),
        ("sanity-tag-extraction", '["QZ-04","GX-17"]'),
        (
            "coding-tool-call",
            '{"tool":"sort_numbers","arguments":{"numbers":[3,1,2],"descending":0}}',
        ),
        ("science-pooled-rate", "0.4"),
        ("science-ci-decision", "REJECT"),
        ("general-exact-steps", "STEP-1\\nSTEP-2\\nDONE\\n"),
    ],
)
def test_known_bad_cannot_pass(case_id: str, answer: str) -> None:
    p = load_packet(PACKET)
    assert evaluate_response(p, case_id, answer).disposition == "FAIL"


@pytest.mark.parametrize(
    "case_id,answer",
    [
        ("sanity-record-map", '{"Cora":5,"Beni":7,"Ada":12}'),
        ("sanity-tag-extraction", '[ "GX-17", "QZ-04" ]'),
        (
            "coding-tool-call",
            '{"arguments":{"descending":false,"numbers":[3,1,2]},"tool":"sort_numbers"}',
        ),
        ("science-pooled-rate", "18/50"),
        ("science-pooled-rate", "0.36"),
    ],
)
def test_alternative_correct_answers_not_falsely_rejected(
    case_id: str, answer: str
) -> None:
    assert evaluate_response(load_packet(PACKET), case_id, answer).disposition == "PASS"


@pytest.mark.parametrize(
    "case_id,answer",
    [
        ("coding-median", "def median(xs): return 0"),
        ("coding-clamp-debug", "def clamp(x,low,high): return min(max(x,low),high-1)"),
        ("science-causal-claim", "Coffee causes higher scores."),
        ("general-faithful-summary", "All people prefer A."),
        ("general-multistep-followup", "Deploy everything immediately."),
    ],
)
def test_deferred_cases_never_misreport_objective_success(
    case_id: str, answer: str
) -> None:
    assert (
        evaluate_response(load_packet(PACKET), case_id, answer).disposition
        == "NOT_AUTOMATICALLY_SCORABLE"
    )


@pytest.mark.parametrize(
    "answer",
    [
        '{"Ada":12,"Ada":12,"Beni":7,"Cora":5}',
        '{"Ada":12,"Beni":7,"Cora":5,}',
        '{"Ada":12,"Beni":7,"Cora":NaN}',
        '{"Ada":12,"Beni":7,"Cora":5,"other":true}',
    ],
)
def test_json_malformed_or_ambiguous_rejected(answer: str) -> None:
    assert (
        evaluate_response(load_packet(PACKET), "sanity-record-map", answer).disposition
        == "FAIL"
    )


def test_conflicting_reference_rejected_even_with_parseable_fixture() -> None:
    packet = json.loads(PACKET.read_bytes())
    packet["cases"][0]["reference"]["expected"] = "391"
    with pytest.raises(ValidationError, match="reference"):
        Q0Packet.model_validate_json(json.dumps(packet))


def test_unknown_case_extra_fields_and_control_packet_mismatch_rejected(
    tmp_path: Path,
) -> None:
    p = load_packet(PACKET)
    with pytest.raises(ValueError, match="case"):
        evaluate_response(p, "unregistered-case", "hi")
    changed = json.loads(PACKET.read_bytes())
    changed["cases"][0]["unapproved_field"] = True
    with pytest.raises(ValidationError):
        Q0Packet.model_validate(changed)
    controls = json.loads(CONTROLS.read_bytes())
    controls["packet_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="packet"):
        load_controls(CONTROLS, p, source_bytes=json.dumps(controls).encode())


def test_control_expected_classification_cannot_launder_bad_answer() -> None:
    p = load_packet(PACKET)
    controls = json.loads(CONTROLS.read_bytes())
    bad = next(
        c
        for c in controls["controls"]
        if c["case_id"] == "sanity-arithmetic" and c["response"] == "391"
    )
    bad["expected_disposition"] = "PASS"
    forged = Q0ControlSet.model_validate_json(json.dumps(controls))
    with pytest.raises(ValueError, match="control"):
        evaluate_all_controls(p, forged)


@given(n=st.integers(min_value=-10000, max_value=10000).filter(lambda n: n != 372))
def test_integer_oracle_rejects_any_other_integer(n: int) -> None:
    assert (
        evaluate_response(load_packet(PACKET), "sanity-arithmetic", str(n)).disposition
        == "FAIL"
    )


@given(n=st.integers(min_value=1, max_value=999))
def test_valid_equivalent_fraction_forms_never_false_reject(n: int) -> None:
    assert (
        evaluate_response(
            load_packet(PACKET), "science-pooled-rate", f"{9 * n}/{25 * n}"
        ).disposition
        == "PASS"
    )
