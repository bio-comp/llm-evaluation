"""R1 controls: independent oracle truth, source-bound qualification, UTF-8 boundary."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from llm_evaluation.q0_multidomain_dev import (
    Q0ControlSet,
    evaluate_response,
    load_controls,
    load_packet,
    qualification_report,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "fixtures/q0_multidomain_packet_dev_v1.json"
CONTROLS = ROOT / "fixtures/q0_multidomain_controls_dev_v1.json"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("9/25", "PASS"),
        ("0.36", "PASS"),
        ("0.360", "PASS"),
        ("18/50", "FAIL"),
        ("36/100", "FAIL"),
        ("0.4", "FAIL"),
        ("9/0", "FAIL"),
    ],
)
def test_simplified_fraction_or_terminating_decimal_oracle(
    answer: str, expected: str
) -> None:
    # Truth comes from the literal problem and reduced fraction 18/50 = 9/25,
    # not a value calculated by the implementation under test.
    assert (
        evaluate_response(
            load_packet(PACKET), "science-pooled-rate", answer
        ).disposition
        == expected
    )


@given(multiplier=st.integers(min_value=1, max_value=10000))
def test_fraction_simplification_is_not_just_numerical_equality(
    multiplier: int,
) -> None:
    # Independently constructed equivalent ratios: gcd(9*k, 25*k) = k.
    numerator, denominator = 9 * multiplier, 25 * multiplier
    assert math.gcd(numerator, denominator) == multiplier
    expected = "PASS" if multiplier == 1 else "FAIL"
    result = evaluate_response(
        load_packet(PACKET), "science-pooled-rate", f"{numerator}/{denominator}"
    )
    assert result.disposition == expected


@given(trailing_zeros=st.integers(min_value=0, max_value=120))
def test_exact_terminating_decimal_alternatives_remain_valid(
    trailing_zeros: int,
) -> None:
    response = "0.36" + "0" * trailing_zeros
    assert (
        evaluate_response(
            load_packet(PACKET), "science-pooled-rate", response
        ).disposition
        == "PASS"
    )


def make_weakened_24() -> bytes:
    """Retain one PASS/FAIL for objective cases and one deferred per other case."""
    data = json.loads(CONTROLS.read_bytes())
    chosen: set[str] = set()
    packet = load_packet(PACKET)
    for case in packet.packet.cases:
        labels = (
            ("NOT_AUTOMATICALLY_SCORABLE",)
            if case.scoring_method.endswith("DEFERRED")
            else ("PASS", "FAIL")
        )
        for label in labels:
            control = next(
                item
                for item in data["controls"]
                if item["case_id"] == case.case_id
                and item["expected_disposition"] == label
            )
            chosen.add(control["control_id"])
    for item in data["controls"]:
        if len(chosen) >= 24:
            break
        chosen.add(item["control_id"])
    data["controls"] = [
        item for item in data["controls"] if item["control_id"] in chosen
    ]
    assert len(data["controls"]) == 24
    return (json.dumps(data, separators=(",", ":")) + "\n").encode()


def test_canonical_controls_have_exact_source_identity() -> None:
    p = load_packet(PACKET)
    source = CONTROLS.read_bytes()
    controls = load_controls(CONTROLS, p)
    result = qualification_report(p, controls, source)
    assert result.controls_qualified is True
    assert result.controls_count == len(controls.controls)
    assert result.controls_sha256 == hashlib.sha256(source).hexdigest()


def test_reduced_24_control_set_fails_v1_qualification(tmp_path: Path) -> None:
    p = load_packet(PACKET)
    replacement = tmp_path / "weakened.json"
    replacement.write_bytes(make_weakened_24())
    with pytest.raises(ValueError, match="control.*source|qualified"):
        load_controls(replacement, p)


def test_actual_offline_cli_rejects_weakened_control_source(
    tmp_path: Path,
) -> None:
    replacement = tmp_path / "weakened.json"
    replacement.write_bytes(make_weakened_24())
    run = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/evaluate_q0_multidomain_dev.py",
            "--controls",
            str(replacement),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert '"controls_qualified": true' not in run.stdout.lower()


def test_qualification_cannot_attach_unrelated_source_digest() -> None:
    packet = load_packet(PACKET)
    controls = load_controls(CONTROLS, packet)
    with pytest.raises(ValueError, match="control.*source|qualified"):
        qualification_report(packet, controls, b"unrelated synthetic bytes")


def test_qualification_cannot_attach_changed_typed_controls() -> None:
    packet = load_packet(PACKET)
    raw = json.loads(CONTROLS.read_bytes())
    raw["controls"][0]["control_purpose"] = "changed from reviewed source"
    altered = Q0ControlSet.model_validate_json(json.dumps(raw))
    with pytest.raises(ValueError, match="control.*source|qualified"):
        qualification_report(packet, altered, CONTROLS.read_bytes())


@pytest.mark.parametrize("answer", ["\ud800", "\udfff", "hello\ud800world"])
@pytest.mark.parametrize("case_id", ["sanity-arithmetic", "coding-median"])
def test_direct_api_rejects_invalid_unicode_explicitly(
    case_id: str, answer: str
) -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        evaluate_response(load_packet(PACKET), case_id, answer)


@pytest.mark.parametrize(
    ("case_id", "answer", "expected"),
    [
        ("sanity-arithmetic", "372", "PASS"),
        ("sanity-arithmetic", "372é", "FAIL"),
        ("coding-median", "def f(): return '🐍'", "NOT_AUTOMATICALLY_SCORABLE"),
    ],
)
def test_valid_unicode_source_bytes_are_hashed_exactly(
    case_id: str, answer: str, expected: str
) -> None:
    actual = evaluate_response(load_packet(PACKET), case_id, answer)
    assert actual.disposition == expected
    assert actual.answer_sha256 == hashlib.sha256(answer.encode("utf-8")).hexdigest()


def test_historical_sources_remain_frozen() -> None:
    assert hashlib.sha256(
        (ROOT / "fixtures/local_runtime_preflight_v1.json").read_bytes()
    ).hexdigest() == (
        "6f879e8ee1bca440306af91c6a6169c97be2c1473acffc0c1882cc20bf9a7bbc"
    )
    assert hashlib.sha256(
        (ROOT / "fixtures/e002_objective_packet_dev_v1.json").read_bytes()
    ).hexdigest() == (
        "16547692cd5c4350790fb57cc7cf1936f706dc82af7d770d441bcc28d5a2ae26"
    )
