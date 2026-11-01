"""Live acceptance is opt-in and never part of ordinary CI."""

import os
import pytest
from llm_evaluation.local_runtime_preflight_records import (
    PreflightRecord,
    PreflightBudget,
)


def test_preflight_record_rejects_secret_and_scientific_sample_fields():
    for key in (
        "secret",
        "authorization",
        "contributes_to_sample",
        "global_gpu_exclusivity",
    ):
        with pytest.raises(ValueError):
            PreflightRecord(**{key: "forbidden"})
    with pytest.raises(ValueError):
        PreflightBudget(generation_submissions=7)


@pytest.mark.live_llama
@pytest.mark.skipif(
    os.environ.get("LLM_EVAL_LLAMA_PREFLIGHT") != "1",
    reason="bounded local preflight requires explicit opt-in",
)
def test_standalone_llama_preflight(tmp_path):
    from llm_evaluation.local_runtime_preflight import run_preflight

    record = run_preflight(
        executable=os.environ["LLM_EVAL_LLAMA_SERVER"],
        artifact=os.environ["LLM_EVAL_LLAMA_GGUF_PATH"],
        expected_sha256=os.environ["LLM_EVAL_LLAMA_GGUF_SHA256"],
        output=tmp_path / "preflight.json",
        authorized=True,
    )
    assert record.status == "passed"
