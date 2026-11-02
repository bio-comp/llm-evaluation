from __future__ import annotations

import tomllib
from pathlib import Path


def test_drift_lab_cli_is_packaged_as_an_installed_entrypoint() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert (
        project["project"]["scripts"]["llm-evaluation-drift-lab"]
        == "llm_evaluation.drift_lab_cli:main"
    )


def test_smoke_mode_runs_tiny_deterministic_pipeline_without_publishing(
    capsys,
) -> None:
    from llm_evaluation.drift_lab_cli import main

    assert main(["--phase", "smoke", "--smoke-paths", "8"]) == 0
    output = capsys.readouterr().out
    assert "E-004 smoke" in output
    assert "paths=8" in output
    assert "published=false" in output


def test_installed_style_smoke_does_not_depend_on_repository_fixture(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from llm_evaluation.drift_lab_cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["--phase", "smoke", "--smoke-paths", "4"]) == 0
    assert "published=false" in capsys.readouterr().out


def test_scientific_phases_require_explicit_frozen_design_path() -> None:
    from llm_evaluation.drift_lab_cli import main

    import pytest

    with pytest.raises(SystemExit, match="--frozen-design is required"):
        main(["--phase", "development", "--output", "unused.json"])
    with pytest.raises(SystemExit, match="--frozen-design is required"):
        main(["--phase", "final", "--calibration", "unused.json", "--output", "unused2.json"])
