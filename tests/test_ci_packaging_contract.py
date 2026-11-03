from pathlib import Path


def test_root_wheel_verifies_external_canonical_sync_git_provenance() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    install = '"$build_dir/venv-root/bin/pip" install "$build_dir"/dist/root/*.whl'
    assert install in workflow
    assert 'distribution("canonical-sync").read_text("direct_url.json")' in workflow
    assert 'provenance["url"] == "https://github.com/bio-comp/canonical-sync.git"' in workflow
    assert 'provenance["vcs_info"]["commit_id"] == expected' in workflow
    assert 'provenance["vcs_info"]["requested_revision"] == expected' in workflow
    assert 'expected = "a157898a4c2fd56984afb410d426f913ae7b2f09"' in workflow
    assert "python -m build packages/canonical-sync" not in workflow


def test_metric_lab_installed_entrypoint_is_exercised_in_ci() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert '"$build_dir/venv-root/bin/llm-evaluation-metric-lab" --help' in workflow
    assert 'grep -q "known-truth metric/reference-error" "$build_dir/metric-lab-help.txt"' in workflow
