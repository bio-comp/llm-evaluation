from pathlib import Path


def test_root_wheel_dependencies_are_prefetched_before_offline_install() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    prefetch = (
        '"$build_dir/venv-root/bin/pip" wheel --wheel-dir="$build_dir/wheelhouse" '
        '--find-links="$build_dir/wheelhouse" "$build_dir"/dist/root/*.whl'
    )
    offline_install = (
        '"$build_dir/venv-root/bin/pip" install --no-index '
        '--find-links="$build_dir/wheelhouse" "$build_dir"/dist/root/*.whl'
    )
    assert prefetch in workflow
    assert offline_install in workflow
    assert workflow.index(prefetch) < workflow.index(offline_install)
