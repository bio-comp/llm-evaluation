from __future__ import annotations

import string
import pytest
from hypothesis import given, strategies as st

from canonical_sync.git_publication import _safe_git_path
from canonical_sync.records import _portable_path_key, _safe_logical_path

# Windows reserved stems
RESERVED_STEMS = [
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
]

ILLEGAL_CHARS = list('<>:"|?*\\')
CONTROL_CHARS = [chr(c) for c in range(0x20)] + [chr(0x7F)]


# --- Positive Generation ---

valid_component_chars = string.ascii_lowercase + string.digits + "_-"

valid_component = st.text(
    alphabet=valid_component_chars,
    min_size=1,
    max_size=20,
).filter(lambda s: s.lower() not in RESERVED_STEMS)

valid_extensions = st.sampled_from(["", ".json", ".txt", ".md", ".csv", ".toml"])

valid_path_part = st.builds(
    lambda stem, ext: f"{stem}{ext}",
    valid_component,
    valid_extensions,
)

valid_relative_paths = st.lists(valid_path_part, min_size=1, max_size=4).map(lambda parts: "/".join(parts))


@given(path=valid_relative_paths)
def test_fuzz_paths_valid_paths_accepted(path: str) -> None:
    assert _safe_logical_path(path) == path
    assert _safe_git_path(path) == path
    key = _portable_path_key(path)
    assert key == path.casefold()


# --- Negative Generation ---

# 1. Paths with control characters
paths_with_control_chars = st.builds(
    lambda base, ctrl: f"{base}{ctrl}",
    valid_relative_paths,
    st.sampled_from(CONTROL_CHARS),
)

# 2. Paths with Windows illegal characters
paths_with_illegal_chars = st.builds(
    lambda base, char: f"{base}/{char}file.txt",
    valid_relative_paths,
    st.sampled_from(ILLEGAL_CHARS),
)

# 3. Paths with backslashes
paths_with_backslashes = st.builds(
    lambda p: p.replace("/", "\\"),
    valid_relative_paths.filter(lambda p: "/" in p),
)

# 4. Paths with traversal sequences or empty parts
traversal_components = st.sampled_from(["..", ".", "", ".git", ".GIT"])
paths_with_traversal = st.builds(
    lambda prefix, comp, suffix: f"{prefix}/{comp}/{suffix}",
    valid_relative_paths,
    traversal_components,
    valid_relative_paths,
)

# 5. Absolute paths
absolute_paths = st.builds(lambda p: f"/{p}", valid_relative_paths)

# 6. Paths with trailing dot or space on component
trailing_bad_char = st.sampled_from([" ", "."])
paths_with_trailing_bad = st.builds(
    lambda base, bad: f"{base}{bad}/file.txt",
    valid_relative_paths,
    trailing_bad_char,
)

# 7. Reserved stem filenames
reserved_stems_strategy = st.sampled_from(RESERVED_STEMS)
reserved_paths = st.builds(
    lambda stem, ext: f"dir/{stem}{ext}",
    reserved_stems_strategy,
    valid_extensions,
)

# 8. Leading/trailing whitespace on entire path
whitespace_padded_paths = st.builds(
    lambda p, ws: f"{ws}{p}{ws}",
    valid_relative_paths,
    st.sampled_from([" ", "  ", "\t", "\n"]),
)


@given(path=paths_with_control_chars)
def test_fuzz_paths_rejects_control_chars(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=paths_with_illegal_chars)
def test_fuzz_paths_rejects_illegal_chars(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=paths_with_backslashes)
def test_fuzz_paths_rejects_backslashes(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=paths_with_traversal)
def test_fuzz_paths_rejects_traversals(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=absolute_paths)
def test_fuzz_paths_rejects_absolute_paths(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=paths_with_trailing_bad)
def test_fuzz_paths_rejects_trailing_dot_or_space(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=reserved_paths)
def test_fuzz_paths_rejects_reserved_stems(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)


@given(path=whitespace_padded_paths)
def test_fuzz_paths_rejects_whitespace_padding(path: str) -> None:
    with pytest.raises(ValueError):
        _portable_path_key(path)
