import hashlib
import math

import pytest

from fp4bench.core.files import file_sha256, write_text_atomic
from fp4bench.core.types import is_finite, is_number


def test_file_sha256_is_the_digest_of_the_whole_file_past_one_chunk(tmp_path):
    path = tmp_path / "blob.bin"
    data = bytes(range(256)) * ((1 << 20) // 256 + 3)
    path.write_bytes(data)
    assert file_sha256(path) == file_sha256(str(path)) == hashlib.sha256(data).hexdigest()
    assert type(file_sha256(path)) is str


def test_file_sha256_of_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.json")


def test_write_text_atomic_replaces_the_file_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("old")
    write_text_atomic(path, "[[1, 2]]")
    assert path.read_text() == "[[1, 2]]"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prompts.json"]


@pytest.mark.parametrize("value", [0, 3, -2.5, 1e308])
def test_finite_numbers(value):
    assert is_number(value) and is_finite(value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_numbers_but_not_finite(value):
    assert is_number(value) and not is_finite(value)


@pytest.mark.parametrize("value", [True, False, None, "1.5", [1.0], {"mean": 1.0}])
def test_bools_and_non_numbers_are_neither(value):
    assert not is_number(value) and not is_finite(value)
