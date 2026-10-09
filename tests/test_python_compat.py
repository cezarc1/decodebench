"""No source may use syntax newer than the requires-python floor in pyproject.toml."""

import ast
import tomllib

import pytest

from tests import REPO

REQUIRES_PYTHON = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["requires-python"]
MIN_PYTHON = (3, 12)
SOURCES = sorted(p for d in ("fp4bench", "tests") for p in (REPO / d).rglob("*.py"))


def test_there_are_sources_to_check():
    assert REPO / "fp4bench" / "core" / "schema.py" in SOURCES


def test_floor_matches_requires_python():
    assert ">=" + ".".join(map(str, MIN_PYTHON)) == REQUIRES_PYTHON


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(REPO)))
def test_parses_as_the_oldest_supported_python(path):
    ast.parse(path.read_text(encoding="utf-8"), str(path), feature_version=MIN_PYTHON)
