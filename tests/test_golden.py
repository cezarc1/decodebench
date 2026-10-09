"""The current analysis reproduces the golden views of the committed runs (tests/golden/<run>.json).
Regenerating the goldens is not a way to make this pass."""

import json

import pytest

from tests.golden.make_golden import RUNS, golden_path, run_view
from tests.golden.view import first_difference

REL_TOL, ABS_TOL = 1e-9, 1e-12


@pytest.mark.parametrize("run", RUNS)
def test_analysis_matches_golden(run):
    expected = json.loads(golden_path(run).read_text())
    diff = first_difference(expected, run_view(run), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    assert diff is None, f"{run}: {diff}"


@pytest.mark.parametrize(
    "expected, actual, where",
    [
        ({"a": [1.0, 2]}, {"a": [1.0 + 1e-12, 2]}, None),
        ({"a": [1.0, 2]}, {"a": [1.0 + 1e-6, 2]}, "$.a[0]"),
        ({"a": [1.0, 2]}, {"a": [1.0, 3]}, "$.a[1]"),
        ({"a": [1.0, 2]}, {"a": [1.0, 2.0]}, "$.a[1]"),
        ({"a": [1.0, 2]}, {"a": [1.0]}, "$.a"),
        ({"a": 1}, {"a": True}, "$.a"),
        ({"a": None}, {"a": 0.0}, "$.a"),
        ({"a": "x"}, {"a": "y"}, "$.a"),
        ({"a": {"b": 0.0}}, {"a": {"c": 0.0}}, "$.a.b"),
        ({"a": 0.0}, {"a": 1e-13}, None),
    ],
)
def test_first_difference(expected, actual, where):
    diff = first_difference(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    assert (diff is None) if where is None else (diff is not None and diff.startswith(f"{where}: "))
