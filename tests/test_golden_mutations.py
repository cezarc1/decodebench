"""The current analysis on mutated copies of the committed runs writes what mutations.json pins
(PNGs as tests/golden/pngs.py compares them), and departs from it only where a recipe says so.

Each case analyses its own recipe. Under pytest-xdist (`-n auto`) the workers share the cases out
and analyse them in-process; a serial session analyses its selected cases in a process pool."""

import json
import os
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest

from tests import GOLDEN_DIR
from tests.golden.mutations import RECIPES, head_result
from tests.golden.pngs import differences

MUTATIONS = json.loads((GOLDEN_DIR / "mutations.json").read_text())

type Analysed = Callable[[str], dict[str, Any]]


def _cases(session: pytest.Session) -> list[str]:
    """The recipes of the session's test_the_analysis_writes_the_tags_files cases, in run order."""
    return [
        str(item.callspec.params["name"])
        for item in session.items
        if isinstance(item, pytest.Function)
        and item.function is test_the_analysis_writes_the_tags_files
    ]


@pytest.fixture(scope="module")
def analysed(request: pytest.FixtureRequest) -> Iterator[Analysed]:
    """head_result of a case's recipe. An xdist worker analyses each case it is handed; a serial
    session submits its cases to a process pool up front, in the order they run."""
    names = _cases(request.session)
    if hasattr(request.config, "workerinput") or len(names) < 2:
        yield head_result
        return
    with ProcessPoolExecutor(min(len(names), os.cpu_count() or 1)) as pool:
        futures = {name: pool.submit(head_result, name) for name in names}
        yield lambda name: futures[name].result()


def test_every_recipe_has_the_tags_outputs():
    assert sorted(MUTATIONS) == sorted(RECIPES)


def test_a_recipe_the_tag_refused_is_refused():
    assert [n for n, old in MUTATIONS.items() if "error" in old and not RECIPES[n].refuses] == []


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_the_analysis_writes_the_tags_files(analysed: Analysed, name: str):
    rec, old = RECIPES[name], MUTATIONS[name]
    new = analysed(name)
    if rec.refuses is not None:
        assert "error" in new, f"{name}: expected a refusal matching {rec.refuses!r}"
        assert re.search(rec.refuses, new["error"]), new["error"]
    elif rec.changed is not None:
        assert "files" in new and differences(old.get("files", {}), new) != [], rec.changed
        assert rec.shows is not None and rec.shows in new["summary"]
    else:
        assert "error" not in new, new["error"]
        assert differences(old["files"], new) == []
        assert differences(old["files"], new, strict=False) == []
