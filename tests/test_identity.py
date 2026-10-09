"""What is measured (tests/golden/identity.json) must not change in a refactor: only
make_identity.identity_view may be adapted to new code. The JSON is regenerated only when a new
study, treatment or constant is added on purpose, and then the regenerated view may only add to it:
python -m tests.golden.make_identity --check-additive."""

import contextlib
import copy
import inspect
import json
import textwrap
from unittest import mock

import pytest

from fp4bench import decode_step, runner
from tests.golden import make_identity
from tests.golden.make_identity import IDENTITY_PATH, additive_diff, identity_view
from tests.golden.view import dumps, first_difference

SNAPSHOT = json.loads(IDENTITY_PATH.read_text())


def test_identity_matches_the_snapshot():
    view = identity_view()
    assert first_difference(SNAPSHOT, view) is None, first_difference(SNAPSHOT, view)
    assert dumps(view) == IDENTITY_PATH.read_text()


def test_the_committed_runs_were_served_and_prompted_as_the_snapshot_says():
    runs = SNAPSHOT["committed_runs"]
    assert sorted(runs) == ["expb-1", "expc-1", "full-1", "smoke-c-1", "smoke-nf-2", "smoke-r2-2"]
    for run, recorded in runs.items():
        (protocol,) = recorded["protocol"]
        sessions = SNAPSHOT["protocols"][protocol]["sessions"]
        assert sorted(recorded["server_argv"]) == sorted(sessions), run
        for treatment, session in sessions.items():
            assert recorded["server_argv"][treatment] == [session["argv"]], (run, treatment)
            assert recorded["server_env_keys"][treatment] == [session["env_keys"]], (run, treatment)
        schedule = SNAPSHOT["protocols"][protocol]
        orders = schedule.get("round_order_extended_to_10", schedule["round_order"])
        for r, treatments in recorded["session_order"]:
            assert treatments == orders[r][: len(treatments)], (run, r, treatments)
    for key in ("m1_prompts_sha256", "nll_prompts_sha256"):
        assert {sha for r in runs.values() for sha in r[key]} == {runs["full-1"][key][0]}
        assert all(len(r[key]) == 1 for r in runs.values())
    cell_runs = {
        run
        for run, r in runs.items()
        if SNAPSHOT["protocols"][r["protocol"][0]]["protocol"]["cells"]
    }
    assert cell_runs == {"expc-1", "smoke-c-1"}
    assert {sha for r in runs.values() for sha in r["c_prompts_sha256"]} == {
        runs["expc-1"]["c_prompts_sha256"][0]
    }
    assert all(len(runs[run]["c_prompts_sha256"]) == 1 for run in cell_runs)
    assert runs["full-1"]["m1_prompts_sha256"][0].startswith("9930b747")
    assert runs["full-1"]["nll_prompts_sha256"][0].startswith("dc20a835")
    assert runs["expc-1"]["c_prompts_sha256"][0].startswith("6439900e")


def recompiled(module, name: str, old: str, new: str):
    """`module.<name>` compiled from its source with `old` (there once) replaced by `new`."""
    source = textwrap.dedent(inspect.getsource(getattr(module, name)))
    assert source.count(old) == 1, f"{old!r} is not in {module.__name__}.{name} exactly once"
    namespace: dict = {}
    exec(compile(source.replace(old, new), module.__file__, "exec"), vars(module), namespace)  # noqa: S102
    return namespace[name]


def difference_with(*patches) -> str | None:
    with contextlib.ExitStack() as stack:
        for target, name, value in patches:
            stack.enter_context(mock.patch.object(target, name, value))
        return first_difference(SNAPSHOT, identity_view())


def test_a_block_served_the_last_c_prompts_of_each_set_fails_the_snapshot():
    block = recompiled(decode_step, "decode_block", "prompt_sets[s][:c]", "prompt_sets[s][-c:]")
    diff = difference_with((decode_step, "decode_block", block), (runner, "decode_block", block))
    assert diff is not None and ".session_trace." in diff, diff


def test_cells_run_in_another_order_fail_the_snapshot():
    session = recompiled(
        runner,
        "run_server_session",
        "for cell, sets in blocks:",
        "for cell, sets in reversed(blocks):",
    )
    diff = difference_with((runner, "run_server_session", session))
    assert diff is not None and ".session_trace." in diff, diff


def test_a_view_that_only_adds_keys_and_list_items_is_additive():
    old = {"a": 1, "names": ["MX", "NV"], "d": {"k": [[1, 2]]}}
    new = {
        "a": 1,
        "b": {"c": 2},
        "names": ["MX", "MXp", "NV", "NVq"],
        "d": {"k": [[1, 2]], "k2": 3},
    }
    assert additive_diff(old, new) == (["$.b", "$.d.k2", "$.names[1]", "$.names[3]"], [])
    assert additive_diff(old, copy.deepcopy(old)) == ([], [])


@pytest.mark.parametrize(
    "old, new, where",
    [
        ({"a": 1, "b": 2}, {"a": 1}, "$.b: removed"),
        ({"a": 1}, {"a": 2}, "$.a: 1 -> 2"),
        ({"a": 1}, {"a": 1.0}, "$.a: int 1 -> float 1.0"),
        ({"a": 1}, {"a": True}, "$.a: int 1 -> bool True"),
        ({"a": [1, 2]}, {"a": [2, 1]}, "$.a: [1, 2] is not kept, in order, in [2, 1]"),
        ({"a": [1, 2]}, {"a": [1]}, "$.a: [1, 2] is not kept, in order, in [1]"),
        ({"a": [{"k": 1}]}, {"a": [{"k": 1, "j": 2}]}, "$.a: [{'k': 1}] is not kept, in order, in"),
        ({"a": [[1, True]]}, {"a": [[1.0, True]]}, "$.a: [[1, True]] is not kept, in order, in"),
        ({"a": [[1, True]]}, {"a": [[1, 1]]}, "$.a: [[1, True]] is not kept, in order, in"),
        ({"a": {"k": 1}}, {"a": [1]}, "$.a: dict {'k': 1} -> list [1]"),
    ],
)
def test_removing_or_changing_anything_is_not_additive(old, new, where):
    _, changed = additive_diff(old, new)
    assert len(changed) == 1 and changed[0].startswith(where), changed


def _with_a_new_study() -> dict:
    view = copy.deepcopy(SNAPSHOT)
    view["protocols"]["new-study"] = {"sessions": {"MX": {"argv": ["vllm", "serve"]}}}
    return view


def test_check_additive_lists_what_a_new_study_adds_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "identity.json"
    path.write_text(IDENTITY_PATH.read_text())
    monkeypatch.setattr(make_identity, "IDENTITY_PATH", path)
    monkeypatch.setattr(make_identity, "identity_view", _with_a_new_study)
    make_identity.main(["--check-additive"])
    assert "added $.protocols.new-study" in capsys.readouterr().out
    assert path.read_text() == IDENTITY_PATH.read_text()
    make_identity.main([])
    assert path.read_text() == dumps(_with_a_new_study())


def test_regenerating_refuses_a_view_that_is_not_additive(tmp_path, monkeypatch, capsys):
    path = tmp_path / "identity.json"
    path.write_text(IDENTITY_PATH.read_text())
    view = _with_a_new_study()
    view["protocols"]["full"]["m1_reps"] += 1
    monkeypatch.setattr(make_identity, "IDENTITY_PATH", path)
    monkeypatch.setattr(make_identity, "identity_view", lambda: view)
    for argv in (["--check-additive"], []):
        with pytest.raises(SystemExit, match="not additive"):
            make_identity.main(argv)
        assert "changed $.protocols.full.m1_reps:" in capsys.readouterr().out
    assert path.read_text() == IDENTITY_PATH.read_text()
