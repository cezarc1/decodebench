"""What tests/golden/pins.json pins (tests/golden/make_pins.py) must not change in a refactor: only
make_pins.pins_view may be adapted to new code, and pins.json is regenerated only to add pins:
python -m tests.golden.make_pins --check-additive. Plus checks that need no snapshot because they
hold against the committed kernel data itself."""

import json

import pytest

from fp4bench import microbench, profiling
from fp4bench.core.kernels import ActImpl, BenchKind, GemmImpl
from fp4bench.core.types import Format, Treatment
from tests.golden import make_pins
from tests.golden.make_pins import (
    MICROBENCH_RESULTS,
    PINS_PATH,
    PROFILES,
    REPO,
    ordered,
    pin_changes,
    pins_view,
)
from tests.golden.view import dumps, first_difference

PINS = json.loads(PINS_PATH.read_text())


@pytest.fixture(scope="module")
def view() -> dict:
    return pins_view()


def test_the_view_has_exactly_the_snapshots_pins(view):
    assert list(view) == list(PINS), pin_changes(PINS, view)


@pytest.mark.parametrize("name", list(PINS))
def test_the_pin_is_unchanged_key_order_included(view, name):
    assert ordered(view[name]) == ordered(PINS[name]), first_difference(
        PINS[name], view[name], key_order=True
    )


def test_pins_json_is_what_make_pins_writes(view):
    assert dumps(view, key_order=True) == PINS_PATH.read_text()


@pytest.mark.parametrize("path", MICROBENCH_RESULTS)
def test_make_cell_rebuilds_every_committed_microbench_cell(path):
    """Every row of the committed results.json, key order and values, from its own inputs."""
    identity = (
        "kind",
        "impl",
        "fmt",
        "shape",
        "M",
        "N",
        "K",
        "bytes",
        "flops",
        "median_us",
        "min_us",
        "max_us",
        "samples_us",
        "gbps",
        "frac_hbm",
        "tflops",
        "rel_err",
        "error",
    )
    cells = json.loads((REPO / path).read_text())["cells"]
    assert len(cells) == 462
    for c in cells:
        fmt = Format(c["fmt"])
        case = (
            microbench.GemmCase(
                GemmImpl(c["impl"]), fmt, c["shape"], c["M"], c["N"], c["K"], c["bytes"], c["flops"]
            )
            if c["kind"] == BenchKind.GEMM
            else microbench.ActCase(ActImpl(c["impl"]), fmt, c["shape"], c["M"], c["K"], c["bytes"])
        )
        extras = microbench.CellExtras(**{k: v for k, v in c.items() if k not in identity})
        rebuilt = microbench.make_cell(
            case, c["samples_us"], error=c["error"], rel_err=c["rel_err"], extras=extras
        )
        assert ordered(rebuilt) == ordered(c), first_difference(c, rebuilt, key_order=True)


def test_the_stored_comparisons_are_recomputed_from_the_stored_summaries():
    report = json.loads((REPO / PROFILES).read_text())
    summaries = {name: s["summary"] for name, s in report["sessions"].items()}
    recomputed = {
        "pairwise": (
            profiling.pairwise_deltas(summaries)
            + profiling.pairwise_deltas(
                summaries,
                config="B",
                concurrencies=(512,),
                pairs=(profiling.TreatmentPair(Treatment.NV, Treatment.MX),),
            )
        ),
        "gemm_share_c512": profiling.gemm_share(summaries),
    }
    assert len(recomputed["pairwise"]) == 7 and len(recomputed["gemm_share_c512"]) == 2
    assert ordered(recomputed) == ordered(report["comparisons"]), first_difference(
        report["comparisons"], json.loads(ordered(recomputed)), key_order=True
    )


def summary_keys(summary: dict) -> dict:
    """The key order of a summarize_trace summary and of the records in it."""
    rows = [*summary["top_kernels"], *summary["kernels"]]
    return {
        "summary": list(summary),
        "categories": list(summary["categories"]),
        "category": sorted({tuple(c) for c in summary["categories"].values()}),
        **{
            key: list(summary[key])
            for key in (
                "max_kernel_us",
                "memcpy",
                "memset",
                "graph_replays",
                "by_linear_layer",
                "step_annotations",
                "decode_check",
                "fusion",
            )
        },
        "kernel_rows": sorted({tuple(row) for row in rows}),
    }


def test_the_committed_session_summaries_have_summarize_traces_key_order():
    pinned = summary_keys(PINS["profiling.summarize_trace"]["graph_trace"])
    sessions = json.loads((REPO / PROFILES).read_text())["sessions"]
    assert len(sessions) == 8
    for name, session in sessions.items():
        assert summary_keys(session["summary"]) == pinned, name


def test_a_pin_may_be_added_but_not_changed_reordered_retyped_or_removed():
    old = {"a": {"x": 1, "y": [1.0, None]}, "b": "text"}
    assert pin_changes(old, {**old, "c": 3}) == (["c"], [])
    for new, why in (
        ({"a": {"y": [1.0, None], "x": 1}, "b": "text"}, "keys ['x', 'y']"),
        ({"a": {"x": 1.0, "y": [1.0, None]}, "b": "text"}, "$.x: int 1 != float"),
        ({"a": {"x": 1, "y": [1.0, None], "z": 0}, "b": "text"}, "keys"),
        ({"a": {"x": 1, "y": [1.0]}, "b": "text"}, "$.y: 2 items != 1"),
        ({"a": old["a"]}, "b: removed"),
    ):
        added, changed = pin_changes(old, new)
        assert added == [] and len(changed) == 1 and why in changed[0], changed


def test_make_pins_writes_only_an_additive_view(tmp_path, monkeypatch, capsys):
    path = tmp_path / "pins.json"
    monkeypatch.setattr(make_pins, "PINS_PATH", path)
    path.write_text(dumps({"a": {"x": 1}}, key_order=True))
    monkeypatch.setattr(make_pins, "pins_view", lambda: {"a": {"x": 1}, "b": [2]})
    make_pins.main(["--check-additive"])
    assert "added b" in capsys.readouterr().out
    assert json.loads(path.read_text()) == {"a": {"x": 1}}
    make_pins.main([])
    assert path.read_text() == dumps({"a": {"x": 1}, "b": [2]}, key_order=True)
    monkeypatch.setattr(make_pins, "pins_view", lambda: {"a": {"x": 2}, "b": [2]})
    for argv in (["--check-additive"], []):
        with pytest.raises(SystemExit, match="not additive"):
            make_pins.main(argv)
    assert path.read_text() == dumps({"a": {"x": 1}, "b": [2]}, key_order=True)
