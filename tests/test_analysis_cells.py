import math
import statistics

import pytest

from fp4bench.analysis import cells as cl
from fp4bench.core.types import Cell, Treatment, ValueKey
from tests.analysis_runs import (
    DROP,
    FAIL_ERROR,
    FOUR,
    cell_steps,
    fail,
    failed_row,
    m1_row,
    smoke,
    synthetic,
    typed_m1,
    typed_m2,
    typed_servers,
)

MX, NV, MXP = Treatment.MX, Treatment.NV, Treatment.MXP


def test_m1_steps_are_medians_per_round_treatment_and_cell_without_warmup_or_dead_sessions():
    servers, m1, _ = synthetic(1.03)
    steps = cell_steps(servers, m1)
    assert steps[ValueKey(0, MX, Cell(8, 1024))] == pytest.approx(0.004)
    assert len(steps) == 5 * 4 * 2


def test_an_unmeasurable_rep_makes_its_cell_nan_not_a_median_of_the_rest():
    servers, m1, _ = synthetic(1.0)
    victim = next(r for r in m1 if r["session_id"] == "r1_NV-x" and r["c"] == 8 and not r["warmup"])
    victim["step_s"] = None
    steps = cell_steps(servers, m1)
    assert math.isnan(steps[ValueKey(1, NV, Cell(8, 1024))])
    assert steps[ValueKey(1, NV, Cell(32, 1024))] == pytest.approx(0.004 * 1.002 * 0.999)
    assert cl.unusable(steps) == [(1, NV, Cell(8, 1024))]


def test_m1_steps_key_rows_by_their_prompt_length():
    rows = [m1_row(c=1, prompt_len=p, step_s=s) for p, s in ((1024, 0.006), (32768, 0.009))]
    steps = cl.m1_steps(typed_m1(rows), {"r0_MX-x"})
    assert steps == {(0, MX, Cell(1, 1024)): 0.006, (0, MX, Cell(1, 32768)): 0.009}


@pytest.mark.parametrize(
    "change", [{"c": 0}, {"c": None}, {"prompt_len": -1}, {"warmup": None}, {"round": -1}]
)
def test_m1_steps_leave_out_a_row_without_a_cell_a_bool_warmup_or_a_round(change):
    rows = [m1_row(), m1_row(set=2, **change)]
    assert cl.m1_steps(typed_m1(rows), {"r0_MX-x"}) == {(0, MX, Cell(8, 1024)): 0.004}


def test_a_row_stored_without_its_prompt_length_has_a_cell_only_in_a_study_by_batch():
    stored = m1_row(c=1)
    rows = typed_m1([m1_row(c=1, prompt_len=4096, step_s=0.007), stored])
    assert cl.row_cell(rows[1]) == Cell(1, 1024)
    assert cl.row_cell(rows[1], by_cell=True) is None
    assert cl.row_cell(rows[0], by_cell=True) == Cell(1, 4096)
    assert cl.m1_steps(rows, {"r0_MX-x"}) == {
        (0, MX, Cell(1, 4096)): 0.007,
        (0, MX, Cell(1, 1024)): 0.004,
    }
    assert cl.m1_steps(rows, {"r0_MX-x"}, by_cell=True) == {(0, MX, Cell(1, 4096)): 0.007}
    assert cl.row_cell(typed_m1([m1_row(c=1, prompt_len=1024)])[0], by_cell=True) == Cell(1, 1024)


def test_m2_values_keep_valid_cells_only_and_a_missing_metric_as_nan():
    servers, _, m2 = synthetic(1.0)
    m2[0]["valid"] = False
    m2[1]["itl_p50_ms"] = None
    sids = cl.session_ids(typed_servers(servers))
    tput = cl.m2_values(typed_m2(m2), sids, "aggregate_tps")
    assert (m2[0]["round"], m2[0]["treatment"], m2[0]["c"]) not in tput
    assert len(tput) == 5 * 4 * 2 - 1
    itl = cl.m2_values(typed_m2(m2), sids, "itl_p50_ms")
    assert math.isnan(itl[ValueKey(m2[1]["round"], m2[1]["treatment"], m2[1]["c"])])


@pytest.mark.parametrize("bad", [0.0, -0.001, float("nan"), float("inf"), None, True])
def test_usable_is_a_finite_positive_number(bad):
    assert cl.usable(0.004) and not cl.usable(bad)


def test_paired_rounds_need_both_treatments_with_usable_values():
    values = {ValueKey(r, t, 8): 0.004 for r in range(4) for t in (MX, NV)}
    values[ValueKey(1, NV, 8)] = float("nan")
    del values[ValueKey(3, MX, 8)]
    assert cl.paired_rounds(values, NV, MX, 8) == [0, 2]


def test_median_value_is_over_usable_rounds_only():
    values = {
        ValueKey(0, MX, 8): 0.004,
        ValueKey(1, MX, 8): 0.010,
        ValueKey(2, MX, 8): 0.0041,
        ValueKey(3, MX, 8): float("nan"),
    }
    assert cl.median_value(values, MX, 8) == pytest.approx(0.0041)
    assert cl.usable_rounds_of(values, MX, 8) == 3
    assert cl.median_value(values, NV, 8) is None


def test_order_effect_splits_the_median_step_by_which_wave_ran_first():
    servers, m1, _ = synthetic(1.0)
    for row in m1:
        if row["treatment"] == "MX" and row["c"] == 8 and not row["warmup"]:
            row["step_s"] = 0.0040 if row["first"] == "n1" else 0.0044
    eff = cl.order_effect(typed_m1(m1), cl.session_ids(typed_servers(servers)))
    cell = eff[MX][Cell(8, 1024)]
    assert cell.n1 == pytest.approx(0.0040) and cell.n2 == pytest.approx(0.0044)
    assert cell.n2_over_n1 == pytest.approx(1.1)
    assert cell.count_n1 == 5 * 1 and cell.count_n2 == 5 * 2
    assert eff[NV][Cell(8, 1024)].n2_over_n1 == pytest.approx(1.0, abs=2e-3)
    assert set(eff) == set(FOUR) and set(eff[NV]) == {Cell(8, 1024), Cell(32, 1024)}


@pytest.mark.parametrize("first", ["", None, DROP])
def test_order_effect_ignores_warmup_dead_sessions_and_rows_of_neither_wave(first):
    servers, m1, _ = synthetic(1.0)
    for row in m1:
        if row["session_id"] == "r0_MX-x":
            row["first"] = first
            if first is DROP:
                del row["first"]
    eff = cl.order_effect(typed_m1(m1), cl.session_ids(typed_servers(servers)))
    assert (eff[MX][Cell(8, 1024)].count_n1, eff[MX][Cell(8, 1024)].count_n2) == (4, 8)
    assert eff[NV][Cell(8, 1024)].count_n1 == 5


def test_a_failed_session_is_not_a_session():
    servers, _, _ = smoke()
    rows = typed_servers(fail(servers, "NVc"))
    assert Treatment.NVC not in {t for _, t in cl.latest_sessions(rows)}
    assert "r0_NVc-x" not in cl.session_ids(rows)
    assert [(s.round, s.treatment, s.error) for s in cl.failed_sessions(rows)] == [
        (0, Treatment.NVC, FAIL_ERROR)
    ]
    assert cl.failed_sessions(typed_servers(servers)) == []


def test_a_failed_rerun_replaces_the_earlier_good_session_instead_of_resurrecting_it():
    servers, _, _ = smoke()
    good = next(s for s in servers if s["treatment"] == "NVc")
    rows = typed_servers([*servers, failed_row({**good, "session_id": "r0_NVc-rerun"})])
    assert Treatment.NVC not in {t for _, t in cl.latest_sessions(rows)}
    assert [s.session_id for s in cl.failed_sessions(rows)] == ["r0_NVc-rerun"]


def test_a_good_rerun_replaces_an_earlier_failed_session():
    servers, _, _ = smoke()
    rows = typed_servers(
        [*fail(servers, "NVc"), next(s for s in servers if s["treatment"] == "NVc")]
    )
    assert Treatment.NVC in {t for _, t in cl.latest_sessions(rows)}
    assert cl.failed_sessions(rows) == []


def test_the_values_leave_out_the_rows_a_failed_session_wrote_before_it_died():
    servers, m1, m2 = smoke()
    sids = cl.session_ids(typed_servers(fail(servers, "NVc")))
    left = {"MX", "NV", "NVx", "NVt", "NVd", "NVv"}
    assert {t for _, t, _ in cl.m1_steps(typed_m1(m1), sids)} == left
    assert {t for _, t, _ in cl.m2_values(typed_m2(m2), sids, "aggregate_tps")} == left
    assert set(cl.order_effect(typed_m1(m1), sids)) == left


def test_mean_nll_needs_a_usable_nll_in_every_session():
    servers, _, _ = synthetic(1.0)
    sessions = cl.sessions_of(typed_servers(servers))
    assert cl.mean_nll(sessions, NV) == pytest.approx(
        statistics.fmean(s["nll"] for s in servers if s["treatment"] == "NV")
    )
    next(s for s in servers if s["treatment"] == "NV" and s["round"] == 3)["nll"] = None
    assert cl.mean_nll(cl.sessions_of(typed_servers(servers)), NV) is None
    assert cl.mean_nll(sessions, Treatment.NVX) is None


def test_cell_key_and_label():
    assert Cell(128, 360).json_key == "C=128,P=360"
    assert Cell(1, 127360).label == "(1, 127,360)"
