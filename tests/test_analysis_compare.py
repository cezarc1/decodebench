import statistics

import pytest

from fp4bench import settings
from fp4bench.analysis import cells as cl
from fp4bench.analysis import compare as cp
from fp4bench.analysis import stats
from fp4bench.core.types import Cell, Treatment, ValueKey, Verdict
from tests.analysis_runs import FIVE, batch_steps, synthetic, typed_m2, typed_servers

MX, NV, NVA, MXP, NVNF = (Treatment.MX, Treatment.NV, Treatment.NVA, Treatment.MXP, Treatment.NVNF)


def _by_batch(nv=1.0, nva=1.0, nvnf=1.0, cs=(8, 32), treatments=("MX", "NV", "NVa", "MXp")):
    """M1 steps by batch, M2 tok/s and ITL of a synthetic run."""
    servers, m1, m2 = synthetic(nv, cs, nva_factor=nva, nvnf_factor=nvnf, treatments=treatments)
    sids = cl.session_ids(typed_servers(servers))
    return (
        batch_steps(servers, m1),
        cl.m2_values(typed_m2(m2), sids, "aggregate_tps"),
        cl.m2_values(typed_m2(m2), sids, "itl_p50_ms"),
    )


def _pair(rounds=5, c=8, nv_factor=1.02):
    return {
        ValueKey(r, t, c): 0.004 * (1 + 0.003 * r) * (nv_factor if t == NV else 1.0)
        for r in range(rounds)
        for t in (MX, NV)
    }


def test_a_three_percent_mxfp4_advantage_is_detected_and_the_aa_is_equivalent():
    step, tput, _ = _by_batch(nv=1.03)
    r = cp.ratio_table(step, NV, MX, (8, 32))
    assert r[8].ratio == pytest.approx(1.03, abs=2e-3) and r[8].verdict == Verdict.MXFP4_FASTER
    assert cp.ratio_table(step, MXP, MX, (8, 32))[8].verdict == Verdict.EQUIVALENT
    assert cp.ratio_table(tput, MX, NV, (8, 32))[32].ratio == pytest.approx(1.03, abs=2e-3)
    assert cp.ratio_table(_by_batch()[0], NV, MX, (8, 32))[32].verdict == Verdict.EQUIVALENT


def test_a_ratio_is_the_geometric_mean_with_the_paired_log_ratio_ci():
    values = _pair()
    r = cp.ratio(values, NV, MX, 8)
    nums = [values[ValueKey(i, NV, 8)] for i in range(5)]
    dens = [values[ValueKey(i, MX, 8)] for i in range(5)]
    est, lo, hi = stats.ratio_ci(stats.paired_log_ratios(nums, dens), settings.CI_LEVEL)
    assert r is not None and (r.ratio, r.lo, r.hi) == (est, lo, hi)
    assert r.n_rounds == 5 and r.rounds == (0, 1, 2, 3, 4) and r.skipped == ()


@pytest.mark.parametrize("bad", [0.0, -0.001, float("nan"), float("inf")])
def test_an_unusable_value_is_skipped_with_a_warning_instead_of_crashing(bad, capsys):
    values = _pair()
    values[ValueKey(2, NV, 8)] = bad
    res = cp.ratio_table(values, NV, MX, (8,))
    assert res[8].n_rounds == 4 and res[8].skipped == ((2, NV, 8),)
    assert res[8].ratio == pytest.approx(1.02, abs=1e-6)
    err = capsys.readouterr().err
    assert "warning" in err and "round 2" in err and "NV" in err


def test_a_ratio_table_leaves_out_a_key_with_fewer_than_two_usable_rounds(capsys):
    values = _pair(rounds=2)
    values[ValueKey(0, MX, 8)] = 0.0
    assert cp.ratio_table(values, NV, MX, (8,)) == {}
    assert "warning" in capsys.readouterr().err


def test_an_absent_treatment_gives_an_empty_table_not_an_error():
    step, _, _ = _by_batch(treatments=("MX", "NV", "MXp"))
    assert cp.ratio_table(step, NVA, MX, (8, 32)) == {}
    assert cp.ratio(step, NVA, NV, 8, stats.relabel_for_k) is None


def test_r_d_k_and_aa_are_ratios_of_their_pairs():
    step, _, _ = _by_batch(nv=1.03, nva=1.06)
    r, d = cp.ratio_table(step, NV, MX, (8, 32)), cp.ratio_table(step, NVA, MX, (8, 32))
    k = cp.ratio_table(step, NVA, NV, (8, 32))
    assert r[8].ratio == pytest.approx(1.03, abs=3e-3) and d[8].ratio == pytest.approx(
        1.06, abs=3e-3
    )
    assert k[8].ratio == pytest.approx(1.06 / 1.03, abs=3e-3)
    assert (r[8].verdict, d[8].verdict) == (Verdict.MXFP4_FASTER, Verdict.MXFP4_FASTER)


@pytest.mark.parametrize(
    "nva, verdict",
    [(1.06, Verdict.PINNED_FASTER), (0.94, Verdict.ALT_FASTER), (1.0, Verdict.EQUIVALENT)],
)
def test_k_uses_pinned_and_alt_labels(nva, verdict):
    step, _, _ = _by_batch(nva=nva)
    k = cp.ratio_table(step, NVA, NV, (8, 32), relabel=stats.relabel_for_k)
    assert {v.verdict for v in k.values()} == {verdict}
    unlabelled = cp.ratio_table(step, NVA, NV, (8, 32))
    assert (
        unlabelled[8].verdict
        == {
            Verdict.PINNED_FASTER: Verdict.MXFP4_FASTER,
            Verdict.ALT_FASTER: Verdict.NVFP4_FASTER,
            Verdict.EQUIVALENT: Verdict.EQUIVALENT,
        }[verdict]
    )


def test_f_and_phi_split_r_into_the_format_and_the_fusion():
    step, _, _ = _by_batch(nv=1.03, nvnf=1.01, treatments=FIVE)
    f = cp.ratio_table(step, NVNF, MX, (8, 32))
    phi = cp.ratio_table(step, NV, NVNF, (8, 32), relabel=stats.relabel_for_phi)
    r = cp.ratio_table(step, NV, MX, (8, 32))
    assert f[8].ratio == pytest.approx(1.01, abs=3e-3)
    assert phi[8].ratio == pytest.approx(1.03 / 1.01, abs=3e-3)
    assert f[8].ratio * phi[8].ratio == pytest.approx(r[8].ratio, rel=1e-12)


@pytest.mark.parametrize(
    "nv, verdict",
    [(1.06, Verdict.FUSION_SLOWER), (0.94, Verdict.FUSION_FASTER), (1.0, Verdict.EQUIVALENT)],
)
def test_phi_uses_the_fusion_labels_and_f_the_format_labels(nv, verdict):
    step, _, _ = _by_batch(nv=nv, treatments=FIVE)
    phi = cp.ratio_table(step, NV, NVNF, (8, 32), relabel=stats.relabel_for_phi)
    assert {v.verdict for v in phi.values()} == {verdict}
    assert {v.verdict for v in cp.ratio_table(step, NVNF, MX, (8, 32)).values()} == {
        Verdict.EQUIVALENT
    }
    slow_nvnf, _, _ = _by_batch(nvnf=1.05, treatments=FIVE)
    assert {v.verdict for v in cp.ratio_table(slow_nvnf, NVNF, MX, (8, 32)).values()} == {
        Verdict.MXFP4_FASTER
    }


def test_the_r_decomposition_compares_r_with_f_times_phi():
    step, _, _ = _by_batch(nv=1.03, nvnf=1.01, treatments=FIVE)
    r, f = cp.ratio_table(step, NV, MX, (8, 32)), cp.ratio_table(step, NVNF, MX, (8, 32))
    phi = cp.ratio_table(step, NV, NVNF, (8, 32), relabel=stats.relabel_for_phi)
    check = cp.r_decomposition(r, f, phi)
    assert set(check) == {8, 32}
    assert check[8].r == r[8].ratio and check[8].f == f[8].ratio
    assert check[8].f_times_phi == pytest.approx(f[8].ratio * phi[8].ratio)
    assert check[8].f_times_phi_over_r == pytest.approx(1.0, abs=1e-12)
    assert check[8].n_rounds == {"R": 5, "F": 5, "Phi": 5}
    assert cp.r_decomposition(r, {}, {}) == {}


def test_the_r_decomposition_differs_from_r_when_the_rounds_differ():
    step, _, _ = _by_batch(nv=1.03, nvnf=1.01, treatments=FIVE)
    step[ValueKey(2, NVNF, 8)] = 0.004 * 1.5
    del step[ValueKey(3, NVNF, 8)]
    r, f = cp.ratio_table(step, NV, MX, (8, 32)), cp.ratio_table(step, NVNF, MX, (8, 32))
    phi = cp.ratio_table(step, NV, NVNF, (8, 32), relabel=stats.relabel_for_phi)
    check = cp.r_decomposition(r, f, phi)
    assert check[8].n_rounds == {"R": 5, "F": 4, "Phi": 4}
    assert check[8].f_times_phi_over_r != pytest.approx(1.0, abs=1e-6)
    assert check[32].f_times_phi_over_r == pytest.approx(1.0, abs=1e-12)


CELL = Cell(1, 1024)


def test_delta_and_r_on_known_data():
    mx, nv = [0.0062, 0.0061, 0.0063], [0.0058, 0.0057, 0.0058]
    values: dict[ValueKey[Cell], float] = {}
    for r, (a, b) in enumerate(zip(mx, nv, strict=True)):
        values[ValueKey(r, MX, CELL)], values[ValueKey(r, NV, CELL)] = a, b
    d = cp.delta_ms(values, MX, NV, CELL)
    diffs = [(a - b) * 1000 for a, b in zip(mx, nv, strict=True)]
    _, lo, hi = stats.mean_ci(diffs, settings.CI_LEVEL)
    assert d is not None and d.n_rounds == 3 and d.rounds == (0, 1, 2)
    assert d.mean_ms == pytest.approx(statistics.fmean(diffs)) == pytest.approx(0.4333333)
    assert (d.lo_ms, d.hi_ms) == pytest.approx((lo, hi))
    assert d.per_round_ms == pytest.approx(dict(enumerate(diffs)))
    r = cp.ratio(values, NV, MX, CELL)
    est, rlo, rhi = stats.ratio_ci(stats.paired_log_ratios(nv, mx), settings.CI_LEVEL)
    assert r is not None and (r.ratio, r.lo, r.hi) == pytest.approx((est, rlo, rhi))
    assert r.verdict == stats.classify(rlo, rhi, settings.DELTA) == Verdict.NVFP4_FASTER


def test_one_round_gives_point_estimates_without_a_ci():
    values = {ValueKey(0, MX, CELL): 0.006, ValueKey(0, NV, CELL): 0.0056}
    r = cp.ratio(values, NV, MX, CELL)
    assert r is not None and r.ratio == pytest.approx(0.0056 / 0.006)
    assert r.lo is None and r.verdict is None
    d = cp.delta_ms(values, MX, NV, CELL)
    assert d is not None and d.mean_ms == pytest.approx(0.4) and d.lo_ms is None
    assert cp.ratio(values, MXP, MX, CELL) is None and cp.delta_ms({}, MX, NV, CELL) is None
    with pytest.raises(ValueError, match="no CI"):
        _ = r.ci


def _contrast_values(n_rounds=5):
    a, b = Cell(1, 127360), Cell(1, 1024)
    values: dict[ValueKey[Cell], float] = {}
    for r in range(n_rounds):
        for cell, delta in ((a, 0.40 + 0.01 * r), (b, 0.42)):
            values[ValueKey(r, MX, cell)] = 0.010
            values[ValueKey(r, NV, cell)] = 0.010 - delta / 1000
    return values, (a, b)


def test_a_contrast_is_the_paired_difference_of_two_cells_deltas():
    values, cells = _contrast_values()
    e = cp.contrast(values, cells, MX, NV, classify=lambda lo, hi: f"{lo:.2f}..{hi:.2f}")
    per_round = [0.40 + 0.01 * r - 0.42 for r in range(5)]
    mean, lo, hi = stats.mean_ci(per_round, settings.CI_LEVEL)
    assert e.n_rounds == 5 and e.missing_cells == () and e.cells == cells
    assert (e.mean_ms, e.lo_ms, e.hi_ms) == pytest.approx((mean, lo, hi))
    assert e.classification == f"{e.lo_ms:.2f}..{e.hi_ms:.2f}"
    assert cp.contrast(values, cells, MX, NV).classification is None


def test_a_contrast_without_a_shared_round_names_the_cells_without_data():
    values, cells = _contrast_values()
    values = {k: v for k, v in values.items() if k[2] != cells[0]}
    e = cp.contrast(values, cells, MX, NV)
    assert (e.n_rounds, e.mean_ms, e.classification) == (0, None, None)
    assert e.missing_cells == (cells[0],)


def test_a_one_round_contrast_has_no_ci_and_no_classification():
    values, cells = _contrast_values(n_rounds=1)
    e = cp.contrast(values, cells, MX, NV, classify=lambda lo, hi: "x")
    assert e.n_rounds == 1 and e.lo_ms is None and e.classification is None


@pytest.mark.parametrize(
    "lo, hi, verdict, ok",
    [
        (0.995, 1.004, Verdict.EQUIVALENT, True),
        (1.001, 1.010, Verdict.EQUIVALENT, False),
        (0.97, 1.03, Verdict.INCONCLUSIVE, False),
        (None, None, None, None),
    ],
)
def test_the_aa_ratio_rule_needs_an_equivalent_ci_that_contains_one(lo, hi, verdict, ok):
    aa = cp.RatioResult(5, (0, 1, 2, 3, 4), 1.0, lo, hi, verdict)
    assert cp.aa_ratio_passes(aa) is ok
    assert cp.aa_ratio_passes(None) is None


@pytest.mark.parametrize(
    "lo, hi, ok",
    [
        (-0.25, 0.25, True),
        (-0.1, 0.05, True),
        (-0.26, 0.1, False),
        (0.01, 0.2, False),
        (-0.2, -0.01, False),
        (None, None, False),
    ],
)
def test_the_aa_effect_rule(lo, hi, ok):
    assert cp.aa_effect_passes(lo, hi, 0.25) is ok


def test_ratio_json_by_batch_and_by_cell():
    r = cp.RatioResult(
        2, (0, 1), 1.01, 1.0, 1.02, Verdict.EQUIVALENT, (ValueKey(3, NV, Cell(8, 1024)),)
    )
    assert cp.ratio_json(r, by_cell=False) == {
        "n_rounds": 2,
        "ratio": 1.01,
        "lo": 1.0,
        "hi": 1.02,
        "verdict": "equivalent",
        "skipped": [(3, "NV", 8)],
    }
    assert cp.ratio_json(r, by_cell=True) == {
        "n_rounds": 2,
        "rounds": [0, 1],
        "ratio": 1.01,
        "lo": 1.0,
        "hi": 1.02,
        "verdict": "equivalent",
    }
