import json
import statistics

import pytest

from fp4bench import bytes_model as bm
from fp4bench import settings
from fp4bench.analysis import cells as cl
from fp4bench.analysis import checks as ck
from fp4bench.analysis import design, stats
from fp4bench.analysis.inputs import kv_cache_dtype
from fp4bench.core.argv import flag_value
from fp4bench.core.types import (
    Cell,
    Format,
    Hypothesis,
    KvDtype,
    Treatment,
    UnknownKvDtype,
    ValueKey,
)
from fp4bench.studies import expc
from fp4bench.studies.base import min_kv_tokens_for
from tests.analysis_runs import (
    BF16_PP,
    EXPB_CS,
    EXPB_NLL_SHIFT,
    FIVE,
    GOOD_REFERENCE,
    N_WINDOWS,
    batch_steps,
    c_argv,
    c_manifest,
    c_session,
    expb,
    expb_manifest,
    fail,
    manifest,
    pp,
    smoke,
    synthetic,
    typed_manifest,
    typed_servers,
    write_run,
)

MX, NV = Treatment.MX, Treatment.NV


def _sessions(servers):
    return cl.sessions_of(typed_servers(servers))


def test_treatment_order_is_the_configured_order_then_the_rest_sorted():
    nvv, nvc, mxp, nvnf = Treatment.NVV, Treatment.NVC, Treatment.MXP, Treatment.NVNF
    assert ck.treatment_order([nvv, nvc, NV, mxp, MX]) == [MX, NV, mxp, nvc, nvv]
    assert ck.treatment_order({nvv: 1, NV: 2, nvnf: 3}) == [NV, nvnf, nvv]


def test_r_ideal_or_none_is_the_bytes_models_or_none_for_an_unknown_kv_dtype():
    assert ck.r_ideal_or_none(8, 1664, KvDtype.FP8) == bm.r_ideal(8, 1664, KvDtype.FP8)
    assert ck.r_ideal_or_none(8, 1664, UnknownKvDtype("auto")) is None


@pytest.mark.parametrize(
    "stored, parsed",
    [
        (None, KvDtype.BF16),
        ("bfloat16", KvDtype.BF16),
        ("fp8_e5m2", KvDtype.FP8_E5M2),
        ("auto", UnknownKvDtype("auto")),
        ("FP8", UnknownKvDtype("FP8")),
        (8, UnknownKvDtype("8")),
    ],
)
def test_the_recorded_kv_cache_dtype_is_parsed_where_it_enters(stored, parsed):
    line = manifest() if stored is None else manifest(kv_cache_dtype=stored)
    kv = kv_cache_dtype(typed_manifest(line))
    assert kv == parsed and type(kv) is type(parsed)
    assert kv_cache_dtype(None) is KvDtype.BF16


def _note(servers, line=None):
    return ck.accuracy_note(_sessions(servers), typed_manifest(line or manifest()))


def test_the_accuracy_note_reports_bf16_mx_and_nv_mean_nll():
    note = _note(synthetic(1.0)[0])
    assert note.bf16 == pytest.approx(GOOD_REFERENCE["mean"])
    assert note.mx == pytest.approx(statistics.fmean(pp("MX")))
    assert note.nv == pytest.approx(statistics.fmean(pp("NV")))
    assert note.n_windows == N_WINDOWS and note.problems == [] and note.reference_run is None


def test_the_interval_is_a_seeded_bootstrap_over_the_paired_per_window_differences():
    servers = synthetic(1.0)[0]
    note = _note(servers)
    mean, lo, hi = stats.paired_bootstrap_ci(
        [nv - mx for nv, mx in zip(pp("NV"), pp("MX"), strict=True)],
        level=0.95,
        resamples=10_000,
        seed=0,
    )
    d = note.nv_minus_mx
    assert d is not None and (d.mean, d.lo, d.hi) == pytest.approx((mean, lo, hi), abs=1e-12)
    assert (d.level, d.resamples, d.seed) == (0.95, 10_000, 0)
    assert note.nv is not None and note.mx is not None
    assert d.lo < d.mean < d.hi < 0 and d.mean == pytest.approx(note.nv - note.mx)
    assert all(_note(servers) == note for _ in range(3))


def test_the_note_averages_a_treatments_sessions_per_window_and_uses_only_mx_and_nv():
    servers = synthetic(1.0)[0]
    base = _note(servers)
    for s in servers:
        if s["treatment"] == "NV" and s["round"] in (0, 1):
            s["nll_per_prompt"] = list(s["nll_per_prompt"])
            s["nll_per_prompt"][0] += 0.02 if s["round"] == 0 else -0.02
        if s["treatment"] in ("MXp", "NVa"):
            s["nll_per_prompt"] = [9.0] * N_WINDOWS
    note = _note(servers)
    assert note.nv_minus_mx is not None and base.nv_minus_mx is not None
    assert note.nv_minus_mx.mean == pytest.approx(base.nv_minus_mx.mean, abs=1e-12)
    assert note.mx == base.mx and note.nv == base.nv


@pytest.mark.parametrize(
    "breaker", ["no_per_prompt", "null_entry", "length_mismatch", "no_nv", "rounds_disagree"]
)
def test_without_usable_per_window_nlls_there_is_no_interval_but_the_means_stay(breaker):
    servers = synthetic(1.0)[0]
    for s in servers:
        if s["treatment"] == "NV":
            if breaker == "no_per_prompt":
                del s["nll_per_prompt"]
            elif breaker == "null_entry":
                s["nll_per_prompt"] = [None, *s["nll_per_prompt"][1:]]
            elif breaker == "length_mismatch" or (breaker == "rounds_disagree" and s["round"] == 2):
                s["nll_per_prompt"] = s["nll_per_prompt"][:-1]
    if breaker == "no_nv":
        servers = [s for s in servers if s["treatment"] != "NV"]
    note = _note(servers)
    assert note.nv_minus_mx is None and note.problems
    assert note.mx == pytest.approx(statistics.fmean(pp("MX")))


def test_the_note_without_a_bf16_reference_or_with_other_windows_says_so():
    servers = synthetic(1.0)[0]
    note = _note(servers, manifest(reference=None))
    assert note.bf16 is None and note.nv_minus_mx is not None
    assert any("bf16" in p.lower() for p in note.problems)
    short = {"per_prompt": BF16_PP[:-2], "mean": GOOD_REFERENCE["mean"]}
    assert any("windows" in p for p in _note(servers, manifest(reference=short)).problems)


def test_the_accuracy_note_ignores_a_failed_session():
    note = _note(fail(smoke()[0], "NV"))
    assert note.nv is None and note.mx == pytest.approx(statistics.fmean(pp("MX")))


def _reference(tmp_path, servers=None, line=None):
    if servers is None:
        servers = synthetic(1.0, (8, 32, 128), treatments=FIVE)[0]
    return write_run(tmp_path, servers, [], [], line or manifest(treatments=FIVE), name="main")


def _expb_sessions(line=None):
    servers = expb()[0]
    for row in servers:
        row["nll"] += EXPB_NLL_SHIFT[row["treatment"]]
    return _sessions(servers), typed_manifest(line or expb_manifest())


def test_the_reference_run_gives_its_mean_nll_per_treatment_and_the_difference(tmp_path):
    ref = _reference(tmp_path)
    cmp = ck.reference_nll_comparison(*_expb_sessions(), ref)
    assert cmp.run == str(ref) and cmp.problems == []
    assert (cmp.kv_cache_dtype, cmp.this_kv_cache_dtype) == ("bfloat16", "fp8")
    assert list(cmp.per_treatment) == ["MX", "NV", "MXp"]
    for t, shift in EXPB_NLL_SHIFT.items():
        v = cmp.per_treatment[Treatment(t)]
        assert v.reference == pytest.approx(statistics.fmean(pp(t)))
        assert v.this == pytest.approx(statistics.fmean(pp(t)) + shift)
        assert v.this_minus_reference == pytest.approx(shift)


def test_a_reference_treatment_without_sessions_is_none_and_listed(tmp_path):
    ref = _reference(tmp_path, synthetic(1.0, (8, 32, 128), treatments=("MX", "NV", "NVa"))[0])
    cmp = ck.reference_nll_comparison(*_expb_sessions(), ref)
    mxp = cmp.per_treatment[Treatment.MXP]
    assert (mxp.this, mxp.reference, mxp.this_minus_reference) == (
        pytest.approx(statistics.fmean(pp("MXp")) + 0.02),
        None,
        None,
    )
    assert any("MXp" in p for p in cmp.problems)


def test_a_reference_with_the_same_kv_dtype_or_other_windows_is_flagged(tmp_path):
    same = expb_manifest()
    same["inputs"]["nll_prompts_sha256"] = "aaa"
    other = expb_manifest()
    other["inputs"]["nll_prompts_sha256"] = "bbb"
    cmp = ck.reference_nll_comparison(*_expb_sessions(other), _reference(tmp_path, line=same))
    assert any("same KV cache dtype" in p for p in cmp.problems)
    assert any("nll_prompts_sha256" in p for p in cmp.problems)


def test_an_empty_reference_run_is_a_problem_not_a_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    cmp = ck.reference_nll_comparison(*_expb_sessions(), empty)
    assert any("no sessions" in p for p in cmp.problems)
    assert all(v.reference is None for v in cmp.per_treatment.values())


def test_the_hbm_peak_is_b200_nominal_8_tb_s():
    assert settings.HBM_PEAK_BYTES_S == 8e12


def test_the_effective_bandwidth_is_the_bytes_models_step_bytes_over_the_median_step():
    servers, m1, _ = synthetic(1.0, (8, 32, 128), treatments=FIVE)
    step = batch_steps(servers, m1)
    bw = ck.effective_bandwidth(step, KvDtype.BF16, settings.M1_MEAN_CONTEXT)
    assert set(bw) == set(FIVE)
    for t in FIVE:
        fmt = Format.MXFP4 if t.startswith("MX") else Format.NVFP4
        for c in (8, 32, 128):
            median = statistics.median(v for (_, tt, cc), v in step.items() if tt == t and cc == c)
            nbytes = bm.step_bytes(c, settings.M1_MEAN_CONTEXT, fmt, kv_dtype=KvDtype.BF16)
            row = bw[Treatment(t)][c]
            assert row.format == fmt and row.rounds == 5
            assert row.step_s == pytest.approx(median) and row.bytes == pytest.approx(nbytes)
            assert row.bytes_per_s == pytest.approx(nbytes / median)
            assert row.fraction_of_peak == pytest.approx(nbytes / median / 8e12)


def test_the_effective_bandwidth_counts_the_kv_dtypes_bytes_and_knows_only_its_formats():
    servers, m1, _ = expb()
    step = batch_steps(servers, m1)
    bw = ck.effective_bandwidth(step, KvDtype.FP8, settings.M1_MEAN_CONTEXT)
    assert set(bw) == {"MX", "NV", "MXp"}
    assert bw[NV][512].bytes == pytest.approx(
        bm.step_bytes(512, settings.M1_MEAN_CONTEXT, Format.NVFP4, kv_dtype=KvDtype.FP8)
    )
    int4 = UnknownKvDtype("int4")
    unknown = ck.effective_bandwidth(step, int4, settings.M1_MEAN_CONTEXT)[NV][512]
    assert unknown.step_s is not None
    assert (unknown.bytes, unknown.bytes_per_s, unknown.fraction_of_peak) == (None, None, None)
    smoke_servers, smoke_m1, _ = smoke()
    assert set(
        ck.effective_bandwidth(
            batch_steps(smoke_servers, smoke_m1), KvDtype.BF16, settings.M1_MEAN_CONTEXT
        )
    ) == {"MX", "NV"}
    assert EXPB_CS == (128, 256, 512)


def test_the_effective_bandwidth_of_a_cell_without_a_usable_step_is_none():
    step = {
        ValueKey(0, MX, 8): float("nan"),
        ValueKey(1, MX, 8): float("nan"),
        ValueKey(0, NV, 8): 0.004,
    }
    bw = ck.effective_bandwidth(step, KvDtype.BF16, settings.M1_MEAN_CONTEXT)
    assert bw[MX][8].step_s is None and bw[MX][8].rounds == 0
    assert bw[MX][8].bytes_per_s is None and bw[MX][8].bytes is not None
    assert bw[NV][8].rounds == 1


def test_the_registered_cells_follow_section_16():
    assert expc.REGISTERED_CELLS == (
        (1, 1024),
        (1, 4096),
        (1, 16384),
        (1, 32768),
        (1, 65536),
        (1, 127360),
        (8, 15360),
        (32, 3360),
        (128, 360),
    )
    assert {design.kv_tokens(cell, expc.MEAN_CONTEXT_EXTRA) for cell in expc.BATCH_ARM} == {128_000}
    assert max(min_kv_tokens_for(*cell) for cell in expc.REGISTERED_CELLS) == 193_536
    assert design.arms_of(Cell(1, 127360)) == ["token", "batch"]
    assert expc.EXPC.cells == expc.REGISTERED_CELLS


def test_the_h_tokens_curve_reproduces_the_preregistered_points():
    assert design.h_tokens_delta_ms(settings.M1_MEAN_CONTEXT) == pytest.approx(0.42)
    assert round(design.h_tokens_delta_ms(128_000), 2) == 0.07
    assert expc.PREDICTED_DELTA_MS[Hypothesis.H_TOKENS][Cell(1, 127360)] == 0.07


def test_the_design_check_passes_section_16_and_flags_every_deviation():
    good = ck.design_check(typed_manifest(c_manifest()))
    assert good.problems == [] and good.kind == "every pre-registered cell"
    bad = c_manifest(
        cells=(*expc.REGISTERED_CELLS, (1, 2048)),
        max_model_len=4096,
        hf_overrides="",
        m2_duration_s=30,
        kv_cache_dtype="fp8",
        treatments=("MX", "NV", "NVa"),
        gpu_memory_utilization=0.95,
    )
    design = ck.design_check(typed_manifest(bad))
    assert design.kind == "deviates from §16" and design.unregistered_cells == [(1, 2048)]
    assert len(design.problems) == 7
    assert any("gpu_memory_utilization is 0.95" in p for p in design.problems)
    as_object = c_manifest(hf_overrides={"max_position_embeddings": 131072})
    assert ck.design_check(typed_manifest(as_object)).problems == []
    smoke_design = ck.design_check(typed_manifest(c_manifest(cells=((1, 1024), (128, 360)))))
    assert smoke_design.kind == "a subset of the cells (smoke)" and smoke_design.problems == []
    assert ck.design_check(None).problems[0] == "no manifest"


def test_the_served_argv_check_passes_section_16_argv():
    sessions = [
        cl.sessions_of(typed_servers([c_session(r, t) for r in range(2) for t in ("MX", "NV")]))
    ]
    assert ck.server_argv_check(sessions[0]) == ck.ArgvCheck(4, [])
    assert ck.server_argv_check([]).problems == ["no sessions: no served argv to check"]


@pytest.mark.parametrize(
    "change, problem",
    [
        (
            lambda a: a.__setitem__(a.index("--max-model-len") + 1, "4096"),
            "served --max-model-len '4096', not 131072",
        ),
        (
            lambda a: a.__setitem__(
                a.index("--hf-overrides") + 1, '{"max_position_embeddings": 65536}'
            ),
            "served --hf-overrides",
        ),
        (
            lambda a: a.__delitem__(
                slice(a.index("--hf-overrides"), a.index("--hf-overrides") + 2)
            ),
            "served --hf-overrides None",
        ),
        (
            lambda a: a.__setitem__(
                a.index("--hf-overrides") + 1,
                json.dumps(
                    {"max_position_embeddings": 131072, "rope_scaling": {"rope_type": "yarn"}}
                ),
            ),
            "the served argv mentions rope_scaling, yarn",
        ),
        (
            lambda a: a.extend(["--rope-parameters", "{}"]),
            "the served argv mentions rope_parameters",
        ),
        (lambda a: a.clear(), "no server_argv"),
    ],
)
def test_the_served_argv_check_flags_deviations(change, problem):
    rows = [c_session(1, "NV"), c_session(1, "MX")]
    change(rows[0]["server_argv"])
    problems = ck.server_argv_check(cl.sessions_of(typed_servers(rows))).problems
    assert problems and all(p.startswith("round 1 NV: ") for p in problems)
    assert any(p.startswith("round 1 NV: " + problem) for p in problems)
    design = ck.design_check(typed_manifest(c_manifest()), cl.sessions_of(typed_servers(rows)))
    assert set(problems) <= set(design.problems)


def test_the_served_argv_flag_forms():
    assert flag_value(["--max-model-len=131072"], "--max-model-len") == "131072"
    assert (
        flag_value(["--max-model-len", "4096", "--max-model-len", "131072"], "--max-model-len")
        == "131072"
    )
    assert flag_value(["--max-model-len-x", "4096"], "--max-model-len") is None
    with pytest.raises(ValueError, match="--max-model-len"):
        flag_value(["--max-model-len"], "--max-model-len")
    assert "--max-model-len" in c_argv("MX")


def test_a_served_flag_without_a_value_is_a_problem_and_the_other_flags_are_still_checked():
    rows = [c_session(1, "NV")]
    argv = rows[0]["server_argv"]
    argv[argv.index("--hf-overrides") + 1] = '{"max_position_embeddings": 65536}'
    argv.extend(["--rope-parameters", "{}", "--max-model-len"])
    window, override, rope = ck.server_argv_check(cl.sessions_of(typed_servers(rows))).problems
    assert window.startswith("round 1 NV: --max-model-len has no value")
    assert override.startswith("round 1 NV: served --hf-overrides")
    assert rope == "round 1 NV: the served argv mentions rope_parameters"


def test_the_nll_crosscheck_compares_with_the_main_run_and_its_windows():
    rows = [c_session(r, t) for r in range(5) for t in ("MX", "NV", "MXp")]
    for row in rows:
        if row["treatment"] == "NV":
            row["nll"] = 1.9
    nll = ck.nll_crosscheck(cl.sessions_of(typed_servers(rows)), typed_manifest(c_manifest()))
    assert nll.windows == "the main run's" and nll.main_run == "results/full-1"
    assert nll.per_treatment[MX].this == pytest.approx(1.9092 + 0.002)
    assert nll.per_treatment[MX].this_minus_main == pytest.approx(0.002)
    assert nll.per_treatment[NV].this_minus_main == pytest.approx(1.9 - 1.8238)
    assert nll.per_treatment[Treatment.MXP].sessions == 5
    assert list(nll.per_treatment) == ["MX", "NV", "MXp"]
    other = typed_manifest({"inputs": {"nll_prompts_sha256": "abc"}})
    assert ck.nll_crosscheck([], other).windows.startswith("DIFFERENT")
    assert ck.nll_crosscheck([], None).windows.startswith("unknown")
