import copy
import json
import pickle
import statistics
from dataclasses import replace
from typing import Any

import pytest

from fp4bench.analysis import cells as cl
from fp4bench.analysis import verdicts as vd
from fp4bench.analysis.compare import ContrastResult, RatioResult, ratio_table
from fp4bench.analysis.gates import g5b_nll
from fp4bench.analysis.inputs import primary_batches
from fp4bench.analysis.stats import OverallVerdict
from fp4bench.core.types import Cell, LinearBackend, RatioName, Treatment, Verdict
from fp4bench.studies import kernel_scan as scan_rule
from fp4bench.studies import model
from fp4bench.studies.expb import EXPB
from fp4bench.studies.expc import EXPC, MAIN_RUN_R_BATCH1, REPRO_CELL
from fp4bench.studies.main import FULL
from fp4bench.studies.smoke import SMOKE
from tests.analysis_runs import (
    CUTE_DSL,
    EXPB_CS,
    FAIL_ERROR,
    SCAN_FACTORS,
    SMOKE_CS,
    batch_steps,
    expb,
    expb_manifest,
    fail,
    manifest,
    pp,
    recording_args,
    set_fields,
    set_nll,
    smoke,
    smoke_manifest,
    synthetic,
    typed_m1,
    typed_manifest,
    typed_servers,
)

MX, NV = Treatment.MX, Treatment.NV
SCAN = scan_rule.NVA_SCAN.treatments


def _verdict(nv: Any = 1.0, cs=(8, 32, 128), rounds=5, line: Any = "ok", treatments=None):
    servers, m1, _ = expb(nv, rounds) if treatments == "B" else synthetic(nv, cs, rounds=rounds)
    step = batch_steps(servers, m1)
    batches = sorted({c for _, _, c in step})
    r = ratio_table(step, NV, MX, batches)
    line = manifest(rounds=5) if line == "ok" else line
    return vd.h_eq_verdict(typed_manifest(line), step, r, FULL.ratio(RatioName.R)), r


@pytest.mark.parametrize(
    "nv, verdict",
    [
        (1.0, "validated"),
        (1.03, "nullified_mxfp4_faster"),
        (0.97, "nullified_nvfp4_faster"),
        ({8: 0.97, 32: 1.03, 128: 1.0}, "nullified_mixed"),
        ({8: 1.0, 32: 1.0, 128: 1.02}, "inconclusive"),
    ],
)
def test_the_h_eq_verdict_over_the_primary_batches(nv, verdict):
    assert _verdict(nv)[0] == verdict


@pytest.mark.parametrize(
    "rounds, line, expected",
    [
        (5, None, "incomplete (no manifest)"),
        (5, {"problems": []}, "incomplete (manifest has no valid protocol.rounds)"),
        (5, manifest(rounds=10), "incomplete (5/10 rounds)"),
        (3, manifest(rounds=5), "incomplete (3/5 rounds)"),
        (6, manifest(rounds=5), "incomplete (6/5 rounds)"),
        (1, manifest(rounds=1), "incomplete (1/1 rounds; a CI needs at least 2)"),
    ],
)
def test_the_h_eq_verdict_needs_every_primary_batch_with_exactly_the_registered_rounds(
    rounds, line, expected
):
    assert _verdict(rounds=rounds, line=line)[0] == expected


def test_an_unusable_primary_cell_makes_the_verdict_incomplete():
    servers, m1, _ = synthetic(1.0, (8, 32, 128))
    for r in m1:
        if r["session_id"] == "r1_NV-x" and r["c"] == 8 and not r["warmup"]:
            r["step_s"] = None
    step = batch_steps(servers, m1)
    verdict = vd.h_eq_verdict(
        typed_manifest(manifest()),
        step,
        ratio_table(step, NV, MX, (8, 32, 128)),
        FULL.ratio(RatioName.R),
    )
    assert verdict == "incomplete (4/5 rounds)"


def test_the_primary_batches_come_from_the_protocol_and_fall_back_to_the_main_runs():
    assert primary_batches(None) == FULL.primary_batches == (8, 32, 128)
    assert primary_batches(typed_manifest(manifest())) == (8, 32, 128)
    assert primary_batches(typed_manifest(expb_manifest())) == (256, 512)


@pytest.mark.parametrize(
    "bad", [[], "256,512", [256, "512"], [256, True], [0, 512], None, [256.0, 512]]
)
def test_invalid_primary_batches_fall_back_to_the_main_runs(bad):
    assert primary_batches(typed_manifest(expb_manifest(primary_concurrencies=bad))) == (8, 32, 128)


def test_the_verdict_uses_the_protocols_primary_batches():
    verdict, _ = _verdict(treatments="B", line=expb_manifest())
    assert verdict == "validated"
    assert _verdict(treatments="B", line=manifest())[0] == "incomplete (0/5 rounds)"
    nv_at_128 = {128: 0.95, 256: 1.0, 512: 1.0}
    verdict, r = _verdict(nv_at_128, treatments="B", line=expb_manifest())
    assert r[128].verdict == Verdict.NVFP4_FASTER and verdict == "validated"
    assert (
        _verdict({128: 1.0, 256: 1.0, 512: 0.95}, treatments="B", line=expb_manifest())[0]
        == "nullified_nvfp4_faster"
    )


def _r(verdicts: dict) -> dict:
    return {c: RatioResult(5, (0, 1, 2, 3, 4), 1.0, 0.9, 1.1, v) for c, v in verdicts.items()}


@pytest.mark.parametrize(
    "verdict, per_c, reading",
    [
        ("validated", {}, "H_B,eq holds"),
        (
            "nullified_nvfp4_faster",
            {256: Verdict.NVFP4_FASTER, 512: Verdict.NVFP4_FASTER},
            "H_B,nv holds",
        ),
        (
            "nullified_nvfp4_faster",
            {256: Verdict.EQUIVALENT, 512: Verdict.NVFP4_FASTER},
            "H_B,nv does not hold",
        ),
        (
            "nullified_mxfp4_faster",
            {256: Verdict.MXFP4_FASTER, 512: Verdict.EQUIVALENT},
            "contradicts H_B,nv",
        ),
        ("inconclusive", {256: Verdict.INCONCLUSIVE, 512: Verdict.EQUIVALENT}, "neither"),
        (vd.Incomplete("3/5 rounds"), {}, "no reading"),
    ],
)
def test_the_experiment_b_reading_of_the_verdict(verdict, per_c, reading):
    assert reading in vd.h_b_reading(verdict, _r(per_c), (256, 512))


def test_an_incomplete_decision_is_its_own_type_and_reads_as_before():
    v = vd.Incomplete("3/5 rounds")
    assert v == "incomplete (3/5 rounds)" and v.why == "3/5 rounds"
    assert f"`{v}`" == "`incomplete (3/5 rounds)`" and json.dumps(v) == '"incomplete (3/5 rounds)"'
    for copied in (pickle.loads(pickle.dumps(v)), copy.deepcopy(v)):  # noqa: S301 - its own bytes
        assert type(copied) is vd.Incomplete and copied == v and copied.why == v.why
    assert type(_verdict(rounds=3)[0]) is vd.Incomplete
    assert type(_verdict(line=None)[0]) is vd.Incomplete
    assert type(_verdict(1.0)[0]) is OverallVerdict


@pytest.mark.parametrize(
    "lo, hi, expected",
    [
        (-0.1, 0.1, vd.Effect.NO_EFFECT),
        (-0.05, 0.02, vd.Effect.NO_EFFECT),
        (-0.7, -0.5, vd.Effect.SHRINKS),
        (-0.4, -0.1000001, vd.Effect.SHRINKS),
        (0.2, 0.3, vd.Effect.GROWS),
        (-0.15, 0.0, vd.Effect.INCONCLUSIVE),
        (-0.12, -0.05, vd.Effect.INCONCLUSIVE),
        (0.05, 0.2, vd.Effect.INCONCLUSIVE),
        (-0.3, 0.3, vd.Effect.INCONCLUSIVE),
    ],
)
def test_classify_effect(lo, hi, expected):
    assert vd.classify_effect(lo, hi, 0.1) == expected


def _answer(e_tok: ContrastResult, e_batch: ContrastResult, rounds):
    return vd.batch_vs_tokens_answer(("E_tok", e_tok), ("E_batch", e_batch), rounds)


def _e(classification, n=5) -> ContrastResult:
    cells = (Cell(1, 127360), Cell(1, 1024))
    return ContrastResult(
        cells, n, tuple(range(n)), (), 0.0 if n else None, None, None, classification, {}
    )


@pytest.mark.parametrize(
    "tok, batch, expected",
    [
        (vd.Effect.NO_EFFECT, vd.Effect.SHRINKS, vd.Answer.BATCH_DRIVES),
        (vd.Effect.SHRINKS, vd.Effect.NO_EFFECT, vd.Answer.TOKENS_DRIVE),
        (vd.Effect.SHRINKS, vd.Effect.SHRINKS, vd.Answer.BOTH),
        (vd.Effect.NO_EFFECT, vd.Effect.NO_EFFECT, vd.Answer.NEITHER),
        (vd.Effect.INCONCLUSIVE, vd.Effect.SHRINKS, vd.Answer.INCONCLUSIVE),
        (vd.Effect.NO_EFFECT, vd.Effect.GROWS, vd.Answer.INCONCLUSIVE),
        (vd.Effect.GROWS, vd.Effect.NO_EFFECT, vd.Answer.INCONCLUSIVE),
    ],
)
def test_the_answer_matrix(tok, batch, expected):
    assert _answer(_e(tok), _e(batch), 5) == expected


def test_the_answer_needs_the_registered_rounds():
    shrinks, none = vd.Effect.SHRINKS, vd.Effect.NO_EFFECT
    assert type(_answer(_e(none), _e(shrinks), 5)) is vd.Answer
    assert type(_answer(_e(none, 4), _e(shrinks), 5)) is vd.Incomplete
    assert type(_answer(_e(none), _e(shrinks), None)) is vd.Incomplete
    assert _answer(_e(none, 4), _e(shrinks), 5) == "incomplete (4/5 rounds)"
    assert _answer(_e(none, 6), _e(shrinks, 6), 5) == "incomplete (6/5 rounds)"
    assert _answer(_e(None, 1), _e(None, 1), 1) == (
        "incomplete (1/1 rounds; a CI needs at least 2)"
    )
    assert _answer(_e(None, 0), _e(shrinks), 5) == ("incomplete (no data for E_tok)")
    assert _answer(_e(None, 0), _e(None, 0), 5) == ("incomplete (no data for E_tok and E_batch)")
    assert _answer(_e(none), _e(shrinks), None).startswith("incomplete (manifest")


def test_the_answer_names_the_effects_it_is_given():
    shrinks = vd.Effect.SHRINKS
    assert vd.batch_vs_tokens_answer(("tokens", _e(None, 0)), ("batches", _e(shrinks)), 5) == (
        "incomplete (no data for tokens)"
    )
    assert (
        vd.batch_vs_tokens_answer(("tokens", _e(vd.Effect.NO_EFFECT)), ("batches", _e(shrinks)), 5)
        == vd.Answer.BATCH_DRIVES
    )


def test_the_extension_note():
    line = vd.extend_line(10)
    assert line == "§16: extend to 10 rounds (`--rounds 10`)" and EXPC.extension_rounds == 10
    assert vd.extension_note(True, vd.Answer.BATCH_DRIVES, 5, True, 10) is None
    assert (
        vd.extension_note(False, vd.Answer.BATCH_DRIVES, 5, True, 10) == f"**{line}** — G3 failed."
    )
    assert vd.extension_note(True, vd.Answer.INCONCLUSIVE, 5, True, 10) == (
        f"**{line}** — the answer is inconclusive."
    )
    both = vd.extension_note(False, vd.Answer.INCONCLUSIVE, 5, True, 10)
    assert both is not None and "G3 failed and the answer is inconclusive" in both
    assert vd.extension_note(False, vd.Answer.INCONCLUSIVE, 5, False, 10) is None
    assert vd.extension_note(True, vd.Incomplete("4/5 rounds"), 5, True, 10) is None
    done = vd.extension_note(False, vd.Answer.BATCH_DRIVES, 10, True, 10)
    assert done is not None and done.startswith(
        "§16's extension to 10 rounds is in this run already"
    )
    assert vd.extension_note(False, vd.Answer.INCONCLUSIVE, 1, True, None) is None
    assert vd.extension_note(False, vd.Answer.BATCH_DRIVES, 5, True, 12) == (
        f"**{vd.extend_line(12)}** — G3 failed."
    )


@pytest.mark.parametrize(
    "r, status",
    [(0.934, "pass"), (0.9435, "pass"), (0.950, "fail"), (0.920, "fail"), (None, "no data")],
)
def test_the_config_reproduction(r, status):
    result = (
        None
        if r is None
        else RatioResult(5, (0, 1, 2, 3, 4), r, r - 0.01, r + 0.01, Verdict.NVFP4_FASTER)
    )
    repro = vd.config_reproduction(result)
    assert repro.status == status and repro.cell == REPRO_CELL
    assert repro.target == MAIN_RUN_R_BATCH1
    if r is not None:
        assert repro.difference == pytest.approx(r - 0.934) and repro.n_rounds == 5


def _scan(servers, m1, line: Any = "ok", spec: scan_rule.KernelScanSpec = scan_rule.NVA_SCAN):
    """kernel_scan as the analysis calls it: with G5b's details and the manifest line."""
    rows = typed_servers(servers)
    sessions = cl.sessions_of(rows)
    typed = typed_manifest(smoke_manifest() if line == "ok" else line)
    scan = vd.kernel_scan(
        batch_steps(servers, m1), rows, SMOKE_CS, spec, g5b=g5b_nll(sessions, typed), manifest=typed
    )
    assert scan is not None
    return scan


SLOW = {1: 1.3, 32: 1.3, 128: 1.3}


def _tied(**per_treatment):
    """SCAN_FACTORS with every scan kernel slow except those given."""
    return {**SCAN_FACTORS, **dict.fromkeys(SCAN, SLOW), **per_treatment}


def test_the_eligibility_bounds_and_the_tie_order_are_the_smokes_scan():
    assert SMOKE.kernel_scan == scan_rule.NVA_SCAN
    spec = scan_rule.NVA_SCAN
    assert spec.max_nll_diff == 0.01 and spec.tie_margin == 0.01 and spec.reference == NV
    assert spec.tiebreak_c == 128 and spec.tiebreak_c in SMOKE.batches
    assert SCAN == ("NVc", "NVt", "NVd", "NVv")


def test_the_scan_gives_the_median_step_per_batch_and_the_ratio_to_nv():
    scan = _scan(*smoke()[:2])
    assert scan.treatments == ["NV", *SCAN]
    assert scan.median_step_s[NV][32] == pytest.approx(0.004)
    assert scan.median_step_s[Treatment.NVC][128] == pytest.approx(0.004 * 0.90)
    assert scan.ratio_to_nv[NV] == {c: pytest.approx(1.0) for c in SMOKE_CS}
    assert scan.ratio_to_nv[Treatment.NVT][32] == pytest.approx(0.97)
    assert scan.ratio_to_nv[Treatment.NVD][1] == pytest.approx(0.80)


def test_the_rule_selects_the_fastest_eligible_non_cute_dsl_kernel_at_c32():
    scan = _scan(*smoke()[:2])
    assert scan.selection_c == 32 == scan_rule.NVA_SCAN.selection_c
    assert scan.selected == "NVt" and scan.selected_kernel == "FlashInferTrtllmNvFp4LinearKernel"
    assert scan.selected_server_args == ("--linear-backend", "flashinfer_trtllm")
    assert scan.candidates == {
        "NVc": pytest.approx(0.0044),
        "NVt": pytest.approx(0.00388),
        "NVd": pytest.approx(0.0042),
        "NVv": pytest.approx(0.0052),
    }
    assert scan.margin is not None and (scan.margin.fastest, scan.margin.runner_up) == (
        "NVt",
        "NVd",
    )
    assert scan.margin.gap == pytest.approx(0.0042 / 0.00388 - 1)
    for t, e in scan.eligibility.items():
        assert e.eligible is True and e.reasons == [], t
        assert (e.kernel_ok, e.failed, e.g5b, e.nll_ok) == (True, False, True, True)
        assert e.nll_diff == pytest.approx(0.0)
    assert scan.eligibility[Treatment.NVT].step_c32_s == pytest.approx(0.00388)
    assert scan.eligibility[Treatment.NVT].nll == pytest.approx(statistics.fmean(pp("NVt")))


@pytest.mark.parametrize("fast", ["NVc", "NVd", "NVv"])
def test_the_rule_follows_whichever_scan_kernel_is_fastest(fast):
    factors = {
        **SCAN_FACTORS,
        **{t: {1: 1.0, 32: 1.2, 128: 1.0} for t in SCAN},
        fast: {1: 1.0, 32: 0.8, 128: 1.0},
    }
    assert _scan(*smoke(factors)[:2]).selected == fast


def test_the_rule_does_not_prefer_nv_or_nvx_even_when_they_are_fastest():
    factors = {**SCAN_FACTORS, "NV": {1: 1.0, 32: 0.5, 128: 1.0}, "NVx": 0.5}
    assert _scan(*smoke(factors)[:2]).selected == "NVt"


@pytest.mark.parametrize("record", [[CUTE_DSL], []])
def test_a_scan_kernel_that_did_not_run_its_kernel_cannot_be_selected(record):
    servers, m1, _ = smoke()
    set_fields(servers, "NVt", linear_kernels=record)
    scan = _scan(servers, m1)
    assert scan.selected == "NVd" and "NVt" not in scan.candidates
    assert scan.kernel_matches[Treatment.NVT] is False and scan.kernel_matches[Treatment.NVD]
    assert scan.observed_kernels[Treatment.NVT] == record


@pytest.mark.parametrize("default", [None, "SomeOtherDefaultKernel"])
def test_the_cute_dsl_kernel_is_never_selected_even_if_a_scan_kernel_expects_it(
    monkeypatch, default
):
    if default:
        monkeypatch.setattr(model, "DEFAULT_NVFP4_KERNEL", default)
    line = recording_args(smoke_manifest(), Treatment.NVT, model.PINNED_KERNEL)
    servers, m1, _ = smoke()
    set_fields(servers, "NVt", linear_kernels=[CUTE_DSL])
    scan = _scan(servers, m1, line)
    assert scan.expected_kernels[Treatment.NVT] == CUTE_DSL
    assert scan.kernel_matches[Treatment.NVT] is True
    assert scan.selected == "NVd" and "NVt" not in scan.candidates


def test_the_scan_expects_the_kernel_of_the_recorded_args_else_the_tables():
    servers, m1, _ = smoke()
    line = recording_args(smoke_manifest(), Treatment.NVC, model.pinned_to(LinearBackend.CUTLASS))
    set_fields(servers, "NVc", linear_kernels=["CutlassNvFp4LinearKernel"])
    scan = _scan(servers, m1, line)
    assert scan.expected_kernels[Treatment.NVC] == "CutlassNvFp4LinearKernel"
    assert scan.kernel_matches[Treatment.NVC] is True
    assert scan.expected_kernels[Treatment.NVT] == model.TREATMENTS[Treatment.NVT].linear_kernel
    assert _scan(servers, m1).kernel_matches[Treatment.NVC] is False


def test_no_selection_without_a_usable_scan_kernel_at_c32():
    servers, m1, _ = smoke()
    m1 = [r for r in m1 if not (r["c"] == 32 and r["treatment"] in SCAN)]
    scan = _scan(servers, m1)
    assert scan.selected is None and scan.candidates == {} and scan.margin is None
    assert scan.reason is not None and "C=32" in scan.reason
    assert all(any("C=32" in r for r in e.reasons) for e in scan.eligibility.values())


def test_a_scan_kernel_missing_from_the_run_is_listed_without_data():
    scan = _scan(*smoke({t: f for t, f in SCAN_FACTORS.items() if t != "NVv"})[:2])
    assert scan.selected == "NVt"
    assert scan.median_step_s[Treatment.NVV] == {} and scan.kernel_matches[Treatment.NVV] is None
    assert scan.eligibility[Treatment.NVV].reasons == ["it has no session"]


def test_the_scan_takes_the_median_over_rounds_not_the_mean():
    servers, m1, _ = smoke(
        _tied(NVc={1: 1.0, 32: 1.0, 128: 1.0}, NVt={1: 1.0, 32: 1.05, 128: 1.0}), rounds=3
    )
    for r in m1:
        if r["treatment"] == "NVc" and r["c"] == 32 and r["round"] == 1 and not r["warmup"]:
            r["step_s"] = 0.0100
    scan = _scan(servers, m1, line=manifest(rounds=3))
    assert scan.candidates[Treatment.NVC] == pytest.approx(0.004 * 1.004)
    assert scan.selected == "NVc"


@pytest.mark.parametrize(
    "degradation, eligible", [(0.039, True), (0.041, False), (0.021, True), (0.019, False)]
)
def test_a_kernel_is_eligible_only_within_the_nll_bound_of_nv_in_either_direction(
    degradation, eligible
):
    servers, m1, _ = smoke()
    set_nll(servers, "NVt", degradation)
    scan = _scan(servers, m1)
    e = scan.eligibility[Treatment.NVT]
    assert e.eligible is eligible and e.nll_ok is eligible
    assert e.nll_diff == pytest.approx(degradation - 0.030)
    assert scan.selected == ("NVt" if eligible else "NVd")
    if not eligible:
        assert any("NLL" in reason for reason in e.reasons)


def test_the_nll_bound_is_the_specs():
    servers, m1, _ = smoke()
    set_nll(servers, "NVt", 0.034)
    assert _scan(servers, m1).selected == "NVt"
    tight = replace(scan_rule.NVA_SCAN, max_nll_diff=0.002)
    assert _scan(servers, m1, spec=tight).selected == "NVd"


def test_without_a_usable_nll_of_nv_no_kernel_can_be_compared_to_it():
    servers, m1, _ = smoke()
    set_nll(servers, "NV", None)
    scan = _scan(servers, m1)
    assert scan.selected is None and scan.nv_nll is None
    assert all(
        not e.eligible and any("NV" in r for r in e.reasons) for e in scan.eligibility.values()
    )


@pytest.mark.parametrize("bad", [None, "missing"])
def test_a_scan_kernel_without_a_usable_nll_is_not_eligible(bad):
    servers, m1, _ = smoke()
    if bad is None:
        set_nll(servers, "NVt", None)
    else:
        for s in servers:
            if s["treatment"] == "NVt":
                del s["nll"]
    scan = _scan(servers, m1)
    assert scan.eligibility[Treatment.NVT].eligible is False and scan.selected == "NVd"


def test_g5b_is_judged_per_kernel_and_a_kernel_failing_it_is_not_eligible():
    servers, m1, _ = smoke()
    set_nll(servers, "MX", 0.9)
    scan = _scan(servers, m1)
    assert scan.selected == "NVt" and scan.eligibility[Treatment.NVT].g5b is True
    set_nll(servers, "NV", 0.6)
    set_nll(servers, "NVt", 0.6)
    e = _scan(servers, m1).eligibility[Treatment.NVT]
    assert e.nll_ok is True and e.g5b is False and e.eligible is False
    assert [r for r in e.reasons if "G5b" in r]


def test_without_a_usable_bf16_reference_no_kernel_passes_g5b():
    servers, m1, _ = smoke()
    scan = _scan(servers, m1, line=smoke_manifest(reference=None))
    assert scan.selected is None
    assert all(e.g5b is False and not e.eligible for e in scan.eligibility.values())
    assert vd.g5b_passes(None, Treatment.NVT) is False


def test_a_kernel_without_an_m1_step_at_c32_is_not_eligible():
    servers, m1, _ = smoke()
    m1 = [r for r in m1 if not (r["treatment"] == "NVt" and r["c"] == 32)]
    e = _scan(servers, m1).eligibility[Treatment.NVT]
    assert e.step_c32_s is None and e.eligible is False and [r for r in e.reasons if "C=32" in r]


def test_a_failed_scan_kernel_is_ineligible_even_with_steps_in_the_cells():
    servers, m1, _ = smoke()
    rows = fail(servers, "NVt")
    sessions = cl.sessions_of(typed_servers(rows))
    every_cell = cl.by_batch(cl.m1_steps(typed_m1(m1), {s["session_id"] for s in servers}))
    scan = vd.kernel_scan(
        every_cell,
        typed_servers(rows),
        SMOKE_CS,
        scan_rule.NVA_SCAN,
        g5b=g5b_nll(sessions, typed_manifest(smoke_manifest())),
    )
    assert scan is not None
    e = scan.eligibility[Treatment.NVT]
    assert e.failed is True and e.eligible is False and e.error == FAIL_ERROR
    assert [r for r in e.reasons if "failed" in r]
    assert scan.selected == "NVd" and scan.kernel_matches[Treatment.NVT] is None
    assert scan.failed == ["NVt"]


def test_there_is_no_scan_or_crosscheck_in_a_run_without_scan_treatments():
    servers, m1, _ = synthetic(1.0)
    rows = typed_servers(servers)
    assert vd.kernel_scan(batch_steps(servers, m1), rows, (8, 32), scan_rule.NVA_SCAN) is None
    assert vd.nvx_crosscheck(batch_steps(servers, m1), (8, 32), scan_rule.NVX_CROSSCHECK) is None


def test_the_scan_is_there_when_its_only_sessions_failed():
    servers, m1, _ = smoke({"MX": 1.0, "NV": 1.0, "NVc": 1.1})
    rows = typed_servers(fail(servers, "NVc"))
    m1 = [r for r in m1 if r["treatment"] != "NVc"]
    scan = vd.kernel_scan(batch_steps(servers, m1), rows, SMOKE_CS, scan_rule.NVA_SCAN)
    assert scan is not None and scan.selected is None


def test_a_clear_winner_at_c32_is_selected_whatever_happens_at_c128():
    scan = _scan(
        *smoke(_tied(NVc={1: 1.0, 32: 1.00, 128: 1.5}, NVt={1: 1.0, 32: 1.0101, 128: 0.8}))[:2]
    )
    m = scan.margin
    assert scan.selected == "NVc" and m is not None
    assert (m.fastest, m.runner_up, m.tie, m.decided_by) == ("NVc", "NVt", False, "C=32")
    assert m.gap == pytest.approx(0.0101) and m.tiebreak is None


def test_a_tie_at_c32_goes_to_the_kernel_faster_at_c128():
    scan = _scan(
        *smoke(_tied(NVc={1: 1.0, 32: 1.000, 128: 1.00}, NVd={1: 1.0, 32: 1.0099, 128: 0.90}))[:2]
    )
    m = scan.margin
    assert scan.selected == "NVd" and m is not None and m.tiebreak is not None
    assert (m.fastest, m.runner_up, m.tie, m.decided_by) == ("NVc", "NVd", True, "C=128")
    assert m.tiebreak.steps == {"NVc": pytest.approx(0.004), "NVd": pytest.approx(0.0036)}
    assert m.tiebreak.tie is False


@pytest.mark.parametrize(
    "factors, selected",
    [
        ({"NVd": {1: 1.0, 32: 1.000, 128: 1.000}, "NVt": {1: 1.0, 32: 1.005, 128: 1.004}}, "NVt"),
        ({"NVc": {1: 1.0, 32: 1.000, 128: 1.0}, "NVv": {1: 1.0, 32: 1.004, 128: 0.995}}, "NVc"),
    ],
)
def test_a_tie_at_both_batches_goes_to_the_earlier_kernel_in_the_fixed_order(factors, selected):
    scan = _scan(*smoke(_tied(**factors))[:2])
    assert scan.selected == selected and scan.margin is not None
    assert scan.margin.decided_by == "order"
    assert scan.margin.tiebreak is not None and scan.margin.tiebreak.tie is True


@pytest.mark.parametrize("fast128, expected", [(0.9899, "NVd"), (0.9905, "NVc")])
def test_the_tie_margin_at_c128_is_one_percent_too(fast128, expected):
    factors = _tied(NVc={1: 1.0, 32: 1.000, 128: 1.0}, NVd={1: 1.0, 32: 1.002, 128: fast128})
    assert _scan(*smoke(factors)[:2]).selected == expected


def test_only_the_fastest_and_the_runner_up_take_part_in_the_tie():
    scan = _scan(
        *smoke(
            _tied(
                NVc={1: 1.0, 32: 1.000, 128: 1.0},
                NVt={1: 1.0, 32: 1.004, 128: 1.0},
                NVd={1: 1.0, 32: 1.006, 128: 0.5},
            )
        )[:2]
    )
    assert scan.selected == "NVc" and scan.margin is not None and scan.margin.runner_up == "NVt"


def test_a_tie_without_a_step_at_c128_is_decided_by_the_order():
    servers, m1, _ = smoke(
        _tied(NVd={1: 1.0, 32: 1.000, 128: 1.0}, NVt={1: 1.0, 32: 1.005, 128: 0.5})
    )
    m1 = [r for r in m1 if not (r["treatment"] == "NVd" and r["c"] == 128)]
    scan = _scan(servers, m1)
    m = scan.margin
    assert scan.selected == "NVt" and m is not None and m.decided_by == "order"
    assert m.tiebreak is not None and m.tiebreak.steps[Treatment.NVD] is None
    assert m.tiebreak.tie is None


def test_an_ineligible_kernel_takes_no_part_in_the_tie_and_a_single_one_has_no_margin():
    servers, m1, _ = smoke(
        _tied(
            NVc={1: 1.0, 32: 1.000, 128: 1.0},
            NVt={1: 1.0, 32: 1.002, 128: 0.5},
            NVd={1: 1.0, 32: 1.30, 128: 1.0},
        )
    )
    set_nll(servers, "NVc", 0.06)
    scan = _scan(servers, m1)
    m = scan.margin
    assert scan.selected == "NVt" and m is not None
    assert (m.fastest, m.runner_up, m.tie) == ("NVt", "NVd", False)
    servers, m1, _ = smoke()
    for t in ("NVc", "NVd", "NVv"):
        set_nll(servers, t, 0.1)
    scan = _scan(servers, m1)
    assert scan.selected == "NVt" and scan.margin is None


def test_the_tie_margin_and_tiebreak_batch_are_the_specs():
    wide = replace(scan_rule.NVA_SCAN, tie_margin=0.05)
    factors = _tied(NVc={1: 1.0, 32: 1.000, 128: 1.0}, NVd={1: 1.0, 32: 1.04, 128: 0.9})
    assert _scan(*smoke(factors)[:2]).selected == "NVc"
    assert _scan(*smoke(factors)[:2], spec=wide).selected == "NVd"
    servers, m1, _ = smoke(
        _tied(NVc={1: 1.0, 32: 1.000, 128: 0.5}, NVd={1: 0.9, 32: 1.04, 128: 1.0})
    )
    assert _scan(servers, m1, spec=wide).selected == "NVc"
    assert _scan(servers, m1, spec=replace(wide, tiebreak_c=1)).selected == "NVd"


def test_the_nvx_crosscheck_is_the_step_ratio_per_batch_and_skips_unusable_steps():
    servers, m1, _ = smoke({**SCAN_FACTORS, "NVx": {1: 1.02, 32: 0.99, 128: 1.04}})
    cross = vd.nvx_crosscheck(batch_steps(servers, m1), SMOKE_CS, scan_rule.NVX_CROSSCHECK)
    assert cross is not None
    assert {c: v.ratio for c, v in cross.items()} == {
        1: pytest.approx(1.02),
        32: pytest.approx(0.99),
        128: pytest.approx(1.04),
    }
    assert cross[32].nvx_step_s == pytest.approx(0.004 * 0.99)
    assert cross[32].nv_step_s == pytest.approx(0.004)
    for r in m1:
        if r["treatment"] == "NVx" and r["c"] == 128 and not r["warmup"]:
            r["step_s"] = None
    cross = vd.nvx_crosscheck(batch_steps(servers, m1), SMOKE_CS, scan_rule.NVX_CROSSCHECK)
    assert cross is not None and set(cross) == {1, 32}


def test_expb_batches_are_the_experiment_b_concurrencies():
    assert EXPB_CS == (128, 256, 512) == EXPB.batches
