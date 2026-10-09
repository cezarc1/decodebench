import json
from pathlib import Path
from typing import Any

import pytest

from fp4bench import settings
from fp4bench.analysis import gates as gt
from fp4bench.analysis.cells import m1_steps, session_ids
from fp4bench.analysis.compare import ContrastResult, contrast
from fp4bench.analysis.inputs import resolve_checkpoint
from fp4bench.core.types import Cell, Gate, RatioName, Treatment, Verdict
from fp4bench.studies import model
from fp4bench.studies.expb import EXPB
from fp4bench.studies.expc import AA_EFFECT_MARGIN_MS, EXPC, REGISTERED_CELLS
from fp4bench.studies.smoke import SMOKE
from tests.analysis_runs import (
    ACT_QUANT_LINE,
    C_REPS,
    CLEAN_CKPT,
    CUTE_DSL,
    DROP,
    FALLBACK_LINE,
    FIVE,
    FOUR,
    GOOD_REFERENCE,
    H_BATCH,
    SMOKE_CS,
    TEL,
    aa_results,
    c_build,
    c_manifest,
    drop_m1,
    expb_manifest,
    expect_kernel,
    fail,
    graph_hash,
    kernel_line,
    manifest,
    set_fields,
    set_nll,
    smoke,
    synthetic,
    typed_m1,
    typed_m2,
    typed_manifest,
    typed_servers,
)

GATE_NAMES = (
    "G1_kernels",
    "G2_bytes",
    "G3_aa",
    "G4_env_throttle",
    "G5a_structure",
    "G5b_nll",
    "G6a_integrity",
    "G6b_m2_validity",
    "G7_same_gpu_per_round",
)

AA = EXPC.ratio(RatioName.AA)


def _gates(
    servers,
    m1,
    m2,
    aa=None,
    batches=(8, 32),
    checkpoint: Any = "ok",
    manifest_line: Any = "ok",
    checkpoint_source=None,
    spec=None,
):
    line = typed_manifest(manifest() if manifest_line == "ok" else manifest_line)
    spec = spec or gt.batch_gate_spec(batches, line)
    return gt.gate_report(
        typed_servers(servers),
        typed_m1(m1),
        typed_m2(m2),
        line,
        CLEAN_CKPT if checkpoint == "ok" else checkpoint,
        checkpoint_source,
        spec,
        gt.AaRatios(aa_results(batches) if aa is None else aa, batches),
    )


def _five(cs=(8, 32)):
    return synthetic(1.0, cs, treatments=FIVE)


def test_a_clean_synthetic_run_passes_every_gate():
    servers, m1, m2 = synthetic(1.0)
    gates = _gates(servers, m1, m2)
    assert {k: v["pass"] for k, v in gates.items()} == dict.fromkeys(GATE_NAMES, True)
    assert tuple(gates) == GATE_NAMES == tuple(Gate)


def test_no_sessions_means_no_gate_passes():
    for spec in (gt.batch_gate_spec((), None), gt.GateSpec(cells=(), m2=False, by_cell=True)):
        gates = gt.gate_report([], [], [], None, None, None, spec, gt.AaRatios({}, ()))
        assert tuple(gates) == spec.names
        assert all(g == {"pass": False, "reason": "no sessions"} for g in gates.values())
    assert Gate.G6B not in gt.GateSpec(cells=(), m2=False, by_cell=True).names


def test_the_gate_report_flags_a_gpu_change_a_capture_size_and_an_invalid_m2_cell():
    servers, m1, m2 = synthetic(1.0)
    servers[1]["gpu_uuid"] = "GPU-2"
    servers[2]["cudagraph_capture_sizes"] = [8]
    m2[3]["valid"], m2[3]["invalid_reasons"] = False, ["underfilled"]
    gates = _gates(servers, m1, m2, aa={})
    assert gates[Gate.G7]["pass"] is False and gates[Gate.G6A]["pass"] is False
    assert gates[Gate.G6A]["capture_size_sessions"] == [servers[2]["session_id"]]
    assert gates[Gate.G6B]["pass"] is False
    assert gates[Gate.G6B]["invalid_m2_cells"] == [
        [m2[3]["round"], m2[3]["treatment"], m2[3]["c"], ["underfilled"]]
    ]
    assert gates[Gate.G4]["pass"] is True


def test_g1_requires_a_manifest_without_problems():
    servers, m1, m2 = synthetic(1.0)
    assert _gates(servers, m1, m2, manifest_line=None)[Gate.G1]["pass"] is False
    bad = _gates(servers, m1, m2, manifest_line=manifest(problems=["torch: expected 2.13.0"]))
    assert bad[Gate.G1]["pass"] is False
    assert bad[Gate.G1]["manifest_problems"] == ["torch: expected 2.13.0"]
    no_field = {"protocol": {"rounds": 5}}
    assert _gates(servers, m1, m2, manifest_line=no_field)[Gate.G1]["pass"] is False
    assert _gates(servers, m1, m2)[Gate.G1]["manifest_problems"] == []


def test_g1_expects_the_configured_kernel_class_of_every_treatment():
    servers, m1, m2 = synthetic(1.0)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is True and g1["mismatched_sessions"] == []
    assert g1["expected"] == {t: model.TREATMENTS[Treatment(t)].linear_kernel for t in FOUR}
    assert g1["observed"]["MX"] == g1["observed"]["MXp"] == ["FlashInferMxFp4LinearKernel"]
    assert g1["observed"]["NV"] == [CUTE_DSL]
    assert g1["observed"]["NVa"] == ["FlashInferCudnnNvFp4LinearKernel"]


def test_g1_fails_a_wrong_kernel_and_names_the_session():
    servers, m1, m2 = synthetic(1.0)
    set_fields(servers, "NV", r=2, linear_kernels=["FlashInferCutlassNvFp4LinearKernel"])
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False
    assert g1["mismatched_sessions"] == [
        [2, "NV", ["FlashInferCutlassNvFp4LinearKernel"], CUTE_DSL]
    ]


@pytest.mark.parametrize(
    "record", [[], None, DROP, ["FlashInferMxFp4LinearKernel", "MarlinMxFp4LinearKernel"]]
)
def test_g1_fails_an_empty_missing_or_double_kernel_record(record):
    servers, m1, m2 = synthetic(1.0)
    set_fields(servers, "MX", r=1, linear_kernels=record)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False and [m[:2] for m in g1["mismatched_sessions"]] == [[1, "MX"]]


def test_g1_fails_the_aa_replica_on_another_kernel_than_mx():
    servers, m1, m2 = synthetic(1.0)
    set_fields(servers, "MXp", linear_kernels=["MarlinMxFp4LinearKernel"])
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False and {m[1] for m in g1["mismatched_sessions"]} == {"MXp"}


def test_g1_follows_nvas_configured_kernel(monkeypatch):
    servers, m1, m2 = synthetic(1.0)
    expect_kernel(monkeypatch, Treatment.NVA, "FlashInferCutlassNvFp4LinearKernel")
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False and {m[1] for m in g1["mismatched_sessions"]} == {"NVa"}
    set_fields(servers, "NVa", linear_kernels=["FlashInferCutlassNvFp4LinearKernel"])
    assert _gates(servers, m1, m2)[Gate.G1]["pass"] is True


def test_g1_fails_a_treatment_with_no_expected_kernel_or_fusion():
    servers, m1, m2 = synthetic(1.0)
    spec = gt.GateSpec(
        cells=(Cell(8, 1024), Cell(32, 1024)),
        m2=True,
        by_cell=False,
        expected_kernel={
            t: s.linear_kernel for t, s in model.TREATMENTS.items() if t != Treatment.NVA
        },
        expected_fusion={
            t: s.act_quant_fusion for t, s in model.TREATMENTS.items() if t != Treatment.NVA
        },
    )
    g1 = _gates(servers, m1, m2, spec=spec)[Gate.G1]
    assert g1["pass"] is False and g1["expected"]["NVa"] is None
    assert g1["expected_act_quant_fusion"]["NVa"] is None
    assert {m[1] for m in g1["mismatched_sessions"]} == {"NVa"}
    assert {m[1] for m in g1["fusion_mismatched_sessions"]} == {"NVa"}


def test_g1_lists_kernel_fallback_and_sampling_lines_per_treatment_for_audit():
    servers, m1, m2 = synthetic(1.0)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["sampling_defaults_lines"] == {t: ["Default sampling parameters: {}"] for t in FOUR}
    assert g1["kernel_lines"]["NV"] == [kernel_line("NV")]
    assert g1["backend_fallback_lines"]["MX"] == [FALLBACK_LINE]
    assert g1["backend_fallback_lines"]["NV"] == []


def test_g1_without_the_replica_treatment_is_not_failed_for_it():
    servers, m1, m2 = synthetic(1.0)
    servers = [s for s in servers if s["treatment"] != "MXp"]
    assert _gates(servers, m1, m2)[Gate.G1]["pass"] is True


def test_g1_records_the_fusion_audit_of_every_treatment():
    servers, m1, m2 = _five()
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is True and g1["fusion_mismatched_sessions"] == []
    assert g1["expected_act_quant_fusion"] == {
        t: model.TREATMENTS[Treatment(t)].act_quant_fusion for t in FIVE
    }
    assert g1["observed_fuse_act_quant"] == {
        "MX": [False],
        "MXp": [False],
        "NV": [True],
        "NVa": [True],
        "NVnf": [False],
    }
    assert g1["custom_fusion_lines"]["NV"] == [ACT_QUANT_LINE]
    assert g1["custom_fusion_lines"]["NVnf"] == g1["custom_fusion_lines"]["MX"] == []
    assert g1["expected"]["NVnf"] == CUTE_DSL
    json.dumps(g1)


def test_g1_fails_nvnf_with_the_fusion_on_and_nv_with_it_off():
    servers, m1, m2 = _five()
    set_fields(servers, "NVnf", r=2, fuse_act_quant=True, custom_fusions=["act_quant"])
    set_fields(servers, "NV", r=0, fuse_act_quant=False, custom_fusions=[])
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False and g1["mismatched_sessions"] == []
    assert g1["fusion_mismatched_sessions"] == [
        [0, "NV", False, [], True],
        [2, "NVnf", True, ["act_quant"], False],
    ]
    assert g1["observed_fuse_act_quant"]["NVnf"] == [False, True]


@pytest.mark.parametrize(
    "fields",
    [
        {"custom_fusions": []},
        {"fuse_act_quant": False},
        {"fuse_act_quant": None},
        {"fuse_act_quant": 1},
        {"pass_config_conflict": True},
        {"fuse_act_quant": DROP},
        {"custom_fusions": DROP},
        {"pass_config_conflict": DROP},
    ],
)
def test_g1_fails_closed_on_inconsistent_or_absent_fusion_evidence(fields):
    servers, m1, m2 = _five()
    set_fields(servers, "NV", r=1, **fields)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False and [m[:2] for m in g1["fusion_mismatched_sessions"]] == [[1, "NV"]]
    json.dumps(g1)


def test_g1_fusion_mismatch_rows_carry_missing_fields_as_null_and_conflicts_are_listed():
    servers, m1, m2 = _five()
    set_fields(servers, "NVnf", r=4, fuse_act_quant=DROP, custom_fusions=DROP)
    set_fields(servers, "NVa", r=0, pass_config_conflict=True)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert [4, "NVnf", None, None, False] in g1["fusion_mismatched_sessions"]
    assert g1["pass_config_conflict_sessions"] == [[0, "NVa", True]]


def test_g1_passes_when_nvnf_and_nv_have_their_own_compiled_graphs():
    servers, m1, m2 = _five()
    g1 = _gates(servers, m1, m2)[Gate.G1]
    check = g1["nvnf_compile_cache"]
    assert g1["pass"] is True and check["status"] == "pass" and check["shared_with_nv"] == []
    assert check["nv_hashes"] == [graph_hash("NV")] and check["nvnf_hashes"] == [graph_hash("NVnf")]
    assert g1["compile_cache_hashes"]["MX"] == g1["compile_cache_hashes"]["MXp"]


def test_g1_fails_an_nvnf_session_that_loaded_any_nv_sessions_graph():
    servers, m1, m2 = _five()
    set_fields(servers, "NVnf", r=3, compile_cache_hashes=[graph_hash("NV")])
    set_fields(servers, "NV", r=4, compile_cache_hashes=[graph_hash("NV"), "recompiled-graph"])
    set_fields(servers, "NVnf", r=0, compile_cache_hashes=["recompiled-graph", graph_hash("NVnf")])
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["pass"] is False and g1["nvnf_compile_cache"]["status"] == "fail"
    assert g1["nvnf_compile_cache"]["shared_with_nv"] == [
        [0, "NVnf", ["recompiled-graph"]],
        [3, "NVnf", [graph_hash("NV")]],
    ]
    assert g1["fusion_mismatched_sessions"] == []


@pytest.mark.parametrize("side", ["NV", "NVnf"])
@pytest.mark.parametrize("missing", [[], DROP])
def test_g1_without_hashes_on_either_side_is_unverified_not_failed(side, missing):
    servers, m1, m2 = _five()
    set_fields(servers, side, compile_cache_hashes=missing)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["nvnf_compile_cache"]["status"] == "unverified" and g1["pass"] is True


def test_g1_lists_nvnf_sessions_without_hashes_and_needs_both_treatments():
    servers, m1, m2 = _five()
    set_fields(servers, "NVnf", r=2, compile_cache_hashes=DROP)
    check = _gates(servers, m1, m2)[Gate.G1]["nvnf_compile_cache"]
    assert check["status"] == "pass" and check["nvnf_sessions_without_hashes"] == [[2, "NVnf"]]
    servers, m1, m2 = synthetic(1.0)
    g1 = _gates(servers, m1, m2)[Gate.G1]
    assert g1["nvnf_compile_cache"]["status"] == "not applicable" and g1["pass"] is True


def test_g2_and_g5a_fail_without_a_checkpoint_report_and_block_interpretation():
    servers, m1, m2 = synthetic(1.0)
    gates = _gates(servers, m1, m2, checkpoint=None)
    assert gates[Gate.G2]["pass"] is False and gates[Gate.G2]["reason"] == "no checkpoint report"
    assert gates[Gate.G5A]["pass"] is False
    assert gates[Gate.G5A]["problems"] == {
        "mx": ["no checkpoint report"],
        "nv": ["no checkpoint report"],
    }
    assert gates[Gate.G5B]["pass"] is True
    assert gt.failed_gates(gates) == ["G2", "G5a"]


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"nv_problems": []},
        {"nv_over_mx": 1.06, "nv_problems": []},
        {"nv_over_mx": None, "expected_nv_over_mx": 1.06, "nv_problems": []},
        {"nv_over_mx": float("nan"), "expected_nv_over_mx": 1.06, "nv_problems": []},
        {"nv_over_mx": 1.06, "expected_nv_over_mx": 0, "nv_problems": []},
    ],
)
def test_g2_fails_on_a_malformed_checkpoint_report_instead_of_raising(broken):
    servers, m1, m2 = synthetic(1.0)
    assert _gates(servers, m1, m2, checkpoint=broken)[Gate.G2]["pass"] is False


def test_g2_uses_the_byte_ratio_and_g2_g5a_name_their_checkpoint_source():
    servers, m1, m2 = synthetic(1.0)
    good = {**CLEAN_CKPT, "nv_over_mx": 1.06}
    assert _gates(servers, m1, m2, checkpoint=good)[Gate.G2]["pass"] is True
    assert _gates(servers, m1, m2, checkpoint={**good, "nv_over_mx": 1.2})[Gate.G2]["pass"] is False
    gates = _gates(servers, m1, m2, checkpoint_source="manifest (last line)")
    assert (
        gates[Gate.G2]["checkpoint_source"]
        == gates[Gate.G5A]["checkpoint_source"]
        == "manifest (last line)"
    )


def test_g3_needs_an_equivalent_aa_with_one_inside_the_ci_at_every_batch():
    servers, m1, m2 = synthetic(1.0)
    assert _gates(servers, m1, m2)[Gate.G3]["pass"] is True
    missing = _gates(servers, m1, m2, aa=aa_results((8,)))[Gate.G3]
    assert missing["pass"] is False and missing["missing_c"] == [32]
    assert _gates(servers, m1, m2, aa={})[Gate.G3]["pass"] is False
    wide = aa_results((8, 32))
    wide[32] = aa_results((32,), verdict=Verdict.INCONCLUSIVE)[32]
    assert _gates(servers, m1, m2, aa=wide)[Gate.G3]["pass"] is False
    off_one = aa_results((8, 32), lo=1.001, hi=1.010)
    assert _gates(servers, m1, m2, aa=off_one)[Gate.G3]["pass"] is False


def test_g4_flags_env_throttling_in_m1_and_m2_cells():
    servers, m1, m2 = synthetic(1.0)
    m1[1]["block_telemetry"] = {**TEL, "env_throttle_us": 5, "sw_power_cap_frac": 0.2}
    g4 = _gates(servers, m1, m2)[Gate.G4]
    assert g4["pass"] is False and g4["max_sw_power_cap_frac"] == 0.2
    assert g4["cells"] == [(0, "MX", 8)]
    servers, m1, m2 = synthetic(1.0)
    m2[1]["telemetry"] = {**TEL, "env_throttle_us": 3}
    assert _gates(servers, m1, m2)[Gate.G4]["cells"] == [
        (m2[1]["round"], m2[1]["treatment"], m2[1]["c"])
    ]


def test_g5a_needs_both_problem_lists_to_exist_and_be_empty():
    servers, m1, m2 = synthetic(1.0)
    assert _gates(servers, m1, m2)[Gate.G5A]["pass"] is True
    mx = _gates(servers, m1, m2, checkpoint={**CLEAN_CKPT, "mx_problems": ["scales are not E8M0"]})
    assert mx[Gate.G5A]["problems"] == {"mx": ["scales are not E8M0"], "nv": []}
    assert gt.failed_gates(mx) == ["G5a"]
    line = gt.gate_status_line(mx)
    assert "G5a FAIL" in line and "G5b pass" in line
    nv = _gates(servers, m1, m2, checkpoint={**CLEAN_CKPT, "nv_problems": ["experts are bf16"]})
    assert nv[Gate.G5A]["problems"] == {"mx": [], "nv": ["experts are bf16"]}
    for missing in ("mx_problems", "nv_problems"):
        no_field = {k: v for k, v in CLEAN_CKPT.items() if k != missing}
        g5a = _gates(servers, m1, m2, checkpoint=no_field)[Gate.G5A]
        assert g5a["pass"] is False
        assert g5a["problems"][missing[:2]] == [f"checkpoint report has no {missing}"]


def test_g5b_is_a_one_sided_bound_on_the_degradation_from_the_bf16_reference():
    servers, m1, m2 = synthetic(1.0)
    g5b = _gates(servers, m1, m2)[Gate.G5B]
    assert g5b["pass"] is True
    assert g5b["bf16_reference_mean"] == pytest.approx(GOOD_REFERENCE["mean"])
    assert g5b["max_degradation"] == settings.NLL_MAX_DEGRADATION == 0.5
    assert g5b["degradation"]["MX"] == pytest.approx(0.060 - 0.0005)
    assert g5b["nll"]["NV"] == pytest.approx(GOOD_REFERENCE["mean"] + 0.030)
    assert set(g5b) == {
        "pass",
        "nll",
        "bf16_reference_mean",
        "reference_problem",
        "max_degradation",
        "degradation",
        "failed_treatments",
    }


@pytest.mark.parametrize(
    "extra, ok", [(0.0, True), (0.49, True), (0.51, False), (0.6, False), (-0.3, True)]
)
def test_g5b_bound_is_the_reference_plus_the_configured_degradation(extra, ok):
    servers, m1, m2 = synthetic(1.0)
    set_nll(servers, "NV", extra)
    g5b = _gates(servers, m1, m2)[Gate.G5B]
    assert g5b["pass"] is ok and g5b["failed_treatments"] == ([] if ok else ["NV"])


def test_g5b_uses_the_reference_mean_it_was_given():
    servers, m1, m2 = synthetic(1.0)
    low = manifest(reference={"per_prompt": [], "mean": GOOD_REFERENCE["mean"] - 0.5})
    assert _gates(servers, m1, m2, manifest_line=low)[Gate.G5B]["pass"] is False


@pytest.mark.parametrize("bad", [float("nan"), None, "one round"])
def test_g5b_fails_on_a_missing_or_nan_nll(bad):
    servers, m1, m2 = synthetic(1.0)
    if bad == "one round":
        set_fields(servers, "NV", r=3, nll=None)
    else:
        set_fields(servers, "NV", nll=bad)
    g5b = _gates(servers, m1, m2)[Gate.G5B]
    assert g5b["pass"] is False and g5b["failed_treatments"] == ["NV"]
    assert g5b["nll"]["NV"] is None


@pytest.mark.parametrize(
    "reference",
    [
        None,
        "error: FileNotFoundError: /data/bf16.json",
        {"per_prompt": [2.0]},
        {"mean": None},
        {"mean": float("nan")},
        [1, 2],
    ],
)
def test_g5b_fails_without_a_usable_bf16_reference(reference):
    servers, m1, m2 = synthetic(1.0)
    g5b = _gates(servers, m1, m2, manifest_line=manifest(reference=reference))[Gate.G5B]
    assert g5b["pass"] is False and g5b["reference_problem"]
    assert g5b["bf16_reference_mean"] is None


def test_g6a_flags_sessions_whose_capture_sizes_miss_a_tested_batch():
    servers, m1, m2 = synthetic(1.0)
    servers[2]["cudagraph_capture_sizes"] = [8]
    servers[4]["cudagraph_capture_sizes"] = []
    servers[5].pop("cudagraph_capture_sizes")
    g6a = _gates(servers, m1, m2)[Gate.G6A]
    assert g6a["pass"] is False and g6a["missing_m1_rows"] == []
    assert g6a["capture_size_sessions"] == [servers[i]["session_id"] for i in (2, 4, 5)]


def test_g6a_does_not_look_at_m2_cells():
    servers, m1, m2 = synthetic(1.0)
    m2[0]["valid"], m2[0]["invalid_reasons"] = False, ["underfilled"]
    m2[1]["aggregate_source"] = "prometheus"
    assert _gates(servers, m1, m2)[Gate.G6A]["pass"] is True


def test_g6a_flags_a_session_and_batch_without_measured_rows_but_not_a_dead_sessions_rows():
    servers, m1, m2 = synthetic(1.0)
    m1 = drop_m1(m1, "r2_NV-x", 32)
    m1 = drop_m1(m1, "r0_MX-x", 8, warmup_too=True)
    m1 = drop_m1(m1, "r0_NV-x", 8)
    g6a = _gates(servers, m1, m2)[Gate.G6A]
    assert g6a["pass"] is False
    assert g6a["missing_m1_rows"] == [[0, "MX", 8, 0, 1], [0, "NV", 8, 0, 1], [2, "NV", 32, 0, 1]]


def test_g6a_expects_every_batch_the_run_registered():
    servers, m1, m2 = synthetic(1.0)
    line = manifest(concurrencies=[8, 32, 128])
    g6a = _gates(servers, m1, m2, manifest_line=line)[Gate.G6A]
    assert g6a["pass"] is False and g6a["expected_concurrencies"] == [8, 32, 128]
    assert len(g6a["missing_m1_rows"]) == 20 and all(m[2] == 128 for m in g6a["missing_m1_rows"])
    assert len(g6a["capture_size_sessions"]) == 20


def test_g6a_expects_the_registered_number_of_measured_reps():
    servers, m1, m2 = synthetic(1.0)
    assert _gates(servers, m1, m2, manifest_line=manifest(m1_reps=3))[Gate.G6A]["pass"] is True
    short = (
        drop_m1(m1, "r1_MX-x", 8)
        + [r for r in m1 if r["session_id"] == "r1_MX-x" and r["c"] == 8 and not r["warmup"]][:2]
    )
    g6a = _gates(servers, short, m2, manifest_line=manifest(m1_reps=3))[Gate.G6A]
    assert g6a["missing_m1_rows"] == [[1, "MX", 8, 2, 3]]
    assert (
        len(_gates(servers, m1, m2, manifest_line=manifest(m1_reps=2))[Gate.G6A]["missing_m1_rows"])
        == 40
    )


def _set_delta(m1, session_id, c, delta, warmup_only=False):
    for row in m1:
        if row["session_id"] == session_id and row["c"] == c and (row["warmup"] or not warmup_only):
            if delta is DROP:
                row.pop("preemptions_delta", None)
            else:
                row["preemptions_delta"] = delta


def test_g6a_lists_each_preempted_m1_block_once_in_order_warmup_included():
    servers, m1, m2 = synthetic(1.0)
    _set_delta(m1, "r3_NVa-x", 32, 1.0)
    _set_delta(m1, "r1_NV-x", 8, 2.0)
    _set_delta(m1, "r0_MX-x", 32, 1.0, warmup_only=True)
    g6a = _gates(servers, m1, m2)[Gate.G6A]
    assert g6a["pass"] is False and g6a["missing_m1_rows"] == []
    assert g6a["preempted_m1_blocks"] == [
        [0, "MX", 32, 1.0],
        [1, "NV", 8, 2.0],
        [3, "NVa", 32, 1.0],
    ]


@pytest.mark.parametrize("delta", [None, DROP, 0.5, float("nan"), True, "0"])
def test_g6a_needs_a_preemption_delta_of_exactly_zero(delta):
    servers, m1, m2 = synthetic(1.0)
    _set_delta(m1, "r2_MXp-x", 8, delta)
    g6a = _gates(servers, m1, m2)[Gate.G6A]
    assert g6a["pass"] is False and [b[:3] for b in g6a["preempted_m1_blocks"]] == [[2, "MXp", 8]]
    json.dumps(g6a["preempted_m1_blocks"])


def test_g6a_ignores_preemptions_in_rows_of_a_dead_session():
    servers, m1, m2 = synthetic(1.0)
    next(r for r in m1 if r["session_id"] == "dead")["preemptions_delta"] = 7.0
    g6a = _gates(servers, m1, m2)[Gate.G6A]
    assert g6a["pass"] is True and g6a["preempted_m1_blocks"] == []


def test_g6a_kv_capacity_needs_the_largest_wave_of_the_expected_batches():
    servers, m1, m2 = synthetic(1.0)
    kv = _gates(servers, m1, m2)[Gate.G6A]["kv_capacity"]
    assert kv["required_tokens"] == 32 * (settings.M1_INPUT_LEN + settings.M1_N2) == 69_632
    assert kv["max_concurrency"] == 32 and kv["short_sessions"] == []
    servers, m1, m2 = synthetic(1.0, (128, 256, 512), treatments=("MX", "NV", "MXp"))
    kv = _gates(servers, m1, m2, batches=(128, 256, 512), manifest_line=expb_manifest())[Gate.G6A][
        "kv_capacity"
    ]
    assert kv["required_tokens"] == EXPB.min_kv_tokens() == 1_114_112


def test_g6a_sizes_the_kv_pool_by_min_kv_tokens_for_of_every_expected_cell(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gt,
        "min_kv_tokens_for",
        lambda c, p=settings.M1_INPUT_LEN: calls.append((c, p)) or 12_345 * c,
    )
    kv = _gates(*synthetic(1.0))[Gate.G6A]["kv_capacity"]
    assert sorted(calls) == [(8, 1024), (32, 1024)] and kv["required_tokens"] == 12_345 * 32


@pytest.mark.parametrize(
    "tokens, ok",
    [
        (69_632, True),
        (69_631, False),
        (None, False),
        (DROP, False),
        (69_632.0, False),
        ("1200000", False),
        (True, False),
    ],
)
def test_g6a_fails_a_session_whose_kv_pool_is_too_small_or_unknown(tokens, ok):
    servers, m1, m2 = synthetic(1.0)
    set_fields(servers, "MXp", r=4, kv_cache_tokens=tokens)
    g6a = _gates(servers, m1, m2)[Gate.G6A]
    assert g6a["pass"] is ok
    if not ok:
        assert g6a["kv_capacity"]["short_sessions"] == [
            [4, "MXp", None if tokens is DROP else tokens]
        ]


def test_g6a_fails_an_experiment_b_run_with_a_bf16_sized_pool():
    servers, m1, m2 = synthetic(1.0, (128, 256, 512), treatments=("MX", "NV", "MXp"))
    set_fields(servers, "NV", kv_cache_tokens=1_117_000 - 10_000)
    g6a = _gates(servers, m1, m2, batches=(128, 256, 512), manifest_line=expb_manifest())[Gate.G6A]
    assert g6a["pass"] is False and len(g6a["kv_capacity"]["short_sessions"]) == 5


def test_g6b_lists_invalid_cells_and_reports_the_paired_rounds_per_batch():
    servers, m1, m2 = synthetic(1.0)
    assert _gates(servers, m1, m2)[Gate.G6B]["m2_paired_rounds"] == {8: 5, 32: 5}
    victim = next(r for r in m2 if (r["round"], r["treatment"], r["c"]) == (1, "MX", 8))
    victim["valid"], victim["invalid_reasons"] = False, ["effective_concurrency"]
    g6b = _gates(servers, m1, m2)[Gate.G6B]
    assert g6b["pass"] is False and g6b["blocks_interpretation"] is False
    assert g6b["invalid_m2_cells"] == [[1, "MX", 8, ["effective_concurrency"]]]
    assert g6b["m2_paired_rounds"] == {8: 4, 32: 5}


@pytest.mark.parametrize("source", ["prometheus", None, DROP])
def test_g6b_flags_valid_rows_not_measured_from_continuous_usage(source):
    servers, m1, m2 = synthetic(1.0)
    m2[4]["aggregate_source"] = source
    if source is DROP:
        del m2[4]["aggregate_source"]
    g6b = _gates(servers, m1, m2)[Gate.G6B]
    assert g6b["pass"] is False and g6b["invalid_m2_cells"] == []
    assert g6b["bad_aggregate_source_cells"] == [
        [m2[4]["round"], m2[4]["treatment"], m2[4]["c"], None if source is DROP else source]
    ]


@pytest.mark.parametrize("delta", [1.0, None, DROP])
def test_g6b_fails_a_valid_m2_cell_whose_preemption_counter_moved(delta):
    servers, m1, m2 = synthetic(1.0)
    if delta is DROP:
        del m2[5]["preemptions_delta"]
    else:
        m2[5]["preemptions_delta"] = delta
    g6b = _gates(servers, m1, m2)[Gate.G6B]
    assert g6b["pass"] is False and g6b["invalid_m2_cells"] == []
    assert g6b["preempted_valid_m2_cells"] == [
        [m2[5]["round"], m2[5]["treatment"], m2[5]["c"], None if delta is DROP else delta]
    ]


def test_g6b_does_not_double_report_an_already_invalid_row():
    servers, m1, m2 = synthetic(1.0)
    m2[4]["valid"], m2[4]["aggregate_source"] = False, "none"
    m2[5].update(valid=False, invalid_reasons=["preemptions"], preemptions_delta=3.0)
    g6b = _gates(servers, m1, m2)[Gate.G6B]
    assert g6b["bad_aggregate_source_cells"] == [] and g6b["preempted_valid_m2_cells"] == []
    assert len(g6b["invalid_m2_cells"]) == 2


def test_a_failed_g6b_is_reported_but_does_not_block_and_every_other_failed_gate_does():
    servers, m1, m2 = synthetic(1.0)
    m2[0]["preemptions_delta"] = 1.0
    gates = _gates(servers, m1, m2)
    assert gt.failed_gates(gates) == [] and gt.nonblocking_failed_gates(gates) == ["G6b"]
    assert "G6b FAIL" in gt.gate_status_line(gates)
    servers[2]["cudagraph_capture_sizes"] = [8]
    gates = _gates(servers, m1, m2)
    assert gt.failed_gates(gates) == ["G6a"] and gt.nonblocking_failed_gates(gates) == ["G6b"]


def test_g7_does_not_accept_two_identical_gpu_query_errors_as_the_same_gpu():
    servers, m1, m2 = synthetic(1.0)
    for s in servers:
        if s["round"] == 1:
            s["gpu_uuid"] = "error: CalledProcessError: nvidia-smi failed"
    g7 = _gates(servers, m1, m2)[Gate.G7]
    assert g7["pass"] is False and len(g7["unreadable_gpu_sessions"]) == 4


def test_g7_fails_a_round_whose_latest_sessions_span_several_starts():
    servers, m1, m2 = synthetic(1.0)
    servers[4]["start_id"] = "start-rerun"
    g7 = _gates(servers, m1, m2)[Gate.G7]
    assert g7["pass"] is False and g7["mixed_start_rounds"] == {1: ["start-1", "start-rerun"]}
    assert g7["rounds_without_start_id"] == [] and g7["gpus"][0] == ["GPU-1"]


def test_g7_passes_when_a_whole_round_was_rerun_under_a_new_start():
    servers, m1, m2 = synthetic(1.0)
    old_round_1 = [dict(s) for s in servers if s["round"] == 1]
    for s in servers:
        if s["round"] == 1:
            s["start_id"], s["session_id"] = "start-rerun", s["session_id"] + "2"
    g7 = _gates(old_round_1 + servers, m1, m2)[Gate.G7]
    assert g7["pass"] is True and g7["mixed_start_rounds"] == {}
    assert _gates(*synthetic(1.0))[Gate.G7]["start_ids"] == {r: [f"start-{r}"] for r in range(5)}


@pytest.mark.parametrize("missing", [DROP, "", None])
def test_g7_fails_a_session_without_a_start_id_or_gpu(missing):
    servers, m1, m2 = synthetic(1.0)
    set_fields(servers, "MX", r=0, start_id=missing)
    g7 = _gates(servers, m1, m2)[Gate.G7]
    assert g7["pass"] is False and g7["rounds_without_start_id"] == [0]
    servers, m1, m2 = synthetic(1.0)
    set_fields(servers, "MX", r=1, gpu_uuid=missing)
    g7 = _gates(servers, m1, m2)[Gate.G7]
    assert g7["pass"] is False
    assert g7["unreadable_gpu_sessions"] == ([] if missing == "" else ["r1_MX-x"])


def test_gate_status_line_names_every_gate():
    gates = {"G1_kernels": {"pass": True}, "G2_bytes": {"pass": None}, "G3_aa": {"pass": False}}
    assert gt.gate_status_line(gates) == "gates: G1 pass · G2 n/a · G3 FAIL"
    assert gt.failed_gates(gates) == ["G3"]


def test_a_failed_session_does_not_fail_or_change_any_gate():
    servers, m1, m2 = smoke()
    line = manifest(rounds=1, treatments=SMOKE.treatments, concurrencies=list(SMOKE_CS), m1_reps=3)
    clean = _gates(servers, m1, m2, aa={}, batches=SMOKE_CS, manifest_line=line)
    rows = fail(servers, "NVc")
    assert set(next(s for s in rows if s.get("failed"))) == {
        "round",
        "treatment",
        "session_id",
        "start_id",
        "failed",
        "error",
    }
    failed = _gates(rows, m1, m2, aa={}, batches=SMOKE_CS, manifest_line=line)
    for name in (Gate.G1, Gate.G5B, Gate.G6A, Gate.G7):
        assert failed[name]["pass"] is True, name
    assert "NVc" not in failed[Gate.G1]["expected"] and "NVc" not in failed[Gate.G5B]["nll"]
    assert failed[Gate.G6A]["missing_m1_rows"] == []
    assert clean[Gate.G1]["expected"].keys() - failed[Gate.G1]["expected"].keys() == {"NVc"}


def _c_gates(mutate=None, manifest_line=None, **kw) -> dict:
    servers, m1 = c_build(H_BATCH, **kw)
    if mutate is not None:
        mutate(servers, m1)
    cells = tuple(Cell(*c) for c in kw.get("cells", REGISTERED_CELLS))
    line = typed_manifest(c_manifest(cells) if manifest_line is None else manifest_line)
    rows, m1_rows = typed_servers(servers), typed_m1(m1)
    steps = m1_steps(m1_rows, session_ids(rows), by_cell=True)
    effects = {
        f"{c.name},AA": contrast(steps, c.cells, Treatment.MXP, Treatment.MX)
        for c in EXPC.contrasts
    }
    checkpoint, source = resolve_checkpoint(Path("/nonexistent/run"), line)
    return gt.gate_report(
        rows,
        m1_rows,
        [],
        line,
        checkpoint,
        source,
        gt.GateSpec(cells=cells, m2=False, by_cell=True),
        gt.AaEffects(effects, AA_EFFECT_MARGIN_MS, AA),
    )


def test_a_clean_experiment_c_run_passes_every_gate_but_g6b_which_it_has_not():
    gates = _c_gates()
    assert tuple(gates) == tuple(g for g in Gate if g != Gate.G6B)
    assert all(g["pass"] for g in gates.values())
    assert gates[Gate.G6A]["kv_capacity"] == {
        "required_tokens": 193_536,
        "largest_wave_cell": [128, 360],
        "short_sessions": [],
    }
    assert gates[Gate.G6A]["required_capture_sizes"] == [1, 8, 32, 128]
    assert gates[Gate.G3]["pass"] and set(gates[Gate.G3]["effects"]) == {"E_tok,AA", "E_batch,AA"}


def test_experiment_c_g1_fails_a_wrong_kernel_a_missing_fusion_field_or_manifest_problems():
    def mutate(servers, m1):
        servers[1]["linear_kernels"] = ["FlashInferCutlassNvFp4LinearKernel"]
        del servers[0]["fuse_act_quant"]

    g1 = _c_gates(mutate)[Gate.G1]
    assert g1["pass"] is False
    assert g1["mismatched_sessions"] == [
        [0, "NV", ["FlashInferCutlassNvFp4LinearKernel"], CUTE_DSL]
    ]
    assert [m[:2] for m in g1["fusion_mismatched_sessions"]] == [[0, "MX"]]
    gates = _c_gates(manifest_line=c_manifest(problems=["vllm 0.30.0 != 0.31.0"]))
    assert gates[Gate.G1]["pass"] is False


def test_experiment_c_g6a_fails_a_preempted_block_or_a_missing_delta():
    def mutate(servers, m1):
        row = next(
            r
            for r in m1
            if r["round"] == 2 and r["treatment"] == "MX" and r["c"] == 128 and r["warmup"]
        )
        row["preemptions_delta"] = 3.0
        del m1[5]["preemptions_delta"]

    g6a = _c_gates(mutate)[Gate.G6A]
    assert g6a["pass"] is False
    assert g6a["preempted_m1_blocks"] == [[0, "MX", 1, 4096, None], [2, "MX", 128, 360, 3.0]]


def test_experiment_c_g6a_needs_the_registered_reps_per_cell():
    def mutate(servers, m1):
        m1.remove(
            next(
                r
                for r in m1
                if r["round"] == 1
                and r["treatment"] == "NV"
                and (r["c"], r["prompt_len"]) == (8, 15360)
                and not r["warmup"]
            )
        )

    g6a = _c_gates(mutate)[Gate.G6A]
    assert g6a["pass"] is False
    assert g6a["missing_m1_rows"] == [[1, "NV", 8, 15360, C_REPS - 1, C_REPS]]


@pytest.mark.parametrize(
    "change",
    [{"m1_reps": None}, {"cells": [[1, 1024], [1, 1024]]}, {"cells": [[1, 0]]}, {"cells": None}],
)
def test_experiment_c_g6a_fails_without_the_registered_reps_or_usable_cells(change):
    line = c_manifest()
    line["protocol"].update(change)
    g6a = _c_gates(manifest_line=line)[Gate.G6A]
    assert g6a["pass"] is False
    if "m1_reps" in change:
        assert g6a["expected_m1_reps"] is None
    else:
        assert g6a["cells_problem"].startswith("protocol.cells")


def test_experiment_c_g6a_lists_malformed_rows_and_cells_outside_the_manifest():
    def mutate(servers, m1):
        m1[3]["prompt_len"] = 0
        m1[4]["warmup"] = None

    line = c_manifest(cells=[c for c in REGISTERED_CELLS if c != (1, 4096)])
    g6a = _c_gates(mutate, manifest_line=line)[Gate.G6A]
    assert g6a["pass"] is False and g6a["cells_not_in_manifest"] == [[1, 4096]]
    assert [m[-1] for m in g6a["malformed_rows"]] == ["no (c, prompt_len) cell", "warmup is None"]


def test_experiment_c_a_row_without_prompt_len_is_malformed():
    victims = []

    def mutate(servers, m1):
        victim = next(r for r in m1 if (r["c"], r["prompt_len"]) == (1, 4096) and not r["warmup"])
        del victim["prompt_len"]
        victim["block_telemetry"] = {**victim["block_telemetry"], "env_throttle_us": 3}
        victims.append(victim)

    gates = _c_gates(mutate)
    g6a = gates[Gate.G6A]
    assert g6a["pass"] is False
    assert g6a["malformed_rows"] == [
        [0, "MX", 1, None, victims[0]["set"], "no (c, prompt_len) cell"]
    ]
    assert g6a["missing_m1_rows"] == [[0, "MX", 1, 4096, C_REPS - 1, C_REPS]]
    assert gates[Gate.G4]["throttled_blocks"] == [[0, "MX", 1, None, 3]]


def test_experiment_c_g6a_needs_capture_sizes_with_every_batch():
    def mutate(servers, m1):
        servers[4]["cudagraph_capture_sizes"] = [1, 8, 32]
        del servers[5]["cudagraph_capture_sizes"]

    g6a = _c_gates(mutate)[Gate.G6A]
    assert g6a["pass"] is False
    assert [row[:2] for row in g6a["capture_size_sessions"]] == [[1, "MXp"], [1, "NV"]]


@pytest.mark.parametrize("tokens, ok", [(193_536, True), (193_535, False), (None, False)])
def test_experiment_c_kv_capacity(tokens, ok):
    def mutate(servers, m1):
        for s in servers:
            s["kv_cache_tokens"] = tokens

    assert _c_gates(mutate)[Gate.G6A]["pass"] is ok


def test_experiment_c_env_throttle_fails_g4_and_power_capping_is_reported_per_cell():
    def mutate(servers, m1):
        for r in m1:
            if r["round"] == 0 and r["treatment"] == "NV" and r["c"] == 128:
                r["block_telemetry"] = {**TEL, "env_throttle_us": 1500, "sw_power_cap_frac": 0.4}

    g4 = _c_gates(mutate)[Gate.G4]
    assert g4["pass"] is False and g4["throttled_blocks"] == [[0, "NV", 128, 360, 1500]]
    assert g4["sw_power_cap_frac"]["C=128,P=360"]["NV"]["max"] == pytest.approx(0.4)
    assert g4["sw_power_cap_frac"]["C=128,P=360"]["NV"]["blocks"] == 5
    assert g4["sw_power_cap_frac"]["C=1,P=1024"]["MX"]["max"] == 0.0
    assert g4["max_sw_power_cap_frac"] == pytest.approx(0.4)


def test_experiment_c_g4_and_g6a_ignore_rows_of_a_dead_session():
    def mutate(servers, m1):
        dead = {
            **m1[1],
            "session_id": "dead",
            "preemptions_delta": 5.0,
            "block_telemetry": {**TEL, "env_throttle_us": 99},
        }
        m1.append(dead)

    gates = _c_gates(mutate)
    assert gates[Gate.G4]["pass"] and gates[Gate.G6A]["pass"]


def test_experiment_c_non_numeric_telemetry_fails_g4():
    def mutate(servers, m1):
        m1[7]["block_telemetry"]["env_throttle_us"] = None

    g4 = _c_gates(mutate)[Gate.G4]
    assert g4["pass"] is False and len(g4["blocks_without_telemetry"]) == 1


def test_experiment_c_g2_g5a_g5b_fail_closed_and_g5b_bounds_the_degradation():
    gates = _c_gates(manifest_line=c_manifest(checkpoint=None, reference=None))
    assert not gates[Gate.G2]["pass"] and not gates[Gate.G5A]["pass"]
    assert not gates[Gate.G5B]["pass"]
    assert gates[Gate.G5B]["reference_problem"] == "the manifest has no inputs.bf16_reference_nll"

    def mutate(servers, m1):
        for s in servers:
            if s["treatment"] == "NV":
                s["nll"] = 1.8 + 0.6

    g5b = _c_gates(mutate)[Gate.G5B]
    assert not g5b["pass"] and g5b["failed_treatments"] == ["NV"]


def test_experiment_c_g7_needs_one_gpu_and_one_start_per_round():
    def mutate(servers, m1):
        servers[0]["start_id"] = "another-start"
        del servers[3]["gpu_uuid"]

    g7 = _c_gates(mutate)[Gate.G7]
    assert not g7["pass"] and set(g7["mixed_start_rounds"]) == {0}
    assert g7["unreadable_gpu_sessions"] == ["r1_MX-x"]


def test_experiment_c_g3_tests_the_aa_effects():
    def offset(servers, m1):
        for r in m1:
            if r["treatment"] == "MXp" and (r["c"], r["prompt_len"]) == (128, 360):
                r["step_s"] += 0.0004

    g3 = _c_gates(offset)[Gate.G3]
    assert not g3["pass"] and g3["effects"]["E_tok,AA"]["pass"]
    assert g3["effects"]["E_batch,AA"]["mean_ms"] == pytest.approx(0.4, abs=0.02)

    def no_aa(servers, m1):
        m1[:] = [
            r
            for r in m1
            if not (r["treatment"] == "MXp" and (r["c"], r["prompt_len"]) == (1, 1024))
        ]

    g3 = _c_gates(no_aa)[Gate.G3]
    assert not g3["pass"] and g3["effects"]["E_tok,AA"]["reason"] == "no data"
    g3 = _c_gates(treatments=("MX", "NV"))[Gate.G3]
    assert not g3["pass"] and g3["reason"] == "no A/A replica"


def _aa_effect(n_rounds, lo, hi, mean=0.0) -> ContrastResult:
    cells = (Cell(1, 127360), Cell(1, 1024))
    return ContrastResult(
        cells, n_rounds, tuple(range(n_rounds)), (), mean if n_rounds else None, lo, hi, None, {}
    )


def test_g3_on_the_aa_effects_fails_closed():
    ok = _aa_effect(5, -0.1, 0.1)
    assert gt.g3_aa_effects({"E_tok,AA": ok, "E_batch,AA": ok}, True, 0.25, AA)["pass"] is True
    one = _aa_effect(1, None, None)
    g3 = gt.g3_aa_effects({"E_tok,AA": one, "E_batch,AA": ok}, True, 0.25, AA)
    assert not g3["pass"] and g3["effects"]["E_tok,AA"]["reason"] == "needs at least 2 rounds"
    none = _aa_effect(0, None, None)
    assert (
        gt.g3_aa_effects({"E_tok,AA": none}, True, 0.25, AA)["effects"]["E_tok,AA"]["reason"]
        == "no data"
    )
    wide = gt.g3_aa_effects({"E_tok,AA": _aa_effect(5, -0.3, 0.1)}, True, 0.25, AA)
    assert wide["effects"]["E_tok,AA"]["reason"].startswith("the CI does not contain 0")
    no_replica = gt.g3_aa_effects({"E_tok,AA": ok}, False, 0.25, AA)
    assert not no_replica["pass"] and no_replica["reason"] == "no A/A replica"
    assert no_replica["effects"]["E_tok,AA"]["reason"] == "no A/A replica (MXp)"
    assert not gt.g3_aa_effects({}, True, 0.25, AA)["pass"]
