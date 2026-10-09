import json

import pytest

from fp4bench import lib_bench, telemetry
from fp4bench.core.types import (
    Arm,
    Cell,
    CheckpointKind,
    ClockEvent,
    ContrastName,
    DecidedBy,
    DesignKind,
    Estimate,
    FetchKind,
    Format,
    FormatSpec,
    Gate,
    Hypothesis,
    KernelMatch,
    KvDtype,
    M2InvalidReason,
    ProbeError,
    Raised,
    RatioName,
    ReproStatus,
    SessionSlot,
    SkippedMetric,
    Treatment,
    UnknownKvDtype,
    ValueKey,
    Verdict,
    VerdictLabels,
    WaveOrder,
    attempt,
    error_as_object,
    error_as_text,
    probe,
)
from fp4bench.studies import model
from fp4bench.studies.main import FULL
from fp4bench.studies.registry import FP8_KV_CACHE_DTYPES, STUDIES
from fp4bench.studies.smoke import SMOKE_ONLY_TREATMENTS
from tests import GOLDEN_DIR, RUN_DIRS
from tests.analysis_runs import load_jsonl

GOLDENS = sorted(GOLDEN_DIR.glob("*-*.json"))


def _values_under(obj, key: str, out: list, top: bool = True) -> list:
    """Every value stored under `key` below the top level of `obj`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and not top:
                out.append(v)
            _values_under(v, key, out, top=False)
    elif isinstance(obj, list):
        for v in obj:
            _values_under(v, key, out, top=False)
    return out


def test_enums_compare_and_hash_as_their_values():
    assert Treatment.MXP == "MXp" and hash(Treatment.MXP) == hash("MXp")
    assert {"NVnf": 1}[Treatment.NVNF] == 1
    assert f"{Treatment.NVA}" == "NVa" and str(Format.NVFP4) == "nvfp4"
    assert json.dumps({Gate.G5B: [Verdict.EQUIVALENT, KvDtype.FP8]}) == (
        '{"G5b_nll": ["equivalent", "fp8"]}'
    )


def test_an_unknown_kv_dtype_renders_as_the_name_it_was_stored_with():
    auto = UnknownKvDtype("auto")
    assert str(auto) == f"{auto}" == "auto" and auto == UnknownKvDtype("auto")
    assert auto != KvDtype.BF16


def test_each_format_has_the_spec_its_llm_compressor_preset_writes():
    assert Format.MXFP4.spec == FormatSpec(
        scheme="MXFP4",
        ct_format="mxfp4-pack-quantized",
        group_size=32,
        strategy="group",
        scale_dtype="U8",
        global_scales=(),
    )
    assert Format.NVFP4.spec == FormatSpec(
        scheme="NVFP4",
        ct_format="nvfp4-pack-quantized",
        group_size=16,
        strategy="tensor_group",
        scale_dtype="F8_E4M3",
        global_scales=("weight_global_scale", "input_global_scale"),
    )


def test_each_checkpoint_kind_has_its_format_and_each_format_our_checkpoint():
    assert {kind: kind.fmt for kind in CheckpointKind} == {
        CheckpointKind.MX: Format.MXFP4,
        CheckpointKind.NV: Format.NVFP4,
        CheckpointKind.NVX: Format.NVFP4,
    }
    assert {fmt: CheckpointKind.ours(fmt) for fmt in Format} == {
        Format.MXFP4: CheckpointKind.MX,
        Format.NVFP4: CheckpointKind.NV,
    }


def test_each_fetch_kind_names_a_checkpoint_and_nvidia_is_nvx():
    assert {kind: kind.checkpoint for kind in FetchKind} == {
        FetchKind.MX: CheckpointKind.MX,
        FetchKind.NV: CheckpointKind.NV,
        FetchKind.NVIDIA: CheckpointKind.NVX,
    }


def test_decided_by_renders_the_text_a_margin_stores_and_is_never_serialised_as_is():
    rendered = [d.to_json(selection_c=32, tiebreak_c=128) for d in DecidedBy]
    assert rendered == ["C=32", "C=128", "order"]
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(DecidedBy.SELECTION_C)


def test_the_scan_tables_kernel_states_render_as_today():
    assert [f"{m}" for m in KernelMatch] == ["no session", "ok", "MISMATCH", "failed"]


def test_treatments_parse_every_committed_session_and_protocol():
    seen = set()
    for run in RUN_DIRS:
        seen |= {row["treatment"] for row in load_jsonl(run / "servers.jsonl")}
        seen |= {
            t
            for line in load_jsonl(run / "manifests.jsonl")
            for t in line["protocol"]["treatments"]
        }
    assert {Treatment(t) for t in seen} == set(Treatment)
    assert seen == {t.value for t in Treatment}


def test_treatment_values_are_the_configs():
    configured = {*FULL.treatments, *SMOKE_ONLY_TREATMENTS}
    assert configured == set(Treatment)
    assert set(model.TREATMENTS) == set(Treatment)


def test_kv_dtypes_and_cells_parse_every_committed_protocol():
    for run in RUN_DIRS:
        for line in load_jsonl(run / "manifests.jsonl"):
            proto = line["protocol"]
            KvDtype(proto.get("kv_cache_dtype", "bfloat16"))
            assert all(Cell(*cell) == tuple(cell) for cell in proto.get("cells", ()))
    assert {s.server.kv_dtype for s in STUDIES.values()} == {KvDtype.BF16, KvDtype.FP8}


def test_every_kv_dtype_but_bfloat16_is_fp8():
    assert {"fp8", "fp8_e4m3", "fp8_e5m2"} == FP8_KV_CACHE_DTYPES
    assert [d for d in KvDtype if not d.is_fp8] == [KvDtype.BF16]


def test_m1_wave_orders_m2_reasons_and_clock_events_parse_every_committed_row():
    firsts, reasons, counters = set(), set(), set()
    for run in RUN_DIRS:
        for row in load_jsonl(run / "m1.jsonl"):
            firsts.add(row.get("first"))
            counters.add(tuple(row["block_telemetry"]["counters_delta"]))
        if (m2 := run / "m2.jsonl").exists():
            reasons |= {r for row in load_jsonl(m2) for r in row["invalid_reasons"]}
    assert {WaveOrder(f) for f in firsts - {None}} == set(WaveOrder)
    assert reasons and all(M2InvalidReason(r) in M2InvalidReason for r in reasons)
    assert counters == {tuple(ClockEvent)} and tuple(ClockEvent) == telemetry.COUNTER_NAMES


def test_m2_reasons_are_the_strings_lib_bench_and_the_runner_stored():
    assert [r.value for r in M2InvalidReason] == [
        *("underfilled", "capacity_limited", "loop_detected", "warmup_timed_out"),
        *("aggregate_tps_not_finite", "aggregate_tps<=0", "num_errors", "aggregate_source"),
        *("effective_concurrency", "missing_cell", "preemptions"),
    ]
    assert json.dumps(lib_bench.INVALID_FLAGS) == (
        '["underfilled", "capacity_limited", "loop_detected", "warmup_timed_out"]'
    )


def test_environmental_throttling_is_the_thermal_and_power_braking_events():
    assert [e for e in ClockEvent if e.is_env_throttle] == [
        "SW Thermal Slowdown",
        "HW Thermal Slowdown",
        "HW Power Braking",
    ]


def test_checkpoint_kinds_parse_every_committed_staging_record():
    kinds = {
        k
        for run in RUN_DIRS
        for line in load_jsonl(run / "manifests.jsonl")
        for k in (line["inputs"].get("local_staging") or {})
    }
    assert {CheckpointKind(k) for k in kinds} == set(CheckpointKind)
    assert CheckpointKind.NVX == "nvx"


def test_gates_are_the_analysis_gate_keys():
    keys = [set(json.loads(p.read_text())["gates"]) for p in GOLDENS]
    assert set().union(*keys) == set(Gate)
    assert all(Gate(k) in Gate for ks in keys for k in ks)


def test_verdicts_parse_every_per_comparison_label():
    labels = {
        v
        for p in GOLDENS
        for v in _values_under(json.loads(p.read_text()), "verdict", [])
        if v is not None
    }
    assert labels and all(Verdict(v) in Verdict for v in labels)


def test_the_analysis_names_parse_every_golden_payload():
    cell_views = 0
    for p in GOLDENS:
        view = json.loads(p.read_text())
        if "design" not in view:
            assert all(SkippedMetric(m) in SkippedMetric for m in view["skipped"])
            assert {RatioName(n) for d in view["m1_r_f_phi"].values() for n in d["n_rounds"]} <= {
                RatioName.R,
                RatioName.F,
                RatioName.PHI,
            }
            continue
        cell_views += 1
        assert DesignKind(view["design"]["kind"]) in DesignKind
        assert ReproStatus(view["config_reproduction"]["status"]) in ReproStatus
        assert {ContrastName(n) for n in view["effects"]} == set(ContrastName)
        predictions = view["predictions"]
        assert {Hypothesis(h) for h in predictions["delta_ms"]} == set(Hypothesis)
        assert all(
            set(map(ContrastName, e)) == set(ContrastName)
            for e in predictions["effects_ms"].values()
        )
    assert cell_views == 2


def test_ratio_symbols_and_verdict_labels():
    from fp4bench.analysis import plots, run

    assert [r.symbol for r in RatioName] == ["R", "D", "K", "F", "Φ", "AA"]
    assert set(run.RELABELS) == set(plots.ABOVE_ONE) == set(VerdictLabels)
    assert {r.name: r.labels for r in FULL.ratios if r.labels is not VerdictLabels.FORMAT} == {
        RatioName.K: VerdictLabels.KERNEL,
        RatioName.PHI: VerdictLabels.FUSION,
    }


def test_arms_parse_every_committed_cell():
    arms = {
        a for p in GOLDENS for v in _values_under(json.loads(p.read_text()), "arms", []) for a in v
    }
    assert {Arm(a) for a in arms} == set(Arm)


@pytest.mark.parametrize(
    "treatment, fmt",
    [
        ("MX", Format.MXFP4),
        ("MXp", Format.MXFP4),
        ("NV", Format.NVFP4),
        ("NVa", Format.NVFP4),
        ("NVnf", Format.NVFP4),
        ("NVx", Format.NVFP4),
        ("NVc", Format.NVFP4),
        ("NVt", Format.NVFP4),
        ("NVd", Format.NVFP4),
        ("NVv", Format.NVFP4),
    ],
)
def test_the_format_of_a_treatment(treatment, fmt):
    assert Treatment(treatment).fmt is fmt


def test_a_cell_is_keyed_and_labelled_as_the_files_and_summaries_store_it():
    cell = Cell(1, 127360)
    assert cell.json_key == "C=1,P=127360"
    assert cell.prompt_file_key == "1x127360"
    assert cell.label == "(1, 127,360)"
    assert json.dumps({cell.json_key: 1}) == '{"C=1,P=127360": 1}'


def test_cell_and_estimate_are_tuples():
    cell = Cell(128, 360)
    assert cell == (128, 360) and hash(cell) == hash((128, 360))
    assert (cell.batch, cell.prompt_len) == (128, 360)
    value, lo, hi = Estimate(1.0, 0.9, 1.1)
    assert (value, lo, hi) == (1.0, 0.9, 1.1)


PROBED = [
    RuntimeError("nvidia-smi hung"),
    FileNotFoundError(2, "No such file or directory", "nvidia-smi"),
    KeyError("uuid"),
    ValueError(""),
    ValueError("a: b: c\nsecond line"),
]


@pytest.mark.parametrize("exc", PROBED)
def test_a_probe_error_is_stored_as_the_probes_wrote_it(exc):
    error = ProbeError.of(exc)
    assert error.to_json() == f"error: {type(exc).__name__}: {exc}"
    assert error.to_json_object() == {"error": f"{type(exc).__name__}: {exc}"}
    assert error.detail == f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("exc", PROBED)
def test_a_probe_error_reads_back_from_either_stored_form(exc):
    error = ProbeError.of(exc)
    for stored in (
        error.to_json(),
        error.to_json_object(),
        json.loads(json.dumps(error.to_json())),
    ):
        assert ProbeError.from_json(stored) == error


@pytest.mark.parametrize(
    "stored",
    [
        "GPU-1",
        "error:RuntimeError: no space",
        "error: no type",
        "error: : no type",
        {"error": "no type"},
        {"error": "RuntimeError: x", "uuid": "GPU-1"},
        {"error": ["RuntimeError: x"]},
        None,
        3,
        ["error: RuntimeError: x"],
    ],
)
def test_other_values_are_not_probe_errors(stored):
    assert ProbeError.from_json(stored) is None


def test_probe_returns_the_value_or_what_was_raised():
    assert probe(lambda: "GPU-1") == "GPU-1"
    assert probe(lambda: {}["uuid"]) == ProbeError("KeyError", "'uuid'")
    assert probe(lambda: None) is None


def test_attempt_tags_what_was_raised_and_returns_the_rest_as_is():
    exc = OSError(28, "No space left on device")

    def fail():
        raise exc

    raised = attempt(fail)
    assert raised == Raised(exc) and raised.exc is exc
    assert attempt(lambda: 1.5) == 1.5
    assert attempt(lambda: exc) is exc


def test_an_exception_returned_is_a_value_not_a_failed_probe():
    exc = ValueError("a value")
    assert probe(lambda: exc) is exc


def test_a_probe_lets_an_interrupt_through():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        probe(interrupted)


def test_error_as_text_stores_only_a_probe_error_as_text():
    assert error_as_text(ProbeError("RuntimeError", "x")) == "error: RuntimeError: x"
    assert error_as_text("GPU-1") == "GPU-1"
    assert error_as_text({"mean": 1.5}) == {"mean": 1.5}


def test_error_as_object_stores_only_a_probe_error_as_an_object():
    assert error_as_object(ProbeError("RuntimeError", "x")) == {"error": "RuntimeError: x"}
    assert error_as_object({"uuid": "GPU-1"}) == {"uuid": "GPU-1"}
    assert error_as_object(None) is None


def test_value_keys_and_session_slots_are_the_tuples_they_replace():
    key = ValueKey(2, Treatment.NV, Cell(1, 1024))
    assert key == (2, "NV", (1, 1024)) and hash(key) == hash((2, "NV", (1, 1024)))
    assert (key.round, key.treatment, key.key) == (2, Treatment.NV, Cell(1, 1024))
    assert json.dumps([key, ValueKey(0, Treatment.MX, 8)]) == '[[2, "NV", [1, 1024]], [0, "MX", 8]]'
    slots = [SessionSlot(1, Treatment.MX), SessionSlot(0, Treatment.NV)]
    assert sorted(slots) == [(0, "NV"), (1, "MX")] and (1, Treatment.MX) in {slots[0]: 1}
