import copy
import dataclasses
import json
import math
import re
import sys

import pytest

from fp4bench import lib_bench, manifest, runner, settings, telemetry
from fp4bench.core.schema import (
    SCHEMA_VERSION,
    BlockTelemetry,
    M1Row,
    M2Row,
    ManifestLine,
    ServerRow,
    append_jsonl,
    append_row,
    load_rows,
)
from fp4bench.core.types import Cell, JsonObject, Treatment
from fp4bench.decode_step import decode_block
from fp4bench.server import parse_server_log
from tests import RUN_DIRS, RUNS_DIR, vllm_logs
from tests.analysis_runs import load_jsonl, write_jsonl

FILES = {
    "m1.jsonl": M1Row,
    "m2.jsonl": M2Row,
    "servers.jsonl": ServerRow,
    "manifests.jsonl": ManifestLine,
}
COMMITTED = [
    (run.name, name, cls)
    for run in RUN_DIRS
    for name, cls in FILES.items()
    if (run / name).exists()
]

TEL = {
    "window_s": 2.5,
    "counters_delta": {"SW Power Capping": 0},
    "env_throttle_us": 0,
    "sw_power_cap_frac": 0.0,
}
M1_V1 = {
    "round": 0,
    "treatment": "MX",
    "session_id": "r0_MX-1",
    "c": 8,
    "set": 1,
    "warmup": False,
    "first": "n2",
    "n1": 128,
    "n2": 1152,
    "t1_s": 1.0,
    "t2_s": 9.0,
    "step_s": 0.0078125,
    "decode_tok_s": 1024.0,
    "block_telemetry": TEL,
}


@pytest.mark.parametrize("run, name, cls", COMMITTED)
def test_every_committed_row_round_trips_byte_for_byte(run, name, cls):
    lines = (RUNS_DIR / run / name).read_text().splitlines()
    assert lines
    for line in lines:
        row = json.loads(line)
        out = cls.from_json(row).to_json()
        assert out.pop("schema_version") == SCHEMA_VERSION
        assert out == row
        assert json.dumps(out, allow_nan=False) == line


def test_committed_rows_know_every_key_of_today():
    for run, name, cls in COMMITTED:
        for row in load_rows(RUNS_DIR / run / name, cls):
            assert row.extra == {}, (run, name, sorted(row.extra))


def test_a_v1_m1_row_of_the_main_run_reads_as_1024_token_prompts_and_writes_back_without_them():
    row = M1Row.from_json(M1_V1)
    assert row.prompt_len == 1024 and row.preemptions_delta is None
    assert row.missing == {"prompt_len", "preemptions_delta"}
    assert row.treatment is Treatment.MX and row.block_telemetry == BlockTelemetry(**TEL)
    assert {k: v for k, v in row.to_json().items() if k != "schema_version"} == M1_V1


def test_a_null_preemption_delta_is_kept_apart_from_a_missing_one():
    row = M1Row.from_json({**M1_V1, "prompt_len": 360, "preemptions_delta": None})
    assert row.preemptions_delta is None and row.missing == frozenset()
    assert row.to_json()["preemptions_delta"] is None and row.to_json()["prompt_len"] == 360


def test_an_m1_row_may_lack_first_and_an_m2_row_its_aggregate_source():
    m1 = M1Row.from_json({k: v for k, v in M1_V1.items() if k != "first"})
    assert m1.first is None and "first" not in m1.to_json()
    m2_stored = {
        "round": 0,
        "treatment": "MX",
        "session_id": "r0_MX-1",
        "c": 8,
        "valid": True,
        "invalid_reasons": [],
        "failure_reason": "",
        "aggregate_tps": 2000.0,
        "measurement_seconds": 30.0,
        "effective_concurrency": 8.0,
        "avg_running_reqs": 8.0,
        "itl_p50_ms": 4.0,
        "tps_per_user_p50": 250.0,
        "ttft_p50_ms": 50.0,
        "server_gen_throughput": 2000.0,
        "telemetry": TEL,
    }
    m2 = M2Row.from_json(m2_stored)
    assert m2.aggregate_source is None
    assert {k: v for k, v in m2.to_json().items() if k != "schema_version"} == m2_stored


def test_new_rows_carry_the_schema_version_first():
    out = M1Row.from_json(M1_V1).to_json()
    assert next(iter(out)) == "schema_version" and SCHEMA_VERSION == 2
    assert M1Row.from_json(out) == M1Row.from_json(M1_V1)


@pytest.mark.parametrize("version", [1, 3, "2", None])
def test_an_unknown_schema_version_is_refused(version):
    with pytest.raises(ValueError, match="schema_version"):
        M1Row.from_json({**M1_V1, "schema_version": version})


def test_unknown_keys_are_kept_and_written_back_after_the_known_ones():
    row = M1Row.from_json({**M1_V1, "moe_backends": ["x"], "later": 1})
    assert row.extra == {"moe_backends": ["x"], "later": 1}
    assert list(row.to_json())[-2:] == ["moe_backends", "later"]


def test_a_row_without_a_required_key_is_refused_by_name():
    with pytest.raises(ValueError, match=r"M1Row.*\['step_s'\]"):
        M1Row.from_json({k: v for k, v in M1_V1.items() if k != "step_s"})


def test_an_unknown_treatment_is_refused():
    with pytest.raises(ValueError, match="'NVz' is not a valid Treatment"):
        M1Row.from_json({**M1_V1, "treatment": "NVz"})


def _m1_kwargs(**over) -> dict:
    row = M1Row.from_json(M1_V1)
    return {**{k: getattr(row, k) for k in M1_V1}, **over}


def test_extra_may_not_shadow_a_field():
    for key in ("step_s", "schema_version"):
        with pytest.raises(ValueError, match=key):
            M1Row(**_m1_kwargs(extra={key: 1}))


def test_a_constructed_row_writes_only_the_fields_it_was_given():
    row = M1Row(**_m1_kwargs())
    assert "prompt_len" not in row.to_json() and row.prompt_len == 1024
    assert M1Row(**_m1_kwargs(prompt_len=4096)).to_json()["prompt_len"] == 4096


def test_a_failed_scan_session_row_round_trips():
    stored = {
        "round": 0,
        "treatment": "NVt",
        "session_id": "r0_NVt-1",
        "start_id": "s",
        "failed": True,
        "error": "RuntimeError('boom')",
    }
    row = ServerRow.from_json(stored)
    assert row.failed is True and row.server_argv is None and row.linear_kernels is None
    assert {k: v for k, v in row.to_json().items() if k != "schema_version"} == stored
    assert ServerRow.from_json({**stored, "failed": False}).failed is False
    assert (
        ServerRow.from_json({k: stored[k] for k in ("round", "treatment", "session_id")}).failed
        is False
    )


def test_a_row_read_from_a_line_is_written_back_as_the_line_was():
    stored = {
        "round": 0,
        "treatment": "NVt",
        "session_id": "r0_NVt-1",
        "failed": True,
        "error": "RuntimeError('x')",
    }
    v1, v2 = ServerRow.from_json(stored), ServerRow.from_json({"schema_version": 2, **stored})
    assert v1 == v2 and (v1.stored_version, v2.stored_version) == (None, 2)
    assert v1.as_stored() == stored
    assert v2.as_stored() == {"schema_version": 2, **stored} == v2.to_json()
    assert v2.replace(error="y").as_stored() == {"schema_version": 2, **stored, "error": "y"}
    assert ServerRow(**dict(stored)).as_stored() == stored


def test_a_partial_manifest_line_round_trips():
    stored = {"utc": "2026-10-04T00:00:00+00:00", "protocol": {"rounds": 2}, "problems": []}
    line = ManifestLine.from_json(stored)
    assert line.protocol == {"rounds": 2} and line.inputs is None
    assert {k: v for k, v in line.to_json().items() if k != "schema_version"} == stored


def test_append_row_and_load_rows(tmp_path):
    path = tmp_path / "m1.jsonl"
    append_row(path, M1Row(**_m1_kwargs(step_s=math.nan, prompt_len=360)))
    (raw,) = load_jsonl(path)
    assert raw["schema_version"] == 2 and raw["step_s"] is None and raw["prompt_len"] == 360
    (row,) = load_rows(path, M1Row)
    assert row.step_s is None and row.prompt_len == 360
    assert load_rows(tmp_path / "absent.jsonl", M1Row) == []


def test_a_bad_json_line_names_the_file_and_the_line(tmp_path):
    path = tmp_path / "m1.jsonl"
    path.write_text(f'{json.dumps(M1_V1)}\n\n{{"round": 0,\n{json.dumps(M1_V1)}\n')
    where = re.escape(f"{path}, line 3: Expecting property name")
    for load in (load_jsonl, lambda p: load_rows(p, M1Row)):
        with pytest.raises(ValueError, match=f"^{where}") as info:
            load(path)
        assert isinstance(info.value.__cause__, json.JSONDecodeError)


def test_load_jsonl_skips_a_truncated_last_line_with_a_warning(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3, "b"')
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]
    err = capsys.readouterr().err
    assert "warning" in err and "rows.jsonl" in err


def test_load_jsonl_only_line_truncated_is_empty(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1')
    assert load_jsonl(path) == []
    assert "warning" in capsys.readouterr().err


def test_load_jsonl_last_line_without_newline_still_parses(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}')
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert capsys.readouterr().err == ""


def test_append_jsonl_writes_strict_json_with_nan_and_inf_as_null(tmp_path):
    path = tmp_path / "rows.jsonl"
    row: JsonObject = {
        "step_s": math.nan,
        "tps": math.inf,
        "nested": {"x": -math.inf, "ok": 1.5},
        "lst": [math.nan, 2, (math.inf, "s")],
        "flag": True,
        "none": None,
        "n": 3,
    }
    append_jsonl(path, row)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text

    def reject(constant):
        raise AssertionError(f"non-strict JSON constant {constant}")

    assert json.loads(text, parse_constant=reject) == {
        "step_s": None,
        "tps": None,
        "nested": {"x": None, "ok": 1.5},
        "lst": [None, 2, [None, "s"]],
        "flag": True,
        "none": None,
        "n": 3,
    }
    assert isinstance(step_s := row["step_s"], float) and math.isnan(step_s)


def test_append_jsonl_appends_one_line_per_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert path.read_text().count("\n") == 2


def test_append_jsonl_does_not_glue_a_row_onto_a_truncated_tail(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2, "b"')
    append_jsonl(path, {"a": 3})
    assert load_jsonl(path) == [{"a": 1}, {"a": 3}]
    assert path.read_text().endswith('{"a": 3}\n')


def test_append_jsonl_keeps_a_complete_last_row_that_lacks_its_newline(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}')
    append_jsonl(path, {"a": 3})
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]


@pytest.mark.parametrize(
    "bad, why",
    [
        ({k: v for k, v in M1_V1.items() if k != "step_s"}, r"M1Row row lacks required keys"),
        ({**M1_V1, "treatment": "NVz"}, r"'NVz' is not a valid Treatment"),
        ({**M1_V1, "schema_version": 3}, r"M1Row: unknown schema_version 3"),
        ({**M1_V1, "block_telemetry": [1]}, r"BlockTelemetry: expected a JSON object"),
        ([1, 2], r"M1Row: expected a JSON object"),
    ],
)
def test_a_row_that_does_not_load_names_the_file_and_the_line(tmp_path, bad, why):
    path = tmp_path / "m1.jsonl"
    path.write_text(f"{json.dumps(M1_V1)}\n\n{json.dumps(bad)}\n{json.dumps(M1_V1)}\n")
    with pytest.raises(ValueError, match=rf"^{re.escape(f'{path}, line 3: ')}{why}") as info:
        load_rows(path, M1Row)
    assert isinstance(info.value.__cause__, ValueError)


def test_load_rows_leaves_out_a_row_that_does_not_load_only_if_told_to(tmp_path):
    path = tmp_path / "m1.jsonl"
    ghost = {k: v for k, v in M1_V1.items() if k != "block_telemetry"} | {"session_id": "ghost"}
    write_jsonl(path, (M1_V1, ghost, [1], M1_V1))

    def not_counted(row) -> bool:
        return not isinstance(row, dict) or row.get("session_id") != "r0_MX-1"

    assert load_rows(path, M1Row, skip_if_bad=not_counted) == [M1Row.from_json(M1_V1)] * 2
    with pytest.raises(ValueError, match=r"m1.jsonl, line 3: M1Row: expected a JSON object"):
        load_rows(path, M1Row, skip_if_bad=lambda row: isinstance(row, dict))
    path.write_text(json.dumps({**ghost, "session_id": "r0_MX-1"}) + "\n")
    with pytest.raises(ValueError, match=r"m1.jsonl, line 1: M1Row row lacks"):
        load_rows(path, M1Row, skip_if_bad=not_counted)


def test_a_constructed_row_takes_its_types_from_plain_json_values():
    row = M1Row(**_m1_kwargs(treatment="NVnf", block_telemetry=TEL))
    assert row.treatment is Treatment.NVNF and row.block_telemetry == BlockTelemetry(**TEL)
    assert row.to_json()["treatment"] == "NVnf" and type(row.to_json()["treatment"]) is str


def test_a_v1_row_equals_the_same_row_with_its_defaults_written_out():
    v1 = M1Row.from_json(M1_V1)
    written = M1Row.from_json({**M1_V1, "prompt_len": 1024, "preemptions_delta": None})
    assert v1 == written and v1.missing != written.missing
    assert v1 != M1Row.from_json({**M1_V1, "prompt_len": 360})


def test_replace_keeps_the_missing_fields_it_does_not_change():
    row = M1Row.from_json({**M1_V1, "later": 1})
    new = row.replace(c=32, treatment="NV")
    assert new.c == 32 and new.treatment is Treatment.NV and new.extra == {"later": 1}
    assert new.missing == row.missing == {"prompt_len", "preemptions_delta"}
    assert {k: v for k, v in new.to_json().items() if k != "schema_version"} == {
        **M1_V1,
        "c": 32,
        "treatment": "NV",
        "later": 1,
    }
    assert row.c == 8


@pytest.mark.parametrize("prompt_len", [1024, 2048])
def test_a_missing_field_that_replace_changes_is_written(prompt_len):
    new = M1Row.from_json(M1_V1).replace(prompt_len=prompt_len)
    assert new.missing == {"preemptions_delta"} and new.to_json()["prompt_len"] == prompt_len


def test_replace_refuses_what_the_constructor_refuses():
    row = M1Row.from_json(M1_V1)
    for changes in ({"missing": frozenset()}, {"stored_version": 2}, {"no_such_field": 1}):
        with pytest.raises(TypeError):
            row.replace(**changes)
    with pytest.raises(ValueError, match="'NVz' is not a valid Treatment"):
        row.replace(treatment="NVz")


def test_replace_works_on_nested_records():
    tel = BlockTelemetry(**TEL).replace(window_s=5.0)
    assert tel.window_s == 5.0 and tel.env_throttle_us == 0


@pytest.mark.parametrize(
    "row",
    [
        M1Row.from_json(M1_V1),
        BlockTelemetry(**TEL),
        ServerRow.from_json({"round": 0, "treatment": "NV", "session_id": "r0_NV-1"}),
    ],
    ids=lambda row: type(row).__name__,
)
def test_copy_replace_is_replace_on_every_record_class(row):
    if sys.version_info < (3, 13):
        pytest.skip("copy.replace is new in Python 3.13")
    new = copy.replace(row, extra={"later": 1})
    assert new == row.replace(extra={"later": 1}) and new.missing == row.missing
    assert new.to_json() == {**row.to_json(), "later": 1}


def test_an_m1_row_names_its_cell():
    assert M1Row.from_json(M1_V1).cell == Cell(8, 1024)
    assert M1Row.from_json({**M1_V1, "c": 1, "prompt_len": 127360}).cell == Cell(1, 127360)
    assert type(M1Row.from_json(M1_V1).cell) is Cell


def field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)} - {"extra", "missing"}


@pytest.mark.parametrize("probes_fail", [False, True])
def test_collect_manifest_keys_are_manifest_line_fields(monkeypatch, probes_fail):
    def failing():
        raise RuntimeError("no GPU here")

    monkeypatch.setattr(manifest, "gpu_info", failing if probes_fail else lambda: {"uuid": "GPU-0"})
    monkeypatch.setattr(
        manifest, "lib_commit", failing if probes_fail else lambda: settings.LIB_COMMIT
    )
    monkeypatch.setattr(manifest, "package_versions", lambda: dict.fromkeys(manifest.PACKAGES))
    monkeypatch.setattr(manifest, "cuda_version", lambda: None)
    passed_by_the_runner = {
        "study",
        "start_id",
        "inputs",
        "checkpoint_report",
        "protocol",
        "problems",
    }
    collected = manifest.collect_manifest()
    assert set(collected) <= field_names(ManifestLine) - passed_by_the_runner
    assert {"code_commit", "code_dirty"} <= set(collected)


def test_decode_block_keys_are_m1_row_fields():
    rows = decode_block(lambda prompts, n: 1.0 + n / 1000, [[[1], [2]]] * 3, 2, 128, 1152, 2)
    passed_by_the_runner = {
        "round",
        "treatment",
        "session_id",
        "prompt_len",
        "block_telemetry",
        "preemptions_delta",
    }
    assert rows and all(set(row) <= field_names(M1Row) - passed_by_the_runner for row in rows)


BENCH_CELL = {
    "concurrency": 8,
    "context_tokens": 0,
    "aggregate_tps": 400.0,
    "aggregate_source": lib_bench.VALID_AGGREGATE_SOURCE,
    "measurement_seconds": 30.0,
    "effective_concurrency": 8.0,
    "avg_running_reqs": 8.0,
    "inter_token_latency_p50": 0.0125,
    "output_tps_per_user_p50": 80.0,
    "ttft_p50": 0.25,
    "server_gen_throughput": 392.0,
    "num_errors": 0,
}


@pytest.mark.parametrize(
    "record",
    [
        lib_bench.parse_lib_result({"results": [BENCH_CELL]}, 8),
        lib_bench.parse_lib_result({"results": [{**BENCH_CELL, "underfilled": True}]}, 8),
        lib_bench.parse_lib_result({"results": []}, 8),
        lib_bench.missing_cell_record(),
    ],
    ids=["valid", "invalid", "no_cell", "missing_cell_record"],
)
def test_m2_bench_record_keys_are_m2_row_fields(record):
    guarded = runner.preemption_guard(record, 1.0)
    passed_by_the_runner = {"round", "treatment", "session_id", "c", "telemetry"}
    assert set(guarded) <= field_names(M2Row) - passed_by_the_runner


@pytest.mark.parametrize("log", ["", vllm_logs.NV_LOG, vllm_logs.MX_LOG], ids=["empty", "NV", "MX"])
def test_server_log_fact_keys_are_server_row_fields(log):
    passed_by_the_runner = {
        "round",
        "treatment",
        "session_id",
        "start_id",
        "gpu_uuid",
        "server_argv",
        "nll",
        "nll_per_prompt",
        "server_env",
        "autotune_cache_dir",
        "autotune_fresh",
        "autotune_saved_files",
        "failed",
        "error",
    }
    assert set(parse_server_log(log)) <= field_names(ServerRow) - passed_by_the_runner


def test_throttle_summary_keys_are_block_telemetry_fields():
    summary = telemetry.throttle_summary(dict.fromkeys(telemetry.COUNTER_NAMES, 0), 1.0)
    assert set(summary) <= field_names(BlockTelemetry) - {"window_s", "counters_delta"}
