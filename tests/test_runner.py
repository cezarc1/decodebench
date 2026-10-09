import hashlib
import json
import math
import shutil
import types
from dataclasses import replace
from pathlib import Path

import pytest

from fp4bench import lib_bench, metrics, runner, settings
from fp4bench.core.schema import (
    M1Row,
    M2Row,
    ManifestLine,
    ServerRow,
    append_jsonl,
    load_rows,
)
from fp4bench.core.types import Cell, KvDtype, ProbeError, RetryPolicy, Treatment
from fp4bench.runner import rounds_to_run
from fp4bench.studies import model
from fp4bench.studies.base import Prompts, ServerSettings, Study, cells_at
from fp4bench.studies.expb import EXPB
from fp4bench.studies.expc import EXPC
from fp4bench.studies.kernel_scan import NVA_SCAN
from fp4bench.studies.main import FULL
from fp4bench.studies.registry import STUDIES
from fp4bench.studies.smoke import SMOKE, SMOKE_NF
from fp4bench.telemetry import COUNTER_NAMES
from tests import RUNS_DIR
from tests.analysis_runs import CLEAN_CKPT, load_jsonl, write_jsonl

T = Treatment


def small_study(
    treatments: tuple[Treatment, ...], batches: tuple[int, ...], rounds: int = 1, **fields
) -> Study:
    """A main-run-like study at `batches` x 1,024 tokens: one M1 pair, 1 s of M2, local weights."""
    return Study(
        name="test",
        treatments=treatments,
        cells=cells_at(settings.M1_INPUT_LEN, batches),
        rounds=rounds,
        **{
            "m1_reps": 1,
            "m2_duration_s": 1,
            "require_published": False,
            "primary_batches": batches,
            **fields,
        },
    )


STUDY = small_study((T.MX, T.NV), (1, 8), rounds=2)
VLLM_MISMATCH = "vllm: expected 0.31.0, got 0.32.0"
BF16_REFERENCE = {
    "per_prompt": [1.5, 2.5],
    "mean": 2.0,
    "source_revision": model.SRC_MODEL_REVISION,
    "nll_prompts_sha256": "f" * 64,
}
KERNEL_LINE = (
    "(EngineCore pid=4242) INFO 10-05 12:00:01 [linear/__init__.py:1186] "
    "Using FlashInferCuteDslNvFp4LinearKernel for NVFP4 GEMM"
)
KV_LINE = (
    "(EngineCore pid=4242) INFO 10-05 12:00:02 [kv_cache_utils.py:2464] GPU KV cache size: "
    "558,496 tokens, Maximum concurrency for 4,096 tokens per request: 136.35x"
)
SERVER_LOG = (
    f"INFO Model loading took 13.2051 GiB memory and 41.2 seconds\n{KERNEL_LINE}\n{KV_LINE}\n"
)


def server_row(round_, treatment, **fields):
    return ServerRow(
        round=round_,
        treatment=treatment,
        session_id=f"r{round_}_{treatment}-{fields.get('start_id')}",
        **fields,
    )


def sessions(round_, start_id="A", treatments=FULL.treatments):
    return [server_row(round_, t, start_id=start_id) for t in treatments]


def test_rounds_to_run_skips_only_fully_completed_rounds():
    done = sessions(0) + sessions(1, treatments=("MX",))
    assert rounds_to_run(done, FULL) == [1, 2, 3, 4]


def test_rounds_to_run_fresh_start():
    assert rounds_to_run([], SMOKE) == [0]


def test_rounds_from_different_starts_are_each_complete_on_their_own():
    done = sessions(0, "A") + sessions(1, "B")
    assert rounds_to_run(done, replace(FULL, rounds=2)) == []


def test_a_round_whose_latest_sessions_span_two_starts_is_not_complete():
    done = [*sessions(0, "A"), server_row(0, "MX", start_id="B")]
    assert rounds_to_run(done, replace(FULL, rounds=1)) == [0]


def test_a_round_rerun_whole_under_one_start_is_complete_again():
    done = [*sessions(0, "A"), server_row(0, "MX", start_id="B"), *sessions(0, "C")]
    assert rounds_to_run(done, replace(FULL, rounds=1)) == []


def test_only_the_latest_row_per_treatment_decides_which_start_a_round_belongs_to():
    done = sessions(0, "A") + sessions(0, "B")
    assert rounds_to_run(done, replace(FULL, rounds=1)) == []
    assert rounds_to_run(sessions(0, "A")[:1] + sessions(0, "B"), replace(FULL, rounds=1)) == []


@pytest.mark.parametrize("missing", [None, ""])
def test_a_session_row_without_a_start_id_cannot_prove_the_round_was_atomic(missing):
    done = sessions(0, "A")
    done[1] = server_row(0, done[1].treatment, **({} if missing is None else {"start_id": missing}))
    assert rounds_to_run(done, replace(FULL, rounds=1)) == [0]
    assert rounds_to_run([server_row(0, t) for t in FULL.treatments], replace(FULL, rounds=1)) == [
        0
    ]


def m1_pair(c: int, s: int, step_s: float) -> dict:
    """A decode_step.decode_block row: pair `s` (0 = warmup) at concurrency c."""
    return {
        "c": c,
        "set": s,
        "warmup": s == 0,
        "first": "n1" if s % 2 == 0 else "n2",
        "n1": settings.M1_N1,
        "n2": settings.M1_N2,
        "t1_s": 1.0,
        "t2_s": 1.0 + step_s * 1024,
        "step_s": step_s,
        "decode_tok_s": c / step_s,
    }


def m2_cell(**fields) -> dict:
    """A lib_bench.run_lib_decode record: a valid cell unless `fields` say otherwise."""
    return {**lib_bench.missing_cell_record(), "valid": True, "invalid_reasons": [], **fields}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Replace GPU, server and bench with fakes; state is exposed for the tests to steer."""
    st = types.SimpleNamespace(
        run_dir=tmp_path / "run",
        calls=[],
        commits=[],
        sessions=[],
        polls=0,
        die_at_poll=None,
        proc_none=False,
        server=None,
        gpu_error=None,
        counters_error=None,
        problems=[],
        counter_reads=0,
        m2_hook=None,
        fail_session=None,
        report=None,
        report_error=None,
        fingerprints={},
        session_starts=[],
        servers=[],
        nlls=[2.0, 3.0],
        nll_calls=[],
        copies=[],
        server_hook=None,
        server_log=SERVER_LOG,
        nll_prompts_sha256=None,
        preemption_reads=[],
        preemptions_hook=None,
        events=[],
        preemption_sleeps=[],
        blocks=[],
        m2_calls=[],
        cell_prompts=[],
    )

    class FakeProc:
        returncode = None

        def poll(self):
            st.polls += 1
            if st.die_at_poll is not None and st.polls >= st.die_at_poll:
                self.returncode = -9
            return self.returncode

    class FakeServer:
        def __init__(self, model_dir, log_path, args=(), env=None):
            self.model_dir, self.log_path, self.args = model_dir, Path(log_path), tuple(args)
            self.env = dict(env or {})
            self.cmd = ["vllm", "serve", model_dir, *args]
            self.proc = None if st.proc_none else FakeProc()
            st.server = self
            st.servers.append(self)

        def __enter__(self):
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(st.server_log)
            if st.server_hook:
                st.server_hook(self)
            return self

        def __exit__(self, *exc):
            return None

        def log_text(self):
            return self.log_path.read_text()

    class FakeSampler:
        def __init__(self, csv_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def decode_block(wave, prompt_sets, c, n1, n2, reps):
        st.blocks.append((c, prompt_sets, n1, n2, reps))
        return [m1_pair(c, 0, math.nan), m1_pair(c, 1, 0.01)]

    def run_lib_decode(c, duration_s, output_path, log_path):
        st.m2_calls.append(c)
        if st.m2_hook:
            st.m2_hook(c)
        return m2_cell(aggregate_tps=100.0 * c)

    def read_counters():
        st.counter_reads += 1
        st.calls.append("counters")
        if st.counters_error:
            raise st.counters_error
        return dict.fromkeys(COUNTER_NAMES, 0)

    def gpu_info():
        if st.gpu_error:
            raise st.gpu_error
        return {"uuid": "GPU-1"}

    def collect_manifest():
        st.calls.append("collect")
        return {"utc": "2026-10-04T00:00:00+00:00", "gpu": {"name": "NVIDIA B200"}}

    def verify_manifest(manifest):
        st.calls.append("verify")
        return list(st.problems)

    def checkpoint_report(mx_dir, nv_dir):
        st.calls.append("report")
        st.report_dirs = (mx_dir, nv_dir)
        if st.report_error:
            raise st.report_error
        return CLEAN_CKPT if st.report is None else st.report

    data = tmp_path / "data"
    data.mkdir()
    (data / "m1.json").write_text(json.dumps([[[1, 2, 3]]]))
    (data / "nll.json").write_text(json.dumps([[4, 5, 6]]))
    nll_sha = hashlib.sha256((data / "nll.json").read_bytes()).hexdigest()
    (data / "bf16_ref.json").write_text(
        json.dumps({**BF16_REFERENCE, "nll_prompts_sha256": nll_sha})
    )
    st.nll_prompts_sha256 = nll_sha
    volume = {"mx": tmp_path / "mx", "nv": tmp_path / "nv", "nvx": tmp_path / "nvx"}
    for d in volume.values():
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": d.name}))
        (d / "model.safetensors").write_bytes(d.name.encode() * 100)
    local_root = tmp_path / "local"
    st.volume, st.local_root = volume, local_root
    for name, value in {
        "M1_PROMPTS_PATH": data / "m1.json",
        "NLL_PROMPTS_PATH": data / "nll.json",
        "C_PROMPTS_PATH": data / "c.json",
        "BF16_REF_NLL_PATH": data / "bf16_ref.json",
    }.items():
        monkeypatch.setattr(settings, name, str(value))
    for name, kind in {"MX_MODEL_DIR": "mx", "NV_MODEL_DIR": "nv", "NVX_MODEL_DIR": "nvx"}.items():
        monkeypatch.setattr(model, name, str(volume[kind]))
    monkeypatch.setattr(
        model,
        "TREATMENTS",
        {
            t: replace(
                spec,
                volume_dir=str(volume[spec.checkpoint]),
                model_dir=str(local_root / spec.checkpoint),
            )
            for t, spec in model.TREATMENTS.items()
        },
    )

    def content_fingerprint(d):
        """By directory name; a local copy has the volume's value unless the test set its own."""
        d = Path(d)
        if d.parent == local_root and ("local-content", d.name) in st.fingerprints:
            return st.fingerprints[("local-content", d.name)]
        return st.fingerprints.get(("content", d.name), f"content:{d.name}")

    real_copytree = shutil.copytree

    def copytree(src, dst, *args, **kwargs):
        st.calls.append("copy")
        st.copies.append((Path(src).name, Path(dst).name))
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(runner.shutil, "copytree", copytree)
    for name, value in {
        "VllmServer": FakeServer,
        "GpuSampler": FakeSampler,
        "decode_block": decode_block,
        "run_lib_decode": run_lib_decode,
        "read_counters": read_counters,
        "gpu_info": gpu_info,
        "collect_manifest": collect_manifest,
        "verify_manifest": verify_manifest,
        "checkpoint_report": checkpoint_report,
        "layout_fingerprint": lambda d: st.fingerprints.get(
            ("layout", Path(d).name), f"layout:{Path(d).name}"
        ),
        "content_fingerprint": content_fingerprint,
    }.items():
        monkeypatch.setattr(runner, name, value)

    def prompt_nlls(*args):
        st.nll_calls.append(args)
        st.events.append("nll")
        return list(st.nlls)

    monkeypatch.setattr(runner, "prompt_nlls", prompt_nlls)

    def read_preemptions(base_url):
        """0.0 unless a test steers it with preemptions_hook(n), n the 1-based read number."""
        st.preemption_reads.append(base_url)
        st.events.append("preemptions")
        if st.preemptions_hook:
            return st.preemptions_hook(len(st.preemption_reads))
        return 0.0

    monkeypatch.setattr(runner, "read_preemptions", read_preemptions)
    monkeypatch.setattr(runner, "_preemption_sleep", st.preemption_sleeps.append)

    def commit():
        st.commits.append((st.run_dir / "errors.jsonl").exists())

    st.commit = commit
    return st


@pytest.fixture
def stubbed(env, monkeypatch):
    """`env` with whole sessions replaced: each records (round, treatment) and is complete."""

    def fake_session(
        run_dir, r, treatment, study, prompt_sets, nll_prompts, start_id, cell_prompts=None
    ):
        env.sessions.append((r, treatment))
        env.cell_prompts.append(cell_prompts)
        env.session_starts.append(start_id)
        if env.fail_session == len(env.sessions):
            raise ValueError("boom")
        append_jsonl(
            Path(run_dir) / "servers.jsonl",
            {
                "round": r,
                "treatment": treatment,
                "session_id": f"s{len(env.sessions)}",
                "start_id": start_id,
            },
        )

    monkeypatch.setattr(runner, "run_server_session", fake_session)
    return env


def start(env, study=STUDY, **kw) -> list[dict]:
    """One start of `study`; the manifest lines the run has after it."""
    runner.run_experiment(env.run_dir, study, env.commit, **kw)
    return load_jsonl(env.run_dir / "manifests.jsonl")


def refused_start(
    env, match, study=STUDY, error: type[BaseException] = RuntimeError, **kw
) -> tuple[str, list[dict]]:
    """A start of `study` that raises `error` matching `match`: its message, the manifest lines."""
    with pytest.raises(error, match=match) as exc:
        runner.run_experiment(env.run_dir, study, env.commit, **kw)
    return str(exc.value), load_jsonl(env.run_dir / "manifests.jsonl")


def run_session(env, treatment=Treatment.MX, study=STUDY, cell_prompts=None):
    runner.run_server_session(
        env.run_dir, 0, treatment, study, [[[1]]], [[1]], "start-1", cell_prompts=cell_prompts
    )


def test_healthy_session_is_recorded_as_complete(env):
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert (row["round"], row["treatment"], row["gpu_uuid"], row["nll"]) == (0, "MX", "GPU-1", 2.5)
    assert row["nll_per_prompt"] == [2.0, 3.0]
    assert row["start_id"] == "start-1"
    assert row["session_id"].startswith("r0_MX-") and row["weights_gib"] == 13.2051
    assert [r["c"] for r in load_jsonl(env.run_dir / "m2.jsonl")] == [1, 8]


def test_session_raises_and_is_not_recorded_when_server_dies_during_m2(env):
    def kill_during_c8(c):
        if c == 8:
            env.server.proc.returncode = -9

    env.m2_hook = kill_during_c8
    with pytest.raises(RuntimeError) as exc:
        run_session(env)
    session_id = load_jsonl(env.run_dir / "m2.jsonl")[0]["session_id"]
    assert session_id in str(exc.value) and str(env.server.log_path) in str(exc.value)
    assert not (env.run_dir / "servers.jsonl").exists()
    assert rounds_to_run(load_rows(env.run_dir / "servers.jsonl", ServerRow), STUDY) == [0, 1]
    assert [r["c"] for r in load_jsonl(env.run_dir / "m2.jsonl")] == [1, 8]


def test_session_checks_the_server_before_every_m2_cell(env):
    def bench_without_a_server(c):
        raise FileNotFoundError(f"m2_raw/c{c}.json")

    env.m2_hook = bench_without_a_server
    env.die_at_poll = 1
    with pytest.raises(RuntimeError, match="died") as exc:
        run_session(env)
    assert str(env.server.log_path) in str(exc.value)
    assert not (env.run_dir / "m2.jsonl").exists()
    assert not (env.run_dir / "servers.jsonl").exists()


def test_session_does_not_start_the_next_m2_cell_on_a_server_that_died_in_the_last_one(env):
    def kill_during_c1(c):
        if c == 1:
            env.server.proc.returncode = -9
        else:
            raise FileNotFoundError(f"m2_raw/c{c}.json")

    env.m2_hook = kill_during_c1
    with pytest.raises(RuntimeError, match="died"):
        run_session(env)
    assert [r["c"] for r in load_jsonl(env.run_dir / "m2.jsonl")] == [1]
    assert not (env.run_dir / "servers.jsonl").exists()


def test_session_checks_the_server_once_per_boundary_without_redundant_calls(env):
    run_session(env)
    assert env.polls == len(STUDY.batches) + 1


def test_session_checks_the_server_again_right_after_the_last_cell(env):
    env.die_at_poll = 3
    with pytest.raises(RuntimeError, match="died"):
        run_session(env)
    assert [r["c"] for r in load_jsonl(env.run_dir / "m2.jsonl")] == [1, 8]
    assert not (env.run_dir / "servers.jsonl").exists()


def test_session_raises_when_there_is_no_server_process(env):
    env.proc_none = True
    with pytest.raises(RuntimeError, match="died"):
        run_session(env)
    assert not (env.run_dir / "servers.jsonl").exists()


def test_measured_returns_the_result_and_window_telemetry(env):
    result, tel = runner.measured(lambda: {"answer": 42})
    assert result == {"answer": 42}
    assert set(tel.to_json()) == {
        "window_s",
        "counters_delta",
        "env_throttle_us",
        "sw_power_cap_frac",
    }
    assert env.counter_reads == 2


def test_run_experiment_logs_a_failed_session_commits_and_reraises(stubbed):
    stubbed.fail_session = 2
    refused_start(stubbed, "boom", error=ValueError)
    (row,) = load_jsonl(stubbed.run_dir / "errors.jsonl")
    assert set(row) == {"utc", "session_hint", "error", "traceback"}
    assert row["error"] == repr(ValueError("boom"))
    assert "Traceback" in row["traceback"] and "ValueError: boom" in row["traceback"]
    assert row["session_hint"] == {"round": 0, "treatment": "NV"}
    assert stubbed.commits[-1] is True


def test_run_experiment_commits_in_finally_on_success_without_an_error_log(stubbed):
    start(stubbed)
    assert not (stubbed.run_dir / "errors.jsonl").exists()
    assert len(stubbed.commits) == 1 + len(stubbed.sessions) + 1
    assert stubbed.sessions == [(0, "MX"), (0, "NV"), (1, "NV"), (1, "MX")]


def test_run_experiment_logs_errors_that_are_not_session_failures(stubbed):
    stubbed.counters_error = ValueError("no 'Clocks Event Reasons Counters' section")
    refused_start(stubbed, "Clocks Event", error=ValueError)
    (row,) = load_jsonl(stubbed.run_dir / "errors.jsonl")
    assert row["session_hint"] is None and "Clocks Event" in row["error"]


def test_run_experiment_reads_counters_as_a_preflight_after_verifying(stubbed):
    start(stubbed)
    assert stubbed.calls[:6] == ["collect", "verify", "report", "copy", "copy", "counters"]
    assert stubbed.calls.count("counters") == 1


def test_run_experiment_fails_before_any_session_on_a_counter_format_mismatch(stubbed):
    stubbed.counters_error = ValueError("missing counters: ['Sync Boost']")
    _, lines = refused_start(stubbed, "Sync Boost", error=ValueError)
    assert stubbed.sessions == []
    assert len(lines) == 1


def test_run_experiment_environment_problems_stop_before_the_preflight(stubbed):
    stubbed.problems = [VLLM_MISMATCH]
    _, lines = refused_start(stubbed, "environment check failed")
    assert stubbed.counter_reads == 0 and stubbed.sessions == []
    assert lines[0]["problems"] == stubbed.problems
    assert len(lines[0]["start_id"]) == 32


def with_rounds(rounds):
    return replace(STUDY, rounds=rounds)


def test_restart_may_raise_the_number_of_rounds(stubbed):
    start(stubbed)
    stubbed.sessions.clear()
    lines = start(stubbed, with_rounds(3))
    assert [r for r, _ in stubbed.sessions] == [2, 2]
    assert len(lines) == 2


def test_restart_with_the_same_protocol_is_fine_and_each_line_records_its_inputs(stubbed):
    start(stubbed)
    lines = start(stubbed)
    assert len(lines) == 2
    assert [m["inputs"]["bf16_reference_nll"]["mean"] for m in lines] == [2.0, 2.0]
    assert [m["inputs"]["treatment_server_args"] for m in lines] == [
        {"MX": list(PIN), "NV": list(PIN)}
    ] * 2


def with_server(study: Study, **fields) -> Study:
    return replace(study, server=replace(study.server, **fields))


CHANGES = {
    "m1_reps": lambda s: replace(s, m1_reps=3),
    "m2_duration_s": lambda s: replace(s, m2_duration_s=60),
    "concurrencies": lambda s: replace(s, cells=cells_at(1024, (1, 8, 32))),
    "treatments": lambda s: replace(s, treatments=(T.NV, T.MX)),
    "kv_cache_dtype": lambda s: with_server(s, kv_dtype=KvDtype.FP8),
    "gpu_memory_utilization": lambda s: with_server(s, gpu_memory_utilization=0.95),
    "primary_concurrencies": lambda s: replace(s, primary_batches=(8,)),
    "cells": lambda s: replace(s, prompts=Prompts.CELLS),
    "max_model_len": lambda s: with_server(s, max_model_len=8192),
    "hf_overrides": lambda s: with_server(s, hf_overrides='{"max_position_embeddings": 8192}'),
}


@pytest.mark.parametrize("key", list(CHANGES))
def test_restart_with_a_changed_protocol_field_is_refused(stubbed, key):
    start(stubbed)
    stubbed.sessions.clear()
    _, lines = refused_start(stubbed, key, CHANGES[key](with_rounds(10)))
    assert stubbed.sessions == []
    assert len(lines) == 1


def test_protocol_mismatch_lists_every_differing_field(stubbed):
    start(stubbed)
    message, _ = refused_start(stubbed, None, replace(STUDY, m1_reps=3, m2_duration_s=60))
    assert "['m1_reps', 'm2_duration_s']" in message
    assert "treatments" not in message


def test_protocol_is_compared_with_the_first_manifest_line_only(stubbed):
    start(stubbed)
    start(stubbed, with_rounds(10))
    lines = start(stubbed, with_rounds(10))
    assert [m["protocol"]["rounds"] for m in lines] == [2, 10, 10]


def test_protocol_check_survives_a_truncated_manifest_tail(stubbed):
    start(stubbed)
    with (stubbed.run_dir / "manifests.jsonl").open("a") as f:
        f.write('{"utc": "2026-10-04T01:')
    assert len(start(stubbed)) == 2


def test_a_nan_warmup_row_round_trips_through_a_session(env):
    run_session(env)
    rows = load_jsonl(env.run_dir / "m1.jsonl")
    assert rows[0]["step_s"] is None and rows[1]["step_s"] == 0.01
    assert "NaN" not in (env.run_dir / "m1.jsonl").read_text()


def complete_round(env, r, start_id="old-start"):
    for t in STUDY.treatments:
        append_jsonl(
            env.run_dir / "servers.jsonl",
            {"round": r, "treatment": t, "session_id": "old", "start_id": start_id},
        )


@pytest.mark.parametrize(
    "completed, rerun_rounds, rounds_run",
    [
        pytest.param(0, (0,), [0, 0, 1, 1], id="reruns_a_completed_round"),
        pytest.param(1, (1, 1), [0, 0, 1, 1], id="the_sorted_union_with_the_pending_rounds"),
        pytest.param(0, (), [1, 1], id="none_and_completed_rounds_stay_skipped"),
    ],
)
def test_rerun_rounds(stubbed, completed, rerun_rounds, rounds_run):
    stubbed.run_dir.mkdir(parents=True)
    complete_round(stubbed, completed)
    start(stubbed, rerun_rounds=rerun_rounds)
    assert [r for r, _ in stubbed.sessions] == rounds_run


@pytest.mark.parametrize("bad", [(2,), (0, 5), (-1,)])
def test_rerun_rounds_outside_the_protocol_are_refused_up_front(stubbed, bad):
    refused_start(stubbed, "rerun", error=ValueError, rerun_rounds=bad)
    assert stubbed.sessions == [] and stubbed.calls == []
    assert not (stubbed.run_dir / "manifests.jsonl").exists()


def test_gpu_uuid_helper_returns_the_uuid(env):
    assert runner._gpu_uuid() == "GPU-1"


def test_gpu_uuid_helper_reports_the_probe_error(env):
    env.gpu_error = RuntimeError("nvidia-smi hung")
    assert runner._gpu_uuid() == ProbeError("RuntimeError", "nvidia-smi hung")


def test_session_is_recorded_even_if_the_gpu_query_fails_at_the_end(env):
    env.gpu_error = FileNotFoundError("nvidia-smi")
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["gpu_uuid"] == "error: FileNotFoundError: nvidia-smi"
    rows = load_rows(env.run_dir / "servers.jsonl", ServerRow)
    assert rounds_to_run(rows, STUDY) == [0, 1]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_manifest_line_records_input_identities(stubbed):
    (line,) = start(stubbed)
    assert line["inputs"] == {
        "m1_prompts_sha256": sha(settings.M1_PROMPTS_PATH),
        "nll_prompts_sha256": sha(settings.NLL_PROMPTS_PATH),
        "mx_layout_fingerprint": "layout:mx",
        "nv_layout_fingerprint": "layout:nv",
        "mx_content_fingerprint": "content:mx",
        "nv_content_fingerprint": "content:nv",
        "mx_fetch": None,
        "nv_fetch": None,
        "bf16_reference_nll": {"per_prompt": [1.5, 2.5], "mean": 2.0},
        "treatment_server_args": {"MX": list(PIN), "NV": list(PIN)},
        "local_staging": line["inputs"]["local_staging"],
    }
    assert set(line["inputs"]["local_staging"]) == {"mx", "nv"}
    assert line["protocol"]["rounds"] == 2 and line["problems"] == []


def test_manifest_line_records_the_whole_protocol_with_the_server_settings(stubbed):
    fp8 = replace(
        with_server(STUDY, kv_dtype=KvDtype.FP8, gpu_memory_utilization=0.95), primary_batches=(8,)
    )
    (line,) = start(stubbed, fp8)
    assert line["protocol"] == {
        "rounds": 2,
        "treatments": ["MX", "NV"],
        "concurrencies": [1, 8],
        "m1_reps": 1,
        "m2_duration_s": 1,
        "require_published": False,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.95,
        "primary_concurrencies": [8],
        "cells": [],
        "max_model_len": 4096,
        "hf_overrides": "",
    }


@pytest.mark.parametrize("kind, model_dir", [("mx", "MX_MODEL_DIR"), ("nv", "NV_MODEL_DIR")])
def test_manifest_line_includes_the_fetch_record_of_each_checkpoint_when_present(
    stubbed, kind, model_dir
):
    record = {"repo_id": "someone/name", "revision": "a" * 40, "utc": "2026-10-04T00:00:00+00:00"}
    (Path(getattr(model, model_dir)) / "fp4bench_fetch.json").write_text(json.dumps(record))
    inputs = start(stubbed)[0]["inputs"]
    other = "nv" if kind == "mx" else "mx"
    assert inputs[f"{kind}_fetch"] == record and inputs[f"{other}_fetch"] is None


def test_a_missing_bf16_reference_is_recorded_as_an_error_and_stops_the_run_early(stubbed):
    Path(settings.BF16_REF_NLL_PATH).unlink()
    _, (line,) = refused_start(stubbed, "environment check failed")
    assert line["inputs"]["bf16_reference_nll"].startswith("error: FileNotFoundError")
    assert [p for p in line["problems"] if p.startswith("input bf16_reference_nll: error:")]
    assert stubbed.sessions == []


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[1, 2]",
        '{"mean": 2.0}',
        '{"per_prompt": [1.0, 2.0]}',
        '{"per_prompt": "x", "mean": 2.0}',
        '{"per_prompt": [1.0, null], "mean": 2.0}',
        '{"per_prompt": [], "mean": 2.0}',
        '{"per_prompt": [1.0, 2.0], "mean": null}',
    ],
)
def test_a_malformed_bf16_reference_is_an_error_string_not_a_crash(stubbed, content):
    Path(settings.BF16_REF_NLL_PATH).write_text(content)
    _, (line,) = refused_start(stubbed, "environment check failed")
    assert line["inputs"]["bf16_reference_nll"].startswith("error: ")
    assert stubbed.sessions == []


def write_reference(env, **changes):
    """The BF16 reference file with `changes` (None drops a key)."""
    ref = {**BF16_REFERENCE, "nll_prompts_sha256": env.nll_prompts_sha256, **changes}
    Path(settings.BF16_REF_NLL_PATH).write_text(
        json.dumps({k: v for k, v in ref.items() if v is not None})
    )


def test_a_reference_computed_on_other_prompts_is_an_error_string_and_a_problem(stubbed):
    other = "0" * 64
    write_reference(stubbed, nll_prompts_sha256=other)
    _, (line,) = refused_start(stubbed, "environment check failed")
    value = line["inputs"]["bf16_reference_nll"]
    assert isinstance(value, str) and value.startswith("error: ValueError")
    assert other in value and stubbed.nll_prompts_sha256 in value and "nll_prompts_sha256" in value
    assert [p for p in line["problems"] if p.startswith("input bf16_reference_nll: error:")]
    assert stubbed.sessions == [] and stubbed.copies == []


def test_a_reference_without_the_prompts_hash_is_not_accepted_either(stubbed):
    write_reference(stubbed, nll_prompts_sha256=None)
    _, lines = refused_start(stubbed, "environment check failed")
    value = lines[0]["inputs"]["bf16_reference_nll"]
    assert value.startswith("error: ValueError") and "None" in value


def test_the_reference_is_compared_with_the_prompts_file_as_it_is_now(stubbed):
    Path(settings.NLL_PROMPTS_PATH).write_text(json.dumps([[7, 8, 9]]))
    _, lines = refused_start(stubbed, "environment check failed")
    assert lines[0]["inputs"]["bf16_reference_nll"].startswith("error: ValueError")


def test_reading_the_reference_directly_checks_the_hash_too(env):
    ref = runner.read_bf16_reference_nll(settings.BF16_REF_NLL_PATH)
    assert ref == {"per_prompt": [1.5, 2.5], "mean": 2.0}
    write_reference(env, nll_prompts_sha256="1" * 64)
    with pytest.raises(ValueError, match="hash"):
        runner.read_bf16_reference_nll(settings.BF16_REF_NLL_PATH)


def test_a_restart_with_a_different_bf16_reference_is_refused(stubbed):
    start(stubbed)
    stubbed.sessions.clear()
    write_reference(stubbed, mean=2.1, per_prompt=[1.6, 2.6])
    message, _ = refused_start(stubbed, "inputs differ")
    assert "['bf16_reference_nll']" in message
    assert stubbed.sessions == []


def test_unreadable_inputs_are_recorded_and_stop_the_run_early(stubbed, monkeypatch):
    Path(settings.NLL_PROMPTS_PATH).unlink()

    def fingerprint(model_dir):
        raise ValueError(f"no safetensors files found in {model_dir}")

    monkeypatch.setattr(runner, "layout_fingerprint", fingerprint)
    _, (line,) = refused_start(stubbed, "environment check failed")
    assert line["inputs"]["nll_prompts_sha256"].startswith("error: FileNotFoundError")
    assert line["inputs"]["nv_layout_fingerprint"].startswith("error: ValueError")
    assert len(line["problems"]) == 3 and all(p.startswith("input ") for p in line["problems"])
    assert stubbed.sessions == []


def test_every_manifest_line_of_a_restart_carries_its_own_inputs(stubbed):
    write_fetch_record()
    start(stubbed)
    write_fetch_record({**FETCH, "utc": "2026-10-05T00:00:00+00:00"})
    lines = start(stubbed)
    for kind in ("mx", "nv"):
        utcs = [m["inputs"][f"{kind}_fetch"]["utc"] for m in lines]
        assert utcs == ["2026-10-04T00:00:00+00:00", "2026-10-05T00:00:00+00:00"]


@pytest.mark.parametrize("side", ["mx", "nv"])
def test_an_unreadable_content_fingerprint_stops_the_run_early(stubbed, monkeypatch, side):
    def fingerprint(model_dir):
        if Path(model_dir).name == side:
            raise FileNotFoundError(f"{model_dir}/config.json")
        return f"content:{Path(model_dir).name}"

    monkeypatch.setattr(runner, "content_fingerprint", fingerprint)
    _, (line,) = refused_start(stubbed, "environment check failed")
    assert line["inputs"][f"{side}_content_fingerprint"].startswith("error: FileNotFoundError")
    assert [p for p in line["problems"] if f"{side}_content_fingerprint" in p]
    assert stubbed.sessions == []


def test_manifest_line_records_the_checkpoint_report_computed_for_this_run(stubbed):
    (line,) = start(stubbed)
    assert line["checkpoint_report"] == CLEAN_CKPT
    assert stubbed.report_dirs == (model.MX_MODEL_DIR, model.NV_MODEL_DIR)
    assert line["problems"] == []


def test_every_restart_computes_a_fresh_checkpoint_report(stubbed):
    start(stubbed)
    stubbed.report = {**CLEAN_CKPT, "nv_expert_bytes": 107}
    reports = [m["checkpoint_report"] for m in start(stubbed)]
    assert [r["nv_expert_bytes"] for r in reports] == [106, 107]


@pytest.mark.parametrize(
    "side, problems, quoted",
    [
        (
            "nv",
            ["no packed FP4 (U8) expert weights", "config differs from MX: sliding_window"],
            "sliding_window",
        ),
        ("mx", ["ignore is ['lm_head', 'x'], expected ['lm_head']"], "expected ['lm_head']"),
    ],
)
def test_checkpoint_problems_are_added_to_the_problems_and_stop_the_run_before_any_session(
    stubbed, side, problems, quoted
):
    bad = {**CLEAN_CKPT, f"{side}_problems": problems}
    stubbed.report = bad
    message, (line,) = refused_start(stubbed, "environment check failed")
    assert quoted in message
    assert stubbed.sessions == [] and stubbed.counter_reads == 0
    assert line["checkpoint_report"] == bad and line["problems"] == problems
    assert stubbed.commits == [False]


def test_both_formats_problems_are_reported_together_mx_first(stubbed):
    stubbed.report = {
        **CLEAN_CKPT,
        "mx_problems": ["mx one", "mx two"],
        "nv_problems": ["nv one"],
    }
    _, lines = refused_start(stubbed, None)
    assert lines[0]["problems"] == ["mx one", "mx two", "nv one"]


def test_checkpoint_problems_are_listed_after_the_environment_problems(stubbed):
    stubbed.problems = [VLLM_MISMATCH]
    stubbed.report = {
        **CLEAN_CKPT,
        "mx_problems": ["no config_groups"],
        "nv_problems": ["unquantized expert tensors: ['x']"],
    }
    _, lines = refused_start(stubbed, None)
    assert lines[0]["problems"] == [
        VLLM_MISMATCH,
        "no config_groups",
        "unquantized expert tensors: ['x']",
    ]


def test_an_unreadable_checkpoint_report_is_recorded_and_stops_the_run(stubbed):
    stubbed.report_error = ValueError("NV checkpoint /data/nv has no config.json")
    _, (line,) = refused_start(stubbed, "environment check failed")
    assert (
        line["checkpoint_report"] == "error: ValueError: NV checkpoint /data/nv has no config.json"
    )
    assert line["problems"] == [
        "checkpoint_report: error: ValueError: NV checkpoint /data/nv has no config.json"
    ]
    assert stubbed.sessions == []


@pytest.mark.parametrize("missing", ["mx_problems", "nv_problems"])
def test_a_checkpoint_report_without_a_problems_list_is_not_accepted(stubbed, missing):
    stubbed.report = {k: v for k, v in CLEAN_CKPT.items() if k != missing}
    refused_start(stubbed, missing)
    assert stubbed.sessions == []


def test_a_checkpoint_report_with_neither_problems_list_names_both(stubbed):
    stubbed.report = {"nv_over_mx": 1.06}
    refused_start(stubbed, r"mx_problems.*nv_problems")
    assert stubbed.sessions == []


FULLISH = replace(STUDY, require_published=True)
FETCH = {"repo_id": "someone/name", "revision": "a" * 40, "utc": "2026-10-04T00:00:00+00:00"}
KIND_DIRS = {"mx": "MX_MODEL_DIR", "nv": "NV_MODEL_DIR"}


def write_fetch_record(record=FETCH, kinds=("mx", "nv")):
    for kind in kinds:
        d = Path(getattr(model, KIND_DIRS[kind]))
        d.mkdir(exist_ok=True)
        (d / "fp4bench_fetch.json").write_text(json.dumps(record))


def test_full_mode_refuses_to_start_without_fetched_checkpoints_and_names_both(stubbed):
    message, (line,) = refused_start(stubbed, "environment check failed", FULLISH)
    assert "nv_fetch" in message and "fp4bench fetch --kind nv" in message
    assert "mx_fetch" in message and "fp4bench fetch --kind mx" in message
    assert "fetch-nv" not in message
    assert stubbed.sessions == [] and stubbed.counter_reads == 0
    assert line["inputs"]["nv_fetch"] is None and line["inputs"]["mx_fetch"] is None
    assert len(line["problems"]) == 2 and "mx_fetch" in line["problems"][0]


@pytest.mark.parametrize("fetched, missing", [("nv", "mx"), ("mx", "nv")])
def test_full_mode_needs_both_checkpoints_fetched_not_just_one(stubbed, fetched, missing):
    write_fetch_record(kinds=(fetched,))
    message, (line,) = refused_start(stubbed, "environment check failed", FULLISH)
    assert f"{missing}_fetch" in message and f"{fetched}_fetch" not in message
    assert stubbed.sessions == [] and stubbed.counter_reads == 0
    assert len(line["problems"]) == 1


def test_full_mode_runs_when_both_checkpoints_were_fetched(stubbed):
    write_fetch_record()
    (line,) = start(stubbed, FULLISH)
    assert len(stubbed.sessions) == 4
    assert line["problems"] == []
    assert line["inputs"]["mx_fetch"] == FETCH and line["inputs"]["nv_fetch"] == FETCH


@pytest.mark.parametrize("kind", ["mx", "nv"])
@pytest.mark.parametrize("record", [{}, {"repo_id": "someone/name"}, {"revision": "a" * 40}, []])
def test_full_mode_needs_a_fetch_record_naming_repo_and_revision(stubbed, kind, record):
    write_fetch_record(kinds=("mx", "nv"))
    write_fetch_record(record, kinds=(kind,))
    refused_start(stubbed, f"{kind}_fetch", FULLISH)
    assert stubbed.sessions == []


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_a_fetch_record_that_cannot_be_read_is_one_problem_not_two(stubbed, kind):
    write_fetch_record()
    (Path(getattr(model, KIND_DIRS[kind])) / "fp4bench_fetch.json").write_text("{not json")
    _, (line,) = refused_start(stubbed, "environment check failed", FULLISH)
    assert len(line["problems"]) == 1
    assert line["problems"][0].startswith(f"input {kind}_fetch: error:")


def test_restart_cannot_switch_the_published_requirement(stubbed):
    start(stubbed)
    write_fetch_record()
    refused_start(stubbed, "require_published", FULLISH)


def change_m1_prompts(env):
    Path(settings.M1_PROMPTS_PATH).write_text(json.dumps([[[9, 9, 9]]]))


def change_nll_prompts(env):
    """prepare --force and a re-quantize: new NLL prompts and a reference computed on them."""
    Path(settings.NLL_PROMPTS_PATH).write_text(json.dumps([[9, 9, 9]]))
    env.nll_prompts_sha256 = sha(settings.NLL_PROMPTS_PATH)
    write_reference(env)


def change_fingerprint(kind, side):
    return lambda env: env.fingerprints.update({(kind, side): "changed"})


@pytest.mark.parametrize(
    "change, key",
    [
        (change_m1_prompts, "m1_prompts_sha256"),
        (change_nll_prompts, "nll_prompts_sha256"),
        (change_fingerprint("layout", "mx"), "mx_layout_fingerprint"),
        (change_fingerprint("layout", "nv"), "nv_layout_fingerprint"),
        (change_fingerprint("content", "mx"), "mx_content_fingerprint"),
        (change_fingerprint("content", "nv"), "nv_content_fingerprint"),
    ],
)
def test_restart_with_a_changed_input_is_refused_and_names_the_key(stubbed, change, key):
    start(stubbed)
    stubbed.sessions.clear()
    change(stubbed)
    message, lines = refused_start(stubbed, "inputs differ")
    assert f"['{key}']" in message
    assert stubbed.sessions == []
    assert len(lines) == 1


def test_every_differing_input_is_named_and_no_other(stubbed):
    start(stubbed)
    change_m1_prompts(stubbed)
    change_fingerprint("content", "nv")(stubbed)
    message, _ = refused_start(stubbed, None)
    assert "['m1_prompts_sha256', 'nv_content_fingerprint']" in message
    assert "nll_prompts_sha256" not in message and "mx_" not in message


def test_a_refetched_checkpoint_of_the_same_revision_is_the_same_input(stubbed):
    write_fetch_record()
    start(stubbed)
    write_fetch_record({**FETCH, "utc": "2026-10-05T00:00:00+00:00"})
    assert len(start(stubbed)) == 2


@pytest.mark.parametrize("kind", ["mx", "nv"])
@pytest.mark.parametrize("changed", [{"revision": "b" * 40}, {"repo_id": "other/name"}])
def test_a_different_published_checkpoint_is_refused(stubbed, kind, changed):
    write_fetch_record()
    start(stubbed)
    write_fetch_record({**FETCH, **changed}, kinds=(kind,))
    refused_start(stubbed, rf"\['{kind}_fetch'\]")


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_a_locally_built_checkpoint_cannot_replace_a_fetched_one_mid_run(stubbed, kind):
    write_fetch_record()
    start(stubbed)
    (Path(getattr(model, KIND_DIRS[kind])) / "fp4bench_fetch.json").unlink()
    refused_start(stubbed, f"{kind}_fetch")


def test_the_baseline_is_the_first_start_that_passed_the_environment_check(stubbed):
    stubbed.problems = [VLLM_MISMATCH]
    refused_start(stubbed, "environment check failed")
    stubbed.problems = []
    change_m1_prompts(stubbed)
    start(stubbed)
    assert len(stubbed.sessions) == 4
    Path(settings.M1_PROMPTS_PATH).write_text(json.dumps([[[7, 7, 7]]]))
    refused_start(stubbed, "m1_prompts_sha256")


def test_unreadable_inputs_on_a_restart_are_an_environment_failure_with_an_audit_line(stubbed):
    start(stubbed)
    Path(settings.NLL_PROMPTS_PATH).unlink()
    _, lines = refused_start(stubbed, "environment check failed")
    assert len(lines) == 2 and lines[1]["inputs"]["nll_prompts_sha256"].startswith("error:")


def test_the_identity_check_runs_before_the_checkpoint_report_and_any_session(stubbed):
    start(stubbed)
    stubbed.calls.clear()
    change_m1_prompts(stubbed)
    refused_start(stubbed, "inputs differ")
    assert "report" not in stubbed.calls and "counters" not in stubbed.calls


def test_restart_with_fewer_rounds_than_an_earlier_start_is_refused(stubbed):
    start(stubbed)
    start(stubbed, with_rounds(10))
    stubbed.sessions.clear()
    message, lines = refused_start(stubbed, "rounds", with_rounds(5))
    assert "rounds 5 is lower than the 10 registered by an earlier start" in message
    assert stubbed.sessions == []
    assert len(lines) == 2


def test_restart_with_fewer_rounds_than_the_first_start_is_refused(stubbed):
    start(stubbed)
    refused_start(stubbed, "rounds", with_rounds(1))


def test_a_start_that_failed_its_environment_check_still_counts_for_the_round_floor(stubbed):
    start(stubbed)
    stubbed.problems = [VLLM_MISMATCH]
    refused_start(stubbed, "environment check failed", with_rounds(10))
    stubbed.problems = []
    refused_start(stubbed, "rounds", with_rounds(5))


def test_lower_rounds_and_another_changed_field_are_both_reported(stubbed):
    start(stubbed)
    message, _ = refused_start(stubbed, None, replace(STUDY, rounds=1, m1_reps=3))
    assert "['m1_reps']" in message
    assert "rounds 1 is lower than the 2 registered by an earlier start" in message


def test_raising_rounds_again_after_an_extension_is_fine(stubbed):
    start(stubbed)
    start(stubbed, with_rounds(10))
    lines = start(stubbed, with_rounds(12))
    assert [m["protocol"]["rounds"] for m in lines] == [2, 10, 12]


def test_every_manifest_line_and_servers_row_of_a_start_carries_its_start_id(stubbed):
    (line,) = start(stubbed)
    start_id = line["start_id"]
    assert len(start_id) == 32 and int(start_id, 16) >= 0
    assert stubbed.session_starts == [start_id] * 4
    rows = load_jsonl(stubbed.run_dir / "servers.jsonl")
    assert {row["start_id"] for row in rows} == {start_id}


def test_each_start_gets_its_own_start_id(stubbed):
    start(stubbed)
    ids = [m["start_id"] for m in start(stubbed, with_rounds(3))]
    assert len(set(ids)) == 2
    starts = stubbed.session_starts
    assert starts[:4] == [ids[0]] * 4 and starts[4:] == [ids[1]] * 2


def test_an_interrupted_rerun_leaves_a_mixed_round_that_reruns_whole_on_the_next_start(stubbed):
    start(stubbed)
    stubbed.sessions.clear()
    stubbed.fail_session = 2
    refused_start(stubbed, "boom", error=ValueError, rerun_rounds=(0,))
    assert stubbed.sessions == [(0, "MX"), (0, "NV")]
    rows = load_rows(stubbed.run_dir / "servers.jsonl", ServerRow)
    latest = {(r.round, r.treatment): r.start_id for r in rows}
    assert latest[(0, Treatment.MX)] != latest[(0, Treatment.NV)]
    assert rounds_to_run(rows, STUDY) == [0]

    stubbed.fail_session = None
    stubbed.sessions.clear()
    start(stubbed)
    assert stubbed.sessions == [(0, "MX"), (0, "NV")]
    assert rounds_to_run(load_rows(stubbed.run_dir / "servers.jsonl", ServerRow), STUDY) == []


def test_a_completed_rerun_is_complete_and_is_not_rerun_again(stubbed):
    start(stubbed)
    start(stubbed, rerun_rounds=(1,))
    stubbed.sessions.clear()
    start(stubbed)
    assert stubbed.sessions == []


PIN = ("--linear-backend", "flashinfer_cutedsl")


@pytest.mark.parametrize(
    "treatment", ["MX", "NV", "NVa", "MXp", "NVnf", "NVx", "NVc", "NVt", "NVd", "NVv"]
)
def test_session_serves_the_local_copy_with_the_treatments_args_after_the_common_ones(
    env, treatment
):
    run_session(env, treatment=treatment)
    server = env.server
    assert server.model_dir == model.TREATMENTS[treatment].model_dir
    assert Path(server.model_dir).parent == env.local_root
    assert server.model_dir not in {spec.volume_dir for spec in model.TREATMENTS.values()}
    assert server.args == (*STUDY.server.args(), *model.TREATMENTS[treatment].server_args)
    assert "--linear-backend" in server.args


def test_servers_row_records_the_full_argv_of_the_server(env):
    run_session(env, treatment=Treatment.NV)
    run_session(env, treatment=Treatment.NVA)
    run_session(env, treatment=Treatment.NVNF)
    nv, nva, nvnf = load_jsonl(env.run_dir / "servers.jsonl")
    common = STUDY.server.args()
    assert nv["server_argv"] == [
        "vllm",
        "serve",
        model.TREATMENTS[Treatment.NV].model_dir,
        *common,
        *PIN,
    ]
    assert nva["server_argv"] == [
        "vllm",
        "serve",
        model.TREATMENTS[Treatment.NVA].model_dir,
        *common,
        "--linear-backend",
        "flashinfer_cudnn",
    ]
    assert nvnf["server_argv"] == [
        "vllm",
        "serve",
        model.TREATMENTS[Treatment.NV].model_dir,
        *common,
        *PIN,
        "--compilation-config",
        '{"pass_config": {"fuse_act_quant": false}}',
    ]


def test_the_common_args_come_from_the_sessions_protocol(env):
    expb_like = with_server(STUDY, kv_dtype=KvDtype.FP8, gpu_memory_utilization=0.95)
    run_session(env, Treatment.NV, expb_like)
    args = env.server.args
    assert args == (*expb_like.server.args(), *PIN)
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"
    run_session(env)
    args = env.server.args
    assert args[args.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.90"


def test_servers_row_records_the_linear_kernel_audit_facts_from_the_log(env):
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["linear_kernels"] == ["FlashInferCuteDslNvFp4LinearKernel"]
    assert row["linear_kernel_lines"] == [KERNEL_LINE]
    assert row["linear_backend_fallback_lines"] == []


BF16_GEMM_LINE = (
    "(EngineCore pid=4242) INFO 10-05 12:00:01 [layers/utils.py:626] "
    "Using FlashInfer cute-dsl for eligible unquantized BF16 GEMMs."
)


def test_servers_row_records_the_bf16_gemm_lines_from_the_log(env):
    env.server_log = SERVER_LOG + BF16_GEMM_LINE + "\n"
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["bf16_gemm_lines"] == [BF16_GEMM_LINE]


def test_servers_row_records_the_fusion_and_kv_cache_facts_from_the_log(env):
    from tests import vllm_logs as real

    env.server_log = real.NV_LOG
    run_session(env, treatment=Treatment.NV)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["fuse_act_quant"] is True and row["pass_config"]["fuse_act_quant"] is True
    assert row["pass_config_conflict"] is False
    assert row["custom_fusions"] == ["act_quant"] and len(row["custom_fusion_lines"]) == 6
    assert (row["kv_cache_tokens"], row["kv_cache_memory_gib"]) == (558_496, 136.35)
    assert len(row["attention_backend_lines"]) == 2


def test_servers_row_has_no_fusion_evidence_when_the_log_has_none(env):
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["pass_config"] is None and row["fuse_act_quant"] is None
    assert row["custom_fusions"] == []
    assert row["kv_cache_tokens"] == 558_496


def kv_log(tokens: int) -> str:
    return SERVER_LOG.replace("558,496 tokens", f"{tokens:,} tokens")


def test_a_session_without_a_kv_cache_size_in_its_log_stops_before_measuring(env):
    env.server_log = SERVER_LOG.replace(KV_LINE + "\n", "")
    with pytest.raises(RuntimeError, match="GPU KV cache size") as exc:
        run_session(env)
    assert str(env.server.log_path) in str(exc.value)
    assert env.preemption_reads == [] and env.nll_calls == []
    for name in ("m1.jsonl", "m2.jsonl", "servers.jsonl"):
        assert not (env.run_dir / name).exists()


def test_a_session_whose_kv_pool_cannot_hold_the_largest_m1_wave_stops_before_measuring(env):
    big = replace(STUDY, cells=cells_at(1024, (128, 512)), primary_batches=(512,))
    with pytest.raises(RuntimeError) as exc:
        run_session(env, Treatment.NV, big)
    message = str(exc.value)
    assert "558,496" in message and "1,114,112" in message and "C=512, P=1024" in message
    assert env.nll_calls == [] and not (env.run_dir / "m1.jsonl").exists()


@pytest.mark.parametrize("tokens, ok", [(17_408, True), (17_407, False)])
def test_the_kv_pool_must_hold_c_max_times_prompt_plus_n2_tokens(env, tokens, ok):
    assert STUDY.min_kv_tokens() == 8 * (1024 + 1152) == 17_408
    env.server_log = kv_log(tokens)
    if ok:
        run_session(env)
        (row,) = load_jsonl(env.run_dir / "servers.jsonl")
        assert row["kv_cache_tokens"] == tokens
    else:
        with pytest.raises(RuntimeError, match="17,407"):
            run_session(env)


def test_kv_capacity_problem_says_what_is_missing_or_short():
    assert runner.kv_capacity_problem(17_408, STUDY) is None
    missing = runner.kv_capacity_problem(None, STUDY)
    assert missing is not None and "GPU KV cache size" in missing
    short = runner.kv_capacity_problem(1_000_000, EXPB)
    assert short is not None
    assert "1,000,000" in short and "1,114,112" in short and "C=512, P=1024" in short
    assert "512 x (1024 + 1152)" in short


def test_a_short_kv_pool_on_a_main_treatment_aborts_the_run(env):
    env.server_log = SERVER_LOG.replace(KV_LINE + "\n", "")
    refused_start(env, "GPU KV cache size")
    (row,) = load_jsonl(env.run_dir / "errors.jsonl")
    assert "nonfatal" not in row and row["session_hint"] == {"round": 0, "treatment": "MX"}
    assert len(env.servers) == 1


def test_servers_row_has_no_bf16_gemm_lines_when_the_log_has_none(env):
    run_session(env, treatment=Treatment.NVC)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["bf16_gemm_lines"] == []


def test_servers_row_has_the_per_prompt_nlls_and_their_mean(env):
    env.nlls = [1.0, 2.0, 4.5]
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["nll_per_prompt"] == [1.0, 2.0, 4.5]
    assert row["nll"] == 2.5
    assert env.nll_calls == [(settings.BASE_URL, settings.SERVED_NAME, [[1]])]


def test_a_non_finite_prompt_nll_is_stored_as_null_not_as_a_token(env):
    env.nlls = [1.0, math.nan]
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["nll_per_prompt"] == [1.0, None] and row["nll"] is None
    assert "NaN" not in (env.run_dir / "servers.jsonl").read_text()


def tree_fingerprint(model_dir) -> str:
    """Stand-in for content_fingerprint over config.json and the shards."""
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir}/config.json")
    digest = hashlib.sha256()
    for f in sorted(model_dir.iterdir()):
        if f.name == "config.json" or f.suffix == ".safetensors":
            digest.update(f.name.encode() + f.read_bytes())
    return digest.hexdigest()


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()
    }


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "content_fingerprint", tree_fingerprint)
    volume = tmp_path / "vol" / "ckpt"
    (volume / ".cache" / "huggingface").mkdir(parents=True)
    (volume / "config.json").write_text('{"a": 1}')
    (volume / "model-00001.safetensors").write_bytes(b"shard-one" * 100)
    (volume / "README.md").write_text("card")
    (volume / ".cache" / "huggingface" / "meta").write_text("x")
    return types.SimpleNamespace(
        volume=volume, local=tmp_path / "local" / "ckpt", fp=tree_fingerprint(volume)
    )


def test_stage_copies_the_whole_directory_and_records_both_fingerprints(stage):
    record = runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert tree(stage.local) == tree(stage.volume)
    assert record["copied"] is True
    assert isinstance(record["copy_s"], float) and record["copy_s"] >= 0
    assert record["volume_content_fingerprint"] == stage.fp
    assert record["local_content_fingerprint"] == tree_fingerprint(stage.local) == stage.fp
    assert (record["volume_dir"], record["local_dir"]) == (str(stage.volume), str(stage.local))


def test_stage_leaves_only_the_final_directory_behind(stage):
    runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert [d.name for d in stage.local.parent.iterdir()] == ["ckpt"]


def test_stage_skips_the_copy_when_the_local_dir_already_matches(stage, monkeypatch):
    runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    (stage.local / "marker").write_text("untouched")

    def no_copy(*args, **kwargs):
        raise AssertionError("a matching local copy must not be copied again")

    monkeypatch.setattr(runner.shutil, "copytree", no_copy)
    record = runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert record["copied"] is False and record["copy_s"] == 0.0
    assert record["local_content_fingerprint"] == record["volume_content_fingerprint"] == stage.fp
    assert (stage.local / "marker").read_text() == "untouched"


def test_stage_replaces_a_local_dir_whose_fingerprint_differs(stage):
    stage.local.mkdir(parents=True)
    (stage.local / "config.json").write_text('{"a": 2}')
    (stage.local / "stale.safetensors").write_bytes(b"old")
    record = runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert record["copied"] is True
    assert tree(stage.local) == tree(stage.volume)


def test_stage_replaces_a_half_written_local_dir_that_has_no_fingerprint(stage):
    stage.local.mkdir(parents=True)
    (stage.local / "model-00001.safetensors").write_bytes(b"part")
    record = runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert record["copied"] is True and tree(stage.local) == tree(stage.volume)


def test_stage_removes_the_leftover_temp_dir_of_a_killed_copy(stage):
    leftover = stage.local.with_name(stage.local.name + runner.STAGING_SUFFIX)
    leftover.mkdir(parents=True)
    (leftover / "model-00001.safetensors").write_bytes(b"part")
    runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert tree(stage.local) == tree(stage.volume)
    assert not leftover.exists()


def test_a_copy_that_dies_midway_never_appears_under_the_final_name(stage, monkeypatch):
    real_copytree = shutil.copytree

    def dying_copytree(src, dst, *args, **kwargs):
        real_copytree(src, dst, *args, **kwargs)
        (Path(dst) / "model-00001.safetensors").write_bytes(b"cut short")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner.shutil, "copytree", dying_copytree)
    with pytest.raises(OSError, match="No space"):
        runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert not stage.local.exists()
    monkeypatch.setattr(runner.shutil, "copytree", real_copytree)
    record = runner.stage_checkpoint(stage.volume, stage.local, stage.fp)
    assert record["copied"] is True and tree(stage.local) == tree(stage.volume)
    assert [d.name for d in stage.local.parent.iterdir()] == ["ckpt"]


def test_stage_returns_a_mismatch_for_the_caller_to_judge(stage):
    wrong = "0" * 64
    record = runner.stage_checkpoint(stage.volume, stage.local, wrong)
    assert record["volume_content_fingerprint"] == wrong
    assert record["local_content_fingerprint"] == stage.fp != wrong


ALL_FULL = small_study(FULL.treatments, (1,))


def test_each_distinct_volume_dir_is_copied_once_whichever_treatments_share_it(
    stubbed, monkeypatch
):
    pin_nva(monkeypatch)
    start(stubbed, ALL_FULL)
    assert {"NVa", "NVnf"} <= set(ALL_FULL.treatments)
    assert sorted(src for src, _ in stubbed.copies) == ["mx", "nv"]
    for kind in ("mx", "nv"):
        assert tree(stubbed.local_root / kind) == tree(stubbed.volume[kind])


def test_the_manifest_records_both_fingerprints_and_the_copy_duration(stubbed):
    (line,) = start(stubbed)
    for kind in ("mx", "nv"):
        record = line["inputs"]["local_staging"][kind]
        assert record["volume_content_fingerprint"] == line["inputs"][f"{kind}_content_fingerprint"]
        assert record["volume_content_fingerprint"] == f"content:{kind}"
        assert record["local_content_fingerprint"] == f"content:{kind}"
        assert record["volume_dir"] == str(stubbed.volume[kind])
        assert record["local_dir"] == str(stubbed.local_root / kind)
        assert record["copied"] is True and record["copy_s"] >= 0
    assert line["problems"] == []


def test_sessions_serve_only_the_local_copies(env):
    start(env)
    served = {server.model_dir for server in env.servers}
    assert served == {str(env.local_root / "mx"), str(env.local_root / "nv")}
    assert all(Path(d).is_dir() for d in served)
    assert len(load_jsonl(env.run_dir / "servers.jsonl")) == 4
    assert {r["server_argv"][2] for r in load_jsonl(env.run_dir / "servers.jsonl")} == served


def test_a_local_copy_that_differs_from_the_volume_aborts_before_any_session(stubbed):
    stubbed.fingerprints[("local-content", "nv")] = "bitrot"
    message, (line,) = refused_start(stubbed, "environment check failed")
    assert "bitrot" in message and "content:nv" in message
    assert stubbed.sessions == [] and stubbed.counter_reads == 0
    assert len(line["problems"]) == 1 and "NV" in line["problems"][0]
    record = line["inputs"]["local_staging"]["nv"]
    assert (record["volume_content_fingerprint"], record["local_content_fingerprint"]) == (
        "content:nv",
        "bitrot",
    )
    assert stubbed.commits == [False]


def test_a_copy_that_fails_is_a_problem_with_an_audit_line_not_a_crash(stubbed, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner.shutil, "copytree", disk_full)
    message, (line,) = refused_start(stubbed, "environment check failed")
    assert "No space left on device" in message
    assert stubbed.sessions == [] and stubbed.counter_reads == 0
    assert len(line["problems"]) == 1
    assert "No space left" in line["inputs"]["local_staging"]["mx"]["error"]


def test_staging_waits_for_the_other_checks_and_precedes_the_gpu_preflight(stubbed):
    start(stubbed)
    calls = stubbed.calls
    assert calls.index("report") < calls.index("copy") < calls.index("counters")
    assert calls[:2] == ["collect", "verify"]


def test_nothing_is_copied_when_an_earlier_check_already_failed(stubbed):
    stubbed.problems = [VLLM_MISMATCH]
    _, (line,) = refused_start(stubbed, "environment check failed")
    assert stubbed.copies == [] and not stubbed.local_root.exists()
    assert "local_staging" not in line["inputs"]


def test_a_checkpoint_report_problem_also_skips_the_copy(stubbed):
    stubbed.report = {**CLEAN_CKPT, "mx_problems": ["no config_groups"]}
    refused_start(stubbed, None)
    assert stubbed.copies == []


def test_a_restart_in_the_same_container_skips_the_copies_and_is_not_refused(stubbed):
    start(stubbed)
    stubbed.copies.clear()
    first, second = start(stubbed)
    assert stubbed.copies == []
    assert first["inputs"]["local_staging"]["mx"]["copied"] is True
    assert second["inputs"]["local_staging"]["mx"]["copied"] is False
    assert second["inputs"]["local_staging"]["mx"]["local_content_fingerprint"] == "content:mx"
    assert second["problems"] == []


def test_a_restart_in_a_fresh_container_copies_again_and_is_not_refused(stubbed):
    start(stubbed)
    shutil.rmtree(stubbed.local_root)
    stubbed.copies.clear()
    start(stubbed)
    assert sorted(src for src, _ in stubbed.copies) == ["mx", "nv"]


SMOKE_LIKE = small_study((T.MX, T.NV, T.NVA, T.NVX), (1,))


def test_nvx_is_staged_and_fingerprinted_but_never_goes_through_checkpoint_report(
    stubbed, monkeypatch
):
    pin_nva(monkeypatch)
    (line,) = start(stubbed, SMOKE_LIKE)
    assert line["inputs"]["nvx_content_fingerprint"] == "content:nvx"
    assert line["inputs"]["local_staging"]["nvx"]["copied"] is True
    assert sorted(src for src, _ in stubbed.copies) == ["mx", "nv", "nvx"]
    assert stubbed.report_dirs == (model.MX_MODEL_DIR, model.NV_MODEL_DIR)
    assert stubbed.calls.count("report") == 1
    assert line["problems"] == [] and ("NVx" in [t for _, t in stubbed.sessions])
    assert set(line["checkpoint_report"]) == set(CLEAN_CKPT)


def test_the_real_smoke_protocol_runs_nvx_and_the_scan_on_one_nv_copy_without_structure_checks(
    stubbed,
):
    (line,) = start(stubbed, SMOKE)
    assert sorted(t for _, t in stubbed.sessions) == sorted(SMOKE.treatments)
    assert line["problems"] == [] and "nvx_content_fingerprint" in line["inputs"]
    assert sorted(src for src, _ in stubbed.copies) == ["mx", "nv", "nvx"]
    assert set(line["inputs"]["local_staging"]) == {"mx", "nv", "nvx"}
    assert stubbed.report_dirs == (model.MX_MODEL_DIR, model.NV_MODEL_DIR)
    assert set(line["checkpoint_report"]) == set(CLEAN_CKPT)


def test_protocols_without_nvx_do_not_touch_its_directory(stubbed, monkeypatch):
    seen = []
    real = runner.content_fingerprint
    monkeypatch.setattr(
        runner, "content_fingerprint", lambda d: (seen.append(Path(d).name), real(d))[1]
    )
    pin_nva(monkeypatch)
    start(stubbed, ALL_FULL)
    assert "nvx" not in seen


def test_a_missing_nvx_directory_aborts_the_smoke_run_before_any_session(stubbed, monkeypatch):
    pin_nva(monkeypatch)
    real = runner.content_fingerprint

    def fingerprint(model_dir):
        if Path(model_dir).name == "nvx":
            raise FileNotFoundError(f"{model_dir}/config.json")
        return real(model_dir)

    monkeypatch.setattr(runner, "content_fingerprint", fingerprint)
    _, (line,) = refused_start(stubbed, "environment check failed", SMOKE_LIKE)
    assert line["inputs"]["nvx_content_fingerprint"].startswith("error: FileNotFoundError")
    assert [p for p in line["problems"] if "nvx_content_fingerprint" in p]
    assert stubbed.sessions == [] and stubbed.copies == []


def test_a_nvx_checkpoint_that_changes_mid_run_is_refused(stubbed, monkeypatch):
    pin_nva(monkeypatch)
    start(stubbed, SMOKE_LIKE)
    change_fingerprint("content", "nvx")(stubbed)
    refused_start(stubbed, r"\['nvx_content_fingerprint'\]", SMOKE_LIKE)


def test_nvnf_shares_the_nv_copy_like_nva(stubbed):
    (line,) = start(stubbed, replace(STUDY, rounds=1, treatments=(T.MX, T.NV, T.NVNF)))
    assert sorted(src for src, _ in stubbed.copies) == ["mx", "nv"]
    assert line["problems"] == [] and sorted(t for _, t in stubbed.sessions) == ["MX", "NV", "NVnf"]
    assert line["inputs"]["treatment_server_args"]["NVnf"] == list(
        model.TREATMENTS[Treatment.NVNF].server_args
    )


def test_the_real_fusion_smoke_protocol_stages_mx_and_nv_only(stubbed):
    (line,) = start(stubbed, replace(SMOKE_NF, require_published=False))
    assert sorted(t for _, t in stubbed.sessions) == ["MX", "NV", "NVnf"]
    assert set(line["inputs"]["local_staging"]) == {"mx", "nv"} and line["problems"] == []


def set_nva_args(monkeypatch, args: tuple[str, ...]) -> None:
    """NVa's server args in the treatment table, and the kernel G1 then expects."""
    nva = replace(
        model.TREATMENTS[T.NVA],
        server_args=args,
        linear_kernel=model.expected_linear_kernel(T.NVA, args),
    )
    monkeypatch.setattr(model, "TREATMENTS", {**model.TREATMENTS, T.NVA: nva})


def pin_nva(monkeypatch, backend="flashinfer_cutlass"):
    """What setting NVa from the smoke scan does."""
    set_nva_args(monkeypatch, ("--linear-backend", backend))


def unpin_nva(monkeypatch):
    """The state before the smoke scan: no NVa pin, so vLLM's default (NV's CuTe-DSL kernel)."""
    set_nva_args(monkeypatch, ())


def test_a_protocol_with_nva_aborts_before_any_session_or_copy_while_nva_has_no_pin(
    stubbed, monkeypatch
):
    unpin_nva(monkeypatch)
    message, (line,) = refused_start(stubbed, "environment check failed", ALL_FULL)
    assert "NVa" in message and "--linear-backend" in message
    assert stubbed.sessions == [] and stubbed.counter_reads == 0
    assert [p for p in line["problems"] if "NVa" in p and "--linear-backend" in p]
    assert stubbed.copies == []


def test_a_protocol_with_nva_aborts_when_nva_is_pinned_to_the_kernel_of_nv(stubbed, monkeypatch):
    pin_nva(monkeypatch, "flashinfer_cutedsl")
    assert (
        model.TREATMENTS[Treatment.NVA].linear_kernel
        == model.TREATMENTS[Treatment.NV].linear_kernel
    )
    message, _ = refused_start(stubbed, "environment check failed", ALL_FULL)
    assert "NVa" in message and "same kernel" in message
    assert stubbed.sessions == []


@pytest.mark.parametrize(
    "backend", ["flashinfer_cutlass", "flashinfer_trtllm", "flashinfer_cudnn", "cutlass"]
)
def test_a_protocol_with_nva_runs_once_nva_is_pinned_to_another_kernel(
    stubbed, monkeypatch, backend
):
    pin_nva(monkeypatch, backend)
    (line,) = start(stubbed, ALL_FULL)
    assert line["problems"] == [] and "NVa" in [t for _, t in stubbed.sessions]


@pytest.mark.parametrize("study", [STUDY, SMOKE])
def test_protocols_without_nva_are_not_held_to_the_nva_rule(stubbed, study):
    assert "NVa" not in study.treatments
    (line,) = start(stubbed, study)
    assert line["problems"] == []


def test_the_manifest_line_records_the_server_args_of_the_protocols_treatments(
    stubbed, monkeypatch
):
    pin_nva(monkeypatch)
    (line,) = start(stubbed, ALL_FULL)
    recorded = line["inputs"]["treatment_server_args"]
    assert recorded == {t: list(model.TREATMENTS[t].server_args) for t in ALL_FULL.treatments}
    assert recorded["NVa"] == ["--linear-backend", "flashinfer_cutlass"]
    assert recorded["NVnf"] == [
        *PIN,
        "--compilation-config",
        '{"pass_config": {"fuse_act_quant": false}}',
    ]
    assert recorded["NV"] == list(PIN)


def test_a_restart_after_a_mid_run_pin_change_is_refused_and_names_the_key(stubbed, monkeypatch):
    pin_nva(monkeypatch, "flashinfer_cutlass")
    start(stubbed, ALL_FULL)
    stubbed.sessions.clear()
    pin_nva(monkeypatch, "flashinfer_trtllm")
    message, _ = refused_start(stubbed, "inputs differ", ALL_FULL)
    assert "['treatment_server_args']" in message and "flashinfer_trtllm" in message
    assert stubbed.sessions == []


def test_the_pin_change_diagnostic_names_the_treatments_as_the_stored_line_does(
    stubbed, monkeypatch
):
    pin_nva(monkeypatch, "flashinfer_cutlass")
    start(stubbed, ALL_FULL)
    pin_nva(monkeypatch, "flashinfer_trtllm")
    server_args = runner.collect_inputs(ALL_FULL)["treatment_server_args"]
    assert isinstance(server_args, dict) and all(type(t) is str for t in server_args)
    message, _ = refused_start(stubbed, "inputs differ", ALL_FULL)
    assert "Treatment" not in message
    for when, backend in (("first start", "flashinfer_cutlass"), ("now", "flashinfer_trtllm")):
        assert f"{when} {{'MX': {list(PIN)!r}, 'NV': " in message
        assert f"'NVa': ['--linear-backend', '{backend}']" in message


def test_a_restart_with_unchanged_pins_is_not_refused(stubbed, monkeypatch):
    pin_nva(monkeypatch)
    start(stubbed, ALL_FULL)
    assert len(start(stubbed, ALL_FULL)) == 2


def test_an_aborted_first_start_does_not_fix_the_pins_a_later_start_must_keep(stubbed, monkeypatch):
    unpin_nva(monkeypatch)
    refused_start(stubbed, None, ALL_FULL)
    pin_nva(monkeypatch)
    assert len(start(stubbed, ALL_FULL)) == 2


SCAN_STUDY = small_study(
    (T.MX, T.NV, T.NVC, T.NVT),
    (1, 8),
    kernel_scan=replace(NVA_SCAN, treatments=(T.NVC, T.NVT), selection_c=8, tiebreak_c=1),
)
BACKEND_OF = {"NVc": "flashinfer_cutlass", "NVt": "flashinfer_trtllm"}


def fail_when_serving(env, treatment, exc=None):
    """Make the server of `treatment` (told apart by its --linear-backend) fail to start."""
    backend = BACKEND_OF.get(treatment) or "flashinfer_cutedsl"

    def hook(server):
        if backend in server.args:
            raise exc or RuntimeError(f"vllm serve exited with 1; see {server.log_path}")

    env.server_hook = hook


def run_scan_proto(env):
    start(env, SCAN_STUDY)
    return {
        t: [r for r in load_jsonl(env.run_dir / "servers.jsonl") if r["treatment"] == t]
        for t in SCAN_STUDY.treatments
    }


def test_a_scan_treatment_whose_server_fails_to_start_does_not_end_the_run(env):
    fail_when_serving(env, "NVc")
    rows = run_scan_proto(env)
    assert [r["treatment"] for r in load_jsonl(env.run_dir / "servers.jsonl")] == [
        "MX",
        "NV",
        "NVc",
        "NVt",
    ]
    assert [len(rows[t]) for t in SCAN_STUDY.treatments] == [1, 1, 1, 1]
    assert "failed" not in rows[Treatment.NVT][0] and rows[Treatment.NVT][0]["nll"] == 2.5


def test_the_failure_is_in_errors_jsonl_in_the_usual_shape_plus_nonfatal(env):
    fail_when_serving(env, "NVc")
    run_scan_proto(env)
    (row,) = load_jsonl(env.run_dir / "errors.jsonl")
    assert set(row) == {"utc", "session_hint", "error", "traceback", "nonfatal"}
    assert row["nonfatal"] is True
    assert row["session_hint"] == {"round": 0, "treatment": "NVc"}
    assert row["error"].startswith("RuntimeError('vllm serve exited with 1")
    assert "Traceback" in row["traceback"] and "vllm serve exited" in row["traceback"]


def test_the_failure_is_a_servers_row_with_failed_and_the_error(env):
    fail_when_serving(env, "NVc")
    rows = run_scan_proto(env)
    (failed,) = rows[Treatment.NVC]
    assert failed["failed"] is True and failed["round"] == 0 and failed["treatment"] == "NVc"
    assert failed["error"] == load_jsonl(env.run_dir / "errors.jsonl")[0]["error"]
    assert failed["start_id"] == load_jsonl(env.run_dir / "manifests.jsonl")[0]["start_id"]
    assert all(
        "failed" not in r for t in (Treatment.MX, Treatment.NV, Treatment.NVT) for r in rows[t]
    )


def test_the_failed_row_names_the_session_whose_log_and_rows_exist(env):
    fail_when_serving(env, "NVc")
    rows = run_scan_proto(env)
    sid = rows[Treatment.NVC][0]["session_id"]
    assert sid.startswith("r0_NVc-") and (env.run_dir / "servers" / f"{sid}.log").exists()


def test_a_crash_after_some_cells_keeps_the_partial_rows_under_the_failed_session_id(env):
    def kill_during_c1(c):
        if c == 1 and env.server.args[-1] == "flashinfer_cutlass":
            env.server.proc.returncode = -9

    env.m2_hook = kill_during_c1
    rows = run_scan_proto(env)
    sid = rows[Treatment.NVC][0]["session_id"]
    assert rows[Treatment.NVC][0]["failed"] is True and "died" in rows[Treatment.NVC][0]["error"]
    assert [r["c"] for r in load_jsonl(env.run_dir / "m2.jsonl") if r["session_id"] == sid] == [1]
    assert rows[Treatment.NVT][0]["nll"] == 2.5


@pytest.mark.parametrize(
    "where", ["prompt_nlls", "decode_block", "run_lib_decode", "read_preemptions"]
)
def test_a_crash_in_warmup_graph_capture_or_a_sweep_is_nonfatal_for_a_scan_treatment(
    env, monkeypatch, where
):
    def boom(*args, **kwargs):
        if env.server.args[-1] == "flashinfer_cutlass":
            raise RuntimeError(f"CUDA error in {where}")
        return real[where](*args, **kwargs)

    real = {
        "prompt_nlls": runner.prompt_nlls,
        "decode_block": runner.decode_block,
        "run_lib_decode": runner.run_lib_decode,
        "read_preemptions": runner.read_preemptions,
    }
    monkeypatch.setattr(runner, where, boom)
    rows = run_scan_proto(env)
    assert rows[Treatment.NVC][0]["failed"] is True and where in rows[Treatment.NVC][0]["error"]
    assert "failed" not in rows[Treatment.NVT][0]
    assert load_jsonl(env.run_dir / "errors.jsonl")[0]["nonfatal"] is True


def test_the_failed_scan_treatment_still_completes_the_round_and_is_committed(env):
    fail_when_serving(env, "NVt")
    rows = run_scan_proto(env)
    assert rounds_to_run(load_rows(env.run_dir / "servers.jsonl", ServerRow), SCAN_STUDY) == []
    assert env.commits == [False, False, False, False, True, True]
    assert rows[Treatment.NVT][0]["failed"] is True


def test_every_scan_treatment_may_fail_and_the_run_still_ends_normally(env):
    env.server_hook = lambda server: (
        (_ for _ in ()).throw(RuntimeError("boom"))
        if "--linear-backend" in server.args
        and server.args[-1] in ("flashinfer_cutlass", "flashinfer_trtllm")
        else None
    )
    rows = run_scan_proto(env)
    assert [rows[t][0].get("failed") for t in SCAN_STUDY.treatments] == [None, None, True, True]
    assert len(load_jsonl(env.run_dir / "errors.jsonl")) == 2


@pytest.mark.parametrize("treatment", ["MX", "NV"])
def test_a_failure_of_any_other_treatment_still_fails_fast(env, treatment):
    fail_when_serving(env, treatment)
    refused_start(env, "vllm serve exited", SCAN_STUDY)
    (row,) = load_jsonl(env.run_dir / "errors.jsonl")
    assert "nonfatal" not in row
    assert not (env.run_dir / "servers.jsonl").exists() or all(
        "failed" not in r for r in load_jsonl(env.run_dir / "servers.jsonl")
    )


@pytest.mark.parametrize("treatment", ["NVx", "MXp", "NVa", "NVnf"])
def test_nvx_and_the_full_run_treatments_are_not_exempt(env, monkeypatch, treatment):
    pin_nva(monkeypatch)
    study = small_study((Treatment(treatment),), (1,))
    env.server_hook = lambda server: (_ for _ in ()).throw(RuntimeError("boom"))
    refused_start(env, "boom", study)
    assert "nonfatal" not in load_jsonl(env.run_dir / "errors.jsonl")[0]


def test_a_scan_kernel_outside_a_study_with_a_kernel_scan_fails_fast(env):
    fail_when_serving(env, "NVc")
    refused_start(env, "vllm serve exited", replace(SCAN_STUDY, kernel_scan=None))
    assert "nonfatal" not in load_jsonl(env.run_dir / "errors.jsonl")[0]


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1)])
def test_a_cancelled_container_is_never_swallowed_by_the_scan_exemption(env, exc):
    fail_when_serving(env, "NVc", exc)
    refused_start(env, None, SCAN_STUDY, type(exc))
    (row,) = load_jsonl(env.run_dir / "errors.jsonl")
    assert "nonfatal" not in row and row["session_hint"] == {"round": 0, "treatment": "NVc"}
    assert env.commits[-1] is True


def test_a_stubbed_session_that_raises_for_a_scan_treatment_is_nonfatal_too(stubbed):
    stubbed.fail_session = 3
    start(stubbed, SCAN_STUDY)
    assert stubbed.sessions == [(0, "MX"), (0, "NV"), (0, "NVc"), (0, "NVt")]
    rows = load_jsonl(stubbed.run_dir / "servers.jsonl")
    failed = [r for r in rows if r.get("failed")]
    assert [(r["round"], r["treatment"], r["error"]) for r in failed] == [
        (0, "NVc", repr(ValueError("boom")))
    ]
    assert failed[0]["session_id"].startswith("r0_NVc-")
    assert failed[0]["start_id"] == stubbed.session_starts[0]


def test_a_failed_scan_session_is_redone_only_by_an_explicit_rerun(env):
    fail_when_serving(env, "NVc")
    run_scan_proto(env)
    env.server_hook = None
    env.sessions.clear()
    run_scan_proto(env)
    assert len(load_jsonl(env.run_dir / "servers.jsonl")) == 4
    start(env, SCAN_STUDY, rerun_rounds=(0,))
    latest = {r["treatment"]: r for r in load_jsonl(env.run_dir / "servers.jsonl")}
    assert "failed" not in latest["NVc"]


AUTOTUNE_ENV = "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"


def autotune_log(cache_dir, loaded=False, ran=True, file_dir=None):
    """The relevant lines of a vLLM log, as vLLM v0.31.0 and FlashInfer 0.7.0.post1 write them."""
    cache_file = f"{file_dir or cache_dir}/0123abcd/autotune_configs.json"
    lines = [
        f"(EngineCore pid=1) INFO 10-05 12:00:02 [kernel_warmup.py:437] "
        f"Using FlashInfer autotune cache file: {cache_file}"
    ]
    if loaded:
        lines.append(
            f"2026-10-05 12:00:02,100 - INFO - autotuner.py:3692 - flashinfer.jit: "
            f"[Autotuner]: Loaded 12 configs from {cache_file}"
        )
    if ran:
        lines.append(
            "(EngineCore pid=1) INFO 10-05 12:00:03 [kernel_warmup.py:360] Running "
            "FlashInfer autotune with 16384 tokens and token buckets (1, 2, 4, 16384)."
        )
    return "\n".join(lines) + "\n"


def serve_like_vllm(env, log_for=None, writes=None):
    """A server_hook that logs `log_for(cache_dir)` and writes `writes` into the cache dir."""
    seen = []

    def hook(server):
        cache_dir = Path(server.env[AUTOTUNE_ENV])
        seen.append((server, cache_dir, sorted(p.name for p in cache_dir.iterdir())))
        for rel, data in (writes or {}).items():
            (cache_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (cache_dir / rel).write_bytes(data)
        if log_for:
            with server.log_path.open("a") as f:
                f.write(log_for(cache_dir))

    env.server_hook = hook
    return seen


def test_every_server_start_gets_its_own_autotune_cache_dir_under_the_run(env):
    seen = serve_like_vllm(env)
    run_session(env)
    run_session(env)
    run_session(env, treatment=Treatment.NV)
    dirs = [d for _, d, _ in seen]
    assert len(set(dirs)) == 3
    for server, d, content in seen:
        assert d.parent == env.run_dir / "autotune" and d.is_dir()
        assert content == []
        assert server.env == {AUTOTUNE_ENV: str(d)}


def test_the_cache_dir_is_named_after_the_session_whose_log_it_belongs_to(env):
    seen = serve_like_vllm(env)
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert seen[0][1] == env.run_dir / "autotune" / row["session_id"]
    assert row["autotune_cache_dir"] == str(env.run_dir / "autotune" / row["session_id"])
    assert row["server_env"] == {AUTOTUNE_ENV: row["autotune_cache_dir"]}


def test_a_stale_cache_in_the_vllm_cache_volume_is_neither_read_nor_deleted(
    env, tmp_path, monkeypatch
):
    vllm_cache = tmp_path / "vllm-cache"
    stale = (
        vllm_cache
        / "flashinfer_autotune_cache"
        / "0.7.0.post1"
        / "100a"
        / "abc"
        / "autotune_configs.json"
    )
    compiled = vllm_cache / "torch_compile_cache" / "deadbeef" / "rank_0_0" / "computation_graph.py"
    for f in (stale, compiled):
        f.parent.mkdir(parents=True)
        f.write_text("kept")
    monkeypatch.setattr(settings, "VLLM_CACHE", str(vllm_cache))
    seen = serve_like_vllm(env)
    run_session(env)
    assert stale.read_text() == compiled.read_text() == "kept"
    assert vllm_cache not in seen[0][1].parents


def test_servers_row_says_fresh_when_the_log_shows_a_tuning_pass_from_an_empty_cache(env):
    serve_like_vllm(env, log_for=autotune_log)
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["autotune_fresh"] is True
    assert row["autotune_ran"] is True and row["autotune_cache_loaded"] is False
    assert row["autotune_cache_files"] == [
        f"{row['autotune_cache_dir']}/0123abcd/autotune_configs.json"
    ]
    assert len(row["autotune_lines"]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loaded": True},
        {"ran": False},
        {"file_dir": "/root/.cache/vllm/flashinfer_autotune_cache/0.7.0.post1/100a"},
    ],
)
def test_servers_row_says_not_fresh_when_the_log_shows_a_cache_read_or_no_tuning(env, kwargs):
    serve_like_vllm(env, log_for=lambda d: autotune_log(d, **kwargs))
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["autotune_fresh"] is False


def test_servers_row_says_not_fresh_when_the_log_has_no_autotune_lines_at_all(env):
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert row["autotune_fresh"] is False and row["autotune_lines"] == []
    assert row["autotune_cache_files"] == [] and row["autotune_ran"] is False


def test_servers_row_keeps_the_file_each_start_wrote_with_its_checksum(env):
    serve_like_vllm(env, writes={"0123abcd/autotune_configs.json": b'{"tactics": 1}'})
    run_session(env)
    (row,) = load_jsonl(env.run_dir / "servers.jsonl")
    path = f"{row['autotune_cache_dir']}/0123abcd/autotune_configs.json"
    assert row["autotune_saved_files"] == {path: hashlib.sha256(b'{"tactics": 1}').hexdigest()}


def test_servers_row_has_no_saved_files_when_the_server_wrote_none(env):
    serve_like_vllm(env)
    run_session(env)
    assert load_jsonl(env.run_dir / "servers.jsonl")[0]["autotune_saved_files"] == {}


def test_a_failed_scan_session_keeps_its_cache_dir_for_audit_and_is_still_recorded_as_failed(env):
    seen = serve_like_vllm(env, writes={"0123abcd/autotune_configs.json": b"{}"})
    inner = env.server_hook

    def hook(server):
        inner(server)
        if "flashinfer_cutlass" in server.args:
            raise RuntimeError("vllm serve exited with 1")

    env.server_hook = hook
    run_scan_proto(env)
    failed = next(r for r in load_jsonl(env.run_dir / "servers.jsonl") if r.get("failed"))
    assert (env.run_dir / "autotune" / failed["session_id"]).is_dir()
    assert len(seen) == 4 and all(d.exists() for _, d, _ in seen)


def test_the_environment_for_the_server_is_the_only_thing_the_session_adds(env):
    run_session(env, treatment=Treatment.NVT)
    assert env.server.args == (*STUDY.server.args(), *model.TREATMENTS[Treatment.NVT].server_args)
    assert set(env.server.env) == {AUTOTUNE_ENV}


def counter_steps(*increments_by_read):
    """A preemptions_hook whose counter rises by increments_by_read[n - 1] just before read n."""
    total = [0.0]

    def hook(n):
        if n <= len(increments_by_read):
            total[0] += increments_by_read[n - 1]
        return total[0]

    return hook


# STUDY has C (1, 8). Reads: 1 preflight, then before/after each M1 block and each M2 cell:
# 1 | M1 c=1: 2, 3 | M1 c=8: 4, 5 | M2 c=1: 6, 7 | M2 c=8: 8, 9
READS_PER_SESSION = 1 + 2 * len(STUDY.batches) * 2


def test_every_m1_row_and_m2_cell_records_a_zero_delta_when_nothing_was_preempted(env):
    run_session(env)
    assert len(env.preemption_reads) == READS_PER_SESSION
    assert set(env.preemption_reads) == {settings.BASE_URL}
    m1, m2 = load_jsonl(env.run_dir / "m1.jsonl"), load_jsonl(env.run_dir / "m2.jsonl")
    assert len(m1) == 4 and all(r["preemptions_delta"] == 0 for r in m1)
    assert [(r["c"], r["preemptions_delta"], r["valid"]) for r in m2] == [
        (1, 0, True),
        (8, 0, True),
    ]
    assert all("preemptions" not in r.get("invalid_reasons", []) for r in m2)


def test_the_counter_is_read_as_a_preflight_before_the_nll_check(env):
    run_session(env)
    assert env.events[:2] == ["preemptions", "nll"]


def test_an_m1_block_with_preemptions_carries_the_delta_on_every_row_of_that_block(env):
    env.preemptions_hook = counter_steps(0, 0, 0, 0, 3)
    run_session(env)
    m1 = load_jsonl(env.run_dir / "m1.jsonl")
    assert [(r["c"], r["set"], r["preemptions_delta"]) for r in m1] == [
        (1, 0, 0),
        (1, 1, 0),
        (8, 0, 3),
        (8, 1, 3),
    ]
    assert (env.run_dir / "servers.jsonl").exists()


def test_preemptions_between_two_windows_are_charged_to_neither(env):
    env.preemptions_hook = counter_steps(0, 5, 0, 0, 0, 7)
    run_session(env)
    assert all(r["preemptions_delta"] == 0 for r in load_jsonl(env.run_dir / "m1.jsonl"))
    assert all(r["preemptions_delta"] == 0 for r in load_jsonl(env.run_dir / "m2.jsonl"))


def test_an_m2_cell_with_a_preemption_is_invalid_with_reason_preemptions(env):
    env.preemptions_hook = counter_steps(0, 0, 0, 0, 0, 0, 0, 0, 2)
    run_session(env)
    c1, c8 = load_jsonl(env.run_dir / "m2.jsonl")
    assert (c1["valid"], c1["preemptions_delta"]) == (True, 0)
    assert (c8["valid"], c8["preemptions_delta"]) == (False, 2)
    assert c8["invalid_reasons"] == ["preemptions"]
    assert c8["aggregate_tps"] == 800.0


def test_the_preemption_reason_is_added_to_the_cells_own_reasons(env, monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_lib_decode",
        lambda c, *a: m2_cell(valid=False, invalid_reasons=["underfilled"], aggregate_tps=1.0),
    )
    env.preemptions_hook = counter_steps(0, 0, 0, 0, 0, 0, 1)
    run_session(env)
    c1, c8 = load_jsonl(env.run_dir / "m2.jsonl")
    assert c1["invalid_reasons"] == ["underfilled", "preemptions"]
    assert c8["invalid_reasons"] == ["underfilled"]


def test_a_session_whose_counter_cannot_be_read_at_the_preflight_fails_loudly(env):
    def unreadable(n):
        raise ValueError("vllm:num_preemptions is not in /metrics")

    env.preemptions_hook = unreadable
    with pytest.raises(ValueError, match="vllm:num_preemptions"):
        run_session(env)
    assert env.nll_calls == []
    for name in ("m1.jsonl", "m2.jsonl", "servers.jsonl"):
        assert not (env.run_dir / name).exists()


def test_a_failed_preflight_fails_a_full_run_treatment_fast(env):
    env.preemptions_hook = lambda n: (_ for _ in ()).throw(OSError("connection refused"))
    refused_start(env, "connection refused", error=OSError)
    (row,) = load_jsonl(env.run_dir / "errors.jsonl")
    assert "nonfatal" not in row and row["session_hint"] == {"round": 0, "treatment": "MX"}
    assert not (env.run_dir / "servers.jsonl").exists()


def failing_at(*reads):
    """A preemptions_hook that fails the given 1-based attempts and returns 0.0 otherwise."""

    def hook(n):
        if n in reads:
            raise OSError("timed out")
        return 0.0

    return hook


def stays_unreadable(logical_read):
    """Fail every attempt of the `logical_read`-th read, an in-sweep one; earlier reads succeed."""
    attempts = runner.SWEEP_READ.attempts
    return failing_at(*range(logical_read, logical_read + attempts))


SWEEP_PAUSES = [2.0, 4.0, 8.0, 16.0]


def test_the_preflight_reads_3_times_2_s_apart_and_the_sweep_5_times_backing_off_from_2_s():
    assert RetryPolicy(attempts=3, backoff_s=2.0, backoff_factor=1.0) == metrics.READ
    assert RetryPolicy(attempts=5, backoff_s=2.0, backoff_factor=2.0) == runner.SWEEP_READ
    assert runner.SWEEP_READ.pauses == SWEEP_PAUSES


@pytest.mark.parametrize("logical_read", [2, 5, 6, 9])
def test_an_in_sweep_read_backs_off_2_4_8_16_s_and_recovers_at_its_fifth_attempt(env, logical_read):
    env.preemptions_hook = failing_at(*range(logical_read, logical_read + 4))
    run_session(env)
    assert env.preemption_sleeps == SWEEP_PAUSES
    assert len(env.preemption_reads) == READS_PER_SESSION + 4
    m1, m2 = load_jsonl(env.run_dir / "m1.jsonl"), load_jsonl(env.run_dir / "m2.jsonl")
    assert all(r["preemptions_delta"] == 0 for r in m1)
    assert all(r["preemptions_delta"] == 0 and r["valid"] for r in m2)
    assert (env.run_dir / "servers.jsonl").exists()


@pytest.mark.parametrize("failing_read", [1, 2, 5, 6, 9])
def test_one_failed_scrape_is_retried_and_the_session_loses_nothing(env, capsys, failing_read):
    env.preemptions_hook = failing_at(failing_read)
    run_session(env)
    m1, m2 = load_jsonl(env.run_dir / "m1.jsonl"), load_jsonl(env.run_dir / "m2.jsonl")
    assert all(r["preemptions_delta"] == 0 for r in m1)
    assert all(r["preemptions_delta"] == 0 and r["valid"] for r in m2)
    assert len(env.preemption_reads) == READS_PER_SESSION + 1
    assert env.preemption_sleeps == [2.0]
    assert (env.run_dir / "servers.jsonl").exists()
    assert "timed out" in capsys.readouterr().err


@pytest.mark.parametrize("logical_read, unread_c", [(6, 1), (9, 8)])
def test_an_m2_cell_whose_counter_stays_unreadable_is_invalid_and_the_session_goes_on(
    env, capsys, logical_read, unread_c
):
    env.preemptions_hook = stays_unreadable(logical_read)
    run_session(env)
    m1, m2 = load_jsonl(env.run_dir / "m1.jsonl"), load_jsonl(env.run_dir / "m2.jsonl")
    assert all(r["preemptions_delta"] == 0 for r in m1)
    for r in m2:
        unread = r["c"] == unread_c
        assert (r["preemptions_delta"] is None) is unread
        assert r["valid"] is not unread
        assert ("preemptions" in r.get("invalid_reasons", [])) is unread
    assert (env.run_dir / "servers.jsonl").exists()
    assert env.preemption_sleeps == SWEEP_PAUSES
    assert "5 attempts" in capsys.readouterr().err


def test_an_m1_block_whose_counter_stays_unreadable_before_it_stops_the_session_unmeasured(env):
    env.preemptions_hook = stays_unreadable(2)
    with pytest.raises(RuntimeError, match="C=1, P=1024") as exc:
        run_session(env)
    assert "G6a" in str(exc.value) and "before" in str(exc.value)
    assert not (env.run_dir / "m1.jsonl").exists()
    assert not (env.run_dir / "servers.jsonl").exists()


def test_an_m1_block_whose_counter_stays_unreadable_after_it_keeps_the_rows_and_stops(env):
    env.preemptions_hook = stays_unreadable(5)
    with pytest.raises(RuntimeError, match="C=8, P=1024") as exc:
        run_session(env)
    assert "after" in str(exc.value)
    m1 = load_jsonl(env.run_dir / "m1.jsonl")
    assert [(r["c"], r["preemptions_delta"]) for r in m1] == [(1, 0), (1, 0), (8, None), (8, None)]
    assert not (env.run_dir / "m2.jsonl").exists() and not (env.run_dir / "servers.jsonl").exists()


@pytest.mark.parametrize("logical_read, when", [(2, "before"), (5, "after")])
def test_an_unreadable_m1_counter_on_a_dead_server_is_reported_as_a_dead_server(
    env, logical_read, when
):
    env.preemptions_hook = stays_unreadable(logical_read)
    env.die_at_poll = 1
    with pytest.raises(RuntimeError, match="vllm server died") as exc:
        run_session(env)
    assert "exit code -9" in str(exc.value) and "G6a" not in str(exc.value)
    assert env.polls >= 1


def test_an_unreadable_m1_counter_on_a_live_server_still_stops_for_g6a(env):
    env.preemptions_hook = stays_unreadable(5)
    with pytest.raises(RuntimeError, match="G6a"):
        run_session(env)
    assert env.polls >= 1


def test_an_unreadable_m1_counter_aborts_a_main_run(env):
    env.preemptions_hook = stays_unreadable(2)
    refused_start(env, "preemption counter")
    (row,) = load_jsonl(env.run_dir / "errors.jsonl")
    assert "nonfatal" not in row


def test_a_preflight_that_fails_twice_then_reads_lets_the_session_run(env):
    env.preemptions_hook = failing_at(1, 2)
    run_session(env)
    assert (env.run_dir / "servers.jsonl").exists()
    assert len(env.preemption_reads) == READS_PER_SESSION + 2
    assert env.preemption_sleeps == [2.0, 2.0]


def test_the_preflight_is_retried_before_the_session_fails(env):
    env.preemptions_hook = failing_at(1, 2, 3)
    with pytest.raises(OSError, match="timed out") as exc:
        run_session(env)
    assert len(env.preemption_reads) == 3 and env.nll_calls == []
    assert env.preemption_sleeps == [2.0, 2.0]
    assert any("3 attempts" in note for note in exc.value.__notes__)


def test_the_preemption_reads_are_outside_the_telemetry_windows(env, monkeypatch):
    order = []
    monkeypatch.setattr(
        runner, "read_counters", lambda: order.append("counters") or dict.fromkeys(COUNTER_NAMES, 0)
    )
    env.preemptions_hook = lambda n: order.append("preemptions") or 0.0
    run_session(env)
    per_window = ["preemptions", "counters", "counters", "preemptions"]
    assert order == ["preemptions", *per_window * (2 * len(STUDY.batches))]


CSTUDY = Study(
    name="test-c",
    treatments=(T.MX, T.NV),
    cells=(Cell(1, 6), Cell(4, 5), Cell(1, 9)),
    server=ServerSettings(max_model_len=2048, hf_overrides='{"max_position_embeddings": 2048}'),
    rounds=2,
    m1_reps=1,
    m2_duration_s=0,
    require_published=False,
    primary_batches=(1,),
    prompts=Prompts.CELLS,
)
CELL_IDS = {(1, 6): 10, (4, 5): 20, (1, 9): 30, (2, 7): 40}


def cell_sets(cell, n_sets=6):
    c, p = cell
    return [[[CELL_IDS[cell] + s] * p for _ in range(c)] for s in range(n_sets)]


def write_cell_prompts(env, cells=tuple(CELL_IDS), **changes):
    """What prep.prepare_expc writes: built from the volume's M1 prompts."""
    record = {
        "cells": {f"{c}x{p}": cell_sets((c, p)) for c, p in cells},
        "seed": 0,
        "n_sets": 6,
        "m1_prompts_sha256": sha(settings.M1_PROMPTS_PATH),
        **changes,
    }
    Path(settings.C_PROMPTS_PATH).write_text(json.dumps(record))


@pytest.fixture
def cells(env):
    write_cell_prompts(env)
    return env


def run_cell_session(env):
    run_session(env, Treatment.NV, CSTUDY, runner.load_cell_prompts(CSTUDY))


def test_a_cell_session_runs_one_m1_block_per_cell_in_the_protocols_order(cells):
    run_cell_session(cells)
    assert [(c, sets, n1, n2, reps) for c, sets, n1, n2, reps in cells.blocks] == [
        (c, cell_sets((c, p)), settings.M1_N1, settings.M1_N2, CSTUDY.m1_reps)
        for c, p in CSTUDY.cells
    ]
    m1 = load_jsonl(cells.run_dir / "m1.jsonl")
    assert [(r["c"], r["prompt_len"], r["set"]) for r in m1] == [
        (1, 6, 0),
        (1, 6, 1),
        (4, 5, 0),
        (4, 5, 1),
        (1, 9, 0),
        (1, 9, 1),
    ]
    for row in m1:
        assert (row["round"], row["treatment"]) == (0, "NV")
        assert set(row) >= {
            "session_id",
            "warmup",
            "step_s",
            "block_telemetry",
            "preemptions_delta",
        }
    (server_row,) = load_jsonl(cells.run_dir / "servers.jsonl")
    assert {r["session_id"] for r in m1} == {server_row["session_id"]}


def test_a_cell_session_with_m2_duration_0_runs_no_m2_at_all(cells):
    run_cell_session(cells)
    assert cells.m2_calls == [] and not (cells.run_dir / "m2.jsonl").exists()
    assert not (cells.run_dir / "lib_logs").exists()


def test_m2_duration_0_skips_m2_for_a_protocol_without_cells_too(env):
    run_session(env, study=replace(STUDY, m2_duration_s=0))
    assert [r["c"] for r in load_jsonl(env.run_dir / "m1.jsonl")] == [1, 1, 8, 8]
    assert env.m2_calls == [] and not (env.run_dir / "m2.jsonl").exists()
    assert len(env.preemption_reads) == 1 + 2 * len(STUDY.batches)
    assert (env.run_dir / "servers.jsonl").exists()


def test_a_protocol_without_cells_runs_a_1024_token_cell_per_c_on_the_m1_prompts(env):
    assert STUDY.cell_list() == ((1, settings.M1_INPUT_LEN), (8, settings.M1_INPUT_LEN))
    run_session(env)
    assert [b[:2] for b in env.blocks] == [(1, [[[1]]]), (8, [[[1]]])]
    m1 = load_jsonl(env.run_dir / "m1.jsonl")
    assert [(r["c"], r["prompt_len"]) for r in m1] == [(1, 1024), (1, 1024), (8, 1024), (8, 1024)]


def test_a_cell_other_than_1024_tokens_needs_the_cell_prompt_file(env):
    study = replace(STUDY, cells=(Cell(1, 1024), Cell(8, 512)), prompts=Prompts.CELLS)
    with pytest.raises(ValueError, match=r"\['8x512'\] have no prompt sets"):
        run_session(env, Treatment.NV, study)
    assert env.servers == []


def test_a_cell_session_serves_with_the_protocols_window_and_override(cells):
    run_cell_session(cells)
    args = cells.server.args
    assert args == (*CSTUDY.server.args(), *model.TREATMENTS[Treatment.NV].server_args)
    assert args[args.index("--max-model-len") + 1] == "2048"
    assert args[args.index("--hf-overrides") + 1] == '{"max_position_embeddings": 2048}'
    (row,) = load_jsonl(cells.run_dir / "servers.jsonl")
    assert row["server_argv"][3:] == list(args)


# 1 preflight, then before/after each cell's block: 2, 3 | 4, 5 | 6, 7
def test_a_cell_block_with_preemptions_carries_the_delta_on_its_rows_only(cells):
    cells.preemptions_hook = counter_steps(0, 0, 0, 0, 2)
    run_cell_session(cells)
    m1 = load_jsonl(cells.run_dir / "m1.jsonl")
    assert len(cells.preemption_reads) == 1 + 2 * len(CSTUDY.cells)
    assert [(r["prompt_len"], r["preemptions_delta"]) for r in m1] == [
        (6, 0),
        (6, 0),
        (5, 2),
        (5, 2),
        (9, 0),
        (9, 0),
    ]


@pytest.mark.parametrize(
    "logical_read, when, cell", [(4, "before", "C=4, P=5"), (7, "after", "C=1, P=9")]
)
def test_a_cell_block_whose_counter_stays_unreadable_stops_the_session_naming_the_cell(
    cells, logical_read, when, cell
):
    cells.preemptions_hook = stays_unreadable(logical_read)
    with pytest.raises(RuntimeError, match="G6a") as exc:
        run_cell_session(cells)
    assert when in str(exc.value) and cell in str(exc.value)
    assert not (cells.run_dir / "servers.jsonl").exists()


def test_a_cell_session_checks_the_server_after_its_last_block(cells):
    cells.die_at_poll = 1
    with pytest.raises(RuntimeError, match="died"):
        run_cell_session(cells)
    assert len(load_jsonl(cells.run_dir / "m1.jsonl")) == 6
    assert not (cells.run_dir / "servers.jsonl").exists()


def test_the_kv_pool_must_hold_the_largest_cells_wave(cells):
    assert CSTUDY.min_kv_tokens() == 4 * (5 + 1152) == 4_628
    cells.server_log = kv_log(4_627)
    with pytest.raises(RuntimeError) as exc:
        run_cell_session(cells)
    assert "4,627" in str(exc.value) and "4,628" in str(exc.value)
    assert "C=4, P=5" in str(exc.value) and "4 x (5 + 1152)" in str(exc.value)
    assert cells.nll_calls == [] and cells.blocks == []
    cells.server_log = kv_log(4_628)
    run_cell_session(cells)
    assert len(load_jsonl(cells.run_dir / "servers.jsonl")) == 1


def test_the_kv_capacity_problem_of_experiment_c_names_its_largest_cell():
    short = runner.kv_capacity_problem(150_000, EXPC)
    assert short is not None
    assert "150,000" in short and "193,536" in short and "C=128, P=360" in short
    assert runner.kv_capacity_problem(193_536, EXPC) is None
    missing = runner.kv_capacity_problem(None, EXPC)
    assert missing is not None and "GPU KV cache size" in missing and "C=128, P=360" in missing


@pytest.mark.parametrize("prompts", [None, {Cell(1, 6): cell_sets((1, 6))}])
def test_a_cell_session_without_prompts_for_every_cell_stops_before_serving(env, prompts):
    with pytest.raises(ValueError, match="4x5"):
        run_session(env, Treatment.NV, CSTUDY, prompts)
    assert env.servers == []


def test_load_cell_prompts_returns_the_protocols_cells_from_the_file(cells):
    assert runner.load_cell_prompts(CSTUDY) == {cell: cell_sets(cell) for cell in CSTUDY.cells}


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda env: Path(settings.C_PROMPTS_PATH).unlink(), "No such file"),
        (lambda env: Path(settings.C_PROMPTS_PATH).write_text('{"cells": '), "Expecting"),
        (lambda env: Path(settings.C_PROMPTS_PATH).write_text("[]"), "cells"),
        (lambda env: write_cell_prompts(env, cells=((1, 6), (1, 9))), "4x5.*missing"),
        (lambda env: write_cell_prompts(env, m1_prompts_sha256="0" * 64), "m1_prompts_sha256"),
        (
            lambda env: (
                write_cell_prompts(env, cells=())
                or Path(settings.C_PROMPTS_PATH).write_text(
                    json.dumps(
                        {
                            "cells": {
                                "1x6": cell_sets((1, 6), n_sets=1),
                                "4x5": cell_sets((4, 5)),
                                "1x9": cell_sets((1, 9)),
                            },
                            "m1_prompts_sha256": sha(settings.M1_PROMPTS_PATH),
                        }
                    )
                )
            ),
            "1x6.*2 sets",
        ),
    ],
)
def test_load_cell_prompts_refuses_a_file_that_cannot_serve_the_protocol(cells, change, match):
    change(cells)
    with pytest.raises((ValueError, OSError), match=match):
        runner.load_cell_prompts(CSTUDY)


@pytest.mark.usefixtures("stubbed")
def test_run_experiment_loads_the_cell_prompts_once_and_gives_every_session_its_cells(
    cells, monkeypatch
):
    loads = []
    real = runner.load_cell_prompts
    monkeypatch.setattr(
        runner, "load_cell_prompts", lambda study: loads.append(study) or real(study)
    )
    start(cells, CSTUDY)
    assert cells.sessions == [(0, "MX"), (0, "NV"), (1, "NV"), (1, "MX")]
    assert loads == [CSTUDY, CSTUDY]
    expected = {cell: cell_sets(cell) for cell in CSTUDY.cells}
    assert cells.cell_prompts == [expected] * 4


def test_a_run_without_cells_gives_its_sessions_no_cell_prompts(stubbed):
    start(stubbed)
    assert stubbed.cell_prompts == [None] * 4


@pytest.mark.usefixtures("stubbed")
def test_the_manifest_of_a_cell_run_records_the_cell_prompt_files_sha256(cells):
    (line,) = start(cells, CSTUDY)
    assert line["inputs"]["c_prompts_sha256"] == sha(settings.C_PROMPTS_PATH)
    assert line["inputs"]["m1_prompts_sha256"] == sha(settings.M1_PROMPTS_PATH)
    assert line["protocol"]["cells"] == [[1, 6], [4, 5], [1, 9]]
    assert (line["protocol"]["max_model_len"], line["protocol"]["m2_duration_s"]) == (2048, 0)
    assert line["protocol"]["hf_overrides"] == '{"max_position_embeddings": 2048}'
    assert line["problems"] == []


def test_a_run_without_cells_does_not_look_at_the_cell_prompt_file(stubbed):
    (line,) = start(stubbed)
    assert "c_prompts_sha256" not in line["inputs"] and line["problems"] == []


@pytest.mark.parametrize(
    "change",
    [
        lambda env: Path(settings.C_PROMPTS_PATH).unlink(),
        lambda env: write_cell_prompts(env, cells=((1, 6), (1, 9))),
        lambda env: write_cell_prompts(env, m1_prompts_sha256="0" * 64),
    ],
)
@pytest.mark.usefixtures("stubbed")
def test_a_cell_prompt_file_that_cannot_serve_the_run_is_an_input_error_before_staging(
    cells, change
):
    change(cells)
    _, (line,) = refused_start(cells, "environment check failed", CSTUDY)
    assert line["inputs"]["c_prompts_sha256"].startswith("error: ")
    assert [p for p in line["problems"] if p.startswith("input c_prompts_sha256: error: ")]
    assert cells.sessions == [] and cells.copies == []


@pytest.mark.usefixtures("stubbed")
def test_a_restart_with_another_cell_prompt_file_is_refused(cells):
    start(cells, CSTUDY)
    cells.sessions.clear()
    record = json.loads(Path(settings.C_PROMPTS_PATH).read_text())
    record["seed"] = 1
    Path(settings.C_PROMPTS_PATH).write_text(json.dumps(record))
    refused_start(cells, r"inputs differ.*\['c_prompts_sha256'\]", CSTUDY)
    assert cells.sessions == []


def test_a_cell_run_runs_its_sessions_for_real_end_to_end(cells):
    start(cells, CSTUDY)
    servers = load_jsonl(cells.run_dir / "servers.jsonl")
    assert [(r["round"], r["treatment"]) for r in servers] == [
        (0, "MX"),
        (0, "NV"),
        (1, "NV"),
        (1, "MX"),
    ]
    m1 = load_jsonl(cells.run_dir / "m1.jsonl")
    assert len(m1) == 4 * len(CSTUDY.cells) * 2 and all("prompt_len" in r for r in m1)
    assert not (cells.run_dir / "m2.jsonl").exists()


LEGACY = small_study((T.MX, T.NV), (1, 8, 4), m1_reps=2)
AS_CELLS = replace(LEGACY, prompts=Prompts.CELLS)
M1_SETS = [[[100 * s + i] * 3 for i in range(8)] for s in range(LEGACY.m1_reps + 1)]


def session_trace(env, monkeypatch, study, **kwargs) -> tuple[list, dict]:
    """Everything one session does, in order, and its rows, without what differs by design."""
    from fp4bench import decode_step

    events = env.events = []

    def wave(base_url, model, prompts, max_tokens):
        events.append(["wave", prompts, max_tokens])
        return 1.0 + max_tokens / 1000 + len(events) / 1e6

    monkeypatch.setattr(runner, "decode_block", decode_step.decode_block)
    monkeypatch.setattr(runner, "run_wave", wave)
    monkeypatch.setattr(
        runner,
        "read_counters",
        lambda: events.append("counters") or dict.fromkeys(COUNTER_NAMES, 0),
    )
    env.m2_hook = lambda c: events.append(["m2", c])
    shutil.rmtree(env.run_dir, ignore_errors=True)
    runner.run_server_session(
        env.run_dir, 0, Treatment.NV, study, M1_SETS, [[1]], "start-1", **kwargs
    )
    rows = {}
    for name in ("m1", "m2", "servers"):
        rows[name] = [
            {
                k: v
                for k, v in row.items()
                if k not in ("session_id", "autotune_cache_dir", "server_env")
            }
            for row in load_jsonl(env.run_dir / f"{name}.jsonl")
        ]
        for row in rows[name]:
            for key in ("block_telemetry", "telemetry"):
                if key in row:
                    row[key] = {k: v for k, v in row[key].items() if k != "window_s"}
    return events, rows


def test_a_protocol_without_cells_runs_exactly_as_its_1024_token_cells(env, monkeypatch):
    assert LEGACY.cell_list() == AS_CELLS.cell_list() == ((1, 1024), (8, 1024), (4, 1024))
    assert LEGACY.min_kv_tokens() == AS_CELLS.min_kv_tokens()
    legacy = session_trace(env, monkeypatch, LEGACY)
    reused = {cell: [prompts[: cell.batch] for prompts in M1_SETS] for cell in AS_CELLS.cell_list()}
    as_cells = session_trace(env, monkeypatch, AS_CELLS, cell_prompts=reused)
    assert as_cells == legacy
    events, rows = legacy
    waves = [e for e in events if e[0] == "wave"]
    assert [(len(prompts), n) for _, prompts, n in waves] == [
        (c, n)
        for c in (1, 8, 4)
        for s in range(LEGACY.m1_reps + 1)
        for n in (
            (settings.M1_N1, settings.M1_N2) if s % 2 == 0 else (settings.M1_N2, settings.M1_N1)
        )
    ]
    assert [prompts for _, prompts, _ in waves[:2]] == [M1_SETS[0][:1]] * 2
    assert [e for e in events if e[0] == "m2"] == [["m2", 1], ["m2", 8], ["m2", 4]]
    assert [(r["c"], r["set"], r["prompt_len"]) for r in rows["m1"]] == [
        (c, s, 1024) for c in (1, 8, 4) for s in range(LEGACY.m1_reps + 1)
    ]
    assert len(rows["servers"]) == 1


def test_m2_batches_are_the_concurrencies_unless_m2_is_off():
    assert LEGACY.m2_batches() == AS_CELLS.m2_batches() == (1, 8, 4)
    assert replace(LEGACY, m2_duration_s=0).m2_batches() == ()
    assert CSTUDY.m2_batches() == ()


@pytest.mark.parametrize(
    "run, mode",
    [
        ("full-1", "full"),
        ("expb-1", "expb"),
        ("expc-1", "expc"),
        ("smoke-c-1", "smoke-c"),
        ("smoke-nf-2", "smoke-nf"),
    ],
)
def test_every_committed_run_would_restart_under_its_protocol(tmp_path, run, mode):
    shutil.copy(RUNS_DIR / run / "manifests.jsonl", tmp_path / "manifests.jsonl")
    rounds = (load_rows(tmp_path / "manifests.jsonl", ManifestLine)[-1].protocol or {})["rounds"]
    study = replace(STUDIES[mode], rounds=rounds)
    runner.check_protocol_unchanged(tmp_path, study)
    with pytest.raises(RuntimeError, match="m1_reps"):
        runner.check_protocol_unchanged(tmp_path, replace(study, m1_reps=4))


def drop_from_first_manifest_line(env, *fields):
    path = env.run_dir / "manifests.jsonl"
    lines = load_jsonl(path)
    for field in fields:
        lines[0]["protocol"].pop(field)
    write_jsonl(path, lines)


def test_a_run_started_before_the_cell_fields_existed_restarts_with_their_defaults(stubbed):
    start(stubbed)
    drop_from_first_manifest_line(stubbed, "cells", "max_model_len", "hf_overrides")
    assert len(start(stubbed, with_rounds(3))) == 2


@pytest.mark.parametrize("key", ["cells", "max_model_len", "hf_overrides"])
def test_but_not_with_other_values_of_them(stubbed, key):
    start(stubbed)
    drop_from_first_manifest_line(stubbed, "cells", "max_model_len", "hf_overrides")
    refused_start(stubbed, key, CHANGES[key](STUDY))


def test_only_the_cell_fields_may_be_missing_from_a_first_start(stubbed):
    start(stubbed)
    drop_from_first_manifest_line(stubbed, "kv_cache_dtype")
    refused_start(stubbed, "kv_cache_dtype")


M1_KEYS = [
    "round",
    "treatment",
    "session_id",
    "c",
    "set",
    "warmup",
    "first",
    "n1",
    "n2",
    "t1_s",
    "t2_s",
    "step_s",
    "decode_tok_s",
    "prompt_len",
    "block_telemetry",
    "preemptions_delta",
]
M2_KEYS = [
    "round",
    "treatment",
    "session_id",
    "c",
    "valid",
    "invalid_reasons",
    "failure_reason",
    *lib_bench.METRIC_FIELDS,
    "preemptions_delta",
    "telemetry",
]


def test_a_session_writes_schema_2_rows_with_todays_keys_in_todays_order(env):
    run_session(env)
    m1, m2 = load_jsonl(env.run_dir / "m1.jsonl"), load_jsonl(env.run_dir / "m2.jsonl")
    (server,) = load_jsonl(env.run_dir / "servers.jsonl")
    assert {row["schema_version"] for row in m1 + m2 + [server]} == {2}
    assert all(list(row) == ["schema_version", *M1_KEYS] for row in m1)
    assert all(list(row) == ["schema_version", *M2_KEYS] for row in m2)
    assert list(server)[:9] == [
        "schema_version",
        "round",
        "treatment",
        "session_id",
        "start_id",
        "gpu_uuid",
        "server_argv",
        "nll",
        "nll_per_prompt",
    ]
    assert list(server)[-4:] == [
        "server_env",
        "autotune_cache_dir",
        "autotune_fresh",
        "autotune_saved_files",
    ]
    assert ServerRow.from_json(server).extra == {}
    rows = load_rows(env.run_dir / "m1.jsonl", M1Row)
    assert rows[0].treatment is Treatment.MX and rows[0].prompt_len == settings.M1_INPUT_LEN
    assert rows[0].block_telemetry.env_throttle_us == 0
    assert [r.c for r in load_rows(env.run_dir / "m2.jsonl", M2Row)] == [1, 8]


def test_a_cell_session_writes_the_same_keys(cells):
    run_cell_session(cells)
    assert all(
        list(row) == ["schema_version", *M1_KEYS] for row in load_jsonl(cells.run_dir / "m1.jsonl")
    )


def test_the_manifest_line_and_a_failed_scan_row_are_schema_2(env):
    fail_when_serving(env, "NVc")
    run_scan_proto(env)
    (line,) = load_jsonl(env.run_dir / "manifests.jsonl")
    assert list(line) == [
        "schema_version",
        "utc",
        "gpu",
        "study",
        "start_id",
        "inputs",
        "checkpoint_report",
        "protocol",
        "problems",
    ]
    failed = [r for r in load_jsonl(env.run_dir / "servers.jsonl") if r.get("failed")]
    assert [list(r) for r in failed] == [
        ["schema_version", "round", "treatment", "session_id", "start_id", "failed", "error"]
    ]
    (manifest,) = load_rows(env.run_dir / "manifests.jsonl", ManifestLine)
    assert manifest.protocol is not None and manifest.protocol["treatments"] == list(
        SCAN_STUDY.treatments
    )
    assert manifest.study == SCAN_STUDY.name
