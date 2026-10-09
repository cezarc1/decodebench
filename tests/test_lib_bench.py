import json
import subprocess
import sys
from itertools import pairwise

import pytest

from fp4bench import lib_bench
from fp4bench.lib_bench import lib_command, parse_lib_result, run_lib_decode


def _has_pair(cmd: list[str], flag: str, value: str) -> bool:
    return any(a == flag and b == value for a, b in pairwise(cmd))


def test_lib_command_pins_sustained_decode_flags():
    cmd = lib_command(32, duration_s=30, output_path="/r/x.json")
    assert cmd[0] == sys.executable
    assert cmd[1] == "/opt/llm-inference-bench/llm_decode_bench.py"
    for flag, value in (
        ("--host", "127.0.0.1"),
        ("--port", "8000"),
        ("--model", "fp4bench"),
        ("--contexts", "0"),
        ("--concurrency", "32"),
        ("--max-tokens", "2048"),
        ("--duration", "30"),
        ("--display-mode", "plain"),
        ("--output", "/r/x.json"),
    ):
        assert _has_pair(cmd, flag, value), (flag, value)
    for switch in ("--skip-prefill", "--no-hw-monitor", "--no-resume"):
        assert cmd.count(switch) == 1
    assert "--temperature" not in cmd


def test_lib_command_concurrency_is_matched_exactly():
    cmd = lib_command(3, duration_s=30, output_path="/r/x.json")
    assert _has_pair(cmd, "--concurrency", "3")
    assert not _has_pair(cmd, "--concurrency", "32")


def _cell(**overrides) -> dict:
    cell = {
        "concurrency": 32,
        "context_tokens": 0,
        "aggregate_tps": 9000.0,
        "aggregate_source": "openai_continuous_usage",
        "measurement_seconds": 30.0,
        "effective_concurrency": 31.8,
        "avg_running_reqs": 31.8,
        "inter_token_latency_p50": 0.0035,
        "output_tps_per_user_p50": 285.0,
        "ttft_p50": 0.05,
        "server_gen_throughput": 8990.0,
        "underfilled": False,
        "capacity_limited": False,
        "loop_detected": False,
        "warmup_timed_out": False,
        "failure_reason": "",
        "num_errors": 0,
    }
    return cell | overrides


def test_parse_valid_cell():
    out = parse_lib_result({"results": [_cell(concurrency=8), _cell()]}, c=32)
    assert out["valid"] and out["invalid_reasons"] == []
    assert out["aggregate_tps"] == 9000.0
    assert out["itl_p50_ms"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"underfilled": True}, "underfilled"),
        ({"capacity_limited": True}, "capacity_limited"),
        ({"loop_detected": True}, "loop_detected"),
        ({"warmup_timed_out": True}, "warmup_timed_out"),
        ({"aggregate_tps": -4.0, "failure_reason": "boom"}, "aggregate_tps<=0"),
        ({"num_errors": 3}, "num_errors"),
    ],
)
def test_parse_flags_invalid_cells_without_raising(override, reason):
    out = parse_lib_result({"results": [_cell(**override)]}, c=32)
    assert not out["valid"] and reason in out["invalid_reasons"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "9000", True])
def test_parse_marks_a_non_finite_aggregate_tps_invalid(bad):
    out = parse_lib_result({"results": [_cell(aggregate_tps=bad)]}, c=32)
    assert not out["valid"] and out["invalid_reasons"] == ["aggregate_tps_not_finite"]
    assert out["aggregate_tps"] is bad


def test_parse_marks_a_cell_without_aggregate_tps_invalid_instead_of_raising():
    cell = _cell()
    del cell["aggregate_tps"]
    out = parse_lib_result({"results": [cell]}, c=32)
    assert not out["valid"] and out["invalid_reasons"] == ["aggregate_tps_not_finite"]
    assert out["aggregate_tps"] is None


def test_parse_keeps_the_non_positive_reason_for_finite_values():
    zero = parse_lib_result({"results": [_cell(aggregate_tps=0.0)]}, c=32)
    assert zero["invalid_reasons"] == ["aggregate_tps<=0"]


@pytest.mark.parametrize(
    "source",
    [
        "openai_stream_chunks_fallback",
        "prometheus_fallback",
        "none",
        "invalid_stream_failure",
        "",
        None,
    ],
)
def test_parse_requires_continuous_usage_aggregate_source(source):
    out = parse_lib_result({"results": [_cell(aggregate_source=source)]}, c=32)
    assert not out["valid"] and out["invalid_reasons"] == ["aggregate_source"]


@pytest.mark.parametrize("effective", [0.0, None, 31.3, 16.0, float("nan"), float("inf")])
def test_parse_requires_batch_to_have_filled_for_c_above_one(effective):
    out = parse_lib_result({"results": [_cell(effective_concurrency=effective)]}, c=32)
    assert not out["valid"] and out["invalid_reasons"] == ["effective_concurrency"]


def test_parse_effective_concurrency_threshold_is_the_bench_underfill_ratio():
    at = parse_lib_result({"results": [_cell(concurrency=50, effective_concurrency=49.0)]}, c=50)
    below = parse_lib_result({"results": [_cell(concurrency=50, effective_concurrency=48.9)]}, c=50)
    assert at["valid"] and not below["valid"]


def test_parse_does_not_require_effective_concurrency_at_c1():
    out = parse_lib_result({"results": [_cell(concurrency=1, effective_concurrency=0.0)]}, c=1)
    assert out["valid"] and out["invalid_reasons"] == []


def test_parse_missing_cell_is_recorded_as_invalid_not_raised():
    out = parse_lib_result({"results": [_cell(concurrency=8), _cell(context_tokens=4096)]}, c=32)
    assert out["valid"] is False and out["invalid_reasons"] == ["missing_cell"]
    metrics = (
        "aggregate_tps",
        "aggregate_source",
        "measurement_seconds",
        "effective_concurrency",
        "avg_running_reqs",
        "itl_p50_ms",
        "tps_per_user_p50",
        "ttft_p50_ms",
        "server_gen_throughput",
    )
    assert all(out[k] is None for k in metrics)
    real = parse_lib_result({"results": [_cell()]}, c=32)
    assert out.keys() == real.keys()


class _FakeRun:
    """Stands in for subprocess.run: records the call and optionally writes the result file."""

    def __init__(self, output_path, result=None):
        self.output_path, self.result = output_path, result
        self.calls = []
        self.output_existed_at_call = None

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        self.output_existed_at_call = self.output_path.exists()
        if self.result is not None:
            self.output_path.write_text(json.dumps(self.result))
        return subprocess.CompletedProcess(cmd, 0)


def test_run_lib_decode_command_timeout_and_stale_output(tmp_path, monkeypatch):
    out = tmp_path / "raw" / "x.json"
    out.parent.mkdir()
    out.write_text(json.dumps({"results": [_cell(aggregate_tps=1.0)]}))
    fake = _FakeRun(out, {"results": [_cell()]})
    monkeypatch.setattr(lib_bench.subprocess, "run", fake)

    metrics = run_lib_decode(32, 120, out, tmp_path / "logs" / "lib.log")

    ((cmd, kwargs),) = fake.calls
    assert fake.output_existed_at_call is False
    assert cmd[0] == sys.executable and "--no-resume" in cmd
    assert _has_pair(cmd, "--output", str(out))
    assert kwargs["timeout"] == 120 + 900
    assert kwargs["check"] is True and kwargs["env"]["LLM_BENCH_NO_UPDATE_CHECK"] == "1"
    assert metrics["aggregate_tps"] == 9000.0


def test_run_lib_decode_missing_output_file_still_raises(tmp_path, monkeypatch):
    out = tmp_path / "x.json"
    out.write_text(json.dumps({"results": [_cell()]}))
    monkeypatch.setattr(lib_bench.subprocess, "run", _FakeRun(out, result=None))
    with pytest.raises(FileNotFoundError):
        run_lib_decode(32, 120, out, tmp_path / "lib.log")
