import re

import pytest

from fp4bench import telemetry as tm
from fp4bench.core.types import ClockEvent

PERF = """
GPU 00000000:63:00.0
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
    Clocks Event Reasons Counters
        SW Power Capping                  : 9852707882 us
        Sync Boost                        : 0 us
        SW Thermal Slowdown               : 0 us
        HW Thermal Slowdown               : 0 us
        HW Power Braking                  : 0 us
    Sparse Operation Mode                 : N/A
"""


def test_parse_counters():
    counters = tm.parse_counters(PERF)
    assert counters == {
        "SW Power Capping": 9852707882,
        "Sync Boost": 0,
        "SW Thermal Slowdown": 0,
        "HW Thermal Slowdown": 0,
        "HW Power Braking": 0,
    }
    assert [type(event) for event in counters] == [ClockEvent] * 5


def test_parse_counters_requires_section():
    with pytest.raises(ValueError, match="no 'Clocks Event Reasons Counters' section"):
        tm.parse_counters("GPU 0\n    Performance State : P0\n")


def test_delta_and_summary():
    before = tm.parse_counters(PERF)
    sw_power_capping = ClockEvent.SW_POWER_CAPPING
    after = {
        **before,
        sw_power_capping: before[sw_power_capping] + 2_000_000,
        ClockEvent.HW_THERMAL_SLOWDOWN: 5,
    }
    delta = tm.counter_delta(before, after)
    summary = tm.throttle_summary(delta, window_s=10.0)
    assert summary["env_throttle_us"] == 5 and isinstance(summary["env_throttle_us"], int)
    assert summary["sw_power_cap_frac"] == pytest.approx(0.2)


class FakeProc:
    """Stands in for subprocess.Popen: logs calls; `wait_outcomes` is consumed per wait()."""

    cmd: list[str]

    def __init__(self, poll_result=None, wait_outcomes=()):
        self.poll_result = poll_result
        self.wait_outcomes = list(wait_outcomes)
        self.calls = []

    def poll(self):
        self.calls.append("poll")
        return self.poll_result

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        outcome = self.wait_outcomes.pop(0) if self.wait_outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return 0


@pytest.fixture
def sampler_env(monkeypatch):
    """Patch Popen and sleep; returns (made_procs, sleeps)."""
    made, sleeps = [], []

    class Popen:
        def __init__(self):
            self.proc_kwargs = {}

        def __call__(self, cmd, **kwargs):
            proc = FakeProc(**self.proc_kwargs)
            proc.cmd = cmd
            made.append(proc)
            return proc

    popen = Popen()
    monkeypatch.setattr(tm.subprocess, "Popen", popen)
    monkeypatch.setattr(tm.time, "sleep", sleeps.append)
    return popen, made, sleeps


def test_sampler_starts_nvidia_smi_and_checks_it_survived(sampler_env, tmp_path):
    _, made, sleeps = sampler_env
    csv = tmp_path / "telemetry" / "s.csv"
    with tm.GpuSampler(csv) as sampler:
        assert sampler.proc is made[0]
        assert csv.parent.is_dir()
    cmd = made[0].cmd
    assert cmd[0] == "nvidia-smi" and f"--query-gpu={tm.SAMPLER_QUERY}" in cmd
    assert sleeps == [0.5]
    assert made[0].calls[0] == "poll"


def test_sampler_raises_if_nvidia_smi_exits_immediately(sampler_env, tmp_path):
    popen, _, sleeps = sampler_env
    popen.proc_kwargs = {"poll_result": 2}
    dead = (
        "nvidia-smi sampler exited immediately with code 2; "
        f"check the query fields: {tm.SAMPLER_QUERY}"
    )
    with (
        pytest.raises(RuntimeError, match=f"^{re.escape(dead)}$"),
        tm.GpuSampler(tmp_path / "s.csv"),
    ):
        pytest.fail("body must not run when the sampler is dead")
    assert sleeps == [0.5]


def test_sampler_exit_terminates_and_waits_10s(sampler_env, tmp_path):
    _, made, _ = sampler_env
    with tm.GpuSampler(tmp_path / "s.csv"):
        pass
    assert made[0].calls[1:] == ["terminate", ("wait", 10)]


def test_sampler_exit_kills_when_terminate_times_out(sampler_env, tmp_path):
    popen, made, _ = sampler_env
    popen.proc_kwargs = {"wait_outcomes": [tm.subprocess.TimeoutExpired("nvidia-smi", 10)]}
    with tm.GpuSampler(tmp_path / "s.csv"):
        pass
    assert made[0].calls[1:] == ["terminate", ("wait", 10), "kill", ("wait", 10)]


def test_sampler_exit_never_masks_an_in_flight_exception(sampler_env, tmp_path):
    popen, made, _ = sampler_env
    stuck = tm.subprocess.TimeoutExpired("nvidia-smi", 10)
    popen.proc_kwargs = {"wait_outcomes": [stuck, stuck]}
    with pytest.raises(KeyError, match="the real failure"), tm.GpuSampler(tmp_path / "s.csv"):
        raise KeyError("the real failure")
    assert "kill" in made[0].calls


def test_sampler_exit_reports_a_stuck_process_when_nothing_else_failed(sampler_env, tmp_path):
    popen, _, _ = sampler_env
    stuck = tm.subprocess.TimeoutExpired("nvidia-smi", 10)
    popen.proc_kwargs = {"wait_outcomes": [stuck, stuck]}
    with pytest.raises(tm.subprocess.TimeoutExpired), tm.GpuSampler(tmp_path / "s.csv"):
        pass


def test_a_missing_counter_is_named_as_nvidia_smi_prints_it():
    text = PERF.replace("HW Power Braking", "HW Something Else")
    with pytest.raises(ValueError, match=r"^missing counters: \['HW Power Braking'\]$"):
        tm.parse_counters(text)


def test_the_counter_query_is_an_nvidia_smi_query_with_its_timeout(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return tm.subprocess.CompletedProcess(argv, 0, stdout=PERF)

    monkeypatch.setattr(tm.subprocess, "run", run)
    assert tm.read_counters() == tm.parse_counters(PERF)
    kwargs = {"capture_output": True, "text": True, "check": True}
    assert calls == [
        (["nvidia-smi", "-q", "-d", "PERFORMANCE"], {**kwargs, "timeout": tm.NVIDIA_SMI_TIMEOUT_S})
    ]
