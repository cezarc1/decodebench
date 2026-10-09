# METHODOLOGY.md#telemetry
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

from fp4bench.core.types import ClockEvent

COUNTER_NAMES = tuple(ClockEvent)
_COUNTER_LINE = re.compile(r"^\s*(.+?)\s*:\s*(\d+)\s*us\s*$")
SAMPLER_QUERY = (
    "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
    "utilization.gpu,clocks_event_reasons.active"
)
NVIDIA_SMI_TIMEOUT_S = 60  # a healthy nvidia-smi answers in about a second


def parse_counters(text: str) -> dict[ClockEvent, int]:
    _, found, section = text.partition("Clocks Event Reasons Counters")
    if not found:
        raise ValueError("no 'Clocks Event Reasons Counters' section")
    counters: dict[ClockEvent, int] = {}
    for line in section.splitlines()[1:]:
        m = _COUNTER_LINE.match(line)
        if not m:
            if counters:
                break
            continue
        if m.group(1) in COUNTER_NAMES:
            counters[ClockEvent(m.group(1))] = int(m.group(2))
    missing = set(COUNTER_NAMES) - counters.keys()
    if missing:
        raise ValueError(f"missing counters: {sorted(event.value for event in missing)}")
    return counters


def nvidia_smi(args: Sequence[str], timeout_s: float = NVIDIA_SMI_TIMEOUT_S) -> str:
    """nvidia-smi's stdout; a failed query raises CalledProcessError, a hung one TimeoutExpired."""
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=timeout_s
    ).stdout


def read_counters() -> dict[ClockEvent, int]:
    return parse_counters(nvidia_smi(("-q", "-d", "PERFORMANCE")))


def counter_delta(
    before: Mapping[ClockEvent, int], after: Mapping[ClockEvent, int]
) -> dict[ClockEvent, int]:
    return {k: after[k] - before[k] for k in COUNTER_NAMES}


def throttle_summary(delta: Mapping[ClockEvent, int], window_s: float) -> dict[str, int | float]:
    """Environmental throttling invalidates a cell; SW power capping is reported only."""
    env_throttle_us = sum(delta[k] for k in COUNTER_NAMES if k.is_env_throttle)
    power_cap_frac = delta[ClockEvent.SW_POWER_CAPPING] / 1e6 / window_s if window_s > 0 else 0.0
    return {"env_throttle_us": env_throttle_us, "sw_power_cap_frac": power_cap_frac}


class GpuSampler:
    """Context manager: nvidia-smi writes a CSV row every 200 ms while active."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> Self:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={SAMPLER_QUERY}",
                "--format=csv",
                "-lms",
                "200",
                "-f",
                str(self.csv_path),
            ]
        )
        time.sleep(0.5)
        code = self.proc.poll()
        if code is not None:
            self.proc = None
            raise RuntimeError(
                f"nvidia-smi sampler exited immediately with code {code}; "
                f"check the query fields: {SAMPLER_QUERY}"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            _stop(proc)
        except Exception:
            if exc_type is None:
                raise


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
