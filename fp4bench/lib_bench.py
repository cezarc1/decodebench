# METHODOLOGY.md#m2
import json
import os
import subprocess
import sys
from pathlib import Path

from fp4bench import settings
from fp4bench.core.types import M2InvalidReason, is_finite

INVALID_FLAGS = (
    M2InvalidReason.UNDERFILLED,
    M2InvalidReason.CAPACITY_LIMITED,
    M2InvalidReason.LOOP_DETECTED,
    M2InvalidReason.WARMUP_TIMED_OUT,
)
VALID_AGGREGATE_SOURCE = "openai_continuous_usage"
MIN_FILL = 0.98
METRIC_FIELDS = (
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


def lib_command(c: int, duration_s: int, output_path: str) -> list[str]:
    return [
        sys.executable,
        f"{settings.LIB_DIR}/llm_decode_bench.py",
        "--host",
        settings.HOST,
        "--port",
        str(settings.PORT),
        "--model",
        settings.SERVED_NAME,
        "--skip-prefill",
        "--contexts",
        "0",
        "--concurrency",
        str(c),
        "--max-tokens",
        str(settings.M2_MAX_TOKENS),
        "--duration",
        str(duration_s),
        "--display-mode",
        "plain",
        "--no-hw-monitor",
        "--no-resume",
        "--output",
        output_path,
    ]


def missing_cell_record() -> dict:
    """Same keys as a parsed cell, so a run with no cell for C is recorded as invalid."""
    return {
        "valid": False,
        "invalid_reasons": [M2InvalidReason.MISSING_CELL],
        "failure_reason": "",
        **dict.fromkeys(METRIC_FIELDS),
    }


def parse_lib_result(result: dict, c: int) -> dict:
    cells = [x for x in result["results"] if x["concurrency"] == c and x["context_tokens"] == 0]
    if not cells:
        return missing_cell_record()
    cell = cells[-1]
    reasons = [flag for flag in INVALID_FLAGS if cell.get(flag)]
    tps = cell.get("aggregate_tps")
    if not is_finite(tps):
        reasons.append(M2InvalidReason.AGGREGATE_TPS_NOT_FINITE)
    elif tps <= 0:
        reasons.append(M2InvalidReason.AGGREGATE_TPS_NOT_POSITIVE)
    if cell.get("num_errors", 0) > 0:
        reasons.append(M2InvalidReason.NUM_ERRORS)
    if cell.get("aggregate_source") != VALID_AGGREGATE_SOURCE:
        reasons.append(M2InvalidReason.AGGREGATE_SOURCE)
    effective = cell.get("effective_concurrency")
    if c > 1 and not (is_finite(effective) and effective >= MIN_FILL * c):
        reasons.append(M2InvalidReason.EFFECTIVE_CONCURRENCY)
    return {
        "valid": not reasons,
        "invalid_reasons": reasons,
        "failure_reason": cell.get("failure_reason", ""),
        "aggregate_tps": tps,
        "aggregate_source": cell.get("aggregate_source"),
        "measurement_seconds": cell.get("measurement_seconds"),
        "effective_concurrency": cell.get("effective_concurrency"),
        "avg_running_reqs": cell.get("avg_running_reqs"),
        "itl_p50_ms": cell.get("inter_token_latency_p50", 0.0) * 1000,
        "tps_per_user_p50": cell.get("output_tps_per_user_p50"),
        "ttft_p50_ms": cell.get("ttft_p50", 0.0) * 1000,
        "server_gen_throughput": cell.get("server_gen_throughput"),
    }


def run_lib_decode(c: int, duration_s: int, output_path: Path, log_path: Path) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    env = {**os.environ, "LLM_BENCH_NO_UPDATE_CHECK": "1"}
    with log_path.open("a") as log:
        subprocess.run(
            lib_command(c, duration_s, str(output_path)),
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=duration_s + 900,
        )
    return parse_lib_result(json.loads(output_path.read_text()), c)
