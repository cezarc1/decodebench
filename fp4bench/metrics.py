# METHODOLOGY.md#preemption-metric
import math
import re
import sys
import time
import urllib.request
from collections.abc import Callable

from fp4bench.core.types import RetryPolicy

PREEMPTIONS_METRIC = "vllm:num_preemptions"
SCRAPE_TIMEOUT_S = 10
_SAMPLE_RE = re.compile(
    r'vllm:num_preemptions_total(?:\{(?:[^"}]|"(?:[^"\\]|\\.)*")*\})?[ \t]+(\S+)(?:[ \t]+-?[0-9]+)?'
)
_SAMPLE_START_RE = re.compile(r"vllm:num_preemptions_total[{ \t]")
_DECLARED_RE = re.compile(r"# (?:HELP|TYPE) vllm:num_preemptions(?:_total)?(?:[ \t].*)?")
READ = RetryPolicy(attempts=3, backoff_s=2.0)


def parse_preemptions(metrics_text: str) -> float:
    """The preemptions counted so far; a page not read whole raises, never reads as 0."""
    total, declared, found = 0.0, False, False
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if _DECLARED_RE.fullmatch(line):
            declared = True
        elif sample := _SAMPLE_RE.fullmatch(line):
            try:
                value = float(sample.group(1))
            except ValueError:
                raise ValueError(f"not a number in /metrics: {line!r}") from None
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"not a count in /metrics: {line!r}")
            total += value
            found = True
        elif _SAMPLE_START_RE.match(line):
            raise ValueError(
                f"a vllm:num_preemptions_total sample that does not parse in /metrics "
                f"(an exemplar, a timestamp or labels this parser does not take): "
                f"{line!r}"
            )
    if not (found or declared):
        raise ValueError(
            f"{PREEMPTIONS_METRIC} is not in the /metrics page ({len(metrics_text)} "
            "characters): the preemption guard cannot be applied"
        )
    return total


def read_preemptions(base_url: str, timeout_s: float = SCRAPE_TIMEOUT_S) -> float:
    """parse_preemptions of one GET of <base_url>/metrics; raises on any failure."""
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout_s) as resp:
        return parse_preemptions(resp.read().decode("utf-8"))


def read_preemptions_retrying(
    base_url: str,
    policy: RetryPolicy = READ,
    *,
    sleep: Callable[[float], None] = time.sleep,
    read: Callable[[str], float] | None = None,
) -> float:
    """read_preemptions, up to `attempts` times, backoff_s * backoff_factor ** (k - 1) apart."""
    attempts, backoff_s, backoff_factor = policy
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1 (got {attempts})")
    read = read_preemptions if read is None else read
    pauses = policy.pauses
    for attempt in range(1, attempts + 1):
        try:
            return read(base_url)
        except Exception as exc:
            if attempt == attempts:
                spacing = (
                    f"{backoff_s} s apart"
                    if backoff_factor == 1
                    else f"pauses of {', '.join(map(str, pauses))} s"
                )
                exc.add_note(
                    f"vLLM's preemption counter could not be read in {attempts} attempts "
                    f"({spacing})"
                )
                raise
            pause = pauses[attempt - 1]
            print(
                f"warning: reading vLLM's preemption counter failed (attempt {attempt} of "
                f"{attempts}), retrying in {pause} s: {exc!r}",
                file=sys.stderr,
            )
            sleep(pause)
    raise AssertionError("unreachable")
