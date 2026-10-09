from types import SimpleNamespace

import pytest

from fp4bench import metrics
from fp4bench.core.types import RetryPolicy
from fp4bench.metrics import parse_preemptions, read_preemptions
from tests import QuietHandler, local_server

HELP = "# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine."
TYPE = "# TYPE vllm:num_preemptions_total counter"
SAMPLE = 'vllm:num_preemptions_total{engine="0",model_name="fp4bench"} %s'
AROUND = [
    "# HELP vllm:num_requests_running Number of requests in model execution batches.",
    "# TYPE vllm:num_requests_running gauge",
    'vllm:num_requests_running{engine="0",model_name="fp4bench"} 128.0',
    "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.",
    "# TYPE vllm:prompt_tokens_total counter",
    'vllm:prompt_tokens_total{engine="0",model_name="fp4bench"} 524288.0',
]
CREATED = [
    "# HELP vllm:num_preemptions_created Cumulative number of preemption from the engine.",
    "# TYPE vllm:num_preemptions_created gauge",
    'vllm:num_preemptions_created{engine="0",model_name="fp4bench"} 1.7596e+09',
]


def page(*lines: str) -> str:
    return "\n".join([*AROUND[:3], *lines, *AROUND[3:]]) + "\n"


@pytest.mark.parametrize("value, expected", [("0.0", 0.0), ("3.0", 3.0), ("12", 12.0)])
def test_the_counter_of_the_one_engine(value, expected):
    assert parse_preemptions(page(HELP, TYPE, SAMPLE % value)) == expected


def test_the_created_sample_and_other_families_are_not_counted():
    assert parse_preemptions(page(HELP, TYPE, SAMPLE % "2.0", *CREATED)) == 2.0


def test_every_label_set_of_one_scrape_is_summed():
    second = 'vllm:num_preemptions_total{engine="1",model_name="fp4bench"} 5.0'
    assert parse_preemptions(page(HELP, TYPE, SAMPLE % "2.0", second)) == 7.0


def test_a_sample_without_labels_counts_too():
    assert parse_preemptions(page(TYPE, "vllm:num_preemptions_total 4.0")) == 4.0


def test_a_label_value_with_braces_or_spaces_does_not_confuse_the_parser():
    line = 'vllm:num_preemptions_total{engine="0",model_name="a } b {c"} 1.0'
    assert parse_preemptions(page(TYPE, line)) == 1.0


@pytest.mark.parametrize(
    "declaration",
    [
        [HELP, TYPE],
        [
            "# HELP vllm:num_preemptions Cumulative number of preemption from the engine.",
            "# TYPE vllm:num_preemptions counter",
        ],
        ["# TYPE vllm:num_preemptions counter"],
    ],
)
def test_a_declared_family_without_a_sample_has_counted_nothing(declaration):
    assert parse_preemptions(page(*declaration)) == 0.0


@pytest.mark.parametrize(
    "text", ["", page(), page(*CREATED), page("# TYPE vllm:num_preemptions_totally counter")]
)
def test_a_page_without_the_family_raises_naming_the_metric(text):
    with pytest.raises(ValueError, match="vllm:num_preemptions"):
        parse_preemptions(text)


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1.0", "lots"])
def test_a_sample_that_is_not_a_count_raises(value):
    with pytest.raises(ValueError, match="vllm:num_preemptions_total"):
        parse_preemptions(page(TYPE, SAMPLE % value))


def test_crlf_line_endings_are_fine():
    assert parse_preemptions(page(TYPE, SAMPLE % "1.0").replace("\n", "\r\n")) == 1.0


@pytest.fixture
def metrics_server():
    """A local stand-in for vLLM's API server: serves `state.body` with `state.status`."""
    state = SimpleNamespace(body=page(HELP, TYPE, SAMPLE % "6.0"), status=200, paths=[])

    class Handler(QuietHandler):
        def do_GET(self):  # noqa: N802 - the name http.server dispatches to
            state.paths.append(self.path)
            self.send_response(state.status)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(state.body.encode())

    with local_server(Handler) as url:
        state.base_url = url
        yield state


def test_read_preemptions_gets_the_metrics_page_of_the_server(metrics_server):
    assert read_preemptions(metrics_server.base_url) == 6.0
    assert metrics_server.paths == ["/metrics"]


def test_read_preemptions_raises_on_an_http_error(metrics_server):
    metrics_server.status = 500
    with pytest.raises(OSError, match="HTTP Error 500"):
        read_preemptions(metrics_server.base_url)


def test_read_preemptions_raises_when_the_page_lacks_the_counter(metrics_server):
    metrics_server.body = page()
    with pytest.raises(ValueError, match="vllm:num_preemptions"):
        read_preemptions(metrics_server.base_url)


def test_read_preemptions_raises_when_nothing_listens():
    with pytest.raises(OSError, match="Connection refused"):
        read_preemptions("http://127.0.0.1:9", timeout_s=2)


def test_the_scrape_has_a_bounded_timeout():
    assert 0 < metrics.SCRAPE_TIMEOUT_S <= 30


@pytest.mark.parametrize(
    "line",
    [
        'vllm:num_preemptions_total{engine="0"} 3.0 # {trace_id="x"} 1.0',
        'vllm:num_preemptions_total {engine="0"} 3.0',
        'vllm:num_preemptions_total{engine="0"} 3.0 1759600000.5',
        'vllm:num_preemptions_total{engine="0" 3.0',
    ],
)
def test_a_sample_line_of_the_family_that_does_not_parse_raises(line):
    with pytest.raises(ValueError, match="vllm:num_preemptions_total"):
        parse_preemptions(page(HELP, TYPE, line))


def test_an_unparsable_sample_raises_even_next_to_a_good_one():
    bad = 'vllm:num_preemptions_total{engine="1"} 3.0 # {trace_id="x"} 1.0'
    with pytest.raises(ValueError, match="does not parse in /metrics"):
        parse_preemptions(page(HELP, TYPE, SAMPLE % "0.0", bad))


def test_other_families_that_share_the_prefix_are_still_ignored():
    others = [
        'vllm:num_preemptions_totally{engine="0"} 9.0',
        'vllm:num_preemptions_total_created{engine="0"} 1.7e9 # not ours',
    ]
    assert parse_preemptions(page(HELP, TYPE, SAMPLE % "2.0", *others)) == 2.0


TIMED_OUT = OSError("timed out")


class Flaky:
    """A read_preemptions stand-in that fails its first `failures` calls."""

    def __init__(self, failures: int, value: float = 4.0, exc: Exception = TIMED_OUT):
        self.failures, self.value, self.exc, self.calls = failures, value, exc, []

    def __call__(self, base_url):
        self.calls.append(base_url)
        if len(self.calls) <= self.failures:
            raise self.exc
        return self.value


def test_retrying_returns_the_first_successful_read_without_sleeping():
    read, sleeps = Flaky(0), []
    assert metrics.read_preemptions_retrying("http://x", read=read, sleep=sleeps.append) == 4.0
    assert read.calls == ["http://x"] and sleeps == []


def test_retrying_sleeps_between_attempts_and_recovers(capsys):
    read, sleeps = Flaky(2), []
    value = metrics.read_preemptions_retrying(
        "http://x", RetryPolicy(attempts=3, backoff_s=2.0), read=read, sleep=sleeps.append
    )
    assert value == 4.0 and len(read.calls) == 3
    assert sleeps == [2.0, 2.0]
    assert "timed out" in capsys.readouterr().err


def test_retrying_raises_the_last_error_after_every_attempt_failed():
    read, sleeps = Flaky(5, exc=ValueError("vllm:num_preemptions is not in the /metrics page")), []
    with pytest.raises(ValueError, match="vllm:num_preemptions") as exc:
        metrics.read_preemptions_retrying(
            "http://x", RetryPolicy(attempts=3, backoff_s=0.5), read=read, sleep=sleeps.append
        )
    assert len(read.calls) == 3 and sleeps == [0.5, 0.5]
    assert any("3 attempts" in note for note in exc.value.__notes__)


def test_retrying_defaults_to_three_attempts_two_seconds_apart(monkeypatch):
    read, sleeps = Flaky(9), []
    monkeypatch.setattr(metrics, "read_preemptions", read)
    with pytest.raises(OSError, match="could not be read in 3 attempts"):
        metrics.read_preemptions_retrying("http://x", sleep=sleeps.append)
    assert len(read.calls) == 3 and sleeps == [2.0, 2.0]


def test_retrying_backs_off_exponentially_with_a_factor(capsys):
    read, sleeps = Flaky(4), []
    value = metrics.read_preemptions_retrying(
        "http://x", RetryPolicy(5, 2.0, backoff_factor=2.0), read=read, sleep=sleeps.append
    )
    assert value == 4.0 and len(read.calls) == 5
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    err = capsys.readouterr().err
    assert "retrying in 2.0 s" in err and "retrying in 16.0 s" in err


def test_retrying_with_a_backoff_factor_names_every_pause_when_all_attempts_failed():
    read, sleeps = Flaky(9), []
    with pytest.raises(OSError, match="could not be read in 5 attempts") as exc:
        metrics.read_preemptions_retrying(
            "http://x", RetryPolicy(5, 2.0, backoff_factor=2.0), read=read, sleep=sleeps.append
        )
    assert len(read.calls) == 5 and sleeps == [2.0, 4.0, 8.0, 16.0]
    assert any(
        "5 attempts" in note and "2.0, 4.0, 8.0, 16.0 s" in note for note in exc.value.__notes__
    )


def test_retrying_never_swallows_an_interrupt():
    def interrupted(base_url):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        metrics.read_preemptions_retrying("http://x", read=interrupted, sleep=lambda s: None)


def test_retrying_needs_at_least_one_attempt():
    with pytest.raises(ValueError, match="attempts"):
        metrics.read_preemptions_retrying("http://x", RetryPolicy(0, 2.0), read=Flaky(0))


def test_retrying_reads_a_real_server(metrics_server):
    assert metrics.read_preemptions_retrying(metrics_server.base_url, sleep=lambda s: None) == 6.0


def test_a_retry_policy_names_its_pauses():
    assert RetryPolicy(3, 2.0).pauses == [2.0, 2.0]
    assert RetryPolicy(5, 2.0, backoff_factor=2.0).pauses == [2.0, 4.0, 8.0, 16.0]
    assert RetryPolicy(1, 2.0).pauses == []
    assert metrics.READ == (3, 2.0, 1.0)
