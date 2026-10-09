import contextlib
import os
import signal
import socket
import sys
import time

import pytest

from fp4bench import server as srv
from fp4bench.server import VllmServer, parse_server_log
from tests import vllm_logs as real

MX_LOG_SNIPPET = """INFO 10-05 12:00:30 [gpu_model_runner.py:2900] Model loading took 13.2051 GiB memory and 41.2 seconds
INFO 10-05 11:59:00 [core.py:90] Initializing engine with config: compilation_config={"level": 3, "cudagraph_capture_sizes": [512, 256, 128, 32, 8, 1], "max_cudagraph_capture_size": 512}
"""  # noqa: E501 - verbatim vLLM log lines

NV_LOG_SNIPPET = """INFO Model loading took 13.9 GiB memory and 40.0 seconds
INFO CompilationConfig(level=3, cudagraph_capture_sizes=[1, 8, 32], max_cudagraph_capture_size=32)
"""


def test_parse_mx_log():
    facts = parse_server_log(MX_LOG_SNIPPET)
    assert facts["weights_gib"] == 13.2051
    assert facts["cudagraph_capture_sizes"] == [1, 8, 32, 128, 256, 512]


def test_parse_nv_log_reads_the_compilation_config_repr():
    facts = parse_server_log(NV_LOG_SNIPPET)
    assert facts["weights_gib"] == 13.9
    assert facts["cudagraph_capture_sizes"] == [1, 8, 32]


SAMPLING_LINE = (
    "INFO 10-05 12:00:03 [serving_chat.py:140] Default sampling parameters have been "
    "overridden by the model's Hugging Face generation config recommended from the "
    "model creator. If this is not intended, please relaunch vLLM instance with "
    "`--generation-config vllm`."
)


def test_parse_empty_log():
    facts = parse_server_log("")
    assert facts == {
        "weights_gib": None,
        "cudagraph_capture_sizes": None,
        "sampling_defaults_lines": [],
        "linear_kernels": [],
        "linear_kernel_lines": [],
        "linear_backend_fallback_lines": [],
        "bf16_gemm_lines": [],
        "autotune_lines": [],
        "autotune_cache_files": [],
        "autotune_ran": False,
        "autotune_cache_loaded": False,
        "pass_config": None,
        "pass_config_conflict": False,
        "fuse_act_quant": None,
        "custom_fusion_lines": [],
        "custom_fusions": [],
        "kv_cache_tokens": None,
        "kv_cache_memory_gib": None,
        "attention_backend_lines": [],
        "compile_cache_lines": [],
        "compile_cache_hashes": [],
    }


def test_parse_sampling_defaults_line_is_captured_once():
    facts = parse_server_log(f"INFO unrelated\n{SAMPLING_LINE}\n{SAMPLING_LINE}\nINFO other\n")
    assert facts["sampling_defaults_lines"] == [SAMPLING_LINE]


def test_parse_log_without_sampling_override_has_empty_list():
    assert parse_server_log(MX_LOG_SNIPPET)["sampling_defaults_lines"] == []


MXFP4_KERNEL_FMT = "Using %s for MXFP4 GEMM"
NVFP4_KERNEL_FMT = "Using %s for NVFP4 GEMM"
BACKEND_NO_KERNEL_FMT = (
    "--linear-backend=%s has no kernel for this linear layer type; "
    "using automatic selection for that layer type."
)
BACKEND_NO_KERNEL_AT_ALL_FMT = (
    "--linear-backend=%s was requested, but no %s kernel exists for "
    "%s layers; falling back to normal kernel selection for this "
    "layer."
)
MXFP8_KERNEL_FMT = "Using %s for MXFP8 GEMM"
MIXED_PRECISION_FMT = "Using %s for mixed-precision linear"


def vllm_line(
    message: str,
    level: str = "INFO",
    where: str = "linear/__init__.py:1186",
    proc: str = "EngineCore",
    pid: int = 4242,
) -> str:
    """One log record as vllm/logger.py's _FORMAT writes it (no color: the log is a file)."""
    return f"({proc} pid={pid}) {level} 10-05 12:00:01 [{where}] {message}"


NV_CUTEDSL = vllm_line(NVFP4_KERNEL_FMT % "FlashInferCuteDslNvFp4LinearKernel")
MX_FLASHINFER = vllm_line(
    MXFP4_KERNEL_FMT % "FlashInferMxFp4LinearKernel", where="linear/__init__.py:975"
)
MX_PIN_FALLBACK = vllm_line(
    BACKEND_NO_KERNEL_FMT % "flashinfer_cutedsl", level="WARNING", where="linear/__init__.py:365"
)


def test_parse_nvfp4_linear_kernel_line():
    facts = parse_server_log(f"INFO unrelated\n{NV_CUTEDSL}\nINFO other\n")
    assert facts["linear_kernels"] == ["FlashInferCuteDslNvFp4LinearKernel"]
    assert facts["linear_kernel_lines"] == [NV_CUTEDSL]
    assert facts["linear_backend_fallback_lines"] == []


def test_parse_mxfp4_linear_kernel_line_and_the_pin_fallback_warning():
    facts = parse_server_log(f"{MX_PIN_FALLBACK}\n{MX_FLASHINFER}\n")
    assert facts["linear_kernels"] == ["FlashInferMxFp4LinearKernel"]
    assert facts["linear_kernel_lines"] == [MX_FLASHINFER]
    assert facts["linear_backend_fallback_lines"] == [MX_PIN_FALLBACK]


def test_parse_second_fallback_warning_variant():
    line = vllm_line(
        BACKEND_NO_KERNEL_AT_ALL_FMT % ("flashinfer_cutedsl", "flashinfer_cutedsl", "NVFP4"),
        level="WARNING",
        where="linear/__init__.py:394",
    )
    assert parse_server_log(line + "\n")["linear_backend_fallback_lines"] == [line]


def test_parse_linear_kernels_are_sorted_unique_class_names():
    other = vllm_line(
        NVFP4_KERNEL_FMT % "EmulationNvFp4LinearKernel", where="linear/__init__.py:1142"
    )
    log = "\n".join(
        [
            NV_CUTEDSL,
            other,
            NV_CUTEDSL,
            vllm_line(
                NVFP4_KERNEL_FMT % "FlashInferCuteDslNvFp4LinearKernel", proc="Worker", pid=7
            ),
        ]
    )
    facts = parse_server_log(log + "\n")
    assert facts["linear_kernels"] == [
        "EmulationNvFp4LinearKernel",
        "FlashInferCuteDslNvFp4LinearKernel",
    ]
    assert facts["linear_kernel_lines"] == sorted({*log.splitlines()})
    assert len(facts["linear_kernel_lines"]) == 3


def test_parse_linear_kernel_ignores_other_gemm_selection_lines():
    log = "\n".join(
        [
            vllm_line(MXFP8_KERNEL_FMT % "FlashInferCutedslMxfp8LinearKernel"),
            vllm_line(MIXED_PRECISION_FMT % "MarlinLinearKernel"),
            "INFO Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential "
            "backends: ['FLASHINFER_TRTLLM'].",
        ]
    )
    facts = parse_server_log(log + "\n")
    assert facts["linear_kernels"] == [] and facts["linear_kernel_lines"] == []


def test_parse_linear_kernel_does_not_depend_on_the_line_prefix():
    bare = NVFP4_KERNEL_FMT % "CutlassNvFp4LinearKernel"
    colored = f"\x1b[0;36m(EngineCore pid=1)\x1b[0m INFO 10-05 [x.py:1] {bare}"
    for line in (bare, colored):
        facts = parse_server_log(line)
        assert facts["linear_kernels"] == ["CutlassNvFp4LinearKernel"]
        assert facts["linear_kernel_lines"] == [line]


def test_parse_linear_kernel_line_stops_at_the_end_of_the_line():
    facts = parse_server_log(f"{NV_CUTEDSL}\r\nINFO Model loading took 13.2051 GiB memory\n")
    assert facts["linear_kernel_lines"] == [NV_CUTEDSL]
    assert facts["weights_gib"] == 13.2051


BF16_GEMM_FMT = "Using FlashInfer %s for eligible unquantized BF16 GEMMs."
BF16_GEMM_UNAVAILABLE_FMT = (
    "--linear-backend=%s requested FlashInfer mm_bf16 backend %r, but it is "
    "unavailable on the current hardware or environment; using automatic "
    "selection for unquantized linear layers."
)
BF16_CUTEDSL = vllm_line(BF16_GEMM_FMT % "cute-dsl", where="layers/utils.py:626")


def test_parse_captures_the_cutedsl_bf16_gemm_line():
    facts = parse_server_log(f"INFO unrelated\n{BF16_CUTEDSL}\nINFO other\n")
    assert facts["bf16_gemm_lines"] == [BF16_CUTEDSL]
    assert facts["linear_kernels"] == [] and facts["linear_kernel_lines"] == []


def test_parse_bf16_gemm_lines_are_unique_and_sorted():
    other_proc = vllm_line(
        BF16_GEMM_FMT % "cute-dsl", where="layers/utils.py:626", proc="Worker", pid=7
    )
    facts = parse_server_log("\n".join([BF16_CUTEDSL, other_proc, BF16_CUTEDSL]) + "\n")
    assert facts["bf16_gemm_lines"] == sorted({BF16_CUTEDSL, other_proc})


def test_parse_bf16_gemm_line_next_to_the_fp4_kernel_lines_is_kept_apart():
    facts = parse_server_log(f"{NV_CUTEDSL}\n{BF16_CUTEDSL}\n")
    assert facts["bf16_gemm_lines"] == [BF16_CUTEDSL]
    assert facts["linear_kernel_lines"] == [NV_CUTEDSL]


def test_parse_does_not_take_the_unavailable_warning_or_other_gemm_lines_for_it():
    unavailable = vllm_line(
        BF16_GEMM_UNAVAILABLE_FMT % ("flashinfer_cutedsl", "cute-dsl"),
        level="WARNING",
        where="layers/utils.py:617",
    )
    quantized = vllm_line("Using FlashInfer cute-dsl for eligible NVFP4 GEMMs.")
    facts = parse_server_log(f"{unavailable}\n{quantized}\n")
    assert facts["bf16_gemm_lines"] == []


def test_parse_a_log_without_the_line_has_an_empty_list():
    for log in (MX_LOG_SNIPPET, NV_LOG_SNIPPET, ""):
        assert parse_server_log(log)["bf16_gemm_lines"] == []


def test_parse_bf16_gemm_line_does_not_depend_on_the_prefix_or_the_line_ending():
    bare = BF16_GEMM_FMT % "cute-dsl"
    colored = f"\x1b[0;36m(EngineCore pid=1)\x1b[0m INFO 10-05 [x.py:1] {bare}"
    for line in (bare, colored):
        assert parse_server_log(line)["bf16_gemm_lines"] == [line]
    assert parse_server_log(f"{BF16_CUTEDSL}\r\nINFO x\n")["bf16_gemm_lines"] == [BF16_CUTEDSL]


AUTOTUNE_FILE_FMT = "Using FlashInfer autotune cache file: %s"
AUTOTUNE_RUN_FMT = "Running FlashInfer autotune with %d tokens and token buckets %s."
AUTOTUNE_BF16_FMT = "Running FlashInfer BF16-only autotune with %d tokens."
AUTOTUNE_SKIP_OPS_FMT = "Skipping FlashInfer autotuning for ops %s"
AUTOTUNE_DISABLED = "Skipping FlashInfer autotune because it is disabled."
FI_AUTOTUNE_STARTS = "[Autotuner]: Autotuning process starts ..."
FI_AUTOTUNE_LOADED = "[Autotuner]: Loaded 12 configs from %s"
FI_AUTOTUNE_SAVED = "[Autotuner]: Saved 12 configs to %s (12 new, 0 from previous config)"
CACHE_DIR = "/results/smoke-r2-1/autotune/r0_NV-1"
CACHE_FILE = f"{CACHE_DIR}/0123abcd/autotune_configs.json"
FI_PREFIX = "2026-10-05 12:00:03,456 - INFO - autotuner.py:1046 - flashinfer.jit: "

AUTOTUNE_FILE = vllm_line(AUTOTUNE_FILE_FMT % CACHE_FILE, where="kernel_warmup.py:437")
AUTOTUNE_RUN = vllm_line(
    AUTOTUNE_RUN_FMT % (16384, (1, 2, 4, 8, 16, 32, 64, 128, 16384)), where="kernel_warmup.py:360"
)
AUTOTUNE_BF16 = vllm_line(AUTOTUNE_BF16_FMT % 32, where="kernel_warmup.py:391")
AUTOTUNE_SKIP = vllm_line(AUTOTUNE_SKIP_OPS_FMT % "('fp4_gemm',)", where="kernel_warmup.py:430")
FI_STARTS = FI_PREFIX + FI_AUTOTUNE_STARTS
FI_LOADED = FI_PREFIX + FI_AUTOTUNE_LOADED % CACHE_FILE
FI_SAVED = FI_PREFIX + FI_AUTOTUNE_SAVED % CACHE_FILE
FRESH_LOG = (
    "\n".join([AUTOTUNE_FILE, AUTOTUNE_SKIP, AUTOTUNE_RUN, FI_STARTS, AUTOTUNE_BF16, FI_SAVED])
    + "\n"
)
REUSED_LOG = "\n".join([AUTOTUNE_FILE, FI_LOADED, AUTOTUNE_RUN, FI_SAVED]) + "\n"


def test_parse_a_fresh_autotune_log():
    facts = parse_server_log(f"INFO unrelated\n{FRESH_LOG}INFO other\n")
    assert facts["autotune_ran"] is True and facts["autotune_cache_loaded"] is False
    assert facts["autotune_cache_files"] == [CACHE_FILE]
    assert facts["autotune_lines"] == sorted(
        {AUTOTUNE_FILE, AUTOTUNE_SKIP, AUTOTUNE_RUN, FI_STARTS, AUTOTUNE_BF16, FI_SAVED}
    )


def test_parse_an_autotune_log_that_read_an_earlier_cache():
    facts = parse_server_log(REUSED_LOG)
    assert facts["autotune_ran"] is True
    assert facts["autotune_cache_loaded"] is True
    assert FI_LOADED in facts["autotune_lines"]


def test_parse_the_disabled_autotune_line_is_captured_and_nothing_ran():
    line = vllm_line(AUTOTUNE_DISABLED, where="kernel_warmup.py:255")
    facts = parse_server_log(line + "\n")
    assert facts["autotune_lines"] == [line]
    assert facts["autotune_ran"] is False and facts["autotune_cache_files"] == []


def test_parse_autotune_defaults_without_any_autotune_line():
    facts = parse_server_log(MX_LOG_SNIPPET)
    assert (facts["autotune_lines"], facts["autotune_cache_files"]) == ([], [])
    assert facts["autotune_ran"] is False and facts["autotune_cache_loaded"] is False


def test_parse_autotune_lines_are_unique_raw_lines_and_cache_files_are_unique_paths():
    other = vllm_line(
        AUTOTUNE_FILE_FMT % f"{CACHE_DIR}/ffff/autotune_configs.json",
        proc="Worker",
        pid=9,
        where="kernel_warmup.py:437",
    )
    facts = parse_server_log("\n".join([AUTOTUNE_FILE, AUTOTUNE_FILE, other]) + "\n")
    assert facts["autotune_lines"] == sorted({AUTOTUNE_FILE, other})
    assert facts["autotune_cache_files"] == [CACHE_FILE, f"{CACHE_DIR}/ffff/autotune_configs.json"]


def test_parse_autotune_ignores_unrelated_autotune_lines():
    helion = vllm_line("Helion autotune finished: 12 configs")
    inductor = vllm_line("max-autotune enabled for the inductor")
    facts = parse_server_log(f"{helion}\n{inductor}\n")
    assert facts["autotune_lines"] == [] and facts["autotune_ran"] is False


def test_the_autotune_cache_env_name_is_the_one_vllm_reads():
    assert srv.AUTOTUNE_CACHE_ENV == "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"


@pytest.mark.parametrize(
    "log, fresh",
    [(FRESH_LOG, True), (REUSED_LOG, False), ("", False), (AUTOTUNE_FILE + "\n", False)],
)
def test_autotune_ran_fresh_needs_a_tuning_pass_and_no_cache_read(log, fresh):
    assert srv.autotune_ran_fresh(parse_server_log(log), CACHE_DIR) is fresh


def test_autotune_ran_fresh_needs_the_cache_file_to_be_inside_the_session_dir():
    facts = parse_server_log(FRESH_LOG)
    assert srv.autotune_ran_fresh(facts, "/results/smoke-r2-1/autotune/r0_MX-9") is False
    assert srv.autotune_ran_fresh(facts, "/results/smoke-r2-1/autotune/r0_NV-1/") is True
    stale = vllm_line(
        AUTOTUNE_FILE_FMT % "/root/.cache/vllm/flashinfer_autotune_cache/0.7.0.post1"
        "/100a/0123/autotune_configs.json"
    )
    assert (
        srv.autotune_ran_fresh(parse_server_log(f"{stale}\n{AUTOTUNE_RUN}\n"), CACHE_DIR) is False
    )


def test_autotune_ran_fresh_is_false_if_any_cache_file_is_outside():
    stale = vllm_line(AUTOTUNE_FILE_FMT % "/elsewhere/autotune_configs.json", proc="Worker", pid=9)
    facts = parse_server_log(f"{FRESH_LOG}{stale}\n")
    assert srv.autotune_ran_fresh(facts, CACHE_DIR) is False


def test_the_server_gets_the_given_environment_on_top_of_the_inherited_one(
    make_server, monkeypatch
):
    monkeypatch.setenv("FP4_INHERITED", "yes")
    server = make_server(
        'echo "got=$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR inherited=$FP4_INHERITED"',
        env={srv.AUTOTUNE_CACHE_ENV: "/some/dir"},
        timeout_s=5,
    )
    with pytest.raises(RuntimeError, match="exited"):
        server.start()
    assert "got=/some/dir inherited=yes" in server.log_path.read_text()


def test_the_server_inherits_the_environment_unchanged_without_an_override(
    make_server, monkeypatch
):
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", "/from/the/parent")
    server = make_server('echo "got=$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"', timeout_s=5)
    with pytest.raises(RuntimeError, match="exited"):
        server.start()
    assert "got=/from/the/parent" in server.log_path.read_text()


def test_the_override_does_not_leak_into_the_parents_environment(make_server, monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", raising=False)
    server = make_server("true", env={srv.AUTOTUNE_CACHE_ENV: "/some/dir"}, timeout_s=5)
    with pytest.raises(RuntimeError, match="exited"):
        server.start()
    assert "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" not in os.environ


@pytest.fixture
def gpu_waits(monkeypatch):
    calls = []
    monkeypatch.setattr(srv, "wait_gpu_released", lambda: calls.append(1))
    return calls


@pytest.fixture
def make_server(tmp_path, gpu_waits, monkeypatch):
    """Build servers around `sh -c ...` and always kill their process groups afterwards."""
    monkeypatch.setattr(srv, "HEALTH_POLL_S", 0.1)
    made = []

    def make(
        script: str | list[str],
        port: int | None = None,
        timeout_s: float = 1,
        env: dict[str, str] | None = None,
    ) -> VllmServer:
        command = ["sh", "-c", script] if isinstance(script, str) else script
        health_url = f"http://127.0.0.1:{port or _closed_port()}/health"
        server = VllmServer(
            "unused",
            tmp_path / "logs" / "server.log",
            command=command,
            health_url=health_url,
            start_timeout_s=timeout_s,
            env=env,
        )
        made.append(server)
        return server

    yield make
    for server in made:
        if server.proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.proc.pid, signal.SIGKILL)
            server.proc.wait()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _group_gone(pgid: int, timeout_s: float = 5.0) -> bool:
    """killpg(pgid, 0) fails once every member (including orphans) is dead and reaped."""
    deadline = time.time() + timeout_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:  # macOS reports a not-yet-reaped zombie as EPERM
            pass
        if time.time() > deadline:
            return False
        time.sleep(0.05)


HEALTHY_SERVER = """
import http.server, signal, sys
if sys.argv[2] == "ignore-sigint":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _healthy_command(port: int, sigint: str = "default") -> list[str]:
    return [sys.executable, "-c", HEALTHY_SERVER, str(port), sigint]


def test_enter_that_times_out_kills_the_whole_process_group(make_server, gpu_waits):
    server = make_server("sleep 60", timeout_s=1)
    with pytest.raises(TimeoutError), server:
        pytest.fail("__enter__ must raise when the server never becomes healthy")
    assert _group_gone(server.proc.pid)
    assert gpu_waits == [1]


def test_stop_kills_orphaned_group_member_after_leader_exited(make_server):
    server = make_server("sleep 60 & exit 0", timeout_s=5)
    with pytest.raises(RuntimeError, match="exited"):
        server.start()
    pgid = server.proc.pid
    os.killpg(pgid, 0)
    server.stop()
    assert _group_gone(pgid)


def test_enter_that_fails_because_leader_exited_still_sweeps_orphans(make_server):
    server = make_server("sleep 60 & exit 0", timeout_s=5)
    with pytest.raises(RuntimeError, match="exited"):
        server.__enter__()
    assert _group_gone(server.proc.pid)


def test_start_refuses_when_port_already_accepts_connections(make_server, gpu_waits, tmp_path):
    marker = tmp_path / "spawned"
    with socket.socket() as old_server:
        old_server.bind(("127.0.0.1", 0))
        old_server.listen()
        port = old_server.getsockname()[1]
        server = make_server(f"touch {marker}; sleep 60", port=port)
        with pytest.raises(RuntimeError, match=str(port)), server:
            pytest.fail("must not start next to an existing server")
    time.sleep(0.3)
    assert server.proc is None and not marker.exists()
    assert gpu_waits == []


def test_stop_before_start_is_a_noop(make_server, gpu_waits):
    make_server("sleep 60").stop()
    assert gpu_waits == []


def test_healthy_server_context_manager_stops_the_group(make_server, gpu_waits):
    port = _closed_port()
    server = make_server(_healthy_command(port), port=port, timeout_s=20)
    with server:
        assert server.proc.poll() is None
        pgid = server.proc.pid
        os.killpg(pgid, 0)
    assert _group_gone(pgid)
    assert gpu_waits == [1]
    assert "KeyboardInterrupt" in server.log_text()


def test_stop_escalates_to_sigkill_when_sigint_is_ignored(make_server, monkeypatch):
    monkeypatch.setattr(srv, "STOP_GRACE_S", 1)
    port = _closed_port()
    server = make_server(_healthy_command(port, "ignore-sigint"), port=port, timeout_s=20)
    server.start()
    pgid = server.proc.pid
    started = time.time()
    server.stop()
    assert 1 <= time.time() - started < 20
    assert _group_gone(pgid)


NV_PASS_CONFIG = {
    "fuse_norm_quant": False,
    "fuse_act_quant": True,
    "fuse_attn_quant": False,
    "enable_sp": False,
    "fuse_gemm_comms": False,
    "fuse_allreduce_rms": False,
    "enable_qk_norm_rope_fusion": False,
    "fuse_rope_kvcache_cat_mla": False,
    "fuse_act_padding": False,
    "fuse_qk_norm_rope_kvcache": False,
}
NV_FUSION_LINES = [line for line in real.NV_LINES if "Enabled custom fusions: " in line]
NV_ATTENTION_LINES = [real.NV_LINES[7], real.NV_LINES[8]]
MX_ATTENTION_LINES = [real.MX_LINES[2], real.MX_LINES[3]]
NVNF_ENGINE_CONFIG = real.NV_ENGINE_CONFIG.replace(
    "'fuse_act_quant': True", "'fuse_act_quant': False"
)
NVNF_LOG = (
    "\n".join(
        line if line != real.NV_ENGINE_CONFIG else NVNF_ENGINE_CONFIG
        for line in real.NV_LINES
        if line not in NV_FUSION_LINES
    )
    + "\n"
)


def test_the_real_nv_log_has_the_act_quant_fusion_on_and_says_so():
    facts = parse_server_log(real.NV_LOG)
    assert facts["pass_config"] == NV_PASS_CONFIG
    assert facts["fuse_act_quant"] is True and facts["pass_config_conflict"] is False
    assert len(NV_FUSION_LINES) == 6
    assert facts["custom_fusion_lines"] == sorted(NV_FUSION_LINES)
    assert facts["custom_fusions"] == ["act_quant"]


def test_the_real_mx_log_has_the_fusion_off_and_no_fusion_line():
    facts = parse_server_log(real.MX_LOG)
    assert facts["pass_config"] == {**NV_PASS_CONFIG, "fuse_act_quant": False}
    assert facts["fuse_act_quant"] is False and facts["pass_config_conflict"] is False
    assert facts["custom_fusion_lines"] == [] and facts["custom_fusions"] == []


def test_an_nvnf_log_has_the_fusion_off_and_no_fusion_line():
    assert NVNF_ENGINE_CONFIG != real.NV_ENGINE_CONFIG
    facts = parse_server_log(NVNF_LOG)
    assert facts["fuse_act_quant"] is False
    assert facts["pass_config"] == {**NV_PASS_CONFIG, "fuse_act_quant": False}
    assert facts["custom_fusion_lines"] == [] and facts["custom_fusions"] == []
    assert facts["linear_kernels"] == ["FlashInferCuteDslNvFp4LinearKernel"]


def test_the_real_logs_give_the_kv_cache_capacity_and_memory():
    nv, mx = parse_server_log(real.NV_LOG), parse_server_log(real.MX_LOG)
    assert (nv["kv_cache_tokens"], nv["kv_cache_memory_gib"]) == (558_496, 136.35)
    assert (mx["kv_cache_tokens"], mx["kv_cache_memory_gib"]) == (560_288, 136.79)
    assert type(nv["kv_cache_tokens"]) is int


def test_the_real_logs_give_the_attention_backend_lines():
    assert parse_server_log(real.NV_LOG)["attention_backend_lines"] == sorted(NV_ATTENTION_LINES)
    assert parse_server_log(real.MX_LOG)["attention_backend_lines"] == sorted(MX_ATTENTION_LINES)
    assert all(
        "kv_cache_dtype=torch.bfloat16" in line or "FLASHINFER" in line
        for line in NV_ATTENTION_LINES
    )


def test_the_existing_facts_parse_the_real_lines_too():
    nv, mx = parse_server_log(real.NV_LOG), parse_server_log(real.MX_LOG)
    assert nv["linear_kernels"] == ["FlashInferCuteDslNvFp4LinearKernel"]
    assert mx["linear_kernels"] == ["FlashInferMxFp4LinearKernel"]
    assert {1, 8, 32, 64, 128, 256, 512} <= set(nv["cudagraph_capture_sizes"])


def engine_config(
    pass_config_body: str, prefix: str = "(EngineCore pid=1) INFO 10-05 [core.py:129] "
) -> str:
    return (
        f"{prefix}Initializing a V1 LLM engine (v0.31.0) with config: model='/m', "
        f"compilation_config={{'mode': 3, 'pass_config': {{{pass_config_body}}}, 'x': 1}}, "
        "kernel_config=KernelConfig(linear_backend='flashinfer_cutedsl')"
    )


def test_a_log_without_the_engine_config_line_has_no_pass_config():
    for log in ("", MX_LOG_SNIPPET, "\n".join(NV_FUSION_LINES)):
        facts = parse_server_log(log)
        assert facts["pass_config"] is None and facts["fuse_act_quant"] is None
        assert facts["pass_config_conflict"] is False


def test_a_hidden_or_none_fuse_act_quant_is_none_not_false():
    facts = parse_server_log(engine_config("'fuse_norm_quant': False"))
    assert facts["pass_config"] == {"fuse_norm_quant": False} and facts["fuse_act_quant"] is None
    facts = parse_server_log(engine_config("'fuse_act_quant': None"))
    assert facts["pass_config"] == {"fuse_act_quant": None} and facts["fuse_act_quant"] is None
    facts = parse_server_log(engine_config(""))
    assert facts["pass_config"] == {} and facts["fuse_act_quant"] is None


def test_only_the_engine_config_line_is_read_for_the_pass_config():
    elsewhere = "DEBUG dumped {'pass_config': {'fuse_act_quant': True}}"
    assert parse_server_log(elsewhere + "\n")["pass_config"] is None


def test_a_pass_config_that_is_not_a_flat_dict_is_no_pass_config():
    facts = parse_server_log(engine_config("'nested': {'a': 1}, 'fuse_act_quant': True"))
    assert facts["pass_config"] is None and facts["fuse_act_quant"] is None


def test_values_other_than_true_false_none_are_not_taken_and_never_evaluated():
    facts = parse_server_log(
        engine_config(
            "'fuse_norm_quant': False, 'fuse_act_quant': __import__('os'), 'sp_min_token_num': 4096"
        )
    )
    assert facts["pass_config"] == {"fuse_norm_quant": False}
    assert facts["fuse_act_quant"] is None


def test_several_engine_config_lines_keep_the_first_and_flag_a_conflict():
    on, off = engine_config("'fuse_act_quant': True"), engine_config("'fuse_act_quant': False")
    facts = parse_server_log(f"{on}\n{off}\n")
    assert facts["fuse_act_quant"] is True and facts["pass_config_conflict"] is True
    facts = parse_server_log(f"{off}\n{on}\n")
    assert facts["fuse_act_quant"] is False and facts["pass_config_conflict"] is True
    again = engine_config("'fuse_act_quant': True", prefix="(Worker pid=2) INFO ")
    same = parse_server_log(f"{on}\n{again}\n")
    assert same["fuse_act_quant"] is True and same["pass_config_conflict"] is False


def test_custom_fusions_are_the_sorted_unique_names_of_every_fusion_line():
    lines = [
        "(EngineCore pid=1) INFO 10-05 [compilation.py:331] Enabled custom fusions: "
        "norm_quant, act_quant",
        "(ApiServer_0 pid=2) INFO 10-05 [compilation.py:331] Enabled custom fusions: act_quant",
    ]
    facts = parse_server_log("\n".join(lines) + "\n")
    assert facts["custom_fusions"] == ["act_quant", "norm_quant"]
    assert facts["custom_fusion_lines"] == sorted(lines)


def test_the_smallest_kv_cache_is_kept_when_several_are_logged():
    log = "\n".join(
        [
            "INFO [kv_cache_utils.py:2464] GPU KV cache size: 1,117,184 tokens, Maximum "
            "concurrency for 4,096 tokens per request: 272.75x",
            "INFO [kv_cache_utils.py:2464] GPU KV cache size: 999 tokens, Maximum concurrency for "
            "4,096 tokens per request: 0.24x",
            "INFO [gpu_worker.py:692] Available KV cache memory: 136.4 GiB",
            "INFO [gpu_worker.py:692] Available KV cache memory: 12.0 GiB",
        ]
    )
    facts = parse_server_log(log + "\n")
    assert (facts["kv_cache_tokens"], facts["kv_cache_memory_gib"]) == (999, 12.0)
    one = parse_server_log("GPU KV cache size: 1,117,184 tokens, Maximum concurrency")
    assert one["kv_cache_tokens"] == 1_117_184


def test_the_kv_facts_are_none_without_their_lines():
    for log in ("", MX_LOG_SNIPPET, NV_LOG_SNIPPET, real.NV_ENGINE_CONFIG):
        facts = parse_server_log(log)
        assert facts["kv_cache_tokens"] is None and facts["kv_cache_memory_gib"] is None


def test_attention_backend_lines_ignore_other_using_lines():
    log = "\n".join(
        [
            NV_CUTEDSL,
            MX_FLASHINFER,
            BF16_CUTEDSL,
            "INFO Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential backends: [].",
        ]
    )
    assert parse_server_log(log + "\n")["attention_backend_lines"] == []


def test_the_real_aot_load_line_gives_the_compiled_graphs_hash():
    facts = parse_server_log(real.NV_LOG + real.NV_AOT_LOAD_LINE + "\n")
    assert facts["compile_cache_lines"] == [real.NV_AOT_LOAD_LINE]
    assert facts["compile_cache_hashes"] == [real.NV_AOT_HASH]
    mx = parse_server_log(real.MX_LOG + real.MX_AOT_LOAD_LINE + "\n")
    assert mx["compile_cache_hashes"] == [real.MX_AOT_HASH] != facts["compile_cache_hashes"]


def test_a_fresh_compile_gives_the_cache_directory_and_the_saved_graph():
    log = (
        "\n".join(
            [
                real.NVV_CACHE_DIR_LINE,
                "INFO Compiling a graph for compile range (1, 16384)",
                real.NVV_AOT_SAVE_LINE,
            ]
        )
        + "\n"
    )
    facts = parse_server_log(log)
    assert facts["compile_cache_lines"] == sorted([real.NVV_CACHE_DIR_LINE, real.NVV_AOT_SAVE_LINE])
    assert facts["compile_cache_hashes"] == sorted([real.NVV_CACHE_DIR_HASH, real.NVV_AOT_HASH])


def test_the_engine_config_line_is_no_compile_cache_evidence():
    facts = parse_server_log(real.NV_LOG)
    assert facts["compile_cache_lines"] == [] and facts["compile_cache_hashes"] == []


def test_compile_cache_hashes_are_unique_and_sorted_and_never_the_aot_directory_itself():
    log = "\n".join(
        [
            real.NV_AOT_LOAD_LINE,
            real.NV_AOT_LOAD_LINE,
            "INFO cleaning /root/.cache/vllm/torch_compile_cache/torch_aot_compile",
            "INFO cleaning /root/.cache/vllm/torch_compile_cache/torch_aot_compile/ done",
        ]
    )
    facts = parse_server_log(log + "\n")
    assert facts["compile_cache_hashes"] == [real.NV_AOT_HASH]
    assert len(facts["compile_cache_lines"]) == 3
