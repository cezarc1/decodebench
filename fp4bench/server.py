# METHODOLOGY.md#server-log
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Self

from fp4bench import settings

WEIGHTS_GIB_RE = re.compile(r"Model loading took ([0-9.]+) GiB")
CAPTURE_SIZES_RE = re.compile(r"cudagraph_capture_sizes['\"]?\s*[:=]\s*(\[[0-9,\s]*\])")
SAMPLING_DEFAULTS_RE = re.compile(r"[^\n]*[Dd]efault sampling param[^\n]*")
LINEAR_KERNEL_RE = re.compile(r"\bUsing ([A-Za-z0-9_]+) for (?:MXFP4|NVFP4) GEMM\b")
LINEAR_BACKEND_FALLBACK_RE = re.compile(
    r"--linear-backend=\S+ (?:has no kernel for this linear layer type|was requested, but no )"
)
BF16_GEMM_RE = re.compile(
    r"[^\r\n]*\bUsing FlashInfer \S+ for eligible unquantized BF16 GEMMs\.[^\r\n]*"
)
AUTOTUNE_CACHE_ENV = "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"  # METHODOLOGY.md#autotune
AUTOTUNE_FILE_RE = re.compile(r"Using FlashInfer autotune cache file: (\S+)")
AUTOTUNE_RAN_RE = re.compile(r"Running FlashInfer autotune with \d+ tokens and token buckets")
AUTOTUNE_LOADED_RE = re.compile(r"\[Autotuner\]: Loaded \d+ configs from ")
AUTOTUNE_LINE_RE = re.compile(
    r"[^\r\n]*(?:Using FlashInfer autotune cache file: |Running FlashInfer autotune with |"
    r"Running FlashInfer BF16-only autotune with |Skipping FlashInfer autotun(?:e|ing) |"
    r"\[Autotuner\]: (?:Loaded \d+ configs from |Autotuning process starts|Saved \d+ configs to ))"
    r"[^\r\n]*"
)
ENGINE_CONFIG_RE = re.compile(r"Initializing a V1 LLM engine \(v[^)]*\) with config: ")
PASS_CONFIG_RE = re.compile(r"'pass_config': \{([^{}]*)\}")
PASS_CONFIG_ITEM_RE = re.compile(r"'([A-Za-z0-9_]+)': (True|False|None)(?=,|\s*$)")
_PASS_CONFIG_VALUES = {"True": True, "False": False, "None": None}
CUSTOM_FUSIONS_LINE = "Enabled custom fusions: "
CUSTOM_FUSIONS_RE = re.compile(r"Enabled custom fusions: ([A-Za-z0-9_]+(?:, [A-Za-z0-9_]+)*)")
KV_CACHE_TOKENS_RE = re.compile(r"GPU KV cache size: ([0-9][0-9,]*) tokens")
KV_CACHE_GIB_RE = re.compile(r"Available KV cache memory: ([0-9]+(?:\.[0-9]+)?) GiB")
ATTENTION_LINE_RE = re.compile(r"Using \S+ attention backend|FlashInfer resolved query dtypes")
COMPILE_CACHE_LINE = "torch_compile_cache"
COMPILE_CACHE_HASH_RE = re.compile(r"torch_compile_cache/(?:torch_aot_compile/)?([^/\s]+)")
_AOT_DIR = "torch_aot_compile"
STOP_GRACE_S = 120
HEALTH_POLL_S = 5


def parse_pass_config(body: str) -> dict[str, bool | None]:
    """The 'name': True|False|None items of a pass_config repr's body; nothing is evaluated."""
    return {name: _PASS_CONFIG_VALUES[value] for name, value in PASS_CONFIG_ITEM_RE.findall(body)}


def parse_server_log(text: str) -> dict:
    weights = WEIGHTS_GIB_RE.search(text)
    sizes = CAPTURE_SIZES_RE.search(text)
    kernel_names, kernel_lines, fallback_lines = set(), set(), set()
    autotune_files = set()
    pass_configs, fusion_lines, fusions, attention_lines = [], set(), set(), set()
    kv_tokens, kv_gib = [], []
    compile_cache_lines, compile_cache_hashes = set(), set()
    for line in text.splitlines():
        if kernel := LINEAR_KERNEL_RE.search(line):
            kernel_names.add(kernel.group(1))
            kernel_lines.add(line)
        if LINEAR_BACKEND_FALLBACK_RE.search(line):
            fallback_lines.add(line)
        if cache_file := AUTOTUNE_FILE_RE.search(line):
            autotune_files.add(cache_file.group(1))
        if ENGINE_CONFIG_RE.search(line) and (body := PASS_CONFIG_RE.search(line)):
            pass_configs.append(parse_pass_config(body.group(1)))
        if CUSTOM_FUSIONS_LINE in line:
            fusion_lines.add(line)
            if names := CUSTOM_FUSIONS_RE.search(line):
                fusions.update(names.group(1).split(", "))
        if tokens := KV_CACHE_TOKENS_RE.search(line):
            kv_tokens.append(int(tokens.group(1).replace(",", "")))
        if gib := KV_CACHE_GIB_RE.search(line):
            kv_gib.append(float(gib.group(1)))
        if ATTENTION_LINE_RE.search(line):
            attention_lines.add(line)
        if COMPILE_CACHE_LINE in line:
            compile_cache_lines.add(line)
            compile_cache_hashes.update(
                h for h in COMPILE_CACHE_HASH_RE.findall(line) if h != _AOT_DIR
            )
    pass_config = pass_configs[0] if pass_configs else None
    return {
        "weights_gib": float(weights.group(1)) if weights else None,
        "cudagraph_capture_sizes": sorted(json.loads(sizes.group(1))) if sizes else None,
        "sampling_defaults_lines": sorted(
            {m.group(0) for m in SAMPLING_DEFAULTS_RE.finditer(text)}
        ),
        "linear_kernels": sorted(kernel_names),
        "linear_kernel_lines": sorted(kernel_lines),
        "linear_backend_fallback_lines": sorted(fallback_lines),
        "bf16_gemm_lines": sorted({m.group(0) for m in BF16_GEMM_RE.finditer(text)}),
        "autotune_lines": sorted({m.group(0) for m in AUTOTUNE_LINE_RE.finditer(text)}),
        "autotune_cache_files": sorted(autotune_files),
        "autotune_ran": bool(AUTOTUNE_RAN_RE.search(text)),
        "autotune_cache_loaded": bool(AUTOTUNE_LOADED_RE.search(text)),
        "pass_config": pass_config,
        "pass_config_conflict": any(other != pass_config for other in pass_configs),
        "fuse_act_quant": None if pass_config is None else pass_config.get("fuse_act_quant"),
        "custom_fusion_lines": sorted(fusion_lines),
        "custom_fusions": sorted(fusions),
        "kv_cache_tokens": min(kv_tokens) if kv_tokens else None,
        "kv_cache_memory_gib": min(kv_gib) if kv_gib else None,
        "attention_backend_lines": sorted(attention_lines),
        "compile_cache_lines": sorted(compile_cache_lines),
        "compile_cache_hashes": sorted(compile_cache_hashes),
    }


def autotune_ran_fresh(facts: dict, cache_dir) -> bool:
    """Tuned, read no cache, and wrote only inside `cache_dir` (METHODOLOGY.md#autotune)."""
    files = facts["autotune_cache_files"]
    return bool(
        facts["autotune_ran"]
        and not facts["autotune_cache_loaded"]
        and files
        and all(Path(f).is_relative_to(Path(cache_dir)) for f in files)
    )


def gpu_memory_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def wait_gpu_released(threshold_mib: int = 2048, timeout_s: float = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if gpu_memory_used_mib() < threshold_mib:
            return
        time.sleep(2)
    raise RuntimeError("GPU memory was not released after server shutdown")


def accepts_connections(host: str, port: int, timeout_s: float = 1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class VllmServer:
    """One `vllm serve` process group (METHODOLOGY.md#server-lifecycle)."""

    def __init__(
        self,
        model_dir: str,
        log_path: Path,
        args: tuple[str, ...] = (),
        command: list[str] | None = None,
        health_url: str | None = None,
        start_timeout_s: float = 1800,
        env: dict[str, str] | None = None,
    ):
        self.cmd = list(command) if command is not None else ["vllm", "serve", model_dir, *args]
        self.env = dict(env or {})
        self.health_url = health_url or f"{settings.BASE_URL}/health"
        self.start_timeout_s = start_timeout_s
        self.log_path = Path(log_path)
        self.proc: subprocess.Popen | None = None
        self._log = None
        self._pgid: int | None = None

    def _check_port_free(self) -> None:
        parts = urllib.parse.urlsplit(self.health_url)
        host, port = parts.hostname, parts.port
        if host and port and accepts_connections(host, port):
            raise RuntimeError(
                f"port {port} on {host} already accepts connections: another server "
                "is running, and its /health would be mistaken for ours"
            )

    def start(self, timeout_s: float | None = None) -> None:
        timeout_s = self.start_timeout_s if timeout_s is None else timeout_s
        self._check_port_free()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **self.env} if self.env else None,
        )
        self._pgid = self.proc.pid
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with {self.proc.returncode}; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except OSError:
                pass
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=min(HEALTH_POLL_S, max(0.0, deadline - time.time())))
        raise TimeoutError(f"vllm serve not healthy after {timeout_s}s; see {self.log_path}")

    def stop(self) -> None:
        """Kill the whole process group, then wait for the GPU. Safe to call at any point."""
        pgid, self._pgid = self._pgid, None
        proc = self.proc
        if pgid is None or proc is None:
            self._close_log()
            return
        try:
            self._interrupt_leader(proc, pgid)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            proc.wait()
        finally:
            self._close_log()
        wait_gpu_released()

    @staticmethod
    def _interrupt_leader(proc: subprocess.Popen, pgid: int) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(pgid, signal.SIGINT)
            proc.wait(timeout=STOP_GRACE_S)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_text(self) -> str:
        return self.log_path.read_text()

    def __enter__(self) -> Self:
        try:
            self.start()
        except BaseException:  # __exit__ is not called when __enter__ raises
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
