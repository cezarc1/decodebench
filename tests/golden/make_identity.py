"""Write tests/golden/identity.json (python -m tests.golden.make_identity).

identity.json must not change in a refactor: only identity_view may be adapted to new code. It is
regenerated only when a new study, treatment or constant is added on purpose, and then the
regenerated view may only add keys and list items to it. `--check-additive` lists what the
regenerated view adds and fails, without writing, if it removes or changes anything; writing
refuses such a view too."""

import argparse
import contextlib
import functools
import hashlib
import inspect
import json
import random
import re
import runpy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

from tests import GOLDEN_DIR, REPO, RUNS_DIR
from tests.golden.view import dumps

IDENTITY_PATH = GOLDEN_DIR / "identity.json"


def canonical(obj):
    """JSON-shaped data: tuples become lists, keys strings."""
    return json.loads(json.dumps(obj, allow_nan=False))


QWEN3_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
QWEN3_SPECIAL = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}
_SPECIAL_RE = re.compile("(" + "|".join(map(re.escape, QWEN3_SPECIAL)) + ")")
_BOS = 1


class FakeTokenizer:
    """One id per character; control tokens one id unless split; a BOS with add_special_tokens."""

    all_special_tokens: ClassVar[list[str]] = list(QWEN3_SPECIAL)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        assert add_generation_prompt is True and tokenize is False
        return QWEN3_TEMPLATE.format(content=messages[0]["content"])

    def encode(self, text, add_special_tokens=True, split_special_tokens=False):
        ids = [_BOS] if add_special_tokens else []
        for piece in [text] if split_special_tokens else _SPECIAL_RE.split(text):
            if not split_special_tokens and piece in QWEN3_SPECIAL:
                ids.append(QWEN3_SPECIAL[piece])
            else:
                ids.extend(map(ord, piece))
        return ids


_WORDS = (
    "the",
    "kernel",
    "decode",
    "batch",
    "token",
    "weights",
    "scale",
    "block",
    "naïve",
    "数据",
    "<|im_end|>",
    "<|endoftext|>",
    "Grüße",
    "fp4",
    "MXFP4",
    "NVFP4",
    "a",
    "of",
    "memory",
    "bandwidth",
    "step",
    "1024",
    "\n",
    "?",
    "!",
    "café",
    "λ",
)


def synthetic_sharegpt(n: int = 4000, seed: int = 20261006) -> list[dict]:
    """ShareGPT-shaped conversations with enough text for the real M1 and Experiment C prompts."""
    rng = random.Random(seed)
    out = [
        {
            "conversations": [
                {"from": "gpt", "value": "no human turn first"},
                {"from": "human", "value": "a later human turn"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "   "},
                {"from": "human", "value": "skipped: the first is empty"},
            ]
        },
        {"conversations": []},
    ]
    for i in range(n):
        words = [rng.choice(_WORDS) for _ in range(rng.randint(30, 600))]
        text = f"  {i} " + " ".join(words) + " "
        out.append(
            {"conversations": [{"from": "human", "value": text}, {"from": "gpt", "value": "ok"}]}
        )
    return out


@functools.cache
def synthetic_sharegpt_file() -> bytes:
    """synthetic_sharegpt() as the hub's ShareGPT download, built once per process."""
    return json.dumps(synthetic_sharegpt()).encode()


def _describe(value) -> Any:
    """A recorded argument as JSON: Modal objects by the name they were looked up with."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    for attr in ("_identity_recipe", "_identity_lookup"):
        if hasattr(value, attr):
            return getattr(value, attr)
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
    return f"<{type(value).__name__}>"


def record_modal(build) -> dict:
    """Run `build()` with Modal's image, volume, secret and function APIs recording their args."""
    import modal

    depth = [0]
    functions: dict[str, dict] = {}

    def recording_image_method(name, raw):
        """A constructor (staticmethod) starts a recipe; a method extends its image's."""
        static = isinstance(raw, staticmethod)
        fn = raw.__func__ if static else raw

        def wrapper(*args, **kwargs):
            depth[0] += 1
            try:
                out = fn(*args, **kwargs)
            finally:
                depth[0] -= 1
            if depth[0] == 0 and isinstance(out, modal.Image):
                parent = [] if static else getattr(args[0], "_identity_recipe", ["<unrecorded>"])
                step_args = args if static else args[1:]
                recipe = [*parent, [name, _describe(list(step_args)), _describe(kwargs)]]
                setattr(out, "_identity_recipe", recipe)  # noqa: B010 - not an attribute of modal.Image
            return out

        return staticmethod(wrapper) if static else wrapper

    def recording_lookup(kind, raw):
        fn = raw.__func__

        def wrapper(name, *args, **kwargs):
            out = fn(name, *args, **kwargs)
            out._identity_lookup = {kind: name, **_describe(kwargs)}
            return out

        return staticmethod(wrapper)

    raw_function = vars(modal.App)["function"]

    def recording_function(self, *args, **kwargs):
        decorator = raw_function(self, *args, **kwargs)

        def apply(fn):
            functions[fn.__name__] = {"args": _describe(list(args)), **_describe(kwargs)}
            return decorator(fn)

        return apply

    patches: list[tuple[type, str, Any, Any]] = [
        (modal.App, "function", raw_function, recording_function)
    ]
    for name, raw in vars(modal.Image).items():
        if not name.startswith("_") and (isinstance(raw, staticmethod) or inspect.isfunction(raw)):
            patches.append((modal.Image, name, raw, recording_image_method(name, raw)))
    patches.append(
        (
            modal.Volume,
            "from_name",
            vars(modal.Volume)["from_name"],
            recording_lookup("volume", vars(modal.Volume)["from_name"]),
        )
    )
    patches.append(
        (
            modal.Secret,
            "from_name",
            vars(modal.Secret)["from_name"],
            recording_lookup("secret", vars(modal.Secret)["from_name"]),
        )
    )
    try:
        for cls, name, _, new in patches:
            setattr(cls, name, new)
        namespace = build()
    finally:
        for cls, name, raw, _ in reversed(patches):
            setattr(cls, name, raw)
    apps = [v for v in namespace.values() if isinstance(v, modal.App)]
    assert len(apps) == 1, f"expected one modal.App, found {len(apps)}"
    assert sorted(functions) == sorted(apps[0].registered_functions), (
        sorted(functions),
        sorted(apps[0].registered_functions),
    )
    return {"app": apps[0].name, "functions": functions}


def prompt_digest(prompt) -> dict:
    """A prompt as its length, its element types and the sha256 of its JSON."""
    return {
        "len": len(prompt),
        "types": sorted({type(t).__name__ for t in prompt}),
        "sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
    }


def batch_digest(prompts) -> dict:
    """A batch of prompts as its size and the sha256 of its JSON (order and every id count)."""
    return {
        "prompts": len(prompts),
        "sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
    }


def synthetic_prompt_sets(sets: int, size: int, length: int, salt: int) -> list:
    """`sets` x `size` distinct prompts of exactly `length` ids, all below 1,000."""
    ramp = [(7 * j + salt) % 997 for j in range(2 * length)]
    return [
        [[salt, s, i, *ramp[i % length : i % length + length - 3]] for i in range(size)]
        for s in range(sets)
    ]


def completions_reply(body: dict) -> dict:
    """What vLLM's /v1/completions answers to `body`, with prompt logprobs when asked for."""
    prompt = body["prompt"]
    choice = {"index": 0, "text": "", "finish_reason": "length"}
    if body.get("prompt_logprobs") is not None:
        choice["prompt_logprobs"] = [None] + [
            {
                str(t): {"logprob": -(t % 13 + 1) / 8, "rank": 2},
                str(t + 1): {"logprob": -1 / 64, "rank": 1},
            }
            for t in prompt[1:]
        ]
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": body.get("max_tokens")},
    }


class _Reply:
    status = 200

    def __init__(self, body: dict):
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)


def recording_client_session(log: list):
    """A stand-in for aiohttp.ClientSession that logs how it was opened and every POST."""

    class RecordingSession:
        def __init__(self, *args, connector=None, timeout=None, **kwargs):
            self._connector = connector
            log.append(
                {
                    "session": {
                        "positional_args": len(args),
                        "connector": type(connector).__name__,
                        "connector_limit": getattr(connector, "limit", None),
                        "timeout_total": getattr(timeout, "total", None),
                        "other": sorted(kwargs),
                    }
                }
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            if self._connector is not None:
                closing = self._connector.close()
                if inspect.isawaitable(closing):
                    await closing

        def post(self, url, **kwargs):
            body = kwargs.pop("json")
            log.append(
                {
                    "post": {
                        "url": url,
                        "other": sorted(kwargs),
                        "body": {
                            k: prompt_digest(v) if k == "prompt" else v for k, v in body.items()
                        },
                    }
                }
            )
            return _Reply(completions_reply(body))

    return RecordingSession


def describe_run_kwargs(kwargs: dict, root: str) -> dict:
    """subprocess.run's keyword arguments as JSON."""
    import os
    import subprocess

    names = {subprocess.PIPE: "PIPE", subprocess.STDOUT: "STDOUT", subprocess.DEVNULL: "DEVNULL"}
    out = {}
    for key, value in sorted(kwargs.items()):
        if key in ("stdin", "stdout", "stderr") and isinstance(value, int):
            out[key] = f"subprocess.{names[value]}"
        elif hasattr(value, "write") and hasattr(value, "name"):
            out[key] = {"file": str(value.name).removeprefix(root), "mode": value.mode}
        elif key == "env":
            out[key] = {
                "inherits_os_environ": all(value.get(k) == v for k, v in os.environ.items()),
                "sets": {k: v for k, v in sorted(value.items()) if os.environ.get(k) != v},
            }
        else:
            out[key] = value
    return out


def identity_view() -> dict:
    """The identity snapshot of the current code; the adapter later tasks may edit."""
    from fp4bench import lib_bench, prep, prompts, runner, server, settings
    from fp4bench.core.types import LinearBackend
    from fp4bench.decode_step import decode_block
    from fp4bench.schedule import round_order
    from fp4bench.studies import expb, expc, model, smoke
    from fp4bench.studies.base import Prompts
    from fp4bench.studies.registry import STUDIES

    config_constants = (
        "M1_INPUT_LEN",
        "M1_N1",
        "M1_N2",
        "M1_SETS",
        "M1_SET_SIZE",
        "M1_MEAN_CONTEXT",
        "M2_MAX_TOKENS",
        "NLL_PROMPTS",
        "NLL_LEN",
        "NLL_MAX_DEGRADATION",
        "SEED",
        "DELTA",
        "CI_LEVEL",
        "PRIMARY_CONCURRENCIES",
        "BOOTSTRAP_RESAMPLES",
        "BOOTSTRAP_SEED",
        "BOOTSTRAP_LEVEL",
        "HBM_PEAK_BYTES_S",
        "TREATMENTS",
        "KERNEL_SCAN_TREATMENTS",
        "SMOKE_ONLY_TREATMENTS",
        "KERNEL_SELECTION_C",
        "KERNEL_SCAN_MAX_NLL_DIFF",
        "KERNEL_TIE_MARGIN",
        "KERNEL_TIEBREAK_C",
        "NVFP4_KERNEL_OF_BACKEND",
        "DEFAULT_NVFP4_KERNEL",
        "MXFP4_KERNEL",
        "EXPECTED_VERSIONS",
        "VLLM_IMAGE",
        "VLLM_COMMIT",
        "LLMCOMPRESSOR_VERSION",
        "LIB_REPO",
        "LIB_COMMIT",
        "SRC_MODEL_ID",
        "SRC_MODEL_REVISION",
        "NVX_MODEL_ID",
        "NVX_MODEL_REVISION",
        "SHAREGPT_REPO",
        "SHAREGPT_REVISION",
        "SHAREGPT_FILE",
        "SERVED_NAME",
        "HOST",
        "PORT",
        "BASE_URL",
        "EXPB_MIN_KV_TOKENS",
        "EXPC_HF_OVERRIDES",
        "C_CELLS",
    )
    path_constants = (
        "HF_CACHE",
        "VLLM_CACHE",
        "DATA_DIR",
        "RESULTS_DIR",
        "LOCAL_DIR",
        "LIB_DIR",
        "BF16_MODEL_DIR",
        "MX_MODEL_DIR",
        "NV_MODEL_DIR",
        "NVX_MODEL_DIR",
        "VOLUME_DIRS",
        "MODEL_DIRS",
        "NV_PROVENANCE_FILE",
        "SHAREGPT_PATH",
        "M1_PROMPTS_PATH",
        "NLL_PROMPTS_PATH",
        "C_PROMPTS_PATH",
        "BF16_REF_NLL_PATH",
    )
    scan = STUDIES["smoke"].kernel_scan
    assert scan is not None
    elsewhere: dict[str, object] = {
        "KERNEL_SCAN_TREATMENTS": scan.treatments,
        "KERNEL_SELECTION_C": scan.selection_c,
        "KERNEL_SCAN_MAX_NLL_DIFF": scan.max_nll_diff,
        "KERNEL_TIE_MARGIN": scan.tie_margin,
        "KERNEL_TIEBREAK_C": scan.tiebreak_c,
        "SMOKE_ONLY_TREATMENTS": smoke.SMOKE_ONLY_TREATMENTS,
        "PRIMARY_CONCURRENCIES": STUDIES["full"].primary_batches,
        "TREATMENTS": STUDIES["full"].treatments,
        "EXPB_MIN_KV_TOKENS": expb.EXPB.min_kv_tokens(),
        "EXPC_HF_OVERRIDES": STUDIES["expc"].server.hf_overrides,
        "C_CELLS": expc.C_CELLS,
        "NVFP4_KERNEL_OF_BACKEND": {b: b.kernel for b in LinearBackend},
        "VOLUME_DIRS": {t: spec.volume_dir for t, spec in model.TREATMENTS.items()},
        "MODEL_DIRS": {t: spec.model_dir for t, spec in model.TREATMENTS.items()},
    }

    def named(name: str):
        if name in elsewhere:
            return elsewhere[name]
        return getattr(next(m for m in (settings, model) if hasattr(m, name)), name)

    constants = {name: named(name) for name in config_constants}
    constants |= {
        "prompts.CELL_SEED_OFFSET": prompts.CELL_SEED_OFFSET,
        "lib_bench.INVALID_FLAGS": lib_bench.INVALID_FLAGS,
        "lib_bench.VALID_AGGREGATE_SOURCE": lib_bench.VALID_AGGREGATE_SOURCE,
        "lib_bench.MIN_FILL": lib_bench.MIN_FILL,
        "server.AUTOTUNE_CACHE_ENV": server.AUTOTUNE_CACHE_ENV,
    }
    paths = {name: named(name) for name in path_constants}
    paths["prep.FETCH_FILE"] = prep.FETCH_FILE

    class Started(Exception):  # noqa: N818 - a stop signal, not an error
        pass

    def no_sampler(_path):
        return contextlib.nullcontext()

    def served(study, treatment) -> dict:
        seen = {}

        def fake_server(model_dir, log_path, args=(), env=None, **kwargs):
            real = server.VllmServer(model_dir, log_path, args=args, env=env, **kwargs)
            seen.update(argv=list(real.cmd), env_keys=sorted(real.env), model_dir=model_dir)
            raise Started

        cell_prompts = (
            {cell: [] for cell in study.cells} if study.prompts is Prompts.CELLS else None
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(runner, "GpuSampler", no_sampler),
            mock.patch.object(runner, "VllmServer", fake_server),
            contextlib.suppress(Started),
        ):
            runner.run_server_session(
                Path(tmp), 0, treatment, study, [], [], "identity", cell_prompts=cell_prompts
            )
        spec = model.TREATMENTS[treatment]
        return {
            **seen,
            "volume_dir": spec.volume_dir,
            "checkpoint_kind": spec.checkpoint,
            "expected_linear_kernel": spec.linear_kernel,
            "expected_act_quant_fusion": spec.act_quant_fusion,
        }

    def m1_wave_sequence(reps: int) -> list:
        """(prompt set, max_tokens) of every wave of one M1 block, in order."""
        calls = []

        def wave(batch, max_tokens):
            calls.append([batch[0][0], max_tokens])
            return 1.0 + max_tokens / 1000

        decode_block(
            wave, [[[s]] for s in range(reps + 1)], 1, settings.M1_N1, settings.M1_N2, reps
        )
        return calls

    def m2_argv(c: int, duration_s: int) -> list:
        argv = lib_bench.lib_command(c, duration_s, "{output}")
        return ["{python}" if a == sys.executable else a for a in argv]

    from fp4bench.telemetry import COUNTER_NAMES

    m1_sets = synthetic_prompt_sets(
        settings.M1_SETS, settings.M1_SET_SIZE, settings.M1_INPUT_LEN, salt=1
    )
    nll_set = synthetic_prompt_sets(1, settings.NLL_PROMPTS, settings.NLL_LEN, salt=2)[0]
    cell_sets = {
        tuple(cell): synthetic_prompt_sets(settings.M1_SETS, *cell, salt=3 + k)
        for k, cell in enumerate(expc.C_CELLS)
    }

    def session_trace(study, treatment) -> dict:
        calls: list[dict] = []
        counts: dict[str, int] = {}
        seen = {}

        def call(name: str, **fields) -> int:
            """Record a call; how many calls of this name there have been, this one included."""
            calls.append({"call": name, **fields})
            counts[name] = counts.get(name, 0) + 1
            return counts[name]

        def run_wave(base_url, model, prompts, max_tokens):
            k = call(
                "run_wave",
                base_url=base_url,
                model=model,
                **batch_digest(prompts),
                max_tokens=max_tokens,
            )
            return 1.0 + max_tokens / 1000 + k / 1e6

        def prompt_nlls(base_url, model, prompts):
            call("prompt_nlls", base_url=base_url, model=model, **batch_digest(prompts))
            return [1.5 + i / 64 for i in range(len(prompts))]

        def run_lib_decode(c, duration_s, output_path, log_path):
            call(
                "run_lib_decode",
                c=c,
                duration_s=duration_s,
                output_path=str(output_path),
                log_path=str(log_path),
            )
            return {
                "valid": True,
                "invalid_reasons": [],
                "failure_reason": "",
                "aggregate_tps": 50.0 * c,
                "aggregate_source": "openai_continuous_usage",
                "measurement_seconds": float(duration_s),
                "effective_concurrency": float(c),
                "avg_running_reqs": float(c),
                "itl_p50_ms": 12.5,
                "tps_per_user_p50": 80.0,
                "ttft_p50_ms": 250.0,
                "server_gen_throughput": 49.0 * c,
            }

        def read_counters():
            k = call("read_counters")
            return {name: (i + 1) * k * k for i, name in enumerate(COUNTER_NAMES)}

        def read_preemptions(base_url):
            return float(call("read_preemptions", base_url=base_url) // 3)

        def clock():
            return 1000.0 + call("clock") ** 3 / 64

        def parse_server_log(text):
            call("parse_server_log", text=text)
            return {
                "kv_cache_tokens": 2_000_000,
                "autotune_ran": True,
                "autotune_cache_loaded": False,
                "autotune_cache_files": [],
            }

        def gpu_info():
            call("gpu_info")
            return {"uuid": "GPU-identity"}

        class Proc:
            returncode = None

            def poll(self):
                call("server.proc.poll")

        class Server:
            def __init__(self, model_dir, log_path, args=(), env=None, **kwargs):
                call(
                    "VllmServer",
                    model_dir=model_dir,
                    log_path=str(log_path),
                    env=env,
                    other=sorted(kwargs),
                )
                self.cmd = server.VllmServer(model_dir, log_path, args=args, env=env, **kwargs).cmd
                self.log_path, self.proc = Path(log_path), Proc()
                seen["session_id"] = self.log_path.stem

            def __enter__(self):
                call("VllmServer.__enter__")
                return self

            def __exit__(self, exc_type, *exc):
                call("VllmServer.__exit__", exception=exc_type and exc_type.__name__)

            def log_text(self):
                call("VllmServer.log_text")
                return "<the server log>"

        class Sampler:
            def __init__(self, csv_path):
                call("GpuSampler", csv_path=str(csv_path))

            def __enter__(self):
                call("GpuSampler.__enter__")
                return self

            def __exit__(self, exc_type, *exc):
                call("GpuSampler.__exit__", exception=exc_type and exc_type.__name__)

        cell_prompts = (
            {cell: cell_sets[tuple(cell)] for cell in study.cells}
            if study.prompts is Prompts.CELLS
            else None
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.multiple(
                runner,
                VllmServer=Server,
                GpuSampler=Sampler,
                parse_server_log=parse_server_log,
                read_preemptions=read_preemptions,
                read_counters=read_counters,
                prompt_nlls=prompt_nlls,
                run_wave=run_wave,
                run_lib_decode=run_lib_decode,
                gpu_info=gpu_info,
                time=SimpleNamespace(time=clock),
            ),
        ):
            runner.run_server_session(
                Path(tmp),
                0,
                treatment,
                study,
                m1_sets,
                nll_set,
                "identity-start",
                cell_prompts=cell_prompts,
            )
            files = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*"))
            rows = {
                name: [json.loads(line) for line in path.read_text().splitlines()]
                for name in ("m1", "m2", "servers")
                if (path := Path(tmp) / f"{name}.jsonl").exists()
            }
            trace = json.dumps(
                {
                    "calls": calls,
                    "files": files,
                    "rows": rows,
                    "row_keys": {
                        name: sorted({tuple(row) for row in rs}) for name, rs in rows.items()
                    },
                }
            )
        session_id = seen["session_id"]
        assert re.fullmatch(rf"r0_{treatment}-[0-9a-f]{{8}}", session_id), session_id
        return json.loads(
            trace.replace(tmp, "{run_dir}").replace(session_id, f"r0_{treatment}-{{id}}")
        )

    traced = {"expc": ("MX", "NV")}
    protocols = {}
    for name, study in STUDIES.items():
        protocols[name] = {
            "protocol": study.to_protocol_dict(),
            "round_order": [round_order(r, study.treatments) for r in range(study.rounds)],
            "cells": list(study.cell_list()),
            "cells_are_the_legacy_expansion": study.prompts is Prompts.M1,
            "m1_reps": study.m1_reps,
            "m1_wave_sequence": m1_wave_sequence(study.m1_reps),
            "m2_duration_s": study.m2_duration_s,
            "m2_argv": {str(c): m2_argv(c, study.m2_duration_s) for c in study.m2_batches()},
            "min_kv_tokens": study.min_kv_tokens(),
            "m1_largest_wave": study.largest_wave(),
            "sessions": {t: served(study, t) for t in study.treatments},
            "session_trace": {
                t: session_trace(study, t)
                for t in study.treatments
                if t in traced.get(name, study.treatments[:1])
            },
        }
        if study.extension_rounds is not None:
            protocols[name]["round_order_extended_to_10"] = [
                round_order(r, study.treatments) for r in range(study.extension_rounds)
            ]

    sharegpt = synthetic_sharegpt_file()
    calls = []
    with tempfile.TemporaryDirectory() as tmp:

        def rel(path) -> str:
            return str(path).removeprefix(tmp)

        def snapshot_download(repo_id, revision=None, *, local_dir, ignore_patterns=None):
            calls.append(["snapshot_download", repo_id, revision, rel(local_dir), ignore_patterns])
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return local_dir

        def hf_hub_download(repo_id, filename, repo_type=None, revision=None, *, local_dir):
            calls.append(
                ["hf_hub_download", repo_id, filename, repo_type, revision, rel(local_dir)]
            )
            target = Path(local_dir) / filename
            target.write_bytes(sharegpt)
            return str(target)

        def from_pretrained(path):
            calls.append(["AutoTokenizer.from_pretrained", rel(path)])
            return FakeTokenizer()

        moved = (
            (settings, "DATA_DIR"),
            (model, "BF16_MODEL_DIR"),
            (settings, "SHAREGPT_PATH"),
            (settings, "M1_PROMPTS_PATH"),
            (settings, "NLL_PROMPTS_PATH"),
            (settings, "C_PROMPTS_PATH"),
        )
        hub = SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=hf_hub_download)
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(sys.modules, {"huggingface_hub": hub, "transformers": transformers})
            )
            for module, name in moved:
                stack.enter_context(mock.patch.object(module, name, tmp + getattr(module, name)))
            Path(settings.DATA_DIR).mkdir(parents=True)
            prepare = prep.prepare_all()
            prepare_expc = prep.prepare_expc()
    prompt_files = {
        "sharegpt_sha256": hashlib.sha256(sharegpt).hexdigest(),
        "calls": calls,
        "prepare": prepare,
        "prepare_expc": prepare_expc,
    }

    default = canonical(STUDIES["full"].to_protocol_dict())
    committed = {}
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if (p / "manifests.jsonl").exists()):
        lines = [json.loads(x) for x in (run_dir / "manifests.jsonl").read_text().splitlines()]
        rows = [json.loads(x) for x in (run_dir / "servers.jsonl").read_text().splitlines()]
        recorded = {**default, **lines[-1]["protocol"]}
        matches = [
            name
            for name, study in STUDIES.items()
            if all(
                v == recorded[k]
                for k, v in canonical(study.to_protocol_dict()).items()
                if k != "rounds"
            )
        ]
        served_rows = {}
        for row in rows:
            if "server_argv" in row:
                served_rows.setdefault(row["treatment"], []).append(row)
        order = {}
        for row in rows:
            order.setdefault((row["start_id"], row["round"]), []).append(row["treatment"])
        committed[run_dir.name] = {
            "protocol": matches,
            "rounds": lines[-1]["protocol"]["rounds"],
            "session_order": [[r, ts] for (_, r), ts in order.items()],
            **{
                key: sorted({line["inputs"][key] for line in lines if key in line["inputs"]})
                for key in ("m1_prompts_sha256", "nll_prompts_sha256", "c_prompts_sha256")
            },
            "server_argv": {
                t: sorted({tuple(r["server_argv"]) for r in rs}) for t, rs in served_rows.items()
            },
            "server_env_keys": {
                t: sorted({tuple(sorted(r["server_env"])) for r in rs})
                for t, rs in served_rows.items()
            },
        }

    import aiohttp

    from fp4bench import manifest, sanity, wave

    m1_prompts = [
        [(17 * i + 5 * j) % 151_643 for j in range(n)] for i, n in enumerate((1024, 360, 4096))
    ]
    nll_prompts = [[(11 * i + 3 * j) % 151_643 for j in range(512)] for i in range(2)]
    requests = {}
    for name, call in (
        (
            "m1_wave_n1",
            lambda: wave.run_wave(
                settings.BASE_URL, settings.SERVED_NAME, m1_prompts, settings.M1_N1
            ),
        ),
        (
            "m1_wave_n2",
            lambda: wave.run_wave(
                settings.BASE_URL, settings.SERVED_NAME, m1_prompts, settings.M1_N2
            ),
        ),
        ("nll", lambda: sanity.prompt_nlls(settings.BASE_URL, settings.SERVED_NAME, nll_prompts)),
    ):
        log: list = []
        with mock.patch.object(aiohttp, "ClientSession", recording_client_session(log)):
            result = call()
        requests[name] = {"calls": log, "returned": None if name.startswith("m1") else result}

    import subprocess

    def m2_subprocess(c: int, duration_s: int) -> dict:
        seen = {}
        with tempfile.TemporaryDirectory() as tmp:

            def fake_run(cmd, **kwargs):
                out = Path(cmd[cmd.index("--output") + 1])
                seen.update(
                    argv=["{python}" if a == sys.executable else a.removeprefix(tmp) for a in cmd],
                    kwargs=describe_run_kwargs(kwargs, tmp),
                    stale_output_present=out.exists(),
                )
                out.write_text(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "concurrency": c,
                                    "context_tokens": 0,
                                    "aggregate_tps": 50.0 * c,
                                    "aggregate_source": "openai_continuous_usage",
                                    "measurement_seconds": float(duration_s),
                                    "effective_concurrency": float(c),
                                    "avg_running_reqs": float(c),
                                    "inter_token_latency_p50": 0.0125,
                                    "output_tps_per_user_p50": 80.0,
                                    "ttft_p50": 0.25,
                                    "server_gen_throughput": 49.0 * c,
                                    "num_errors": 0,
                                }
                            ]
                        }
                    )
                )
                return subprocess.CompletedProcess(cmd, 0)

            output = Path(tmp) / "m2_raw" / f"session_c{c}.json"
            output.parent.mkdir(parents=True)
            output.write_text("stale")
            with mock.patch.object(subprocess, "run", fake_run):
                seen["returned"] = lib_bench.run_lib_decode(
                    c, duration_s, output, Path(tmp) / "lib_logs" / "session.log"
                )
        return seen

    m2 = {f"c{c}_{d}s": m2_subprocess(c, d) for c, d in ((1, 10), (32, 30))}

    pinned = {
        "gpu": {
            "name": "NVIDIA B200",
            "uuid": "GPU-0",
            "compute_cap": "10.0",
            "driver_version": "580.95.05",
            "memory.total": "183359 MiB",
            "power.limit": "1000.00 W",
        },
        "packages": dict(settings.EXPECTED_VERSIONS),
        "vllm_build_commit": settings.VLLM_COMMIT,
        "llm_inference_bench_commit": settings.LIB_COMMIT,
    }

    def varied(gpu=None, packages=None, **top) -> dict:
        return {
            **pinned,
            "gpu": {**pinned["gpu"], **(gpu or {})},
            "packages": {**pinned["packages"], **(packages or {})},
            **top,
        }

    versions = settings.EXPECTED_VERSIONS
    manifests = {
        "as_pinned": pinned,
        "local_version_suffixes": varied(packages={p: f"{v}+cu130" for p, v in versions.items()}),
        "post_release": varied(packages={"vllm": f"{versions['vllm']}.post1"}),
        "prefix_only": varied(packages={"torch": versions["torch"].rsplit(".", 1)[0]}),
        "dev_build": varied(
            packages={"flashinfer-python": f"{versions['flashinfer-python']}.dev0"}
        ),
        "package_missing": {
            **pinned,
            "packages": {p: v for p, v in versions.items() if p != "flashinfer-cubin"},
        },
        "package_none": varied(packages={"torch": None}),
        "gb200": varied(gpu={"name": "NVIDIA GB200"}),
        "b300": varied(gpu={"name": "NVIDIA B300 SXM6 AC", "compute_cap": "10.3"}),
        "h100": varied(gpu={"name": "NVIDIA H100 80GB HBM3", "compute_cap": "9.0"}),
        "compute_cap_10": varied(gpu={"compute_cap": "10"}),
        "name_lowercase": varied(gpu={"name": "nvidia b200"}),
        "gpu_query_failed": {**pinned, "gpu": {"error": "FileNotFoundError: nvidia-smi"}},
        "gpu_missing": {k: v for k, v in pinned.items() if k != "gpu"},
        "build_commit_missing": varied(vllm_build_commit=None),
        "lib_commit_error": varied(llm_inference_bench_commit="error: RuntimeError: no git"),
        "empty": {},
    }
    smi = []

    def fake_smi(cmd, **kwargs):
        smi.append({"argv": list(cmd), "kwargs": describe_run_kwargs(kwargs, "")})
        return subprocess.CompletedProcess(
            cmd, 0, stdout="NVIDIA B200, GPU-0, 10.0, 580.95.05, 183359 MiB, 1000.00 W\n"
        )

    with mock.patch.object(subprocess, "run", fake_smi):
        gpu = manifest.gpu_info()
    environment_check = {
        "verify_manifest": {name: manifest.verify_manifest(m) for name, m in manifests.items()},
        "gpu_info": {"calls": smi, "returned": gpu},
        "packages_recorded": list(manifest.PACKAGES),
    }

    modal_app = record_modal(
        lambda: runpy.run_path(
            str(REPO / "fp4bench" / "executors" / "modal.py"), run_name="modal_app_identity"
        )
    )
    return canonical(
        {
            "constants": constants,
            "paths": paths,
            "protocols": protocols,
            "prompt_files": prompt_files,
            "committed_runs": committed,
            "modal": modal_app,
            "requests": requests,
            "m2_subprocess": m2,
            "environment_check": environment_check,
        }
    )


def _kept_in_order(old: list, new: list) -> list[int] | None:
    """The indices of `new` not matched by the items of `old`, in order, or None if one is lost."""
    unmatched, i = [], 0
    for j, item in enumerate(new):
        if i < len(old) and dumps(item) == dumps(old[i]):
            i += 1
        else:
            unmatched.append(j)
    return unmatched if i == len(old) else None


def additive_diff(old, new, path: str = "$") -> tuple[list[str], list[str]]:
    """(what `new` adds to `old`, what it removes or changes), as JSON paths: a dict may gain keys
    and a list items, as long as the old items stay in order. 1 is not 1.0, True is not 1."""
    if type(old) is not type(new):
        return [], [
            f"{path}: {type(old).__name__} {old!r:.200} -> {type(new).__name__} {new!r:.200}"
        ]
    if isinstance(old, dict):
        added, changed = [], []
        for key in sorted(old.keys() | new.keys()):
            if key not in new:
                changed.append(f"{path}.{key}: removed")
            elif key not in old:
                added.append(f"{path}.{key}")
            else:
                more, broken = additive_diff(old[key], new[key], f"{path}.{key}")
                added += more
                changed += broken
        return added, changed
    if isinstance(old, list):
        unmatched = _kept_in_order(old, new)
        if unmatched is None:
            return [], [f"{path}: {old!r:.200} is not kept, in order, in {new!r:.200}"]
        return [f"{path}[{j}]" for j in unmatched], []
    return ([], []) if old == new else ([], [f"{path}: {old!r:.200} -> {new!r:.200}"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.golden.make_identity")
    parser.add_argument(
        "--check-additive",
        action="store_true",
        help="list what the regenerated view adds to identity.json; write nothing",
    )
    args = parser.parse_args(argv)
    view = identity_view()
    old = json.loads(IDENTITY_PATH.read_text()) if IDENTITY_PATH.exists() else {}
    added, changed = additive_diff(old, view)
    for line in added:
        print(f"added {line}")
    for line in changed:
        print(f"changed {line}")
    if changed:
        raise SystemExit(
            f"the regenerated view is not additive: it removes or changes "
            f"{len(changed)} entries of {IDENTITY_PATH.name}"
        )
    if args.check_additive:
        print(f"additive: the regenerated view adds {len(added)} entries to {IDENTITY_PATH.name}")
        return
    IDENTITY_PATH.write_text(dumps(view))
    print(f"wrote {IDENTITY_PATH} ({IDENTITY_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
