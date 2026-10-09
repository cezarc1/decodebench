import argparse
import contextlib
import gzip
import hashlib
import io
import itertools
import json
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override

import pytest

from fp4bench.core.types import Format
from tests import GOLDEN_DIR, REPO
from tests.golden.view import dumps, first_difference

PINS_PATH = GOLDEN_DIR / "pins.json"
MICROBENCH_RESULTS = (
    "data/kernels/microbench-1/results.json",
    "data/kernels/microbench-1/replicate/results.json",
)
PROFILES = "data/kernels/profile-1/profiles.json"
FIXED_NOW = datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC)
FLOAT = "<float>"


class FixedDatetime(datetime):
    """datetime whose now() is FIXED_NOW."""

    @classmethod
    @override
    def now(cls, tz=None):
        return FIXED_NOW.astimezone(tz) if tz is not None else FIXED_NOW.replace(tzinfo=None)


def ordered(value: Any) -> str:
    """The JSON text a pin is compared by: key order and types count, 1 is not 1.0."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def text_pin(text: str) -> dict:
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)}


def normalised(value: Any, tmp: str) -> Any:
    """`value` as JSON, with the temporary directory (as given and resolved) written {tmp}."""
    text = ordered(value)
    for prefix in sorted({str(Path(tmp).resolve()), tmp}, key=len, reverse=True):
        text = text.replace(prefix, "{tmp}")
    return json.loads(text)


def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# --- kernels: microbench and profiling ---------------------------------------------------------


def microbench_pins() -> dict:
    from fp4bench import microbench as mb
    from fp4bench.core.kernels import ActImpl, GemmImpl

    pins = {
        f"microbench.summarize({path})": text_pin(
            mb.summarize(json.loads((REPO / path).read_text()))
        )
        for path in MICROBENCH_RESULTS
    }
    n, k = mb.SHAPE_NK["gate_up_proj"]
    gemm: mb.CellExtras = {
        "autotuned": True,
        "tuned_at_M": 512,
        "library": "flashinfer",
        "order": 3,
    }
    rotation: mb.CellExtras = {
        "copies": 41,
        "calls_per_sample": 205,
        "bytes_per_copy": 1_234_567,
        "bytes_per_graph": 205 * 1_234_567,
    }
    act: mb.CellExtras = {**rotation, "rotation_capped": False, "width_in": 51200}
    # The pinned row's choice is older than AutotunerChoice: no runner or fallback, an int tactic.
    choice: Any = [{"op": "fp4_gemm", "tactic": 7}]
    cells = {
        "gemm_timed": mb.make_cell(
            mb.GemmCase(
                GemmImpl.VLLM_CUTEDSL,
                mb.NVFP4,
                "gate_up_proj",
                8,
                n,
                k,
                mb.gemm_bytes(mb.NVFP4, 8, n, k)["total"],
                mb.gemm_flops(8, n, k),
            ),
            [41.5, 40.25, 43.0, 40.75, 42.0],
            rel_err=0.0123,
            extras={
                **gemm,
                **rotation,
                "autotuner_choice": choice,
                "kernels": ["kernel_a", "kernel_b"],
            },
        ),
        "gemm_failed": mb.make_cell(
            mb.GemmCase(
                GemmImpl.CUDNN,
                mb.MXFP4,
                "gate_up_proj",
                8,
                n,
                k,
                mb.gemm_bytes(mb.MXFP4, 8, n, k)["total"],
                mb.gemm_flops(8, n, k),
            ),
            error="autotune failed: RuntimeError: no tactic",
            extras=gemm,
        ),
        "act_timed": mb.make_cell(
            mb.ActCase(
                ActImpl.SILU_MUL_NV_FUSED,
                mb.NVFP4,
                "down_proj_input",
                32,
                25600,
                mb.silu_quant_bytes(mb.NVFP4, 32, 25600)["total"],
            ),
            [5.0, 4.5, 6.0],
            rel_err=0.25,
            extras={**act, "kernels_error": "RuntimeError: profiler"},
        ),
        "act_failed": mb.make_cell(
            mb.ActCase(
                ActImpl.MX_QUANT_CUTEDSL,
                mb.MXFP4,
                "K=5120",
                1,
                5120,
                mb.act_quant_bytes(mb.MXFP4, 1, 5120)["total"],
            ),
            error="ValueError: bad shape",
            extras=act,
        ),
    }
    return {**pins, "microbench.make_cell": cells}


def _x(cat: str, name: str, ts: float, dur: float, corr: int | None = None) -> dict:
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": 0,
        "tid": 7,
        "ts": ts,
        "dur": dur,
        "args": {} if corr is None else {"correlation": corr},
    }


GEMM_KERNEL = (
    "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100"
    "BlockScaledPersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000"
)
SILU_QUANT_KERNEL = (
    "void vllm::silu_mul_cvt_fp16_to_fp4<__nv_bfloat16, false>(int, int, int, "
    "__nv_bfloat16 const*, float const*, unsigned int*, unsigned int*)"
)
QUANT_KERNEL = (
    "void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false, false>(int, int, int, int, "
    "__nv_bfloat16 const*, float const*, unsigned int*, unsigned int*)"
)


def graph_trace(gemm_us: float = 20.0) -> dict:
    """Two decode steps of CUDA-graph replays and eager kernels, a memcpy, a memset, a gap."""
    step = "execute_context_0(0)_generation_2(2)"
    ev: list[dict] = [{"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "GPU 0"}}]
    for i, t0 in enumerate((0.0, 100.0)):
        corr = 100 * (i + 1)
        ev += [
            _x("cuda_runtime", "cudaGraphLaunch", t0 + 1, 2.0, corr),
            _x("cuda_runtime", "cudaLaunchKernel", t0 + 3, 1.0, corr + 1),
            _x("cuda_runtime", "cudaLaunchKernel", t0 + 4, 1.0, corr + 2),
            _x("user_annotation", step, t0 + 0.5, 50.0),
            _x("gpu_user_annotation", step, t0 + 10, 42.0 + 8 * i),
            _x("kernel", GEMM_KERNEL, t0 + 10, gemm_us + 2 * i, corr),
            _x("kernel", SILU_QUANT_KERNEL, t0 + 30 + 2 * i, 4.0, corr),
            _x(
                "kernel",
                "fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16",
                t0 + 34 + 2 * i,
                8.0,
                corr,
            ),
            _x("kernel", "triton_red_fused_fused_add_rms_norm_3", t0 + 42 + i, 2.0 + i, corr),
            _x(
                "kernel" if i == 0 else "Kernel",
                "nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT",
                t0 + 44 + 8 * i,
                6.0,
                corr + 1,
            ),
            _x("kernel", "_gumbel_sample_kernel", t0 + 50 + 8 * i, 2.0, corr + 2),
            _x("kernel", "_post_update_kernel", t0 + 52 + 8 * i, 1.0, corr + 2),
        ]
    ev += [
        _x("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", 5.0, 1.0, 103),
        _x("gpu_memset", "Memset (Device)", 6.0, 0.5, 104),
    ]
    return {"schemaVersion": 1, "traceEvents": ev}


def layer_trace(gemm_us: tuple[float, ...] = (2.0, 3.0, 10.0, 8.0)) -> dict:
    """Two graph replays of two decoder layers: activation quantization, then each FP4 GEMM."""
    ev = []
    for i, corr in enumerate((500, 600)):
        ev.append(_x("cuda_runtime", "cudaGraphLaunch", i * 1000.0, 1.0, corr))
        t = i * 1000.0 + 10
        for _ in range(2):
            for us in gemm_us:
                for name, dur in ((QUANT_KERNEL, 1.0), (GEMM_KERNEL, us)):
                    ev.append(_x("kernel", name, t, dur, corr))
                    t += dur + 1.0
    return {"traceEvents": ev}


def profiling_pins() -> dict:
    from fp4bench import profiling as prof

    report = json.loads((REPO / PROFILES).read_text())
    pins: dict[str, Any] = {
        f"profiling.render_summary_md({PROFILES})": text_pin(prof.render_summary_md(report))
    }
    pins["profiling.summarize_trace"] = {
        "graph_trace": prof.summarize_trace(graph_trace(), n_steps=2, top=3),
        "layer_trace": prof.summarize_trace(layer_trace(), n_steps=2),
    }
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        mp.setattr(prof, "datetime", FixedDatetime)
        out = Path(tmp)
        servers = []
        for config, treatment, c, trace in (
            ("A", "MX", 1, graph_trace()),
            ("A", "NV", 1, graph_trace(19.5)),
            ("B", "MX", 512, layer_trace()),
            ("B", "NV", 512, layer_trace((2.0, 3.5, 10.0, 8.5))),
        ):
            with gzip.open(out / prof.trace_name(config, treatment, c), "wt") as f:
                json.dump(trace, f)
            servers.append(
                {
                    "config": config,
                    "treatment": treatment,
                    "server_argv": ["vllm", "serve", f"/local/{treatment}"],
                    "facts": {"linear_kernels": ["K"]},
                    "concurrencies": [
                        {
                            "c": c,
                            "trace_file": prof.trace_name(config, treatment, c),
                            "delay_iterations": 16,
                            "max_iterations": 2,
                            "unprofiled_step_ms": 0.1 * c,
                        }
                    ],
                }
            )
        meta = {
            "environment": {"gpu": {"name": "NVIDIA B200"}},
            "staging": {"MX": "copied"},
            "run": {"started_utc": "2026-10-08T00:00:00+00:00"},
            "servers": servers,
            "errors": [
                {
                    "where": "B_NVnf",
                    "error": "RuntimeError('vllm serve exited')",
                    "traceback": "Traceback (most recent call last): ...",
                }
            ],
        }
        (out / prof.SESSIONS_FILE).write_text(json.dumps(meta))
        prof.write_report(out)
        pins["profiling.write_report: profiles.json"] = json.loads(
            (out / "profiles.json").read_text()
        )
        pins["profiling.write_report: summary.md"] = text_pin((out / "summary.md").read_text())
    return pins


# --- checkpoints: sanity, quantize, publish, prep -------------------------------------------


def checkpoint_pair(
    root: Path, mx_tensors=None, nv_tensors=None, mx_config=None, nv_config=None
) -> tuple[str, str]:
    """A header-only MX and NV checkpoint pair (tests/ct_checkpoints, as tests/test_sanity)."""
    from tests.ct_checkpoints import ct_tensors, qwen3_config, write_checkpoint

    mx_config = mx_config or qwen3_config(
        Format.MXFP4,
        torch_dtype="bfloat16",
        _name_or_path="Qwen/Qwen3-32B",
        transformers_version="4.57.0",
    )
    nv_config = nv_config or qwen3_config(Format.NVFP4, transformers_version="4.57.1")
    mx = write_checkpoint(
        root / "mx", ct_tensors(Format.MXFP4) if mx_tensors is None else mx_tensors, mx_config
    )
    nv = write_checkpoint(
        root / "nv", ct_tensors(Format.NVFP4) if nv_tensors is None else nv_tensors, nv_config
    )
    return str(mx), str(nv)


def checkpoint_report_pins() -> dict:
    from fp4bench import sanity
    from tests.ct_checkpoints import ct_tensors, qwen3_config, unquantize

    pins = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pins["sanity.checkpoint_report: a clean pair"] = normalised(
            sanity.checkpoint_report(*checkpoint_pair(root / "clean")), tmp
        )
        nv = {
            k: v
            for k, v in ct_tensors(Format.NVFP4).items()
            if not k.endswith("layers.3.mlp.down_proj.input_global_scale")
        }
        nv_config = qwen3_config(Format.MXFP4, rope_theta=500000, extra_key=[1, 2])
        report = sanity.checkpoint_report(
            *checkpoint_pair(
                root / "problems",
                mx_tensors=unquantize(ct_tensors(Format.MXFP4), "self_attn"),
                nv_tensors=nv,
                nv_config=nv_config,
            )
        )
        pins["sanity.checkpoint_report: a pair with problems"] = normalised(report, tmp)
    return pins


def quantize_pins() -> dict:
    """quantize() against tests/test_quantize's fake torch, transformers and llm-compressor."""
    from fp4bench import quantize, settings
    from fp4bench.core.types import Format
    from tests import test_quantize

    make_env = getattr(test_quantize.env, "__wrapped__")  # noqa: B009 - untyped on pytest fixtures
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        st = make_env(mp, Path(tmp))
        mp.setattr(quantize, "datetime", FixedDatetime)
        quantize.quantize(Format.NVFP4)
        reference = json.loads(st.ref.read_text())
        quantize.quantize(Format.MXFP4)
        provenance = {
            fmt: json.loads((d / settings.NV_PROVENANCE_FILE).read_text())
            for fmt, d in ((Format.NVFP4, st.nv), (Format.MXFP4, st.mx))
        }
    for prov in provenance.values():
        assert isinstance(prov["bf16_reference_nll"]["mean"], float)
        prov["bf16_reference_nll"]["mean"] = FLOAT
    assert all(isinstance(x, float) for x in [*reference["per_prompt"], reference["mean"]])
    reference["per_prompt"] = [FLOAT] * len(reference["per_prompt"])
    reference["mean"] = FLOAT
    return {
        "quantize.quantize: provenance (nvfp4, the first checkpoint)": normalised(
            provenance[Format.NVFP4], tmp
        ),
        "quantize.quantize: provenance (mxfp4, after nvfp4)": normalised(
            provenance[Format.MXFP4], tmp
        ),
        "quantize.quantize: bf16_reference_nll.json": normalised(reference, tmp),
    }


CARD_REPORT = {
    "mx_quantized_bytes": 18_000_000_000,
    "nv_quantized_bytes": 19_062_000_000,
    "nv_over_mx": 19_062_000_000 / 18_000_000_000,
    "expected_nv_over_mx": 4.5 / 4.25,
    "mx_problems": [],
    "nv_problems": [],
    "mx_quant_format": "mxfp4-pack-quantized",
    "nv_quant_format": "nvfp4-pack-quantized",
}


def card_provenance(fmt: Format) -> dict:
    from fp4bench import settings
    from fp4bench.studies import model

    return {
        "source_model": model.SRC_MODEL_ID,
        "source_revision": model.SRC_MODEL_REVISION,
        "format": fmt,
        "scheme": fmt.upper(),
        "llmcompressor_version": "0.14.0",
        "compressed_tensors_version": "0.19.0",
        "transformers_version": "5.17.0",
        "torch_version": "2.13.0+cu130",
        "recipe": {
            "modifier": "QuantizationModifier",
            "targets": "Linear",
            "scheme": fmt.upper(),
            "ignore": ["lm_head"],
        },
        "ignore": ["lm_head"],
        "calib_dataset": f"{settings.SHAREGPT_REPO}@{settings.SHAREGPT_REVISION}",
        "calib_samples": 256,
        "calib_tokens": 98765,
        "calib_max_len": 1024,
        "calib_seed": 3,
        "calib_sha256": "c" * 64,
        "quantized_modules": [f"model.layers.{i}.mlp.up_proj" for i in range(448)],
        "tokenizer_files": ["tokenizer.json"],
        "config_parity": {},
        "bf16_reference_nll": {
            "path": "/data/bf16_reference_nll.json",
            "mean": 1.23456,
            "computed_by_this_run": True,
            "nll_prompts_sha256": "d" * 64,
        },
        "utc": "2026-10-05T01:02:03+00:00",
    }


def model_card_pins() -> dict:
    from fp4bench import publish

    def card(kind: str, repo_url: str | None) -> str:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(publish, "REPO_URL", repo_url)
            return publish.model_card(
                kind,
                f"someone/Qwen3-32B-{kind.upper()}FP4",
                CARD_REPORT,
                card_provenance({"mx": Format.MXFP4, "nv": Format.NVFP4}[kind]),
            )

    return {
        "publish.model_card(mx)": card("mx", None),
        "publish.model_card(nv)": card("nv", None),
        "publish.model_card(nv), with REPO_URL": card("nv", "https://github.com/someone/fp4/"),
    }


def prompt_file_pins() -> dict:
    """prepare and prepare_expc as make_identity runs them: c_prompts.json, its cells as shapes;
    and fetch_checkpoint."""
    from fp4bench import prep, settings
    from fp4bench.studies import model
    from tests.golden.make_identity import FakeTokenizer, synthetic_sharegpt

    sharegpt = json.dumps(synthetic_sharegpt()).encode()
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:

        def snapshot_download(repo_id, revision=None, *, local_dir, ignore_patterns=None):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return local_dir

        def hf_hub_download(repo_id, filename, repo_type=None, revision=None, *, local_dir):
            target = Path(local_dir) / filename
            target.write_bytes(sharegpt)
            return str(target)

        mp.setitem(
            sys.modules,
            "huggingface_hub",
            SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=hf_hub_download),
        )
        mp.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer())
            ),
        )
        for module, name in (
            (settings, "DATA_DIR"),
            (model, "BF16_MODEL_DIR"),
            (settings, "SHAREGPT_PATH"),
            (settings, "M1_PROMPTS_PATH"),
            (settings, "NLL_PROMPTS_PATH"),
            (settings, "C_PROMPTS_PATH"),
        ):
            mp.setattr(module, name, tmp + getattr(module, name))
        Path(settings.DATA_DIR).mkdir(parents=True)
        prep.prepare_all()
        prep.prepare_expc()
        text = Path(settings.C_PROMPTS_PATH).read_text()
        record = json.loads(text)

        mp.setattr(prep, "datetime", FixedDatetime)
        mp.setattr(model, "MX_MODEL_DIR", tmp + model.MX_MODEL_DIR)
        fetched = prep.fetch_checkpoint("mx", "someone/Qwen3-32B-MXFP4", "a" * 40)
        fetch_file = json.loads((Path(model.MX_MODEL_DIR) / prep.FETCH_FILE).read_text())
        fetch = normalised({"returned": fetched, "written": fetch_file}, tmp)
    record["cells"] = {
        key: {"sets": len(sets), "prompts_per_set": len(sets[0]), "prompt_len": len(sets[0][0])}
        for key, sets in record["cells"].items()
    }
    return {
        "prep.prepare_expc: c_prompts.json": {"file": text_pin(text), "record": record},
        "prep.fetch_checkpoint": fetch,
    }


class _Stop(Exception):  # noqa: N818 - a stop signal, not an error
    """Ends run_experiment at its first commit, right after the manifest line."""


Session = Callable[..., None]
Counters = Callable[[], dict]


def no_counters() -> dict:
    return {}


def run_start(
    study_name: str, session: Session | None = None, read_counters: Counters = no_counters
) -> dict:
    """The rows run_experiment writes on a first start of `study_name`, on synthetic inputs, with
    the real collect_manifest (fake GPU, packages and git), collect_inputs, checkpoint_report (on
    a header-only pair) and staging. Without `session` it stops at its first commit, right after
    the manifest line; else `session` stands in for run_server_session."""
    from fp4bench import manifest, runner, settings
    from fp4bench.core.files import file_sha256
    from fp4bench.prep import FETCH_FILE
    from fp4bench.studies import model
    from fp4bench.studies.base import Prompts
    from fp4bench.studies.registry import STUDIES
    from tests.ct_checkpoints import ct_tensors, qwen3_config, write_checkpoint

    study = STUDIES[study_name]
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        root = Path(tmp)
        volume = root / "volume"
        mx, nv = checkpoint_pair(volume)
        nvx = write_checkpoint(volume / "nvx", ct_tensors(Format.NVFP4), qwen3_config(Format.NVFP4))
        volume_dirs = {"mx": Path(mx), "nv": Path(nv), "nvx": nvx}
        if study.require_published:
            for kind in ("mx", "nv"):
                (volume_dirs[kind] / FETCH_FILE).write_text(
                    json.dumps(
                        {
                            "repo_id": f"someone/Qwen3-32B-{kind.upper()}FP4",
                            "revision": kind * 20,
                            "utc": "2026-10-01T00:00:00+00:00",
                        }
                    )
                )
        data = root / "data"
        data.mkdir()
        m1, nll, c = data / "m1.json", data / "nll.json", data / "c.json"
        m1.write_text(json.dumps([[[1, 2, 3]]]))
        nll.write_text(json.dumps([[4, 5, 6], [7, 8, 9]]))
        (data / "bf16_ref.json").write_text(
            json.dumps(
                {
                    "per_prompt": [1.5, 2.25],
                    "mean": 1.875,
                    "source_revision": "r",
                    "nll_prompts_sha256": file_sha256(nll),
                }
            )
        )
        if study.prompts is Prompts.CELLS:
            c.write_text(
                json.dumps(
                    {
                        "cells": {
                            cell.prompt_file_key: [[[7] * cell[1]] * cell[0]] * (study.m1_reps + 1)
                            for cell in study.cells
                        },
                        "m1_prompts_sha256": file_sha256(m1),
                    }
                )
            )
        for name, path in (
            ("M1_PROMPTS_PATH", m1),
            ("NLL_PROMPTS_PATH", nll),
            ("C_PROMPTS_PATH", c),
            ("BF16_REF_NLL_PATH", data / "bf16_ref.json"),
        ):
            mp.setattr(settings, name, str(path))
        for name, kind in (
            ("MX_MODEL_DIR", "mx"),
            ("NV_MODEL_DIR", "nv"),
            ("NVX_MODEL_DIR", "nvx"),
        ):
            mp.setattr(model, name, str(volume_dirs[kind]))
        mp.setattr(
            model,
            "TREATMENTS",
            {
                t: replace(
                    spec,
                    volume_dir=str(volume_dirs[spec.checkpoint]),
                    model_dir=str(root / "local" / spec.checkpoint),
                )
                for t, spec in model.TREATMENTS.items()
            },
        )

        gpu = {
            "name": "NVIDIA B200",
            "uuid": "GPU-pins",
            "compute_cap": "10.0",
            "driver_version": "580.95.05",
            "memory.total": "183359 MiB",
            "power.limit": "1000.00 W",
        }
        mp.setattr(manifest, "gpu_info", lambda: dict(gpu))
        mp.setattr(manifest, "lib_commit", lambda: settings.LIB_COMMIT)
        mp.setattr(manifest, "datetime", FixedDatetime)
        mp.setattr(
            manifest.importlib.metadata,
            "version",
            lambda pkg: settings.EXPECTED_VERSIONS.get(pkg, f"{len(pkg)}.0"),
        )
        mp.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="13.0")))
        mp.setenv("VLLM_BUILD_COMMIT", settings.VLLM_COMMIT)
        mp.setenv(manifest.CODE_COMMIT_ENV, "c0de" * 10)
        mp.setenv(manifest.CODE_DIRTY_ENV, "0")

        ids = itertools.count(1)
        ticks = itertools.count(1)
        mp.setattr(
            runner, "uuid", SimpleNamespace(uuid4=lambda: uuid.UUID(bytes=bytes([next(ids)]) * 16))
        )
        mp.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(ticks) * 0.25))
        mp.setattr(runner, "datetime", FixedDatetime)
        mp.setattr(runner, "read_counters", read_counters)
        if session is not None:
            mp.setattr(runner, "run_server_session", session)

        def commit():
            if session is None:
                raise _Stop

        run_dir = root / "run"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                runner.run_experiment(run_dir, study, commit)
        except (_Stop, RuntimeError):
            pass
        written = {
            name: jsonl(run_dir / f"{name}.jsonl") for name in ("manifests", "errors", "servers")
        }
        shutil.rmtree(root / "local", ignore_errors=True)
    assert [line["problems"] for line in written["manifests"]] == [[]], written["manifests"]
    for row in written["errors"]:
        assert isinstance(row["traceback"], str) and row["traceback"].startswith("Traceback")
        row["traceback"] = "<traceback>"
    return normalised(written, tmp)


def scan_sessions_fail(run_dir, r, treatment, study, *args, **kwargs) -> None:
    """Kernel-scan sessions fail (non-fatal: recorded, and the run goes on); the others pass."""
    if study.kernel_scan is not None and treatment in study.kernel_scan.treatments:
        raise RuntimeError(f"{treatment} could not start")


def sessions_die(run_dir, r, treatment, *args, **kwargs) -> None:
    raise RuntimeError(f"vllm server died during session r{r}_{treatment}")


def counters_unreadable() -> dict:
    raise RuntimeError("DCGM counters unreadable")


def runner_pins() -> dict:
    pins = {}
    for name in ("smoke", "smoke-c", "full"):
        (line,) = run_start(name)["manifests"]
        pins[f"runner.run_experiment: manifests.jsonl line ({name})"] = line
    scan = run_start("smoke", scan_sessions_fail)
    died = run_start("full", sessions_die)
    preflight = run_start("smoke-c", sessions_die, counters_unreadable)
    pins["runner.run_experiment: errors.jsonl rows"] = {
        "smoke, the kernel-scan sessions fail": scan["errors"],
        "full, the first session dies": died["errors"],
        "smoke-c, the preflight counter read fails": preflight["errors"],
    }
    pins["runner.run_experiment: servers.jsonl rows of failed scan sessions (smoke)"] = scan[
        "servers"
    ]
    return pins


PIN_GROUPS: tuple[Callable[[], dict], ...] = (
    microbench_pins,
    profiling_pins,
    checkpoint_report_pins,
    runner_pins,
    quantize_pins,
    model_card_pins,
    prompt_file_pins,
)


def pins_view() -> dict:
    """The pins of the current code; the adapter later phases may edit."""
    view: dict = {}
    for group in PIN_GROUPS:
        pins = group()
        assert not view.keys() & pins.keys(), sorted(view.keys() & pins.keys())
        view |= pins
    return json.loads(ordered(view))


def pin_changes(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """(the pins `new` adds to `old`, the pins it removes or changes, with the first difference)."""
    added = [name for name in new if name not in old]
    changed = [f"{name}: removed" for name in old if name not in new]
    changed += [
        f"{name}: {first_difference(old[name], new[name], key_order=True)}"
        for name in old
        if name in new and ordered(old[name]) != ordered(new[name])
    ]
    return added, changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.golden.make_pins")
    parser.add_argument(
        "--check-additive",
        action="store_true",
        help="list the pins the regenerated view adds to pins.json; write nothing",
    )
    args = parser.parse_args(argv)
    view = pins_view()
    old = json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else {}
    added, changed = pin_changes(old, view)
    for line in added:
        print(f"added {line}")
    for line in changed:
        print(f"changed {line}")
    if changed:
        raise SystemExit(
            f"the regenerated view is not additive: it removes or changes "
            f"{len(changed)} pins of {PINS_PATH.name}"
        )
    if args.check_additive:
        print(f"additive: the regenerated view adds {len(added)} pins to {PINS_PATH.name}")
        return
    PINS_PATH.write_text(dumps(view, key_order=True))
    print(f"wrote {PINS_PATH} ({PINS_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
