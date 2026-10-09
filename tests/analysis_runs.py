"""Synthetic runs for the analysis tests: rows as the runner writes them (dicts, so a test can
change or drop a field), their typed forms, and run directories on disk."""

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from fp4bench import settings
from fp4bench.analysis import cells as cl
from fp4bench.analysis import design, plots
from fp4bench.analysis.compare import RatioResult
from fp4bench.analysis.inputs import served_args
from fp4bench.core.schema import M1Row, M2Row, ManifestLine, ServerRow, _numbered_rows
from fp4bench.core.types import Treatment, ValueKey, Verdict
from fp4bench.studies import expc, model
from fp4bench.studies.base import ServerSettings
from fp4bench.studies.smoke import SMOKE

TEL = {"window_s": 30.0, "counters_delta": {}, "env_throttle_us": 0, "sw_power_cap_frac": 0.0}
SOURCE = "openai_continuous_usage"
CUTE_DSL = "FlashInferCuteDslNvFp4LinearKernel"
DROP = object()
ACT_QUANT_LINE = (
    "(EngineCore pid=1) INFO 10-05 12:00:02 [compilation.py:331] Enabled custom fusions: act_quant"
)
FALLBACK_LINE = (
    "(EngineCore pid=1) WARNING 10-05 12:00:00 [linear/__init__.py:365] "
    "--linear-backend=flashinfer_cutedsl has no kernel for this linear layer type"
)

N_WINDOWS = 16
BF16_PP = [2.0 + 0.01 * i for i in range(N_WINDOWS)]
WOBBLE = [0.004 * ((i * 7) % 5 - 2) for i in range(N_WINDOWS)]
DEGRADATION = {
    "MX": 0.060,
    "MXp": 0.060,
    "NV": 0.030,
    "NVa": 0.030,
    "NVnf": 0.030,
    "NVx": 0.032,
    "NVc": 0.030,
    "NVt": 0.030,
    "NVd": 0.030,
    "NVv": 0.030,
}
KV_TOKENS = 1_200_000
CLEAN_CKPT = {
    "mx_expert_bytes": 100,
    "nv_expert_bytes": 106,
    "nv_over_mx": 4.5 / 4.25,
    "expected_nv_over_mx": 4.5 / 4.25,
    "config_diff": {},
    "mx_problems": [],
    "nv_problems": [],
}
GOOD_REFERENCE = {"per_prompt": BF16_PP, "mean": statistics.fmean(BF16_PP)}
FOUR = ("MX", "NV", "NVa", "MXp")
TABLE_ARGS = served_args(None)
FIVE = ("MX", "NV", "NVa", "MXp", "NVnf")
SMOKE_CS = (1, 32, 128)
EXPB_CS = (128, 256, 512)
SCAN_FACTORS = {
    "MX": 1.0,
    "NV": 1.0,
    "NVx": 1.01,
    "NVc": {1: 1.30, 32: 1.10, 128: 0.90},
    "NVt": {1: 1.20, 32: 0.97, 128: 1.20},
    "NVd": {1: 0.80, 32: 1.05, 128: 1.10},
    "NVv": {1: 1.10, 32: 1.30, 128: 1.30},
}
GATE_LINE_CLEAN = (
    "gates: G1 pass · G2 pass · G3 pass · G4 pass · G5a pass · G5b pass · G6a pass "
    "· G6b pass · G7 pass"
)
EXPB_NLL_SHIFT = {"MX": 0.020, "NV": 0.010, "MXp": 0.020}


@pytest.fixture(autouse=True)
def _no_figures(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """analyze_run draws no figures unless the test uses `figures`: what it draws is checked by
    test_analysis_figures and the mutation corpus (tests/golden/mutations.py)."""
    if "figures" not in request.fixturenames:
        monkeypatch.setattr(plots, "write", lambda result: None)


@pytest.fixture
def figures() -> None:
    """analyze_run draws its figures in this test."""


def recording_args(line: dict, treatment: Treatment, args: tuple[str, ...]) -> dict:
    """`line` with `treatment` served with `args` (inputs.treatment_server_args), which G1 and
    the kernel scan expect the kernel and fusion of."""
    recorded = {**line["inputs"].get("treatment_server_args", {}), treatment: list(args)}
    return {**line, "inputs": {**line["inputs"], "treatment_server_args": recorded}}


def pp(treatment: str, degradation: float | None = None) -> list[float]:
    d = DEGRADATION[treatment] if degradation is None else degradation
    extra = WOBBLE if treatment.startswith("MX") else [0.0] * N_WINDOWS
    return [b + d + e for b, e in zip(BF16_PP, extra, strict=True)]


def kernel_line(treatment: str) -> str:
    fmt = "MXFP4" if treatment.startswith("MX") else "NVFP4"
    return (
        f"(EngineCore pid=1) INFO 10-05 12:00:01 [linear/__init__.py:1186] "
        f"Using {model.TREATMENTS[Treatment(treatment)].linear_kernel} for {fmt} GEMM"
    )


def graph_hash(t: str) -> str:
    """One torch.compile cache directory per serving configuration (MX and MXp share one)."""
    return f"{'mx' if t.startswith('MX') else t.lower()}-graph-0123456789"


def fusion_facts(t: str) -> dict:
    on = model.TREATMENTS[Treatment(t)].act_quant_fusion
    return {
        "pass_config": {"fuse_norm_quant": False, "fuse_act_quant": on, "fuse_attn_quant": False},
        "pass_config_conflict": False,
        "fuse_act_quant": on,
        "custom_fusion_lines": [ACT_QUANT_LINE] if on else [],
        "custom_fusions": ["act_quant"] if on else [],
    }


def session(r: int, t: str, concurrencies, argv: list[str] | None = None) -> dict:
    h = graph_hash(t)
    return {
        "round": r,
        "treatment": t,
        "session_id": f"r{r}_{t}-x",
        "gpu_uuid": "GPU-1",
        "start_id": f"start-{r}",
        "server_argv": argv
        or [
            "vllm",
            "serve",
            model.TREATMENTS[Treatment(t)].model_dir,
            *ServerSettings().args(),
            *model.TREATMENTS[Treatment(t)].server_args,
        ],
        "nll": statistics.fmean(pp(t)),
        "nll_per_prompt": pp(t),
        "linear_kernels": [model.TREATMENTS[Treatment(t)].linear_kernel],
        "linear_kernel_lines": [kernel_line(t)],
        "linear_backend_fallback_lines": [FALLBACK_LINE] if t.startswith("MX") else [],
        "sampling_defaults_lines": ["Default sampling parameters: {}"],
        "cudagraph_capture_sizes": list(concurrencies),
        "weights_gib": 13.0,
        **fusion_facts(t),
        "compile_cache_lines": [f"Directly load AOT compilation from path .../{h}/rank_0_0"],
        "compile_cache_hashes": [h],
        "kv_cache_tokens": KV_TOKENS,
        "kv_cache_memory_gib": 136.35,
    }


def m1_row(**fields) -> dict:
    row = {
        "round": 0,
        "treatment": "MX",
        "session_id": "r0_MX-x",
        "c": 8,
        "set": 1,
        "warmup": False,
        "first": "n1",
        "n1": settings.M1_N1,
        "n2": settings.M1_N2,
        "t1_s": 1.0,
        "t2_s": 5.0,
        "step_s": 0.004,
        "decode_tok_s": 2000.0,
        "block_telemetry": dict(TEL),
        "preemptions_delta": 0.0,
    }
    return {**row, **fields}


def m2_row(**fields) -> dict:
    row = {
        "round": 0,
        "treatment": "MX",
        "session_id": "r0_MX-x",
        "c": 8,
        "valid": True,
        "invalid_reasons": [],
        "failure_reason": "",
        "aggregate_tps": 2000.0,
        "aggregate_source": SOURCE,
        "measurement_seconds": 30.0,
        "effective_concurrency": 8.0,
        "avg_running_reqs": 8.0,
        "itl_p50_ms": 4.0,
        "tps_per_user_p50": 250.0,
        "ttft_p50_ms": 50.0,
        "server_gen_throughput": 2000.0,
        "telemetry": dict(TEL),
        "preemptions_delta": 0.0,
    }
    return {**row, **fields}


def build(factors: dict, concurrencies, rounds: int, jitter: bool = True):
    """Servers, M1 and M2 rows; `factors[t]` scales MX's 4 ms step (a float or {c: float})."""
    servers, m1, m2 = [], [], []
    for r in range(rounds):
        drift = 1 + 0.002 * r
        wobble = 1 + 0.001 * (-1) ** r if jitter else 1.0
        for t, factor in factors.items():
            sid = f"r{r}_{t}-x"
            servers.append(session(r, t, concurrencies))
            for c in concurrencies:
                base = factor[c] if isinstance(factor, dict) else factor
                shake = {"MX": 1.0, "MXp": 1.0 / wobble, "NV": wobble}.get(t, 1 / wobble**0.7)
                step = 0.004 * drift * base * shake
                m1.append(
                    m1_row(
                        round=r,
                        treatment=t,
                        session_id=sid,
                        c=c,
                        set=0,
                        warmup=True,
                        first="n1",
                        step_s=99.0,
                    )
                )
                m1.extend(
                    m1_row(
                        round=r,
                        treatment=t,
                        session_id=sid,
                        c=c,
                        set=s,
                        first="n1" if s % 2 == 0 else "n2",
                        step_s=step,
                    )
                    for s in (1, 2, 3)
                )
                m2.append(
                    m2_row(
                        round=r,
                        treatment=t,
                        session_id=sid,
                        c=c,
                        aggregate_tps=c / step,
                        tps_per_user_p50=1 / step,
                        itl_p50_ms=step * 1000,
                    )
                )
    if "NV" in factors:
        m1.append(
            m1_row(round=0, treatment="NV", session_id="dead", c=concurrencies[0], step_s=1.0)
        )
    return servers, m1, m2


def synthetic(
    nv_factor: Any,
    concurrencies=(8, 32),
    rounds: int = 5,
    nva_factor: Any = 1.0,
    treatments=FOUR,
    nvnf_factor: Any = 1.0,
):
    factors = {"MX": 1.0, "NV": nv_factor, "NVa": nva_factor, "MXp": 1.0, "NVnf": nvnf_factor}
    return build({t: factors[t] for t in treatments}, concurrencies, rounds)


def smoke(factors=None, rounds: int = 1):
    return build(SCAN_FACTORS if factors is None else factors, SMOKE_CS, rounds, jitter=False)


def expb(nv: Any = 1.0, rounds: int = 5):
    return synthetic(nv, EXPB_CS, rounds=rounds, treatments=("MX", "NV", "MXp"))


def manifest(
    rounds: int = 5,
    problems=(),
    checkpoint: Any = "clean",
    reference: Any = "good",
    treatments=FOUR,
    **protocol,
) -> dict:
    """The last manifests.jsonl line; `checkpoint`, `reference`: "clean"/"good", None or given."""
    line = {
        "protocol": {"rounds": rounds, "treatments": list(treatments), **protocol},
        "problems": list(problems),
        "inputs": {},
    }
    if checkpoint is not None:
        line["checkpoint_report"] = CLEAN_CKPT if checkpoint == "clean" else checkpoint
    if reference is not None:
        line["inputs"]["bf16_reference_nll"] = GOOD_REFERENCE if reference == "good" else reference
    return line


def expb_manifest(rounds: int = 5, **protocol) -> dict:
    fields = {
        "concurrencies": list(EXPB_CS),
        "m1_reps": 3,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.95,
        "primary_concurrencies": [256, 512],
        **protocol,
    }
    return manifest(rounds=rounds, treatments=("MX", "NV", "MXp"), **fields)


def smoke_manifest(**kw) -> dict:
    """A smoke's manifest line, naming its study."""
    return {
        **manifest(
            rounds=1, treatments=SMOKE.treatments, concurrencies=list(SMOKE_CS), m1_reps=3, **kw
        ),
        "study": SMOKE.name,
    }


def set_fields(rows: list[dict], treatment: str, r: int | None = None, **fields) -> None:
    """Set (or with DROP delete) fields of `treatment`'s rows (of round `r`, or all)."""
    for row in rows:
        if row["treatment"] == treatment and (r is None or row["round"] == r):
            for k, v in fields.items():
                if v is DROP:
                    row.pop(k, None)
                else:
                    row[k] = v


def set_nll(servers: list[dict], treatment: str, degradation: float | None) -> None:
    """Every session of `treatment` at the BF16 reference plus `degradation` (None: no NLL)."""
    for s in servers:
        if s["treatment"] == treatment:
            if degradation is None:
                s["nll"], s["nll_per_prompt"] = None, [None] * N_WINDOWS
            else:
                s["nll_per_prompt"] = pp(treatment, degradation)
                s["nll"] = statistics.fmean(s["nll_per_prompt"])


FAIL_ERROR = (
    "RuntimeError('vllm serve exited with 1; see /results/smoke-r2-1/servers/r0_NVc-1.log')"
)


def failed_row(s: dict, error: str = FAIL_ERROR) -> dict:
    return {
        "round": s["round"],
        "treatment": s["treatment"],
        "session_id": s["session_id"],
        "start_id": s["start_id"],
        "failed": True,
        "error": error,
    }


def fail(servers: list[dict], treatment: str, r: int = 0, error: str = FAIL_ERROR) -> list[dict]:
    """`treatment`'s session of round `r` replaced by its failed row; its M1/M2 rows stay."""
    return [
        failed_row(s, error) if (s["treatment"], s["round"]) == (treatment, r) else s
        for s in servers
    ]


def drop_m1(m1: list[dict], session_id: str, c: int, warmup_too: bool = False) -> list[dict]:
    return [
        r
        for r in m1
        if not (r["session_id"] == session_id and r["c"] == c and (warmup_too or not r["warmup"]))
    ]


def typed_servers(rows: list[dict]) -> list[ServerRow]:
    return [ServerRow.from_json(r) for r in rows]


def typed_m1(rows: list[dict]) -> list[M1Row]:
    return [M1Row.from_json(r) for r in rows]


def typed_m2(rows: list[dict]) -> list[M2Row]:
    return [M2Row.from_json(r) for r in rows]


def typed_manifest(line: dict | None) -> ManifestLine | None:
    return None if line is None else ManifestLine.from_json(line)


def aa_results(batches, lo=0.995, hi=1.004, verdict=Verdict.EQUIVALENT) -> dict:
    return {c: RatioResult(5, (0, 1, 2, 3, 4), 1.0, lo, hi, verdict) for c in batches}


def cell_steps(servers: list[dict], m1: list[dict]) -> cl.Steps:
    """M1 steps per (round, treatment, cell) of rows as the runner writes them."""
    return cl.m1_steps(typed_m1(m1), cl.session_ids(typed_servers(servers)))


def batch_steps(servers: list[dict], m1: list[dict]) -> dict[ValueKey[int], float]:
    return cl.by_batch(cell_steps(servers, m1))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, row in _numbered_rows(path):
        if not isinstance(row, dict):
            raise ValueError(f"{path}, line {n}: not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_run(tmp_path: Path, servers, m1, m2=(), manifests=None, name: str = "run") -> Path:
    """`manifests`: None writes no manifests.jsonl; a dict is one line; a list is several."""
    run = tmp_path / name
    run.mkdir()
    write_jsonl(run / "servers.jsonl", servers)
    write_jsonl(run / "m1.jsonl", m1)
    write_jsonl(run / "m2.jsonl", m2)
    if manifests is not None:
        write_jsonl(
            run / "manifests.jsonl", manifests if isinstance(manifests, list) else [manifests]
        )
    return run


def main_and_expb(tmp_path: Path) -> tuple[Path, Path]:
    """An Experiment B run, its NLLs EXPB_NLL_SHIFT above the main run's, and that main run."""
    servers, m1, m2 = synthetic(1.0, (8, 32, 128), treatments=FIVE)
    main = write_run(tmp_path, servers, m1, m2, manifest(treatments=FIVE), name="main")
    servers, m1, m2 = expb()
    for row in servers:
        row["nll"] += EXPB_NLL_SHIFT[row["treatment"]]
    return write_run(tmp_path, servers, m1, m2, expb_manifest(), name="expb"), main


C_REPS = 3
C_SMOKE_CELLS = ((1, 1024), (1, 32768), (1, 127360), (128, 360))
C_CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512]
C_KV_TOKENS = 560_288
C_CKPT = {
    "nv_over_mx": 4.5 / 4.25,
    "expected_nv_over_mx": 4.5 / 4.25,
    "mx_problems": [],
    "nv_problems": [],
}
C_REFERENCE = {"per_prompt": [1.8] * 4, "mean": 1.8}
H_BATCH = {cell: expc.MAIN_RUN_DELTA_MS[cell[0]] for cell in expc.REGISTERED_CELLS}
H_TOKENS = {
    cell: design.h_tokens_delta_ms(design.kv_tokens(cell, expc.MEAN_CONTEXT_EXTRA))
    for cell in expc.REGISTERED_CELLS
}


def c_noise(r: int, i: int) -> float:
    """A deterministic wobble of Δ, at most 0.01 ms, averaging to 0 in every cell."""
    return 0.005 * ((r + 3 * i + i // 5) % 5 - 2)


def mx_step(cell) -> float:
    c, p = cell
    return 0.006 + 1e-8 * c * (p + 640)


def c_argv(t: str) -> list[str]:
    return [
        "vllm",
        "serve",
        model.TREATMENTS[Treatment(t)].model_dir,
        "--served-model-name",
        "fp4bench",
        "--max-model-len",
        "131072",
        "--max-num-seqs",
        "512",
        "--kv-cache-dtype",
        "bfloat16",
        "--hf-overrides",
        '{"max_position_embeddings": 131072}',
        *model.TREATMENTS[Treatment(t)].server_args,
    ]


def c_session(r: int, t: str) -> dict:
    on = model.TREATMENTS[Treatment(t)].act_quant_fusion
    return {
        "round": r,
        "treatment": t,
        "session_id": f"r{r}_{t}-x",
        "gpu_uuid": "GPU-1",
        "start_id": f"start-{r}",
        "nll": expc.MAIN_RUN_NLL[Treatment(t)] + 0.001 * r,
        "server_argv": c_argv(t),
        "linear_kernels": [model.TREATMENTS[Treatment(t)].linear_kernel],
        "linear_kernel_lines": [f"Using {model.TREATMENTS[Treatment(t)].linear_kernel}"],
        "cudagraph_capture_sizes": list(C_CAPTURE_SIZES),
        "weights_gib": 18.6,
        "kv_cache_tokens": C_KV_TOKENS,
        "compile_cache_hashes": [f"{t.lower()}-graph"],
        "pass_config_conflict": False,
        "fuse_act_quant": on,
        "custom_fusion_lines": [ACT_QUANT_LINE] if on else [],
        "custom_fusions": ["act_quant"] if on else [],
    }


def c_build(
    delta_ms: dict,
    cells=expc.REGISTERED_CELLS,
    rounds: int = 5,
    treatments=("MX", "NV", "MXp"),
    aa: dict | None = None,
    noise: float = 1.0,
    aa_noise: float = 0.0,
):
    """Per round and cell: MX's step, NV = MX - Δ(cell), MXp = MX x aa[cell], with small wobbles."""
    servers, m1 = [], []
    for r in range(rounds):
        drift = 1 + 0.001 * r
        for t in treatments:
            servers.append(c_session(r, t))
            for i, cell in enumerate(cells):
                mx = mx_step(cell) * drift
                if t == "MX":
                    step = mx
                elif t == "NV":
                    step = mx - (delta_ms[cell] + noise * c_noise(r, i)) / 1000
                else:
                    step = (
                        mx * (aa or {}).get(cell, 1.0) * (1 + 0.0005 * (-1) ** (r + i))
                        + aa_noise * c_noise(r, i) / 1000
                    )
                base = {
                    "round": r,
                    "treatment": t,
                    "session_id": f"r{r}_{t}-x",
                    "c": cell[0],
                    "prompt_len": cell[1],
                }
                m1.append(m1_row(**base, set=0, warmup=True, first="n1", step_s=99.0))
                m1.extend(
                    m1_row(**base, set=s, first="n2", step_s=step * (1 + 0.001 * (s - 2)))
                    for s in range(1, C_REPS + 1)
                )
    return servers, m1


def c_manifest(
    cells=expc.REGISTERED_CELLS,
    rounds: int = 5,
    treatments=("MX", "NV", "MXp"),
    checkpoint: Any = "clean",
    reference: Any = "good",
    problems=(),
    **protocol,
) -> dict:
    proto = {
        "rounds": rounds,
        "treatments": list(treatments),
        "cells": [list(c) for c in cells],
        "concurrencies": sorted({c for c, _ in cells}),
        "m1_reps": C_REPS,
        "m2_duration_s": 0,
        "max_model_len": 131072,
        "hf_overrides": '{"max_position_embeddings": 131072}',
        "kv_cache_dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "primary_concurrencies": [1],
        **protocol,
    }
    line = {
        "protocol": proto,
        "problems": list(problems),
        "inputs": {"nll_prompts_sha256": expc.MAIN_RUN_NLL_PROMPTS_SHA256},
    }
    if checkpoint is not None:
        line["checkpoint_report"] = C_CKPT if checkpoint == "clean" else checkpoint
    if reference is not None:
        line["inputs"]["bf16_reference_nll"] = C_REFERENCE if reference == "good" else reference
    return line


def c_run(tmp_path: Path, delta=None, manifest=None, mutate=None, **kw) -> Path:
    """An Experiment C run directory (`mutate(servers, m1)` changes rows before writing)."""
    servers, m1 = c_build(H_BATCH if delta is None else delta, **kw)
    if mutate is not None:
        mutate(servers, m1)
    line = (
        c_manifest(
            kw.get("cells", expc.REGISTERED_CELLS),
            kw.get("rounds", 5),
            kw.get("treatments", ("MX", "NV", "MXp")),
        )
        if manifest is None
        else manifest
    )
    return write_run(tmp_path, servers, m1, (), line, name="run-c")
