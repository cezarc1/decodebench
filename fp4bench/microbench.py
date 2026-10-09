"""Kernel-level FP4 GEMM microbenchmark (EXPERIMENT.md §14, METHODOLOGY.md#microbench): what it
times, how it counts bytes, the rows of results.json and its summary. fp4bench.gpu.microbench runs
it on the GPU."""

import math
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypedDict, TypeGuard, overload

from fp4bench.core.kernels import ActImpl, BenchKind, GemmImpl
from fp4bench.core.types import Format

MXFP4, NVFP4 = Format.MXFP4, Format.NVFP4
# METHODOLOGY.md#microbench
HBM_BYTES_PER_S = 8e12
FORMATS = (MXFP4, NVFP4)
BLOCK_SIZE: dict[Format, int] = {MXFP4: 32, NVFP4: 16}
M_VALUES = (1, 8, 32, 64, 128, 256, 512)
N_SAMPLES = 7
MIN_CALLS_PER_SAMPLE = 200
L2_FACTOR = 2.0
MAX_COPIES = 32768
PRACTICAL_BAND = (0.98, 1.02)
BANDWIDTH_BOUND = 0.80
NOT_BANDWIDTH_BOUND = 0.60
SMALL_M = (1, 8, 32)
LARGE_M = (256, 512)
LARGE_SHAPES = ("gate_up_proj", "down_proj", "qkv_proj")

SHAPES = (
    ("qkv_proj", 10240, 5120, "merged"),
    ("o_proj", 5120, 8192, "merged"),
    ("gate_up_proj", 51200, 5120, "merged"),
    ("down_proj", 5120, 25600, "merged"),
    ("q_proj", 8192, 5120, "individual"),
    ("k_proj=v_proj", 1024, 5120, "individual"),
    ("gate_proj=up_proj", 25600, 5120, "individual"),
)
SHAPE_NK = {label: (n, k) for label, n, k, _ in SHAPES}


@dataclass(frozen=True, slots=True, kw_only=True)
class GemmImplSpec:
    formats: tuple[Format, ...]
    autotuned: dict[Format, bool]
    library: str


GEMM_IMPLS = {
    GemmImpl.VLLM_CUTEDSL: GemmImplSpec(
        formats=(MXFP4, NVFP4),
        autotuned={MXFP4: True, NVFP4: False},
        library="FlashInfer mm_fp4 backend=cute-dsl (vLLM's path, as deployed)",
    ),
    GemmImpl.VLLM_CUTEDSL_NV_TUNED: GemmImplSpec(
        formats=(NVFP4,),
        autotuned={NVFP4: True},
        library="FlashInfer mm_fp4 backend=cute-dsl, NVFP4 autotuned",
    ),
    GemmImpl.CUDNN: GemmImplSpec(
        formats=(MXFP4, NVFP4),
        autotuned={MXFP4: True, NVFP4: True},
        library="FlashInfer mm_fp4 backend=cudnn",
    ),
    GemmImpl.TORCH_CUBLASLT_NV1: GemmImplSpec(
        formats=(NVFP4,),
        autotuned={NVFP4: False},
        library="torch._scaled_mm, single-level NVFP4 (cuBLASLt)",
    ),
    GemmImpl.TORCH_CUBLASLT_NV2: GemmImplSpec(
        formats=(NVFP4,),
        autotuned={NVFP4: False},
        library="F.scaled_mm, two-level NVFP4 (BlockWise1x16 + TensorWise)",
    ),
    GemmImpl.TORCH_MSLK_MX: GemmImplSpec(
        formats=(MXFP4,),
        autotuned={MXFP4: False},
        library="F.scaled_mm, MXFP4 BlockWise1x32 (MSLK f4f4bf16 if built in)",
    ),
}
GEMM_CELLS = (
    (GemmImpl.VLLM_CUTEDSL, MXFP4),
    (GemmImpl.VLLM_CUTEDSL, NVFP4),
    (GemmImpl.VLLM_CUTEDSL_NV_TUNED, NVFP4),
    (GemmImpl.CUDNN, MXFP4),
    (GemmImpl.CUDNN, NVFP4),
    (GemmImpl.TORCH_MSLK_MX, MXFP4),
    (GemmImpl.TORCH_CUBLASLT_NV1, NVFP4),
    (GemmImpl.TORCH_CUBLASLT_NV2, NVFP4),
)
FLASHINFER_IMPLS = (GemmImpl.VLLM_CUTEDSL, GemmImpl.VLLM_CUTEDSL_NV_TUNED, GemmImpl.CUDNN)
COMPARISONS = (
    (
        "a",
        "(a) vLLM path as deployed: CuTe-DSL, MXFP4 autotuned vs NVFP4 heuristic",
        (GemmImpl.VLLM_CUTEDSL, MXFP4),
        (GemmImpl.VLLM_CUTEDSL, NVFP4),
    ),
    (
        "a_tuned",
        "(a') vLLM path, both autotuned: CuTe-DSL",
        (GemmImpl.VLLM_CUTEDSL, MXFP4),
        (GemmImpl.VLLM_CUTEDSL_NV_TUNED, NVFP4),
    ),
    (
        "c",
        "(c) same library: cuDNN, both autotuned",
        (GemmImpl.CUDNN, MXFP4),
        (GemmImpl.CUDNN, NVFP4),
    ),
    (
        "b",
        "(b) MAMF path, different libraries: NVFP4 single-level torch._scaled_mm "
        "vs MXFP4 F.scaled_mm BlockWise1x32",
        (GemmImpl.TORCH_MSLK_MX, MXFP4),
        (GemmImpl.TORCH_CUBLASLT_NV1, NVFP4),
    ),
    (
        "b_2lvl",
        "(b') NVFP4 two-level F.scaled_mm vs MXFP4 F.scaled_mm BlockWise1x32 (different libraries)",
        (GemmImpl.TORCH_MSLK_MX, MXFP4),
        (GemmImpl.TORCH_CUBLASLT_NV2, NVFP4),
    ),
)

ACT_QUANT_K = (5120, 8192, 25600)
ACT_IMPLS = {ActImpl.MX_QUANT_CUTEDSL: MXFP4, ActImpl.NV_QUANT_VLLM: NVFP4}
SILU_D = 25600
SILU_IMPLS = {
    ActImpl.SILU_MUL_THEN_MX_QUANT: MXFP4,
    ActImpl.SILU_MUL_NV_FUSED: NVFP4,
    ActImpl.SILU_MUL_C_THEN_MX_QUANT: MXFP4,
    ActImpl.SILU_MUL_THEN_NV_QUANT: NVFP4,
}

FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
SWIZZLE_128x4_PERM = (0, 3, 2, 1, 4)


def pad_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def scale_bytes(fmt: Format, rows: int, k: int) -> int:
    """One byte per block scale (E8M0 or E4M3) in the 128x4-swizzled, padded layout."""
    return pad_up(rows, 128) * pad_up(k // BLOCK_SIZE[fmt], 4)


def gemm_bytes(fmt: Format, m: int, n: int, k: int) -> dict[str, int]:
    """HBM bytes of one (M x K) @ (K x N) FP4 GEMM, each operand once; global scales left out."""
    parts = {
        "weights": n * k // 2,
        "weight_scales": scale_bytes(fmt, n, k),
        "acts": m * k // 2,
        "act_scales": scale_bytes(fmt, m, k),
        "output": 2 * m * n,
    }
    parts["total"] = sum(parts.values())
    return parts


def gemm_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def act_quant_bytes(fmt: Format, m: int, k: int) -> dict[str, int]:
    """BF16 input read once, packed FP4 and its block scales written once."""
    parts = {"input": 2 * m * k, "packed": m * k // 2, "scales": scale_bytes(fmt, m, k)}
    parts["total"] = sum(parts.values())
    return parts


def silu_quant_bytes(fmt: Format, m: int, d: int) -> dict[str, int]:
    """The fused minimum for SiLU-mul + quantize, counted for every implementation."""
    parts = {"input": 2 * m * 2 * d, "packed": m * d // 2, "scales": scale_bytes(fmt, m, d)}
    parts["total"] = sum(parts.values())
    return parts


def bytes_ratio(m: int, n: int, k: int) -> float:
    """t_NV / t_MX if both GEMMs ran at the same bandwidth."""
    return gemm_bytes(NVFP4, m, n, k)["total"] / gemm_bytes(MXFP4, m, n, k)["total"]


def rotation_count(
    bytes_per_copy: int, l2_bytes: int, factor: float = L2_FACTOR, max_copies: int = MAX_COPIES
) -> int:
    """Copies to cycle so that (R - 1) * bytes_per_copy >= factor * L2; at least 2."""
    if bytes_per_copy <= 0 or l2_bytes <= 0:
        raise ValueError("bytes_per_copy and l2_bytes must be positive")
    return min(max_copies, max(2, math.ceil(factor * l2_bytes / bytes_per_copy) + 1))


def calls_per_sample(n_copies: int, min_calls: int = MIN_CALLS_PER_SAMPLE) -> int:
    """Calls in one graph: whole passes over the copies, at least `min_calls`."""
    return n_copies * math.ceil(min_calls / n_copies)


def aggregate(samples_us: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def rates(nbytes: int, flops: int, us: float) -> dict[str, float]:
    seconds = us * 1e-6
    return {
        "gbps": nbytes / seconds / 1e9,
        "frac_hbm": nbytes / seconds / HBM_BYTES_PER_S,
        "tflops": flops / seconds / 1e12,
    }


def practically_different(ratio: float) -> bool:
    lo, hi = PRACTICAL_BAND
    return not (lo <= ratio <= hi)


# --- results.json: what `run` writes and `summarize` reads ---------------------------------------


class AutotunerChoice(TypedDict):
    """A tactic FlashInfer's autotuner chose for one call; -1 is its heuristic fallback."""

    op: str
    runner: str
    tactic: str
    fallback: bool


class CellExtras(TypedDict, total=False):
    """A cell's fields after `error`, in the order the run records them. A GEMM cell has its
    tuning (`autotuned` to `order`), then its rotation (`copies` to `bytes_per_graph`); an
    activation cell has its rotation, `rotation_capped` and `width_in`. Then `autotuner_choice`
    (FlashInfer GEMMs) and `kernels` or `kernels_error`, as far as the cell got."""

    autotuned: bool
    tuned_at_M: int | None
    library: str
    order: int
    copies: int
    calls_per_sample: int
    bytes_per_copy: int
    bytes_per_graph: int
    rotation_capped: bool
    width_in: int
    autotuner_choice: list[AutotunerChoice]
    kernels: list[str]
    kernels_error: str


class _CellBody(TypedDict):
    fmt: Format
    shape: str
    M: int
    N: int
    K: int
    bytes: int
    flops: int
    median_us: float | None
    min_us: float | None
    max_us: float | None
    samples_us: list[float] | None
    gbps: float | None
    frac_hbm: float | None
    tflops: float | None
    rel_err: float | None
    error: str | None


class GemmCell(_CellBody, CellExtras):
    kind: Literal[BenchKind.GEMM]
    impl: GemmImpl


class ActCell(_CellBody, CellExtras):
    kind: Literal[BenchKind.ACT]
    impl: ActImpl


type MicrobenchCell = GemmCell | ActCell


class Tuning(TypedDict):
    impl: GemmImpl
    fmt: Format
    shape: str
    M: int
    tune_s: float


class CopyBandwidth(TypedDict, total=False):
    """The device-to-device copy reference: its three measurements, or the `error` it raised."""

    bytes_per_copy: int
    tb_per_s: float
    frac_hbm: float
    error: str


class EnvironmentProbe(TypedDict):
    """What `run` reads from the device, torch and the installed packages before it starts."""

    gpu: str
    capability: str
    sm_count: int
    l2_bytes: int
    total_memory: int
    cuda: str | None
    cudnn: int | None
    versions: dict[str, str]
    torch_config_mentions_mslk: bool
    torch_config: str
    env: dict[str, str]
    python: str


class Environment(EnvironmentProbe):
    nvidia_smi_start: dict[str, str | None]
    copy_bandwidth_reference: CopyBandwidth
    nvidia_smi_end: dict[str, str | None]


class ResultsConfig(TypedDict):
    shapes: list[str]
    m_values: list[int]
    impls: Sequence[GemmImpl | ActImpl] | None
    n_samples: int
    min_calls: int
    l2_factor: float
    max_copies: int
    hbm_bytes_per_s: float


class MicrobenchResults(TypedDict):
    """results.json. `summarize` reads it with defaults for what an older file may lack."""

    spec: str
    env: Environment
    config: ResultsConfig
    cells: list[MicrobenchCell]
    tuning: list[Tuning]
    notes: list[str]
    wall_s: float


class GemmCase(NamedTuple):
    """A GEMM cell's identity and cost: the fields of its row before the timings."""

    impl: GemmImpl
    fmt: Format
    shape: str
    m: int
    n: int
    k: int
    nbytes: int
    flops: int


class ActCase(NamedTuple):
    """An activation cell's identity and cost; its N and FLOPs are 0."""

    impl: ActImpl
    fmt: Format
    shape: str
    m: int
    k: int
    nbytes: int


def _body(
    case: GemmCase | ActCase,
    samples_us: list[float] | None,
    error: str | None,
    rel_err: float | None,
) -> _CellBody:
    n, flops = (case.n, case.flops) if isinstance(case, GemmCase) else (0, 0)
    stats = aggregate(samples_us) if samples_us and error is None else None
    speed = rates(case.nbytes, flops, stats["median_us"]) if stats is not None else None
    return {
        "fmt": case.fmt,
        "shape": case.shape,
        "M": case.m,
        "N": n,
        "K": case.k,
        "bytes": case.nbytes,
        "flops": flops,
        "median_us": None if stats is None else stats["median_us"],
        "min_us": None if stats is None else stats["min_us"],
        "max_us": None if stats is None else stats["max_us"],
        "samples_us": samples_us,
        "gbps": None if speed is None else speed["gbps"],
        "frac_hbm": None if speed is None else speed["frac_hbm"],
        "tflops": None if speed is None else speed["tflops"],
        "rel_err": rel_err,
        "error": error,
    }


@overload
def make_cell(
    case: GemmCase,
    samples_us: list[float] | None = None,
    *,
    error: str | None = None,
    rel_err: float | None = None,
    extras: CellExtras | None = None,
) -> GemmCell: ...


@overload
def make_cell(
    case: ActCase,
    samples_us: list[float] | None = None,
    *,
    error: str | None = None,
    rel_err: float | None = None,
    extras: CellExtras | None = None,
) -> ActCell: ...


def make_cell(
    case: GemmCase | ActCase,
    samples_us: list[float] | None = None,
    *,
    error: str | None = None,
    rel_err: float | None = None,
    extras: CellExtras | None = None,
) -> MicrobenchCell:
    """One result row. A failed cell keeps its identity and `error`, with null timings."""
    body = _body(case, samples_us, error, rel_err)
    rest = extras if extras is not None else CellExtras()
    if isinstance(case, GemmCase):
        return {"kind": BenchKind.GEMM, "impl": case.impl, **body, **rest}
    return {"kind": BenchKind.ACT, "impl": case.impl, **body, **rest}


class Timing(NamedTuple):
    """A timed cell's summary: one with no error and a median."""

    median_us: float
    min_us: float
    max_us: float
    frac_hbm: float
    tflops: float


def timing(cell: MicrobenchCell | None) -> Timing | None:
    if cell is None or cell.get("error") is not None:
        return None
    median, lo, hi = cell.get("median_us"), cell.get("min_us"), cell.get("max_us")
    frac, tflops = cell.get("frac_hbm"), cell.get("tflops")
    if median is None or lo is None or hi is None or frac is None or tflops is None:
        return None
    return Timing(median, lo, hi, frac, tflops)


# --- the plans ------------------------------------------------------------------------------------


class GemmStep(NamedTuple):
    shape: str
    m: int
    impl: GemmImpl
    fmt: Format


def gemm_plan(
    shapes: Sequence[str] | None = None,
    m_values: Sequence[int] = M_VALUES,
    impls: Collection[GemmImpl] | None = None,
) -> list[GemmStep]:
    """(shape, M, impl, fmt) in execution order."""
    labels = [s[0] for s in SHAPES] if shapes is None else list(shapes)
    cells = [(i, f) for i, f in GEMM_CELLS if impls is None or i in impls]
    plan: list[GemmStep] = []
    for label in labels:
        for j, m in enumerate(m_values):
            plan.extend(GemmStep(label, m, i, f) for i, f in (cells if j % 2 == 0 else cells[::-1]))
    return plan


class ActPlan(NamedTuple):
    impl: ActImpl
    fmt: Format
    label: str
    input_width: int
    quantized_width: int


def act_plan(impls: Collection[ActImpl] | None) -> list[ActPlan]:
    jobs = [
        ActPlan(impl, fmt, f"K={k}", k, k) for k in ACT_QUANT_K for impl, fmt in ACT_IMPLS.items()
    ]
    jobs += [
        ActPlan(impl, fmt, "down_proj_input", 2 * SILU_D, SILU_D)
        for impl, fmt in SILU_IMPLS.items()
    ]
    return [job for job in jobs if impls is None or job.impl in impls]


def is_gemm_impl(impl: GemmImpl | ActImpl) -> TypeGuard[GemmImpl]:
    return impl in GEMM_IMPLS


def is_act_impl(impl: GemmImpl | ActImpl) -> TypeGuard[ActImpl]:
    return impl in ACT_IMPLS or impl in SILU_IMPLS


# --- the summary ----------------------------------------------------------------------------------

type CellKey = tuple[BenchKind, GemmImpl | ActImpl, Format, str, int]
type CellIndex = dict[CellKey, MicrobenchCell]
type ImplFormat = tuple[GemmImpl | ActImpl, Format]


def _index(cells: Sequence[MicrobenchCell]) -> CellIndex:
    return {(c["kind"], c["impl"], c["fmt"], c["shape"], c["M"]): c for c in cells}


class PairRatio(NamedTuple):
    """An MX cell, an NV cell and t_NV / t_MX when both were timed."""

    mx: MicrobenchCell | None
    nv: MicrobenchCell | None
    ratio: float | None


def pair_ratio(
    idx: CellIndex,
    mx: ImplFormat,
    nv: ImplFormat,
    *,
    shape: str,
    m: int,
    kind: BenchKind = BenchKind.GEMM,
) -> PairRatio:
    c_mx, c_nv = idx.get((kind, *mx, shape, m)), idx.get((kind, *nv, shape, m))
    t_mx, t_nv = timing(c_mx), timing(c_nv)
    ratio = t_nv.median_us / t_mx.median_us if t_mx is not None and t_nv is not None else None
    return PairRatio(c_mx, c_nv, ratio)


def bytes_regime(frac_mx: float, frac_nv: float) -> str:
    if frac_mx >= BANDWIDTH_BOUND and frac_nv >= BANDWIDTH_BOUND:
        return "both >= 80%: bytes test applies"
    if frac_mx < NOT_BANDWIDTH_BOUND or frac_nv < NOT_BANDWIDTH_BOUND:
        return "a format < 60%: kernel efficiency, not bytes"
    return "60-80%: in between"


class BandwidthBoundPair(NamedTuple):
    comparison: str
    shape: str
    m: int
    ratio: float
    bytes_ratio: float
    frac_mx: float
    frac_nv: float


def bandwidth_bound_pairs(results: MicrobenchResults) -> list[BandwidthBoundPair]:
    """Cells where both formats reach >= 80% of 8 TB/s (the pre-registered bytes test)."""
    idx = _index(results["cells"])
    m_values = results.get("config", {}).get("m_values", M_VALUES)
    rows: list[BandwidthBoundPair] = []
    for key, _, mx, nv in COMPARISONS:
        for label, n, k, _ in SHAPES:
            for m in m_values:
                pair = pair_ratio(idx, mx, nv, shape=label, m=m)
                t_mx, t_nv = timing(pair.mx), timing(pair.nv)
                if t_mx is None or t_nv is None or pair.ratio is None:
                    continue
                if t_mx.frac_hbm >= BANDWIDTH_BOUND and t_nv.frac_hbm >= BANDWIDTH_BOUND:
                    rows.append(
                        BandwidthBoundPair(
                            key,
                            label,
                            m,
                            pair.ratio,
                            bytes_ratio(m, n, k),
                            t_mx.frac_hbm,
                            t_nv.frac_hbm,
                        )
                    )
    return rows


def _us(cell: MicrobenchCell | None) -> str:
    if cell is None:
        return "—"
    t = timing(cell)
    if t is None:
        return "FAIL"
    return f"{t.median_us:.2f} [{t.min_us:.2f}–{t.max_us:.2f}]"


def _pct(cell: MicrobenchCell | None) -> str:
    t = timing(cell)
    return f"{100 * t.frac_hbm:.0f}%" if t is not None else "—"


def _ratio(ratio: float | None) -> str:
    if ratio is None:
        return "—"
    return f"{ratio:.3f}" + (" ≠" if practically_different(ratio) else "")


def _comparison_table(
    idx: CellIndex,
    mx: ImplFormat,
    nv: ImplFormat,
    *,
    shapes: Sequence[str],
    m_values: Sequence[int],
) -> list[str]:
    lines = [
        "| shape | M | MX µs [min–max] | MX %HBM | NV µs [min–max] | NV %HBM | "
        "t_NV/t_MX | bytes NV/MX |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in shapes:
        n, k = SHAPE_NK[label]
        for m in m_values:
            c_mx, c_nv, ratio = pair_ratio(idx, mx, nv, shape=label, m=m)
            if c_mx is None and c_nv is None:
                continue
            lines.append(
                f"| {label} | {m} | {_us(c_mx)} | {_pct(c_mx)} | {_us(c_nv)} | "
                f"{_pct(c_nv)} | {_ratio(ratio)} | {bytes_ratio(m, n, k):.3f} |"
            )
    return lines


def _headline(idx: CellIndex, m_values: Sequence[int]) -> list[str]:
    lines = [
        "| comparison | shape | M | MX %HBM | NV %HBM | t_NV/t_MX | bytes NV/MX | regime |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, _, mx, nv in COMPARISONS:
        for label in LARGE_SHAPES:
            n, k = SHAPE_NK[label]
            for m in [m for m in m_values if m in SMALL_M]:
                c_mx, c_nv, ratio = pair_ratio(idx, mx, nv, shape=label, m=m)
                t_mx, t_nv = timing(c_mx), timing(c_nv)
                regime = (
                    bytes_regime(t_mx.frac_hbm, t_nv.frac_hbm)
                    if t_mx is not None and t_nv is not None
                    else "missing"
                )
                lines.append(
                    f"| {key} | {label} | {m} | {_pct(c_mx)} | {_pct(c_nv)} | "
                    f"{_ratio(ratio)} | {bytes_ratio(m, n, k):.3f} | {regime} |"
                )
    return lines


def _tflops(cell: MicrobenchCell | None) -> str:
    t = timing(cell)
    return f"{t.tflops:.0f}" if t is not None else "—"


def _large_m(idx: CellIndex, m_values: Sequence[int]) -> list[str]:
    lines = [
        "| comparison | shape | M | MX TFLOPS | NV TFLOPS | t_NV/t_MX |",
        "|---|---|---|---|---|---|",
    ]
    for key, _, mx, nv in COMPARISONS:
        for label in LARGE_SHAPES:
            for m in [m for m in m_values if m in LARGE_M]:
                c_mx, c_nv, ratio = pair_ratio(idx, mx, nv, shape=label, m=m)
                lines.append(
                    f"| {key} | {label} | {m} | {_tflops(c_mx)} | {_tflops(c_nv)} | "
                    f"{_ratio(ratio)} |"
                )
    return lines


def _act_quant_table(idx: CellIndex, m_values: Sequence[int]) -> list[str]:
    lines = [
        "### Activation quantization (BF16 (M, K) -> FP4 + swizzled scales)",
        "",
        'MX = `flashinfer.mxfp4_quantize(backend="cute-dsl")`; NV = vLLM '
        "`scaled_fp4_quant` (static global scale). %HBM counts the BF16 input read and "
        "the FP4 output + scales written once.",
        "",
        "| K | M | MX µs [min–max] | MX %HBM | NV µs [min–max] | NV %HBM | t_NV/t_MX |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in ACT_QUANT_K:
        for m in m_values:
            c_mx, c_nv, ratio = pair_ratio(
                idx,
                (ActImpl.MX_QUANT_CUTEDSL, MXFP4),
                (ActImpl.NV_QUANT_VLLM, NVFP4),
                shape=f"K={k}",
                m=m,
                kind=BenchKind.ACT,
            )
            if c_mx is None and c_nv is None:
                continue
            lines.append(
                f"| {k} | {m} | {_us(c_mx)} | {_pct(c_mx)} | {_us(c_nv)} | "
                f"{_pct(c_nv)} | {_ratio(ratio)} |"
            )
    return lines


def _silu_quant_table(idx: CellIndex, m_values: Sequence[int]) -> list[str]:
    lines = [
        f"### SiLU·mul + quantize on the down_proj input (M, 2 x {SILU_D})",
        "",
        "As deployed (custom_ops none): MX = inductor-compiled native SiLU·mul, then "
        "`mxfp4_quantize(cute-dsl)` (two kernels); NV = fused "
        "`torch.ops._C.silu_and_mul_nvfp4_quant`. Extras: MX with vLLM's CUDA "
        "`silu_and_mul` (MX-C), and NV unfused (inductor SiLU·mul + `scaled_fp4_quant`, "
        "NV-u). %HBM uses the fused minimum (input read once, FP4 + scales written once) "
        "for all. Ratio = NV fused / MX as deployed.",
        "",
        "| M | MX µs [min–max] | MX %HBM | NV fused µs [min–max] | NV %HBM | t_NV/t_MX | "
        "MX-C µs | NV-u µs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    label = "down_proj_input"
    for m in m_values:
        c_mx, c_nv, ratio = pair_ratio(
            idx,
            (ActImpl.SILU_MUL_THEN_MX_QUANT, MXFP4),
            (ActImpl.SILU_MUL_NV_FUSED, NVFP4),
            shape=label,
            m=m,
            kind=BenchKind.ACT,
        )
        if c_mx is None and c_nv is None:
            continue
        c_mxc = idx.get((BenchKind.ACT, ActImpl.SILU_MUL_C_THEN_MX_QUANT, MXFP4, label, m))
        c_nvu = idx.get((BenchKind.ACT, ActImpl.SILU_MUL_THEN_NV_QUANT, NVFP4, label, m))
        lines.append(
            f"| {m} | {_us(c_mx)} | {_pct(c_mx)} | {_us(c_nv)} | {_pct(c_nv)} | "
            f"{_ratio(ratio)} | {_us(c_mxc)} | {_us(c_nvu)} |"
        )
    return lines


def _rel_err_table(cells: Sequence[MicrobenchCell]) -> list[str]:
    groups: dict[tuple[BenchKind, GemmImpl | ActImpl, Format], list[float]] = {}
    for c in cells:
        rel_err = c.get("rel_err")
        if rel_err is not None:
            groups.setdefault((c["kind"], c["impl"], c["fmt"]), []).append(rel_err)
    lines = [
        "| kind | impl | fmt | cells | median rel. err | max rel. err |",
        "|---|---|---|---|---|---|",
    ]
    for kind, impl, fmt in sorted(groups):
        errs = groups[kind, impl, fmt]
        lines.append(
            f"| {kind} | {impl} | {fmt} | {len(errs)} | "
            f"{statistics.median(errs):.2e} | {max(errs):.2e} |"
        )
    return lines


@dataclass(slots=True)
class _ChoiceCounts:
    """One (impl, fmt)'s cells by whether any call fell back to the heuristic, and its tactics."""

    fallback: int = 0
    cached: int = 0
    tactics: Counter[str] = field(default_factory=Counter[str])


def _choice_table(cells: Sequence[MicrobenchCell]) -> list[str]:
    groups: dict[tuple[GemmImpl, Format], _ChoiceCounts] = {}
    for c in cells:
        choices = c.get("autotuner_choice")
        if c["kind"] == BenchKind.GEMM and choices:
            counts = groups.setdefault((c["impl"], c["fmt"]), _ChoiceCounts())
            if any(ch["fallback"] for ch in choices):
                counts.fallback += 1
            else:
                counts.cached += 1
            counts.tactics.update(ch["tactic"] for ch in choices)
    if not groups:
        return ["(not recorded)"]
    lines = [
        "| impl | fmt | autotuned | cells on heuristic fallback | cells on a cached tactic | "
        "tactics (cells) |",
        "|---|---|---|---|---|---|",
    ]
    for impl, fmt in sorted(groups):
        counts = groups[impl, fmt]
        seen = "; ".join(f"`{t}` ({n})" for t, n in counts.tactics.most_common())
        lines.append(
            f"| {impl} | {fmt} | {GEMM_IMPLS[impl].autotuned[fmt]} | {counts.fallback} | "
            f"{counts.cached} | {seen} |"
        )
    return lines


def _kernel_table(cells: Sequence[MicrobenchCell]) -> list[str]:
    seen: dict[tuple[BenchKind, GemmImpl | ActImpl, Format], Counter[str]] = {}
    for c in cells:
        for name in c.get("kernels") or []:
            seen.setdefault((c["kind"], c["impl"], c["fmt"]), Counter()).update([name])
    lines: list[str] = []
    for kind, impl, fmt in sorted(seen):
        lines.append(f"- **{impl} / {fmt}** ({kind}):")
        for name, count in seen[kind, impl, fmt].most_common():
            short = name if len(name) <= 200 else f"{name[:90]}…{name[-100:]}"
            lines.append(f"  - `{short}` ({count} cells)")
    return lines


VERSIONS_SHOWN = (
    "torch",
    "vllm",
    "flashinfer-python",
    "flashinfer-cubin",
    "nvidia-cutlass-dsl",
    "nvidia-cudnn-frontend",
)


def _header(results: MicrobenchResults) -> list[str]:
    env = results.get("env", {})
    config = results.get("config", {})
    versions = env.get("versions", {})
    copy_ref = env.get("copy_bandwidth_reference", {})
    lo, hi = PRACTICAL_BAND
    return [
        "# FP4 GEMM kernel microbenchmark (EXPERIMENT.md §14)",
        "",
        f"- GPU: {env.get('gpu', '?')} (SM {env.get('capability', '?')}, "
        f"{env.get('sm_count', '?')} SMs), L2 {env.get('l2_bytes', 0) / 2**20:.1f} MiB",
        "- Versions: "
        + ", ".join(f"{k} {versions.get(k)}" for k in VERSIONS_SHOWN if versions.get(k))
        + f"; cuDNN {env.get('cudnn')}"
        f"; CUDA {env.get('cuda')}",
        f"- nvidia-smi at start: {env.get('nvidia_smi_start', {})}",
        f"- Reference (not pre-registered): device-to-device copy of 4 GiB reaches "
        f"{copy_ref.get('tb_per_s', float('nan')):.2f} TB/s "
        f"(read + write) = {100 * copy_ref.get('frac_hbm', float('nan')):.0f}% "
        "of 8 TB/s.",
        f"- Method: one CUDA graph per cell with >= {config.get('min_calls', '?')} calls "
        f"cycling through enough input copies (weights, activations and their scales) to "
        f"touch >= {config.get('l2_factor', L2_FACTOR):g}x L2 between two uses of a copy; "
        f"{config.get('n_samples', '?')} graph replays per cell; median [min–max] µs per "
        "call. %HBM = bytes (packed weights + weight scales + packed activations + activation "
        "scales + BF16 output, each once) / time / 8 TB/s. Activations are pre-quantized: "
        "GEMM cells time the GEMM only. One BF16 weight and activation per (shape, M) is "
        "quantized to both formats; all cells of a (shape, M) run back to back (order "
        "reversed on every other M). Autotuned FlashInfer cells are tuned once per shape at "
        "the largest M; the heuristic CuTe-DSL NVFP4 cells use FlashInfer's fallback tactic "
        '(`skip_ops={"fp4_gemm"}`), as in the deployed NVFP4 server whose autotune cache '
        "has no `fp4_gemm` entries.",
        f"- t_NV/t_MX outside [{lo}, {hi}] is marked ≠ (practically different; descriptive "
        "only). 'bytes NV/MX' is the ratio if both ran at equal bandwidth.",
        "",
    ]


def _bound_table(results: MicrobenchResults) -> list[str]:
    bound = bandwidth_bound_pairs(results)
    if not bound:
        return ["None: no (comparison, shape, M) has both formats at >= 80% of 8 TB/s."]
    return [
        "| comparison | shape | M | MX %HBM | NV %HBM | t_NV/t_MX | bytes NV/MX |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {r.comparison} | {r.shape} | {r.m} | {100 * r.frac_mx:.0f}% | "
            f"{100 * r.frac_nv:.0f}% | {_ratio(r.ratio)} | {r.bytes_ratio:.3f} |"
            for r in bound
        ),
    ]


def _tuning_lines(tuning: Sequence[Tuning]) -> list[str]:
    if not tuning:
        return []
    return [
        f"Autotuning (once per implementation, format and shape, at M = "
        f"{max(t['M'] for t in tuning)}): {sum(t['tune_s'] for t in tuning):.0f} s "
        f"over {len(tuning)} tunings.",
        "",
    ]


def _failures(cells: Sequence[MicrobenchCell]) -> list[str]:
    failures = [(c, error) for c in cells if (error := c.get("error"))]
    if not failures:
        return ["None."]
    return [
        f"- {c['kind']} {c['impl']} / {c['fmt']} / {c['shape']} / M={c['M']}: {error[:300]}"
        for c, error in failures
    ]


def summarize(results: MicrobenchResults) -> str:
    """Markdown report of a `run` result (descriptive only: no verdict, no TOST)."""
    cells = results["cells"]
    idx = _index(cells)
    config = results.get("config", {})
    m_values = config.get("m_values", list(M_VALUES))
    shapes = [s for s in (config.get("shapes") or [s[0] for s in SHAPES]) if s in SHAPE_NK]
    out = [
        *_header(results),
        "## Pre-registered bytes test: small M (<= 32) on gate_up / down / qkv",
        "",
        "If both formats reach >= 80% of 8 TB/s, t_NV/t_MX should approach the bytes ratio "
        "(MXFP4 ~5.6% faster on the weights); if either is below 60% the ratio measures "
        "kernel efficiency, not bytes.",
        "",
        *_headline(idx, m_values),
        "",
        "### Every cell where both formats reach >= 80% of 8 TB/s",
        "",
        *_bound_table(results),
        "",
        "## Large M (256, 512): compute-bound regime",
        "",
        *_large_m(idx, m_values),
        "",
        "## Per-comparison tables",
        "",
    ]
    for key, title, mx, nv in COMPARISONS:
        out += [
            f"### {title} [`{key}`]",
            "",
            f"MX = `{mx[0]}`, NV = `{nv[0]}`.",
            "",
            *_comparison_table(idx, mx, nv, shapes=shapes, m_values=m_values),
            "",
        ]
    out += [
        "## Activation quantization",
        "",
        *_act_quant_table(idx, m_values),
        "",
        *_silu_quant_table(idx, m_values),
        "",
        "## Numerical check (relative error, not gated)",
        "",
        "GEMM: ||out - ref|| / ||ref|| with ref = dequantized activations @ dequantized "
        "weights^T in FP32. Activation quantization: ||dequant(q(x)) - x|| / ||x||.",
        "",
        *_rel_err_table(cells),
        "",
        "## Autotuner choices (FlashInfer `mm_fp4` cells)",
        "",
        *_choice_table(cells),
        "",
        *_tuning_lines(results.get("tuning") or []),
        "## Kernels observed (torch profiler, one eager call per cell)",
        "",
        *(_kernel_table(cells) or ["(not recorded)"]),
        "",
        "## Failures",
        "",
        *_failures(cells),
        *(f"- note: {note}" for note in results.get("notes", [])),
    ]
    return "\n".join(out) + "\n"


# --- the run's log, errors and nvidia-smi probe (fp4bench.gpu.microbench) ------------------------


def log(msg: str) -> None:
    print(f"[microbench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def error_text(exc: BaseException) -> str:
    """How a cell, a tuning or a step records what it raised."""
    return f"{type(exc).__name__}: {str(exc).strip()[:1500]}"


NVIDIA_SMI_FIELDS = (
    "name",
    "driver_version",
    "pstate",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "clocks.max.mem",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks_event_reasons.active",
)


def nvidia_smi() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for field_name in NVIDIA_SMI_FIELDS:
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--query-gpu={field_name}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            out[field_name] = r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, IndexError):
            out[field_name] = None
    return out
