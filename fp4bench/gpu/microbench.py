"""The microbenchmark on the GPU (fp4bench.microbench defines what it times): each cell's inputs
are quantized, its kernel is checked against a dequantized FP32 reference, then timed in a CUDA
graph."""

import importlib.metadata
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import NamedTuple, assert_never

import flashinfer
import torch

# Importing vllm._custom_ops registers vLLM's CUDA ops as torch.ops._C, which the activation
# kernels call (silu_and_mul, silu_and_mul_nvfp4_quant); its scaled_fp4_quant is NV's quantizer.
import vllm._custom_ops as vllm_ops
from flashinfer.autotuner import AutoTuner
from torch import Tensor
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from fp4bench import microbench as mb
from fp4bench.core.kernels import ActImpl, BenchKind, GemmImpl
from fp4bench.core.types import Format, Raised, attempt


@dataclass(frozen=True, slots=True)
class GemmOperands:
    """One copy of a GEMM's inputs: packed FP4 activations and weights, and their block scales."""

    x_fp4: Tensor
    x_sf: Tensor
    w_fp4: Tensor
    w_sf: Tensor

    @property
    def nbytes(self) -> int:
        tensors = (self.x_fp4, self.x_sf, self.w_fp4, self.w_sf)
        return sum(t.numel() * t.element_size() for t in tensors)

    def clone(self) -> "GemmOperands":
        return GemmOperands(
            self.x_fp4.clone(), self.x_sf.clone(), self.w_fp4.clone(), self.w_sf.clone()
        )


type GemmFn = Callable[[GemmOperands], Tensor]
type ActFn = Callable[[Tensor], tuple[Tensor, Tensor]]


@dataclass(frozen=True, slots=True, kw_only=True)
class MicrobenchConfig:
    """What `run` times: every pre-registered cell, unless `shapes`, `m_values` or `impls` restrict
    it or `gemm` or `act` turns a step off."""

    shapes: Sequence[str] | None = None
    m_values: Sequence[int] = mb.M_VALUES
    impls: tuple[GemmImpl | ActImpl, ...] | None = None
    gemm: bool = True
    act: bool = True
    n_samples: int = mb.N_SAMPLES
    min_calls: int = mb.MIN_CALLS_PER_SAMPLE
    kernel_names: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchContext:
    """What every cell's timing shares: the L2 its copies rotate past, and how it samples."""

    l2: int
    n_samples: int
    min_calls: int
    kernel_names: bool


@dataclass(frozen=True, slots=True)
class _Recorded:
    """What the steps append to as they go: an aborted step keeps the cells it finished."""

    cells: list[mb.MicrobenchCell] = field(default_factory=list[mb.MicrobenchCell])
    tuning: list[mb.Tuning] = field(default_factory=list[mb.Tuning])
    notes: list[str] = field(default_factory=list[str])


@dataclass(frozen=True, slots=True)
class _Failed:
    """What a step, a tuning, a cell or its kernel listing raised, as the run records it."""

    error: str


def _attempt[T](fn: Callable[[], T]) -> T | _Failed:
    """fn's value, or what it raised as text. The exception goes here: its traceback holds the
    tensors of the frames it left, in a cycle (through attempt's f_back) with any frame that
    keeps it, which only the garbage collector would free."""
    result = attempt(fn)
    if not isinstance(result, Raised):
        return result
    failed = _Failed(mb.error_text(result.exc))
    del result
    return failed


# --- environment ----------------------------------------------------------------------------------

VERSIONS_RECORDED = (
    "torch",
    "vllm",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "nvidia-cutlass-dsl",
    "nvidia-cudnn-frontend",
    "cuda-python",
    "triton",
    "mslk",
)


def _environment() -> mb.EnvironmentProbe:
    props = torch.cuda.get_device_properties(0)
    versions: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name in VERSIONS_RECORDED or name.startswith(("nvidia-cudnn", "nvidia-cublas", "mslk")):
            versions[name] = dist.version
    config_text: str = torch.__config__.show()
    return {
        "gpu": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "total_memory": props.total_memory,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "versions": versions,
        "torch_config_mentions_mslk": "mslk" in config_text.lower(),
        "torch_config": config_text,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("FLASHINFER", "VLLM_", "CUBLAS", "CUDNN", "TORCH_CUDA"))
        },
        "python": sys.version.split()[0],
    }


def _copy_bandwidth(nbytes: int = 4 << 30, iters: int = 20) -> mb.CopyBandwidth:
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1000 / iters
    del src, dst
    torch.cuda.empty_cache()
    return {
        "bytes_per_copy": 2 * nbytes,
        "tb_per_s": 2 * nbytes / seconds / 1e12,
        "frac_hbm": 2 * nbytes / seconds / mb.HBM_BYTES_PER_S,
    }


def _copy_reference() -> mb.CopyBandwidth:
    measured = _attempt(_copy_bandwidth)
    return {"error": measured.error} if isinstance(measured, _Failed) else measured


# --- quantization and the FP32 reference ----------------------------------------------------------


def _unswizzle_128x4(sf: Tensor) -> Tensor:
    rows, cols = sf.shape
    return (
        sf.reshape(rows // 128, cols // 4, 32, 4, 4)
        .permute(*mb.SWIZZLE_128x4_PERM)
        .reshape(rows, cols)
    )


def _dequant(
    fmt: Format, packed: Tensor, sf: Tensor, *, rows: int, k: int, global_scale: Tensor | None
) -> Tensor:
    lut = torch.tensor(mb.FP4_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8)[:rows]
    vals = torch.stack((lut[(p & 0xF).long()], lut[(p >> 4).long()]), dim=-1).reshape(rows, k)
    bs = mb.BLOCK_SIZE[fmt]
    s = _unswizzle_128x4(sf.view(torch.uint8))[:rows, : k // bs]
    match fmt:
        case Format.NVFP4:
            if global_scale is None:
                raise ValueError("an NVFP4 tensor dequantizes with its global scale")
            scale = s.view(torch.float8_e4m3fn).float() / global_scale.float()
        case Format.MXFP4:
            scale = torch.exp2(s.float() - 127.0)
        case _:
            assert_never(fmt)
    return (vals.view(rows, k // bs, bs) * scale[..., None]).reshape(rows, k)


def _rel_err(out: Tensor, ref: Tensor) -> float:
    diff = torch.linalg.vector_norm(out.float() - ref)
    return float(diff / torch.linalg.vector_norm(ref))


def _global_scale(t: Tensor) -> Tensor:
    return ((448.0 * 6.0) / t.float().abs().amax()).to(torch.float32)


class _Weight(NamedTuple):
    """A quantized weight, its NVFP4 global scale (None for MXFP4) and its dequantized values."""

    packed: Tensor
    scales: Tensor
    global_scale: Tensor | None
    dequantized: Tensor


def _quantize_weight(fmt: Format, w: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
    n = w.shape[0]
    if fmt == Format.MXFP4:
        w_fp4, w_sf = flashinfer.mxfp4_quantize(w)
        return w_fp4, w_sf.view(torch.uint8).reshape(mb.pad_up(n, 128), -1), None
    g_w = _global_scale(w)
    w_fp4, w_sf = flashinfer.nvfp4_quantize(
        w, g_w, sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
    )
    return w_fp4, w_sf.view(torch.float8_e4m3fn).reshape(mb.pad_up(n, 128), -1), g_w


def _quantize_act(fmt: Format, x: Tensor, g_x: Tensor | None) -> tuple[Tensor, Tensor]:
    m = x.shape[0]
    if fmt == Format.MXFP4:
        x_fp4, x_sf = flashinfer.mxfp4_quantize(x.contiguous(), backend="cute-dsl")
        return x_fp4, x_sf.view(torch.uint8).reshape(mb.pad_up(m, 128), -1)
    x_fp4, x_sf = vllm_ops.scaled_fp4_quant(
        x, g_x, is_sf_swizzled_layout=True, backend="flashinfer-cutedsl"
    )
    return x_fp4, x_sf.view(torch.float8_e4m3fn).reshape(mb.pad_up(m, 128), -1)


def _quantized_weights(fmts: Sequence[Format], n: int, k: int) -> dict[Format, _Weight]:
    """One BF16 weight per shape, quantized to each format and dequantized for the reference."""
    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    weights: dict[Format, _Weight] = {}
    for fmt in fmts:
        packed, scales, global_scale = _quantize_weight(fmt, w)
        dequantized = _dequant(fmt, packed, scales, rows=n, k=k, global_scale=global_scale)
        weights[fmt] = _Weight(packed, scales, global_scale, dequantized)
    del w
    return weights


@dataclass(frozen=True, slots=True)
class GemmInputs:
    """A GEMM's quantized operands and the scales that undo NVFP4's global ones: `alpha` for
    mm_fp4 and cuBLASLt (1 for MXFP4), `inv_gx` and `inv_gw` for the two-level recipe."""

    base: GemmOperands
    g_x: Tensor | None
    alpha: Tensor
    inv_gx: Tensor | None
    inv_gw: Tensor | None


def _inverse(t: Tensor) -> Tensor:
    """`1.0 / t`, the same torch ops. A function because inline, on a `Tensor | None` that a None
    check narrowed, ty types the quotient as float.__truediv__'s float."""
    return 1.0 / t


def _gemm_operands(fmt: Format, x: Tensor, weight: _Weight) -> GemmInputs:
    g_x = _global_scale(x) if fmt == Format.NVFP4 else None
    x_fp4, x_sf = _quantize_act(fmt, x, g_x)
    base = GemmOperands(x_fp4, x_sf, weight.packed, weight.scales)
    g_w = weight.global_scale
    if g_x is None or g_w is None:
        return GemmInputs(base, None, torch.ones(1, dtype=torch.float32, device="cuda"), None, None)
    alpha = _inverse(g_x * g_w).to(torch.float32)
    return GemmInputs(base, g_x, alpha, _inverse(g_x).reshape(1), _inverse(g_w).reshape(1))


class _Prepared(NamedTuple):
    """One format's inputs at one M, and the FP32 reference its GEMMs are checked against."""

    inputs: GemmInputs
    ref: Tensor


def _prepare(fmt: Format, x: Tensor, weight: _Weight, *, k: int) -> _Prepared:
    inputs = _gemm_operands(fmt, x, weight)
    base, rows = inputs.base, x.shape[0]
    x_ref = _dequant(fmt, base.x_fp4, base.x_sf, rows=rows, k=k, global_scale=inputs.g_x)
    return _Prepared(inputs, x_ref @ weight.dequantized.t())


# --- the kernels ----------------------------------------------------------------------------------


def _flashinfer_fn(impl: GemmImpl, fmt: Format, alpha: Tensor) -> GemmFn:
    bs, nv = mb.BLOCK_SIZE[fmt], fmt == Format.NVFP4
    backend = "cudnn" if impl == GemmImpl.CUDNN else "cute-dsl"

    def fn(o: GemmOperands) -> Tensor:
        x_sf, w_sf = o.x_sf, o.w_sf
        if backend == "cudnn":
            x_sf, w_sf = x_sf.view(torch.uint8), w_sf.view(torch.uint8)
        return flashinfer.mm_fp4(
            o.x_fp4,
            o.w_fp4.t(),
            x_sf,
            w_sf.t(),
            alpha,
            torch.bfloat16,
            block_size=bs,
            use_8x4_sf_layout=False,
            backend=backend,
            use_nvfp4=nv,
        )

    if mb.GEMM_IMPLS[impl].autotuned[fmt]:
        return fn

    def heuristic(o: GemmOperands) -> Tensor:
        with flashinfer.autotune(False, skip_ops={"fp4_gemm"}):
            return fn(o)

    return heuristic


def _cublaslt_nv1_fn() -> GemmFn:
    fp4 = torch.float4_e2m1fn_x2

    def fn(o: GemmOperands) -> Tensor:
        return torch._scaled_mm(  # noqa: SLF001 - the kernel under test
            o.x_fp4.view(fp4),
            o.w_fp4.view(fp4).t(),
            o.x_sf,
            o.w_sf,
            out_dtype=torch.bfloat16,
        )

    return fn


def _cublaslt_nv2_fn(inv_gx: Tensor | None, inv_gw: Tensor | None) -> GemmFn:
    functional = torch.nn.functional
    fp4 = torch.float4_e2m1fn_x2
    recipe = [functional.ScalingType.BlockWise1x16, functional.ScalingType.TensorWise]
    swizzle = [functional.SwizzleType.SWIZZLE_32_4_4, functional.SwizzleType.NO_SWIZZLE]

    def fn(o: GemmOperands) -> Tensor:
        return functional.scaled_mm(
            o.x_fp4.view(fp4),
            o.w_fp4.view(fp4).t(),
            [o.x_sf, inv_gx],
            recipe,
            [o.w_sf, inv_gw],
            recipe,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
            output_dtype=torch.bfloat16,
        )

    return fn


def _mslk_mx_fn() -> GemmFn:
    functional = torch.nn.functional
    fp4 = torch.float4_e2m1fn_x2
    e8m0 = torch.float8_e8m0fnu

    def fn(o: GemmOperands) -> Tensor:
        return functional.scaled_mm(
            o.x_fp4.view(fp4),
            o.w_fp4.view(fp4).t(),
            o.x_sf.view(e8m0),
            functional.ScalingType.BlockWise1x32,
            o.w_sf.view(e8m0),
            functional.ScalingType.BlockWise1x32,
            swizzle_a=functional.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=functional.SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )

    return fn


TORCH_GEMMS: dict[GemmImpl, Callable[[GemmInputs], GemmFn]] = {
    GemmImpl.TORCH_CUBLASLT_NV1: lambda _: _cublaslt_nv1_fn(),
    GemmImpl.TORCH_CUBLASLT_NV2: lambda inputs: _cublaslt_nv2_fn(inputs.inv_gx, inputs.inv_gw),
    GemmImpl.TORCH_MSLK_MX: lambda _: _mslk_mx_fn(),
}
"""The GEMMs torch runs; every other GemmImpl is FlashInfer's mm_fp4 (mb.FLASHINFER_IMPLS)."""


def _gemm_fn(impl: GemmImpl, fmt: Format, inputs: GemmInputs) -> GemmFn:
    if impl in mb.FLASHINFER_IMPLS:
        return _flashinfer_fn(impl, fmt, inputs.alpha)
    return TORCH_GEMMS[impl](inputs)


@cache
def _compiled_silu_mul() -> Callable[[Tensor], Tensor]:
    def silu_and_mul(x: Tensor) -> Tensor:
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

    return torch.compile(silu_and_mul, dynamic=True, fullgraph=True)


@dataclass(frozen=True, slots=True)
class _ActKernels:
    """The activation kernels, with NVFP4's global scale (None for MXFP4)."""

    g_x: Tensor | None

    def mx_quant(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return flashinfer.mxfp4_quantize(h, backend="cute-dsl")

    def nv_quant(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return vllm_ops.scaled_fp4_quant(
            h, self.g_x, is_sf_swizzled_layout=True, backend="flashinfer-cutedsl"
        )

    def nv_fused(self, x: Tensor) -> tuple[Tensor, Tensor]:
        m, d = x.shape[0], x.shape[1] // 2
        result = torch.empty((m, d // 2), dtype=torch.uint8, device=x.device)
        block_scale = torch.empty(
            (mb.pad_up(m, 128), mb.pad_up(d // 16, 4) // 4), dtype=torch.int32, device=x.device
        ).view(torch.float8_e4m3fn)
        torch.ops._C.silu_and_mul_nvfp4_quant(result, block_scale, x, self.g_x)  # noqa: SLF001
        return result, block_scale

    def c_silu_mul(self, x: Tensor) -> Tensor:
        h = torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(h, x)  # noqa: SLF001
        return h


def _act_fn(impl: ActImpl, g_x: Tensor | None) -> ActFn:
    kernels = _ActKernels(g_x)
    match impl:
        case ActImpl.MX_QUANT_CUTEDSL:
            return kernels.mx_quant
        case ActImpl.NV_QUANT_VLLM:
            return kernels.nv_quant
        case ActImpl.SILU_MUL_NV_FUSED:
            return kernels.nv_fused
        case ActImpl.SILU_MUL_THEN_MX_QUANT:
            silu_mul = _compiled_silu_mul()
            return lambda x: kernels.mx_quant(silu_mul(x))
        case ActImpl.SILU_MUL_C_THEN_MX_QUANT:
            return lambda x: kernels.mx_quant(kernels.c_silu_mul(x))
        case ActImpl.SILU_MUL_THEN_NV_QUANT:
            silu_mul = _compiled_silu_mul()
            return lambda x: kernels.nv_quant(silu_mul(x))
        case _:
            assert_never(impl)


# --- timing ---------------------------------------------------------------------------------------


def _time_graph[T](
    fn: Callable[[T], object], copies: Sequence[T], n_calls: int, n_samples: int
) -> list[float]:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for o in copies[:3]:
            fn(o)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(n_calls):
            fn(copies[j % len(copies)])
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(n_samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / n_calls)
    del graph
    return samples


def _kernel_names[T](fn: Callable[[T], object], arg: T) -> list[str]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(arg)
        torch.cuda.synchronize()
    names: list[str] = []
    for evt in prof.events():
        if evt.device_type == DeviceType.CUDA and evt.name[:1000] not in names:
            names.append(evt.name[:1000])
    return names


def _record_kernels[T](extras: mb.CellExtras, fn: Callable[[T], object], arg: T) -> None:
    names = _attempt(lambda: _kernel_names(fn, arg))
    if isinstance(names, _Failed):
        extras["kernels_error"] = names.error
    else:
        extras["kernels"] = names


def _autotuner_choices(fn: GemmFn, arg: GemmOperands) -> list[mb.AutotunerChoice]:
    tuner = AutoTuner.get()
    original = tuner.choose_one
    seen: list[mb.AutotunerChoice] = []

    def recording(
        custom_op: str, runners: object, tuning_config: object, inputs: object, **kwargs: object
    ) -> tuple[object, object]:
        runner, tactic = original(custom_op, runners, tuning_config, inputs, **kwargs)
        seen.append(
            {
                "op": custom_op,
                "runner": type(runner).__name__,
                "tactic": repr(tactic)[:400],
                "fallback": tactic == -1,
            }
        )
        return runner, tactic

    tuner.choose_one = recording
    try:
        fn(arg)
    finally:
        del tuner.choose_one
    return seen


def _autotune(fn: GemmFn, arg: GemmOperands) -> float:
    t0 = time.perf_counter()
    with flashinfer.autotune(True):
        fn(arg)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


class _Rotation(NamedTuple):
    """How many copies of a cell's inputs a graph cycles through, and its calls per sample."""

    copies: int
    calls: int
    bytes_per_copy: int

    def extras(self) -> mb.CellExtras:
        return {
            "copies": self.copies,
            "calls_per_sample": self.calls,
            "bytes_per_copy": self.bytes_per_copy,
            "bytes_per_graph": self.calls * self.bytes_per_copy,
        }


def _rotation(bytes_per_copy: int, ctx: BenchContext) -> _Rotation:
    copies = mb.rotation_count(bytes_per_copy, ctx.l2)
    return _Rotation(copies, mb.calls_per_sample(copies, ctx.min_calls), bytes_per_copy)


def _summary(cell: mb.MicrobenchCell) -> mb.Timing:
    summary = mb.timing(cell)
    if summary is None:
        raise RuntimeError("timed cell has no summary")
    return summary


# --- GEMM cells -----------------------------------------------------------------------------------


class _Shape(NamedTuple):
    """One GEMM shape's steps, its quantized weights and the M its autotuned cells are tuned at."""

    label: str
    n: int
    k: int
    steps: list[mb.GemmStep]
    weights: dict[Format, _Weight]
    m_tune: int


def _timed_gemm_cell(
    fn: GemmFn,
    case: mb.GemmCase,
    prepared: _Prepared,
    *,
    rotation: _Rotation,
    ctx: BenchContext,
    extras: mb.CellExtras,
) -> mb.GemmCell:
    """Check, then time, one GEMM; `extras` records its autotuner choices and kernels as it goes,
    so that a cell that fails later keeps them."""
    base = prepared.inputs.base
    out = fn(base)
    if case.impl == GemmImpl.TORCH_CUBLASLT_NV1:
        out = out.float() * prepared.inputs.alpha
    rel = _rel_err(out, prepared.ref)
    del out
    if case.impl in mb.FLASHINFER_IMPLS:
        extras["autotuner_choice"] = _autotuner_choices(fn, base)
    if ctx.kernel_names:
        _record_kernels(extras, fn, base)
    copies = [base] + [base.clone() for _ in range(rotation.copies - 1)]
    samples = _time_graph(fn, copies, rotation.calls, ctx.n_samples)
    cell = mb.make_cell(case, samples, rel_err=rel, extras=extras)
    summary = _summary(cell)
    mb.log(
        f"  {case.impl:22s} {case.fmt} {case.shape:18s} M={case.m:<4d} {summary.median_us:9.2f} us "
        f"{100 * summary.frac_hbm:5.1f}% HBM  rel_err={rel:.2e}"
    )
    return cell


def _gemm_cell(
    fn: GemmFn,
    case: mb.GemmCase,
    prepared: _Prepared,
    *,
    ctx: BenchContext,
    tuning: mb.CellExtras,
) -> mb.GemmCell:
    rotation = _rotation(prepared.inputs.base.nbytes, ctx)
    extras: mb.CellExtras = {**tuning, **rotation.extras()}
    timed = _attempt(
        lambda: _timed_gemm_cell(fn, case, prepared, rotation=rotation, ctx=ctx, extras=extras)
    )
    if isinstance(timed, _Failed):
        cell = mb.make_cell(case, error=timed.error, extras=extras)
        mb.log(f"  {case.impl} {case.fmt} {case.shape} M={case.m} FAILED {timed.error[:300]}")
        torch.cuda.synchronize()
    else:
        cell = timed
    torch.cuda.empty_cache()
    return cell


def _gemm_step_cell(
    step: mb.GemmStep,
    shape: _Shape,
    prepared: _Prepared,
    *,
    tune_errors: dict[tuple[GemmImpl, Format], str],
    ctx: BenchContext,
    order: int,
) -> mb.GemmCell:
    n, k = shape.n, shape.k
    case = mb.GemmCase(
        step.impl,
        step.fmt,
        step.shape,
        step.m,
        n,
        k,
        mb.gemm_bytes(step.fmt, step.m, n, k)["total"],
        mb.gemm_flops(step.m, n, k),
    )
    spec = mb.GEMM_IMPLS[step.impl]
    autotuned = spec.autotuned[step.fmt]
    tuning: mb.CellExtras = {
        "autotuned": autotuned,
        "tuned_at_M": shape.m_tune if autotuned else None,
        "library": spec.library,
        "order": order,
    }
    tune_error = tune_errors.get((step.impl, step.fmt))
    if tune_error is not None:
        return mb.make_cell(case, error=tune_error, extras=tuning)
    fn = _gemm_fn(step.impl, step.fmt, prepared.inputs)
    return _gemm_cell(fn, case, prepared, ctx=ctx, tuning=tuning)


def _autotune_one(
    shape: _Shape, impl: GemmImpl, fmt: Format, tuning: list[mb.Tuning]
) -> str | None:
    """Tune one autotuned (impl, fmt) at the shape's largest M; the error, if it failed."""
    torch.manual_seed(999)
    x = torch.randn(shape.m_tune, shape.k, device="cuda", dtype=torch.bfloat16)
    inputs = _gemm_operands(fmt, x, shape.weights[fmt])
    tuned = _attempt(lambda: _autotune(_gemm_fn(impl, fmt, inputs), inputs.base))
    if isinstance(tuned, _Failed):
        mb.log(f"  autotune {impl} {fmt} FAILED {tuned.error[:300]}")
        torch.cuda.synchronize()
        return "autotune failed: " + tuned.error
    tuning.append(
        {"impl": impl, "fmt": fmt, "shape": shape.label, "M": shape.m_tune, "tune_s": tuned}
    )
    mb.log(f"  tuned {impl} {fmt} at M={shape.m_tune}: {tuned:.1f} s")
    return None


def _run_at_m(
    shape: _Shape,
    m: int,
    *,
    tune_errors: dict[tuple[GemmImpl, Format], str],
    ctx: BenchContext,
    out: _Recorded,
) -> None:
    """The shape's cells at M, back to back, on one BF16 activation quantized to each format."""
    torch.manual_seed(1000 + m)
    x = torch.randn(m, shape.k, device="cuda", dtype=torch.bfloat16)
    prepared = {fmt: _prepare(fmt, x, w, k=shape.k) for fmt, w in shape.weights.items()}
    for step in [step for step in shape.steps if step.m == m]:
        cell = _gemm_step_cell(
            step,
            shape,
            prepared[step.fmt],
            tune_errors=tune_errors,
            ctx=ctx,
            order=len(out.cells),
        )
        out.cells.append(cell)
    del x, prepared
    torch.cuda.empty_cache()


def _run_shape(steps: list[mb.GemmStep], m_tune: int, ctx: BenchContext, out: _Recorded) -> None:
    label = steps[0].shape
    n, k = mb.SHAPE_NK[label]
    fmts = [f for f in mb.FORMATS if any(step.fmt == f for step in steps)]
    mb.log(f"{label} (N={n}, K={k})")
    shape = _Shape(label, n, k, steps, _quantized_weights(fmts, n, k), m_tune)
    tune_errors: dict[tuple[GemmImpl, Format], str] = {}
    for impl, fmt in dict.fromkeys((step.impl, step.fmt) for step in steps):
        if mb.GEMM_IMPLS[impl].autotuned[fmt]:
            error = _autotune_one(shape, impl, fmt, out.tuning)
            if error is not None:
                tune_errors[impl, fmt] = error
    for m in dict.fromkeys(step.m for step in steps):
        _run_at_m(shape, m, tune_errors=tune_errors, ctx=ctx, out=out)
    del shape
    torch.cuda.empty_cache()


def _run_gemms(plan: Sequence[mb.GemmStep], m_tune: int, ctx: BenchContext, out: _Recorded) -> None:
    for label in dict.fromkeys(step.shape for step in plan):
        _run_shape([step for step in plan if step.shape == label], m_tune, ctx, out)


# --- activation cells -----------------------------------------------------------------------------


class _ActInputs(NamedTuple):
    """The BF16 input, what its quantized output should dequantize to (x, or SiLU(gate) * up) and
    NVFP4's global scale (None for MXFP4)."""

    x: Tensor
    target: Tensor
    g_x: Tensor | None


def _timed_act_cell(
    case: mb.ActCase,
    inputs: _ActInputs,
    *,
    rotation: _Rotation,
    ctx: BenchContext,
    extras: mb.CellExtras,
) -> mb.ActCell:
    x, target, g_x = inputs
    fn = _act_fn(case.impl, g_x)
    q, sf = fn(x)
    rows = case.m
    deq = _dequant(
        case.fmt, q, sf.reshape(mb.pad_up(rows, 128), -1), rows=rows, k=case.k, global_scale=g_x
    )
    rel = _rel_err(deq, target)
    del q, sf, deq
    copies = [x] + [x.clone() for _ in range(rotation.copies - 1)]
    if ctx.kernel_names:
        _record_kernels(extras, fn, x)
    samples = _time_graph(fn, copies, rotation.calls, ctx.n_samples)
    cell = mb.make_cell(case, samples, rel_err=rel, extras=extras)
    summary = _summary(cell)
    mb.log(
        f"  {case.impl:24s} {case.shape:16s} M={case.m:<4d} {summary.median_us:8.2f} us "
        f"{100 * summary.frac_hbm:5.1f}% HBM rel_err={rel:.2e}"
    )
    del copies
    return cell


def _act_cell(job: mb.ActPlan, m: int, ctx: BenchContext) -> mb.ActCell:
    torch.manual_seed(1)
    x = torch.randn(m, job.input_width, device="cuda", dtype=torch.bfloat16)
    k_out = job.quantized_width
    if job.impl.startswith("silu"):
        target = torch.nn.functional.silu(x[:, :k_out].float()) * x[:, k_out:].float()
        nbytes = mb.silu_quant_bytes(job.fmt, m, k_out)["total"]
    else:
        target = x.float()
        nbytes = mb.act_quant_bytes(job.fmt, m, k_out)["total"]
    g_x = _global_scale(target) if job.fmt == Format.NVFP4 else None
    rotation = _rotation(x.numel() * x.element_size(), ctx)
    extras: mb.CellExtras = {
        **rotation.extras(),
        "rotation_capped": rotation.copies == mb.MAX_COPIES,
        "width_in": job.input_width,
    }
    case = mb.ActCase(job.impl, job.fmt, job.label, m, k_out, nbytes)
    inputs = _ActInputs(x, target, g_x)
    timed = _attempt(
        lambda: _timed_act_cell(case, inputs, rotation=rotation, ctx=ctx, extras=extras)
    )
    if not isinstance(timed, _Failed):
        return timed
    cell = mb.make_cell(case, error=timed.error, extras=extras)
    mb.log(f"  {job.impl} {job.label} M={m} FAILED {timed.error[:300]}")
    torch.cuda.synchronize()
    return cell


def _run_act(
    jobs: Sequence[mb.ActPlan], m_values: Sequence[int], ctx: BenchContext, out: _Recorded
) -> None:
    for job in jobs:
        mb.log(f"act {job.impl} {job.label}")
        for m in m_values:
            out.cells.append(_act_cell(job, m, ctx))
            torch.cuda.empty_cache()


# --- the run --------------------------------------------------------------------------------------


def _steps(
    config: MicrobenchConfig, shapes: list[str], ctx: BenchContext, out: _Recorded
) -> list[tuple[BenchKind, Callable[[], None]]]:
    """The steps `config` turns on; one that `impls` leaves no implementation for is off."""
    impls = config.impls
    act_impls = None if impls is None else [i for i in impls if mb.is_act_impl(i)]
    gemm_impls = None if impls is None else [i for i in impls if mb.is_gemm_impl(i)]
    m_values = list(config.m_values)
    steps: list[tuple[BenchKind, Callable[[], None]]] = []
    if config.gemm and gemm_impls != []:
        steps.append(
            (
                BenchKind.GEMM,
                lambda: _run_gemms(
                    mb.gemm_plan(shapes, m_values, gemm_impls), max(m_values), ctx, out
                ),
            )
        )
    if config.act and act_impls != []:
        steps.append((BenchKind.ACT, lambda: _run_act(mb.act_plan(act_impls), m_values, ctx, out)))
    return steps


def run(
    out_path: str | None = None, config: MicrobenchConfig | None = None
) -> mb.MicrobenchResults:
    """Run the §14 microbenchmark on GPU 0 (all of it, unless `config` restricts it)."""
    config = config if config is not None else MicrobenchConfig()
    started = time.time()
    shapes = [s[0] for s in mb.SHAPES] if config.shapes is None else list(config.shapes)
    probe = _environment()
    smi_start = mb.nvidia_smi()
    l2 = probe["l2_bytes"]
    mb.log(f"{probe['gpu']} L2={l2 / 2**20:.1f} MiB torch={torch.__version__}")
    copy_reference = _copy_reference()
    ctx = BenchContext(
        l2=l2,
        n_samples=config.n_samples,
        min_calls=config.min_calls,
        kernel_names=config.kernel_names,
    )
    out = _Recorded()
    for kind, step in _steps(config, shapes, ctx, out):
        done = _attempt(step)
        if isinstance(done, _Failed):
            note = f"{kind} step aborted: {done.error}"
            out.notes.append(note)
            mb.log(note)
    smi_end = mb.nvidia_smi()
    results: mb.MicrobenchResults = {
        "spec": "EXPERIMENT.md §14",
        "env": {
            **probe,
            "nvidia_smi_start": smi_start,
            "copy_bandwidth_reference": copy_reference,
            "nvidia_smi_end": smi_end,
        },
        "config": {
            "shapes": shapes,
            "m_values": list(config.m_values),
            "impls": config.impls,
            "n_samples": config.n_samples,
            "min_calls": config.min_calls,
            "l2_factor": mb.L2_FACTOR,
            "max_copies": mb.MAX_COPIES,
            "hbm_bytes_per_s": mb.HBM_BYTES_PER_S,
        },
        "cells": out.cells,
        "tuning": out.tuning,
        "notes": out.notes,
        "wall_s": time.time() - started,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=1))
    return results
