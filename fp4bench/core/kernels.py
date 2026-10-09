"""The vocabulary of the kernel microbenchmark (EXPERIMENT.md §14) and the profiler (§15)."""

from enum import StrEnum
from typing import NamedTuple


class BenchKind(StrEnum):
    GEMM = "gemm"
    ACT = "act"


class GemmImpl(StrEnum):
    VLLM_CUTEDSL = "vllm_cutedsl"
    VLLM_CUTEDSL_NV_TUNED = "vllm_cutedsl_nv_tuned"
    CUDNN = "cudnn"
    TORCH_CUBLASLT_NV1 = "torch_cublaslt_nv1"
    TORCH_CUBLASLT_NV2 = "torch_cublaslt_nv2"
    TORCH_MSLK_MX = "torch_mslk_mx"


class ActImpl(StrEnum):
    MX_QUANT_CUTEDSL = "mx_quant_cutedsl"
    NV_QUANT_VLLM = "nv_quant_vllm"
    SILU_MUL_THEN_MX_QUANT = "silu_mul_then_mx_quant"
    SILU_MUL_NV_FUSED = "silu_mul_nv_fused"
    SILU_MUL_C_THEN_MX_QUANT = "silu_mul_c_then_mx_quant"
    SILU_MUL_THEN_NV_QUANT = "silu_mul_then_nv_quant"


class KernelGroup(StrEnum):
    FP4_GEMM = "fp4_gemm"
    ACT_QUANT_SILU_MUL = "act_quant_silu_mul"
    EVERYTHING_ELSE = "everything_else"

    @property
    def members(self) -> tuple["KernelCategory", ...]:
        return tuple(c for c in KernelCategory if c.group is self)


class KernelCategory(StrEnum):
    FP4_GEMM = "fp4_gemm"
    ACT_QUANT = "act_quant"
    SILU_MUL = "silu_mul"
    ATTENTION = "attention"
    BF16_GEMM = "bf16_gemm"
    NORM_ROPE_ELEMENTWISE = "norm_rope_elementwise"
    SAMPLING = "sampling"
    OTHER = "other"

    @property
    def group(self) -> KernelGroup:
        if self is KernelCategory.FP4_GEMM:
            return KernelGroup.FP4_GEMM
        if self in (KernelCategory.ACT_QUANT, KernelCategory.SILU_MUL):
            return KernelGroup.ACT_QUANT_SILU_MUL
        return KernelGroup.EVERYTHING_ELSE


class CategoryPattern(NamedTuple):
    category: KernelCategory
    regex: str


class LinearLayer(StrEnum):
    """The linear layers of a Qwen3 decoder layer, in the order a step runs them."""

    QKV_PROJ = "qkv_proj"
    O_PROJ = "o_proj"
    GATE_UP_PROJ = "gate_up_proj"
    DOWN_PROJ = "down_proj"
