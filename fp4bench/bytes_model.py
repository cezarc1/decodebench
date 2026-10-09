# METHODOLOGY.md#bytes-model
from collections.abc import Mapping
from typing import Final

from fp4bench.core.types import Format, KvDtype

N_LAYERS: Final = 64
HIDDEN: Final = 5120
INTERMEDIATE: Final = 25600
N_Q_HEADS: Final = 64
N_KV_HEADS: Final = 8
HEAD_DIM: Final = 128
VOCAB: Final = 151_936

ATTN_PARAMS_PER_LAYER: Final = (
    2 * HIDDEN * N_Q_HEADS * HEAD_DIM + 2 * HIDDEN * N_KV_HEADS * HEAD_DIM
)
MLP_PARAMS_PER_LAYER: Final = 3 * HIDDEN * INTERMEDIATE
QUANT_PARAMS: Final = N_LAYERS * (ATTN_PARAMS_PER_LAYER + MLP_PARAMS_PER_LAYER)
LM_HEAD_BYTES: Final = 2 * VOCAB * HIDDEN
KV_ELEMENTS_PER_TOKEN: Final = N_LAYERS * N_KV_HEADS * HEAD_DIM * 2
BITS: Final[Mapping[Format, float]] = {Format.MXFP4: 4.25, Format.NVFP4: 4.5}


def weight_bytes(fmt: Format) -> float:
    return QUANT_PARAMS * BITS[fmt] / 8


def kv_bytes_per_element(kv: KvDtype) -> int:
    match kv:
        case KvDtype.BF16:
            return 2
        case KvDtype.FP8 | KvDtype.FP8_E4M3 | KvDtype.FP8_E5M2:
            return 1


def kv_bytes_per_token(kv_dtype: KvDtype = KvDtype.BF16) -> int:
    return KV_ELEMENTS_PER_TOKEN * kv_bytes_per_element(kv_dtype)


def kv_bytes_per_seq(ctx: int, kv_dtype: KvDtype = KvDtype.BF16) -> int:
    return kv_bytes_per_token(kv_dtype) * ctx


def step_bytes(c: int, ctx: int, fmt: Format, kv_dtype: KvDtype = KvDtype.BF16) -> float:
    """HBM bytes read by one decode step of `c` sequences, each at context `ctx`."""
    return weight_bytes(fmt) + LM_HEAD_BYTES + c * kv_bytes_per_seq(ctx, kv_dtype)


def r_ideal(c: int, ctx: int, kv_dtype: KvDtype = KvDtype.BF16) -> float:
    """Upper bound on t_NV / t_MX if decode were purely bandwidth-bound with no fixed costs."""
    return step_bytes(c, ctx, Format.NVFP4, kv_dtype) / step_bytes(c, ctx, Format.MXFP4, kv_dtype)
