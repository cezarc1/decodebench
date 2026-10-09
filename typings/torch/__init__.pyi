# Installed only in the Modal image. Declares just the names fp4bench uses; add a name when code
# starts to use it (pyright and ty both read typings/).
from typing import Any

# What fp4bench uses of torch, including _scaled_mm: torch's own entry to the cuBLASLt NVFP4 GEMM,
# which the microbenchmark times (pyright strict would take the underscore for a private name).
__all__ = [
    "Tensor",
    "__config__",
    "__version__",
    "_scaled_mm",
    "backends",
    "bfloat16",
    "compile",
    "cuda",
    "empty",
    "empty_like",
    "exp2",
    "float4_e2m1fn_x2",
    "float8_e4m3fn",
    "float8_e8m0fnu",
    "float32",
    "int32",
    "linalg",
    "manual_seed",
    "nn",
    "no_grad",
    "ones",
    "ops",
    "randn",
    "stack",
    "tensor",
    "uint8",
    "version",
]

__config__: Any
__version__: str
_scaled_mm: Any
backends: Any
bfloat16: Any
compile: Any  # noqa: A001 - torch's name
cuda: Any
empty: Any
empty_like: Any
exp2: Any
float32: Any
float4_e2m1fn_x2: Any
float8_e4m3fn: Any
float8_e8m0fnu: Any
int32: Any
linalg: Any
manual_seed: Any
nn: Any
no_grad: Any
ones: Any
ops: Any
randn: Any
stack: Any
type Tensor = Any
tensor: Any
uint8: Any
version: Any
