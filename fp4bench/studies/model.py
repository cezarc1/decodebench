"""Qwen3-32B: its checkpoints, where they live, and how each treatment serves them."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from fp4bench import settings
from fp4bench.core.argv import flag_value
from fp4bench.core.types import (
    ActQuantFusion,
    CheckpointKind,
    Format,
    LinearBackend,
    LinearKernel,
    Treatment,
)

SRC_MODEL_ID = "Qwen/Qwen3-32B"
SRC_MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
NVX_MODEL_ID = "nvidia/Qwen3-32B-NVFP4"
NVX_MODEL_REVISION = "16426c6eb87be9e27c14cc9fb318f9c7a5f8588c"

BF16_MODEL_DIR = f"{settings.DATA_DIR}/qwen3-32b-bf16"
MX_MODEL_DIR = f"{settings.DATA_DIR}/qwen3-32b-mxfp4"
NV_MODEL_DIR = f"{settings.DATA_DIR}/qwen3-32b-nvfp4"
NVX_MODEL_DIR = f"{settings.DATA_DIR}/nvidia-qwen3-32b-nvfp4"

# METHODOLOGY.md#kernel-selection
DEFAULT_NVFP4_KERNEL = LinearKernel.FLASHINFER_CUTEDSL_NVFP4
MXFP4_KERNEL = LinearKernel.FLASHINFER_MXFP4


def pinned_to(backend: LinearBackend) -> tuple[str, ...]:
    return ("--linear-backend", backend.value)


PINNED_KERNEL = pinned_to(LinearBackend.FLASHINFER_CUTEDSL)
NO_ACT_QUANT_FUSION = ("--compilation-config", '{"pass_config": {"fuse_act_quant": false}}')


def expected_linear_kernel(treatment: Treatment, args: tuple[str, ...]) -> LinearKernel:
    """The class G1 expects (METHODOLOGY.md#kernel-selection); unknown backends raise ValueError."""
    if treatment.fmt is Format.MXFP4:
        return MXFP4_KERNEL
    if treatment in (Treatment.NV, Treatment.NVX, Treatment.NVNF):
        return LinearBackend.FLASHINFER_CUTEDSL.kernel
    backend = flag_value(args, "--linear-backend")
    return DEFAULT_NVFP4_KERNEL if backend is None else LinearBackend(backend).kernel


def expected_act_quant_fusion(treatment: Treatment, args: tuple[str, ...]) -> ActQuantFusion:
    """Off for MXFP4, and where the args' --compilation-config turns pass_config.fuse_act_quant
    off (NO_ACT_QUANT_FUSION); malformed JSON raises ValueError."""
    if treatment.fmt is Format.MXFP4:
        return ActQuantFusion.OFF
    config = flag_value(args, "--compilation-config")
    parsed = None if config is None else json.loads(config)
    passes = parsed.get("pass_config") if isinstance(parsed, dict) else None
    off = isinstance(passes, dict) and passes.get("fuse_act_quant") is False
    return ActQuantFusion.OFF if off else ActQuantFusion.ON


def local_copy(volume_dir: str) -> str:
    """Where a run stages a volume dir and serves it from."""
    return f"{settings.LOCAL_DIR}/{volume_dir.rsplit('/', 1)[-1]}"


@dataclass(frozen=True)
class TreatmentSpec:
    checkpoint: CheckpointKind
    volume_dir: str
    model_dir: str
    server_args: tuple[str, ...]
    linear_kernel: LinearKernel
    act_quant_fusion: bool


def volume_dir(kind: CheckpointKind) -> str:
    match kind:
        case CheckpointKind.MX:
            return MX_MODEL_DIR
        case CheckpointKind.NV:
            return NV_MODEL_DIR
        case CheckpointKind.NVX:
            return NVX_MODEL_DIR


def treatment_spec(
    treatment: Treatment, checkpoint: CheckpointKind, server_args: tuple[str, ...]
) -> TreatmentSpec:
    directory = volume_dir(checkpoint)
    return TreatmentSpec(
        checkpoint,
        directory,
        local_copy(directory),
        server_args,
        expected_linear_kernel(treatment, server_args),
        expected_act_quant_fusion(treatment, server_args) is ActQuantFusion.ON,
    )


_T, _K, _B = Treatment, CheckpointKind, LinearBackend
TREATMENTS: Mapping[Treatment, TreatmentSpec] = MappingProxyType(
    {
        _T.MX: treatment_spec(_T.MX, _K.MX, PINNED_KERNEL),
        _T.NV: treatment_spec(_T.NV, _K.NV, PINNED_KERNEL),
        # METHODOLOGY.md#nv-alt: the kernel scan's choice
        _T.NVA: treatment_spec(_T.NVA, _K.NV, pinned_to(_B.FLASHINFER_CUDNN)),
        _T.MXP: treatment_spec(_T.MXP, _K.MX, PINNED_KERNEL),
        _T.NVNF: treatment_spec(_T.NVNF, _K.NV, (*PINNED_KERNEL, *NO_ACT_QUANT_FUSION)),
        _T.NVX: treatment_spec(_T.NVX, _K.NVX, PINNED_KERNEL),
        _T.NVC: treatment_spec(_T.NVC, _K.NV, pinned_to(_B.FLASHINFER_CUTLASS)),
        _T.NVT: treatment_spec(_T.NVT, _K.NV, pinned_to(_B.FLASHINFER_TRTLLM)),
        _T.NVD: treatment_spec(_T.NVD, _K.NV, pinned_to(_B.FLASHINFER_CUDNN)),
        _T.NVV: treatment_spec(_T.NVV, _K.NV, pinned_to(_B.CUTLASS)),
    }
)
