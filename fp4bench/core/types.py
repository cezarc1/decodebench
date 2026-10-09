"""The vocabulary of the benchmark. Every value is the string the committed data stores."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import NamedTuple, NewType, Self, TypeGuard, override

type JsonValue = bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue] | None
"""It admits any Sequence, so a NamedTuple (dumped as an array) and bytes (which json.dumps refuses)
type-check as JSON: "never a NamedTuple for a JSON object" is a review rule, not a checker's."""
type JsonObject = dict[str, JsonValue]
type Prompt = list[int]
type PromptSet = list[Prompt]

SessionId = NewType("SessionId", str)
StartId = NewType("StartId", str)
GpuUuid = NewType("GpuUuid", str)
Sha256 = NewType("Sha256", str)
Fingerprint = NewType("Fingerprint", str)


def is_number(v: object) -> TypeGuard[int | float]:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_finite(v: object) -> TypeGuard[int | float]:
    return is_number(v) and math.isfinite(v)


class Treatment(StrEnum):
    MX = "MX"
    MXP = "MXp"
    NV = "NV"
    NVA = "NVa"
    NVNF = "NVnf"
    NVX = "NVx"
    NVC = "NVc"
    NVT = "NVt"
    NVD = "NVd"
    NVV = "NVv"

    @property
    def fmt(self) -> "Format":
        return Format.MXFP4 if self in (Treatment.MX, Treatment.MXP) else Format.NVFP4


# METHODOLOGY.md#checkpoint-checks
@dataclass(frozen=True, slots=True, kw_only=True)
class FormatSpec:
    """How llm-compressor's `scheme` preset writes a format, as G5a checks it."""

    scheme: str
    ct_format: str
    group_size: int
    strategy: str
    scale_dtype: str
    global_scales: tuple[str, ...]


class Format(StrEnum):
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"

    @property
    def spec(self) -> FormatSpec:
        match self:
            case Format.MXFP4:
                return FormatSpec(
                    scheme="MXFP4",
                    ct_format="mxfp4-pack-quantized",
                    group_size=32,
                    strategy="group",
                    scale_dtype="U8",
                    global_scales=(),
                )
            case Format.NVFP4:
                return FormatSpec(
                    scheme="NVFP4",
                    ct_format="nvfp4-pack-quantized",
                    group_size=16,
                    strategy="tensor_group",
                    scale_dtype="F8_E4M3",
                    global_scales=("weight_global_scale", "input_global_scale"),
                )


class KvDtype(StrEnum):
    BF16 = "bfloat16"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"

    @property
    def is_fp8(self) -> bool:
        return self in (KvDtype.FP8, KvDtype.FP8_E4M3, KvDtype.FP8_E5M2)


@dataclass(frozen=True, slots=True)
class UnknownKvDtype:
    """A recorded KV cache dtype that is no KvDtype; it renders as the stored name."""

    name: str

    @override
    def __str__(self) -> str:
        return self.name


type RecordedKvDtype = KvDtype | UnknownKvDtype


class WaveOrder(StrEnum):
    """Which wave of an M1 pair ran first."""

    N1 = "n1"
    N2 = "n2"


class M2InvalidReason(StrEnum):
    UNDERFILLED = "underfilled"
    CAPACITY_LIMITED = "capacity_limited"
    LOOP_DETECTED = "loop_detected"
    WARMUP_TIMED_OUT = "warmup_timed_out"
    AGGREGATE_TPS_NOT_FINITE = "aggregate_tps_not_finite"
    AGGREGATE_TPS_NOT_POSITIVE = "aggregate_tps<=0"
    NUM_ERRORS = "num_errors"
    AGGREGATE_SOURCE = "aggregate_source"
    EFFECTIVE_CONCURRENCY = "effective_concurrency"
    MISSING_CELL = "missing_cell"
    PREEMPTIONS = "preemptions"


class ClockEvent(StrEnum):
    """nvidia-smi's clock-event reason counters (METHODOLOGY.md#telemetry)."""

    SW_POWER_CAPPING = "SW Power Capping"
    SYNC_BOOST = "Sync Boost"
    SW_THERMAL_SLOWDOWN = "SW Thermal Slowdown"
    HW_THERMAL_SLOWDOWN = "HW Thermal Slowdown"
    HW_POWER_BRAKING = "HW Power Braking"

    @property
    def is_env_throttle(self) -> bool:
        return self in (
            ClockEvent.SW_THERMAL_SLOWDOWN,
            ClockEvent.HW_THERMAL_SLOWDOWN,
            ClockEvent.HW_POWER_BRAKING,
        )


class Verdict(StrEnum):
    EQUIVALENT = "equivalent"
    MXFP4_FASTER = "mxfp4_faster"
    NVFP4_FASTER = "nvfp4_faster"
    DIFFERENT_SMALL = "different_small"
    INCONCLUSIVE = "inconclusive"
    PINNED_FASTER = "pinned_faster"
    ALT_FASTER = "alt_faster"
    FUSION_SLOWER = "fusion_slower"
    FUSION_FASTER = "fusion_faster"


class Gate(StrEnum):
    G1 = "G1_kernels"
    G2 = "G2_bytes"
    G3 = "G3_aa"
    G4 = "G4_env_throttle"
    G5A = "G5a_structure"
    G5B = "G5b_nll"
    G6A = "G6a_integrity"
    G6B = "G6b_m2_validity"
    G7 = "G7_same_gpu_per_round"


class CheckpointKind(StrEnum):
    MX = "mx"
    NV = "nv"
    NVX = "nvx"

    @property
    def fmt(self) -> Format:
        match self:
            case CheckpointKind.MX:
                return Format.MXFP4
            case CheckpointKind.NV | CheckpointKind.NVX:
                return Format.NVFP4

    @staticmethod
    def ours(fmt: Format) -> "CheckpointKind":
        match fmt:
            case Format.MXFP4:
                return CheckpointKind.MX
            case Format.NVFP4:
                return CheckpointKind.NV


class FetchKind(StrEnum):
    """`fp4bench fetch --kind`: our checkpoints, and NVIDIA's NVFP4 one."""

    MX = "mx"
    NV = "nv"
    NVIDIA = "nvidia"

    @property
    def checkpoint(self) -> CheckpointKind:
        match self:
            case FetchKind.MX:
                return CheckpointKind.MX
            case FetchKind.NV:
                return CheckpointKind.NV
            case FetchKind.NVIDIA:
                return CheckpointKind.NVX


class ExecutorKind(StrEnum):
    MODAL = "modal"
    LOCAL = "local"


class LinearKernel(StrEnum):
    """vLLM's linear-layer kernel classes, as its server log names them."""

    FLASHINFER_CUTEDSL_NVFP4 = "FlashInferCuteDslNvFp4LinearKernel"
    FLASHINFER_CUTLASS_NVFP4 = "FlashInferCutlassNvFp4LinearKernel"
    FLASHINFER_TRTLLM_NVFP4 = "FlashInferTrtllmNvFp4LinearKernel"
    FLASHINFER_CUDNN_NVFP4 = "FlashInferCudnnNvFp4LinearKernel"
    CUTLASS_NVFP4 = "CutlassNvFp4LinearKernel"
    FLASHINFER_MXFP4 = "FlashInferMxFp4LinearKernel"


class LinearBackend(StrEnum):
    """vLLM's --linear-backend for NVFP4 (METHODOLOGY.md#kernel-selection)."""

    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    CUTLASS = "cutlass"

    @property
    def kernel(self) -> LinearKernel:
        return _NVFP4_KERNEL_OF_BACKEND[self]


_NVFP4_KERNEL_OF_BACKEND = {
    LinearBackend.FLASHINFER_CUTEDSL: LinearKernel.FLASHINFER_CUTEDSL_NVFP4,
    LinearBackend.FLASHINFER_CUTLASS: LinearKernel.FLASHINFER_CUTLASS_NVFP4,
    LinearBackend.FLASHINFER_TRTLLM: LinearKernel.FLASHINFER_TRTLLM_NVFP4,
    LinearBackend.FLASHINFER_CUDNN: LinearKernel.FLASHINFER_CUDNN_NVFP4,
    LinearBackend.CUTLASS: LinearKernel.CUTLASS_NVFP4,
}


class ActQuantFusion(StrEnum):
    """Whether vLLM fuses the activation quantization into the kernel before it."""

    ON = "on"
    OFF = "off"


class RatioName(StrEnum):
    """The ratios the summaries, the JSON and the figures render; R is the one the verdict reads."""

    R = "R"
    D = "D"
    K = "K"
    F = "F"
    PHI = "Phi"
    AA = "AA"

    @property
    def symbol(self) -> str:
        return "Φ" if self is RatioName.PHI else self.value


class VerdictLabels(StrEnum):
    """Which pair of verdict labels a ratio's verdict reads as."""

    FORMAT = "format"
    KERNEL = "kernel"
    FUSION = "fusion"


class ContrastName(StrEnum):
    E_TOK = "E_tok"
    E_BATCH = "E_batch"


class Hypothesis(StrEnum):
    H_BATCH = "H_batch"
    H_TOKENS = "H_tokens"


class SkippedMetric(StrEnum):
    M1 = "m1"
    M2_THROUGHPUT = "m2_throughput"
    M2_ITL = "m2_itl"
    M2_PER_USER = "m2_per_user"


class NvnfCacheStatus(StrEnum):
    NOT_APPLICABLE = "not applicable"
    UNVERIFIED = "unverified"
    FAIL = "fail"
    PASS = "pass"  # noqa: S105 - a check's outcome, not a credential


class ReproStatus(StrEnum):
    NO_DATA = "no data"
    PASS = "pass"  # noqa: S105 - a check's outcome, not a credential
    FAIL = "fail"


class DesignKind(StrEnum):
    DEVIATES = "deviates from §16"
    EVERY_CELL = "every pre-registered cell"
    SUBSET = "a subset of the cells (smoke)"


class DecidedBy(Enum):
    """What picked the kernel scan's selection: the selection C, the tie-break C or the order.

    Not a StrEnum: a Margin stores `to_json(...)` ("C=32", "C=128", "order"), never `.value`.
    """

    SELECTION_C = "selection_c"
    TIEBREAK_C = "tiebreak_c"
    ORDER = "order"

    def to_json(self, *, selection_c: int, tiebreak_c: int) -> str:
        match self:
            case DecidedBy.SELECTION_C:
                return f"C={selection_c}"
            case DecidedBy.TIEBREAK_C:
                return f"C={tiebreak_c}"
            case DecidedBy.ORDER:
                return "order"


class KernelMatch(StrEnum):
    """Whether a treatment's sessions ran the kernel G1 expects, as the scan table shows it."""

    NO_SESSION = "no session"
    MATCH = "ok"
    MISMATCH = "MISMATCH"
    FAILED = "failed"


class Arm(StrEnum):
    TOKEN = "token"  # noqa: S105 - the token arm, not a credential
    BATCH = "batch"


class Cell(NamedTuple):
    batch: int
    prompt_len: int

    @property
    def json_key(self) -> str:
        return f"C={self.batch},P={self.prompt_len}"

    @property
    def prompt_file_key(self) -> str:
        return f"{self.batch}x{self.prompt_len}"

    @property
    def label(self) -> str:
        return f"({self.batch}, {self.prompt_len:,})"


class ValueKey[K](NamedTuple):
    round: int
    treatment: Treatment
    key: K


class SessionSlot(NamedTuple):
    round: int
    treatment: Treatment


class ChatFrame(NamedTuple):
    """The chat template's token ids before and after the user content."""

    pre: Prompt
    post: Prompt

    @property
    def n_tokens(self) -> int:
        return len(self.pre) + len(self.post)

    def room(self, length: int) -> int:
        return length - self.n_tokens

    def frames(self, prompt: Prompt) -> bool:
        return (
            prompt[: len(self.pre)] == self.pre
            and prompt[len(prompt) - len(self.post) :] == self.post
        )


class RetryPolicy(NamedTuple):
    attempts: int
    backoff_s: float
    backoff_factor: float = 1.0

    @property
    def pauses(self) -> list[float]:
        return [self.backoff_s * self.backoff_factor**k for k in range(self.attempts - 1)]


class Estimate(NamedTuple):
    value: float
    lo: float
    hi: float


_ERROR_PREFIX = "error: "
_TYPE_SEPARATOR = ": "


@dataclass(frozen=True, slots=True)
class ProbeError:
    """What a probe raised: stored as "error: <type>: <message>" or as {"error": "<type>: …"}."""

    type_name: str
    message: str

    @classmethod
    def of(cls, exc: BaseException) -> Self:
        return cls(type(exc).__name__, str(exc))

    @property
    def detail(self) -> str:
        return f"{self.type_name}{_TYPE_SEPARATOR}{self.message}"

    def to_json(self) -> str:
        return f"{_ERROR_PREFIX}{self.detail}"

    def to_json_object(self) -> JsonObject:
        return {"error": self.detail}

    @classmethod
    def from_json(cls, stored: JsonValue) -> Self | None:
        """The error either stored form holds; None for any other value."""
        if isinstance(stored, Mapping):
            detail = stored.get("error") if stored.keys() == {"error"} else None
        elif isinstance(stored, str) and stored.startswith(_ERROR_PREFIX):
            detail = stored.removeprefix(_ERROR_PREFIX)
        else:
            detail = None
        if not isinstance(detail, str):
            return None
        type_name, separator, message = detail.partition(_TYPE_SEPARATOR)
        return cls(type_name, message) if type_name and separator else None


@dataclass(frozen=True, slots=True)
class Raised:
    """What `attempt`'s call raised: never what it returned, even an exception."""

    exc: Exception


def attempt[T](fn: Callable[[], T]) -> T | Raised:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a probe records whatever went wrong as its value
        return Raised(exc)


def probe[T](fn: Callable[[], T]) -> T | ProbeError:
    result = attempt(fn)
    return ProbeError.of(result.exc) if isinstance(result, Raised) else result


def error_as_text[T](value: T | ProbeError) -> T | str:
    return value.to_json() if isinstance(value, ProbeError) else value


def error_as_object[T](value: T | ProbeError) -> T | JsonObject:
    return value.to_json_object() if isinstance(value, ProbeError) else value
