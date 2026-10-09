"""Typed JSONL rows (METHODOLOGY.md#jsonl)."""

import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import Field, dataclass, field, fields
from dataclasses import replace as dataclass_replace
from enum import Enum
from pathlib import Path
from typing import ClassVar, Final, Self, overload, override

from fp4bench.core.types import Cell, JsonObject, JsonValue, Treatment

SCHEMA_VERSION = 2
VERSION_KEY = "schema_version"
V1_PROMPT_LEN = 1024

_ABSENT: Final = object()


@overload
def absent() -> None: ...
@overload
def absent[T](default: T) -> T: ...
def absent(default: object = None) -> object:
    """A field a stored row may lack: it reads as `default`, and to_json leaves it out."""
    return field(default=_ABSENT, metadata={"absent": default})


def _json_value(value: "JsonValue | Record | Enum") -> JsonValue:
    if isinstance(value, Record):
        return value.to_json()
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True, kw_only=True)
class Record:
    """Fields in stored order; `extra` holds unknown keys, `missing` the fields to_json omits."""

    _BASE: ClassVar[frozenset[str]] = frozenset({"extra", "missing"})

    extra: dict[str, JsonValue] = field(default_factory=dict)
    missing: frozenset[str] = field(init=False, default=frozenset(), compare=False, repr=False)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # @dataclass gives each class it decorates its own __replace__ unless the class has one
        cls.__replace__ = cls.replace

    def replace(self, **changes: object) -> Self:
        if derived := sorted(f.name for f in fields(self) if not f.init and f.name in changes):
            raise TypeError(
                f"{type(self).__name__}.replace() cannot set {derived}: they are not "
                "constructor arguments"
            )
        new = dataclass_replace(self, **changes)
        object.__setattr__(new, "missing", self.missing - changes.keys())
        return new

    __replace__ = replace

    @classmethod
    def _fields(cls) -> tuple[Field[object], ...]:
        return tuple(f for f in fields(cls) if f.name not in cls._BASE)

    def __post_init__(self) -> None:
        names = {f.name for f in self._fields()}
        clash = sorted((names | {VERSION_KEY}) & set(self.extra))
        if clash:
            raise ValueError(f"{type(self).__name__}.extra repeats keys of the schema: {clash}")
        missing = set()
        for f in self._fields():
            value = getattr(self, f.name)
            if value is _ABSENT:
                object.__setattr__(self, f.name, f.metadata["absent"])
                missing.add(f.name)
            elif (parsed := _parse(f, value)) is not value:
                object.__setattr__(self, f.name, parsed)
        object.__setattr__(self, "missing", frozenset(missing))

    @classmethod
    def from_json(cls, obj: JsonValue) -> Self:
        if not isinstance(obj, Mapping):
            raise ValueError(f"{cls.__name__}: expected a JSON object, got {obj!r:.200}")
        declared = {f.name: f for f in cls._fields()}
        lacking = [
            name for name, f in declared.items() if name not in obj and "absent" not in f.metadata
        ]
        if lacking:
            raise ValueError(f"{cls.__name__} row lacks required keys {lacking}: {obj!r:.300}")
        return cls(
            **{k: v for k, v in obj.items() if k in declared},
            extra={k: v for k, v in obj.items() if k not in declared},
        )

    def to_json(self) -> JsonObject:
        out = {
            f.name: _json_value(getattr(self, f.name))
            for f in self._fields()
            if f.name not in self.missing
        }
        return out | self.extra


def _parse(f: Field[object], value: JsonValue) -> object:
    kind = f.type
    if kind is Treatment:
        return value if type(value) is Treatment else Treatment(value)
    if isinstance(kind, type) and issubclass(kind, Record) and not isinstance(value, Record):
        return kind.from_json(value)
    return value


@dataclass(frozen=True, kw_only=True)
class Row(Record):
    """A versioned JSONL line; `stored_version` is the version it was read with (None for v1)."""

    _BASE: ClassVar[frozenset[str]] = Record._BASE | {"stored_version"}  # noqa: SLF001

    stored_version: int | None = field(init=False, default=None, compare=False, repr=False)

    @classmethod
    @override
    def from_json(cls, obj: JsonValue) -> Self:
        version = None
        if isinstance(obj, Mapping) and VERSION_KEY in obj:
            version = obj[VERSION_KEY]
            if type(version) is not int or version != SCHEMA_VERSION:
                raise ValueError(
                    f"{cls.__name__}: unknown schema_version {version!r} (a v1 row "
                    f"has none; this code writes {SCHEMA_VERSION})"
                )
            obj = {k: v for k, v in obj.items() if k != VERSION_KEY}
        row = super().from_json(obj)
        object.__setattr__(row, "stored_version", version)
        return row

    @override
    def replace(self, **changes: object) -> Self:
        new = super().replace(**changes)
        object.__setattr__(new, "stored_version", self.stored_version)
        return new

    @override
    def to_json(self) -> JsonObject:
        return {VERSION_KEY: SCHEMA_VERSION, **super().to_json()}

    def as_stored(self) -> JsonObject:
        """The row as its line holds it: to_json, without schema_version for a v1 line."""
        return self.to_json() if self.stored_version is not None else super().to_json()


@dataclass(frozen=True, kw_only=True)
class BlockTelemetry(Record):
    window_s: float
    counters_delta: dict[str, int]
    env_throttle_us: int
    sw_power_cap_frac: float


@dataclass(frozen=True, kw_only=True)
class M1Row(Row):
    round: int
    treatment: Treatment
    session_id: str
    c: int
    set: int
    warmup: bool
    first: str | None = absent()
    n1: int
    n2: int
    t1_s: float | None
    t2_s: float | None
    step_s: float | None
    decode_tok_s: float | None
    prompt_len: int = absent(V1_PROMPT_LEN)
    block_telemetry: BlockTelemetry
    preemptions_delta: float | None = absent()

    @property
    def cell(self) -> Cell:
        return Cell(self.c, self.prompt_len)


@dataclass(frozen=True, kw_only=True)
class M2Row(Row):
    round: int
    treatment: Treatment
    session_id: str
    c: int
    valid: bool
    invalid_reasons: list[str]
    failure_reason: str
    aggregate_tps: float | None
    aggregate_source: str | None = absent()
    measurement_seconds: float | None
    effective_concurrency: float | None
    avg_running_reqs: float | None
    itl_p50_ms: float | None
    tps_per_user_p50: float | None
    ttft_p50_ms: float | None
    server_gen_throughput: float | None
    preemptions_delta: float | None = absent()
    telemetry: BlockTelemetry


@dataclass(frozen=True, kw_only=True)
class ServerRow(Row):
    """One `vllm serve` session, or a scan session that failed; older rows lack later facts."""

    round: int
    treatment: Treatment
    session_id: str
    start_id: str | None = absent()
    gpu_uuid: str | None = absent()
    server_argv: list[str] | None = absent()
    nll: float | None = absent()
    nll_per_prompt: list[float] | None = absent()
    moe_backends: list[str] | None = absent()
    moe_backend_lines: list[str] | None = absent()
    weights_gib: float | None = absent()
    cudagraph_capture_sizes: list[int] | None = absent()
    sampling_defaults_lines: list[str] | None = absent()
    linear_kernels: list[str] | None = absent()
    linear_kernel_lines: list[str] | None = absent()
    linear_backend_fallback_lines: list[str] | None = absent()
    bf16_gemm_lines: list[str] | None = absent()
    autotune_lines: list[str] | None = absent()
    autotune_cache_files: list[str] | None = absent()
    autotune_ran: bool | None = absent()
    autotune_cache_loaded: bool | None = absent()
    pass_config: dict[str, bool | None] | None = absent()
    pass_config_conflict: bool | None = absent()
    fuse_act_quant: bool | None = absent()
    custom_fusion_lines: list[str] | None = absent()
    custom_fusions: list[str] | None = absent()
    kv_cache_tokens: int | None = absent()
    kv_cache_memory_gib: float | None = absent()
    attention_backend_lines: list[str] | None = absent()
    compile_cache_lines: list[str] | None = absent()
    compile_cache_hashes: list[str] | None = absent()
    server_env: dict[str, str] | None = absent()
    autotune_cache_dir: str | None = absent()
    autotune_fresh: bool | None = absent()
    autotune_saved_files: dict[str, str] | None = absent()
    failed: bool = absent(False)
    error: str | None = absent()


@dataclass(frozen=True, kw_only=True)
class ManifestLine(Row):
    """One line per container start."""

    utc: str | None = absent()
    image: str | None = absent()
    vllm_build_commit: str | None = absent()
    llm_inference_bench_commit: str | None = absent()
    gpu: dict[str, str] | None = absent()
    packages: dict[str, str | None] | None = absent()
    cuda: str | None = absent()
    code_commit: str | None = absent()
    code_dirty: bool | None = absent()
    study: str | None = absent()
    start_id: str | None = absent()
    inputs: dict[str, JsonValue] | None = absent()
    checkpoint_report: dict[str, JsonValue] | str | None = absent()
    protocol: dict[str, JsonValue] | None = absent()
    problems: list[str] | None = absent()


def _numbered_rows(path: Path) -> list[tuple[int, JsonValue]]:
    path = Path(path)
    if not path.exists():
        return []
    lines = [(n, line) for n, line in enumerate(path.read_text().splitlines(), 1) if line.strip()]
    rows = []
    for i, (n, line) in enumerate(lines):
        try:
            rows.append((n, json.loads(line)))
        except json.JSONDecodeError as exc:
            if i < len(lines) - 1:
                raise ValueError(f"{path}, line {n}: {exc}") from exc
            print(
                f"warning: ignoring truncated last line of {path} (line {n}): {line[:80]!r}",
                file=sys.stderr,
            )
    return rows


def load_rows[R: Record](
    path: Path, cls: type[R], skip_if_bad: Callable[[JsonValue], bool] | None = None
) -> list[R]:
    """A row that does not load raises, naming file and line, unless `skip_if_bad(row)`."""
    out = []
    for n, row in _numbered_rows(path):
        try:
            out.append(cls.from_json(row))
        except ValueError as exc:
            if skip_if_bad is None or not skip_if_bad(row):
                raise ValueError(f"{path}, line {n}: {exc}") from exc
    return out


def _finite(obj: JsonValue) -> JsonValue:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _repair_tail(path: Path) -> None:
    if not path.exists():
        return
    with path.open("rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        head, newline, tail = data.rpartition(b"\n")
        try:
            json.loads(tail)
        except ValueError:
            print(f"warning: dropping truncated last line of {path}", file=sys.stderr)
            f.truncate(len(head) + len(newline))
        else:
            f.write(b"\n")


def append_jsonl(path: Path, row: JsonObject) -> None:
    """Non-finite floats are written as null."""
    line = json.dumps(_finite(row), allow_nan=False)
    path = Path(path)
    _repair_tail(path)
    with path.open("a") as f:
        f.write(line + "\n")


def append_row(path: Path, row: Record) -> None:
    append_jsonl(path, row.to_json())
