"""Descriptive checks that decide nothing (METHODOLOGY.md#descriptive-checks)."""

import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fp4bench import settings
from fp4bench.analysis import stats
from fp4bench.analysis.cells import (
    Values,
    mean_nll,
    median_value,
    sessions_of,
    usable_rounds_of,
)
from fp4bench.analysis.inputs import (
    bf16_reference,
    kv_cache_dtype,
    nll_prompts_sha,
    protocol,
    protocol_cells,
)
from fp4bench.bytes_model import r_ideal, step_bytes
from fp4bench.core.schema import ManifestLine, ServerRow, load_rows
from fp4bench.core.types import (
    Cell,
    DesignKind,
    Format,
    KvDtype,
    RecordedKvDtype,
    Treatment,
    UnknownKvDtype,
    is_finite,
)
from fp4bench.studies.expc import (
    EXPC_TREATMENTS,
    HF_OVERRIDES,
    MAIN_RUN_NLL,
    MAIN_RUN_NLL_PROMPTS_SHA256,
    REGISTERED_CELLS,
    ROPE_SCALING_MARKERS,
    SERVER,
)
from fp4bench.studies.main import FULL


@dataclass(frozen=True)
class NllDifference:
    mean: float
    lo: float
    hi: float
    level: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class ReferenceNll:
    this: float | None
    reference: float | None
    this_minus_reference: float | None


@dataclass(frozen=True)
class ReferenceComparison:
    run: str
    kv_cache_dtype: str
    this_kv_cache_dtype: str
    per_treatment: dict[Treatment, ReferenceNll]
    problems: list[str]


@dataclass(frozen=True)
class AccuracyNote:
    bf16: float | None
    mx: float | None
    nv: float | None
    nv_minus_mx: NllDifference | None
    n_windows: int | None
    problems: list[str]
    reference_run: ReferenceComparison | None = None


def _window_nlls(
    sessions: Sequence[ServerRow], treatment: Treatment
) -> tuple[list[float] | None, str | None]:
    mine = [s for s in sessions if s.treatment == treatment]
    if not mine:
        return None, f"no {treatment} sessions"
    vectors = []
    for s in mine:
        v = s.nll_per_prompt
        if not (isinstance(v, list) and v and all(is_finite(x) for x in v)):
            return None, f"{treatment} round {s.round} has no usable nll_per_prompt"
        vectors.append(v)
    if len({len(v) for v in vectors}) != 1:
        return None, f"{treatment} sessions disagree on the number of windows"
    return [statistics.fmean(window) for window in zip(*vectors, strict=True)], None


def accuracy_note(sessions: Sequence[ServerRow], manifest: ManifestLine | None) -> AccuracyNote:
    """MX and NV only: the other treatments serve the same checkpoints."""
    reference, reference_problem = bf16_reference(manifest)
    problems = [] if reference_problem is None else [f"BF16 reference: {reference_problem}"]
    mx_windows, mx_problem = _window_nlls(sessions, Treatment.MX)
    nv_windows, nv_problem = _window_nlls(sessions, Treatment.NV)
    problems += [p for p in (mx_problem, nv_problem) if p]
    diff, n_windows = None, None
    if mx_windows is not None and nv_windows is not None:
        if len(mx_windows) != len(nv_windows):
            problems.append(f"MX has {len(mx_windows)} windows, NV has {len(nv_windows)}")
        else:
            n_windows = len(mx_windows)
            mean, lo, hi = stats.paired_bootstrap_ci(
                [nv - mx for nv, mx in zip(nv_windows, mx_windows, strict=True)],
                settings.BOOTSTRAP_LEVEL,
                settings.BOOTSTRAP_RESAMPLES,
                settings.BOOTSTRAP_SEED,
            )
            diff = NllDifference(
                mean,
                lo,
                hi,
                settings.BOOTSTRAP_LEVEL,
                settings.BOOTSTRAP_RESAMPLES,
                settings.BOOTSTRAP_SEED,
            )
            ref_windows = None if reference is None else reference.get("per_prompt")
            if isinstance(ref_windows, list) and len(ref_windows) != n_windows:
                problems.append(
                    f"the BF16 reference has {len(ref_windows)} windows, the "
                    f"sessions have {n_windows}"
                )
    return AccuracyNote(
        bf16=None if reference is None else reference["mean"],
        mx=mean_nll(sessions, Treatment.MX),
        nv=mean_nll(sessions, Treatment.NV),
        nv_minus_mx=diff,
        n_windows=n_windows,
        problems=problems,
    )


def treatment_order(treatments: Iterable[Treatment]) -> list[Treatment]:
    """The main run's order first, then the rest sorted."""
    present = set(treatments)
    return [t for t in FULL.treatments if t in present] + sorted(present - set(FULL.treatments))


def reference_nll_comparison(
    sessions: Sequence[ServerRow], manifest: ManifestLine | None, reference_dir: Path
) -> ReferenceComparison:
    reference_dir = Path(reference_dir)
    ref_sessions = sessions_of(load_rows(reference_dir / "servers.jsonl", ServerRow))
    ref_manifests = load_rows(reference_dir / "manifests.jsonl", ManifestLine)
    ref_manifest = ref_manifests[-1] if ref_manifests else None
    this_dtype, ref_dtype = kv_cache_dtype(manifest), kv_cache_dtype(ref_manifest)
    problems = [] if ref_sessions else [f"the reference run {reference_dir} has no sessions"]
    if this_dtype == ref_dtype:
        problems.append(
            f"the reference run serves the same KV cache dtype ({ref_dtype}) as this "
            f"run: the difference is not the KV dtype's"
        )
    this_sha, ref_sha = nll_prompts_sha(manifest), nll_prompts_sha(ref_manifest)
    if this_sha is not None and ref_sha is not None and this_sha != ref_sha:
        problems.append(
            f"the runs' NLL windows differ (inputs.nll_prompts_sha256: this run "
            f"{this_sha}, the reference {ref_sha})"
        )
    per_treatment = {}
    for t in treatment_order({s.treatment for s in sessions}):
        this, other = mean_nll(sessions, t), mean_nll(ref_sessions, t)
        if other is None and ref_sessions:
            problems.append(f"the reference run has no usable mean NLL for {t}")
        per_treatment[t] = ReferenceNll(
            this, other, None if this is None or other is None else this - other
        )
    return ReferenceComparison(
        str(reference_dir), str(ref_dtype), str(this_dtype), per_treatment, problems
    )


@dataclass(frozen=True)
class Bandwidth:
    format: Format
    rounds: int
    step_s: float | None
    bytes: float | None
    bytes_per_s: float | None
    fraction_of_peak: float | None


BANDWIDTH_TREATMENTS = (Treatment.MX, Treatment.MXP, Treatment.NV, Treatment.NVA, Treatment.NVNF)


def r_ideal_or_none(c: int, context: int, kv_dtype: RecordedKvDtype) -> float | None:
    match kv_dtype:
        case KvDtype():
            return r_ideal(c, context, kv_dtype)
        case UnknownKvDtype():
            return None


def effective_bandwidth(
    step: Values[int], kv_dtype: RecordedKvDtype, context: int
) -> dict[Treatment, dict[int, Bandwidth]]:
    """Bytes-model step bytes over the median M1 step; bytes None for an unknown KV dtype."""
    out: dict[Treatment, dict[int, Bandwidth]] = {}
    for t in BANDWIDTH_TREATMENTS:
        fmt = t.fmt
        for c in sorted({cc for (_, tt, cc) in step if tt == t}):
            median = median_value(step, t, c)
            nbytes = (
                step_bytes(c, context, fmt, kv_dtype=kv_dtype)
                if isinstance(kv_dtype, KvDtype)
                else None
            )
            rate = None if nbytes is None or median is None else nbytes / median
            out.setdefault(t, {})[c] = Bandwidth(
                fmt,
                usable_rounds_of(step, t, c),
                median,
                nbytes,
                rate,
                None if rate is None else rate / settings.HBM_PEAK_BYTES_S,
            )
    return out


def _parse_overrides(value: object) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def flag_value(argv: Sequence[str], flag: str) -> str | None:
    """The value of the last `flag` ("--flag value" or "--flag=value"), None if absent."""
    value = None
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(flag + "="):
            value = arg[len(flag) + 1 :]
    return value


@dataclass(frozen=True)
class ArgvCheck:
    sessions_checked: int
    problems: list[str]


def server_argv_check(sessions: Sequence[ServerRow]) -> ArgvCheck:
    """Every served argv has §16's window and override and no RoPE scaling."""
    problems = [] if sessions else ["no sessions: no served argv to check"]
    for s in sorted(sessions, key=lambda s: (s.round, s.treatment)):
        who = f"round {s.round} {s.treatment}"
        argv = s.server_argv
        if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
            problems.append(f"{who}: no server_argv")
            continue
        window = flag_value(argv, "--max-model-len")
        if window != str(SERVER.max_model_len):
            problems.append(f"{who}: served --max-model-len {window!r}, not {SERVER.max_model_len}")
        override = flag_value(argv, "--hf-overrides")
        if _parse_overrides(override) != HF_OVERRIDES:
            problems.append(
                f"{who}: served --hf-overrides {override!r}, not {json.dumps(HF_OVERRIDES)}"
            )
        rope = [
            m for m in ROPE_SCALING_MARKERS if any(m in a.lower().replace("-", "_") for a in argv)
        ]
        if rope:
            problems.append(f"{who}: the served argv mentions {', '.join(rope)}")
    return ArgvCheck(len(sessions), problems)


@dataclass(frozen=True)
class DesignCheck:
    cells: list[Cell] | None
    registered_cells: list[Cell]
    unregistered_cells: list[Cell]
    missing_registered_cells: list[Cell]
    kind: DesignKind | None
    server_argv: ArgvCheck | None
    problems: list[str] = field(default_factory=list)


def design_check(
    manifest: ManifestLine | None, sessions: Sequence[ServerRow] | None = None
) -> DesignCheck:
    """The manifest's protocol (and, with `sessions`, what every session served) against §16."""
    cells, problem = protocol_cells(manifest)
    proto = protocol(manifest)
    problems = [] if problem is None else [problem]
    unregistered: list[Cell] = []
    missing: list[Cell] = []
    if cells is not None:
        unregistered = sorted(set(cells) - set(REGISTERED_CELLS))
        missing = [c for c in REGISTERED_CELLS if c not in cells]
        if unregistered:
            problems.append(
                "cells that §16 does not register: " + ", ".join(c.label for c in unregistered)
            )
    treatments = proto.get("treatments")
    if not (
        isinstance(treatments, list) and treatments and set(treatments) <= set(EXPC_TREATMENTS)
    ):
        problems.append(
            f"protocol.treatments {treatments!r} is not a non-empty subset of "
            f"{[t.value for t in EXPC_TREATMENTS]}"
        )
    if proto.get("max_model_len") != SERVER.max_model_len:
        problems.append(
            f"protocol.max_model_len is {proto.get('max_model_len')!r}, §16 "
            f"serves {SERVER.max_model_len}"
        )
    if _parse_overrides(proto.get("hf_overrides")) != HF_OVERRIDES:
        problems.append(
            f"protocol.hf_overrides is {proto.get('hf_overrides')!r}, §16 serves "
            f"{json.dumps(HF_OVERRIDES)}"
        )
    if proto.get("kv_cache_dtype", SERVER.kv_dtype) != SERVER.kv_dtype:
        problems.append(
            f"protocol.kv_cache_dtype is {proto.get('kv_cache_dtype')!r}, §16 "
            f"serves {SERVER.kv_dtype}"
        )
    utilization = SERVER.gpu_memory_utilization
    if proto.get("gpu_memory_utilization", utilization) != utilization:
        problems.append(
            f"protocol.gpu_memory_utilization is "
            f"{proto.get('gpu_memory_utilization')!r}, §16 serves {utilization}"
        )
    if proto.get("m2_duration_s") != 0:
        problems.append(
            f"protocol.m2_duration_s is {proto.get('m2_duration_s')!r}: §16 is M1 only (0)"
        )
    argv = None if sessions is None else server_argv_check(sessions)
    if argv is not None:
        problems += argv.problems
    kind = (
        None
        if cells is None
        else DesignKind.DEVIATES
        if unregistered
        else DesignKind.EVERY_CELL
        if not missing
        else DesignKind.SUBSET
    )
    return DesignCheck(
        cells=None if cells is None else list(cells),
        registered_cells=list(REGISTERED_CELLS),
        unregistered_cells=unregistered,
        missing_registered_cells=missing,
        kind=kind,
        server_argv=argv,
        problems=problems,
    )


@dataclass(frozen=True)
class NllCheck:
    sessions: int
    this: float | None
    main_run: float | None
    this_minus_main: float | None


@dataclass(frozen=True)
class NllCrosscheck:
    main_run: str
    windows: str
    per_treatment: dict[Treatment, NllCheck]


def nll_crosscheck(
    sessions: Sequence[ServerRow], manifest: ManifestLine | None = None
) -> NllCrosscheck:
    """Each treatment's mean session NLL against the main run's; reported only."""
    sha = nll_prompts_sha(manifest)
    windows = (
        "unknown (the manifest has no inputs.nll_prompts_sha256)"
        if sha is None
        else "the main run's"
        if sha == MAIN_RUN_NLL_PROMPTS_SHA256
        else f"DIFFERENT from the main run's ({sha})"
    )
    present = {s.treatment for s in sessions}
    per = {}
    for t in [t for t in EXPC_TREATMENTS if t in present] + sorted(present - set(EXPC_TREATMENTS)):
        this = mean_nll(sessions, t)
        main = MAIN_RUN_NLL.get(t)
        per[t] = NllCheck(
            sum(1 for s in sessions if s.treatment == t),
            this,
            main,
            None if this is None or main is None else this - main,
        )
    return NllCrosscheck("results/full-1", windows, per)
