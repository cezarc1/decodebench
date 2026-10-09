"""The pre-registered decisions: H_eq, Experiment B's and C's readings, the kernel scan."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, override

from fp4bench.analysis import stats
from fp4bench.analysis.cells import (
    Values,
    failed_sessions,
    mean_nll,
    median_value,
    paired_rounds,
    sessions_of,
)
from fp4bench.analysis.compare import ContrastResult, RatioResult
from fp4bench.analysis.inputs import expected_kernels, primary_batches, registered_rounds
from fp4bench.core.schema import ManifestLine, ServerRow
from fp4bench.core.types import (
    Cell,
    LinearBackend,
    LinearKernel,
    ReproStatus,
    Treatment,
    Verdict,
    is_finite,
)
from fp4bench.studies import model
from fp4bench.studies.base import Ratio
from fp4bench.studies.expc import MAIN_RUN_R_BATCH1, REPRO_CELL, REPRO_TOL
from fp4bench.studies.kernel_scan import Crosscheck, KernelScanSpec

_INCOMPLETE = "incomplete ("


class Incomplete(str):
    """A decision the run cannot make, typed so code tells it from one without parsing text."""

    __slots__ = ()

    def __new__(cls, why: str) -> Self:
        return super().__new__(cls, f"{_INCOMPLETE}{why})")

    @override
    def __getnewargs__(self) -> tuple[str]:
        return (self.why,)

    @property
    def why(self) -> str:
        return self[len(_INCOMPLETE) : -1]


def _rounds_short(counts: Sequence[int], rounds: float) -> Incomplete | None:
    if rounds < 2:
        return Incomplete(f"{min(counts)}/{rounds} rounds; a CI needs at least 2")
    if any(n != rounds for n in counts):
        return Incomplete(f"{min(counts) if min(counts) < rounds else max(counts)}/{rounds} rounds")
    return None


def h_eq_verdict(
    manifest: ManifestLine | None, step: Values[int], r_m1: Mapping[int, RatioResult[int]], r: Ratio
) -> stats.OverallVerdict | Incomplete:
    """The verdict on R, or 'incomplete (k/R rounds)' unless every primary C has them all."""
    if manifest is None:
        return Incomplete("no manifest")
    rounds = registered_rounds(manifest)
    if rounds is None:
        return Incomplete("manifest has no valid protocol.rounds")
    primary = primary_batches(manifest)
    paired = [len(paired_rounds(step, r.numer, r.denom, c)) for c in primary]
    short = _rounds_short(paired, rounds)
    if short is not None:
        return short
    return stats.overall_verdict(
        {c: v for c in primary if (v := r_m1[c].verdict) is not None}, primary
    )


def h_b_reading(
    verdict: stats.OverallVerdict | Incomplete,
    r_m1: Mapping[int, RatioResult[int]],
    primary: tuple[int, ...],
) -> str:
    """What Experiment B's verdict (H_eq's rule on R_B) says about H_B,eq and H_B,nv."""
    if isinstance(verdict, Incomplete):
        return "no reading: the run is incomplete"
    if verdict == stats.OverallVerdict.VALIDATED:
        return "H_B,eq holds: R_B is Equivalent at every primary C"
    per_c = {c: r_m1[c].verdict if c in r_m1 else None for c in primary}
    mx_at = [c for c in primary if per_c[c] == Verdict.MXFP4_FASTER]
    nv_at = [c for c in primary if per_c[c] == Verdict.NVFP4_FASTER]
    if mx_at:
        return (
            f"H_B,eq rejected; MXFP4 practically faster at C={', '.join(map(str, mx_at))}: "
            f"contradicts H_B,nv"
        )
    if nv_at and len(nv_at) == len(primary):
        return "H_B,eq rejected; H_B,nv holds: NVFP4 practically faster at every primary C"
    if nv_at:
        return (
            f"H_B,eq rejected; NVFP4 practically faster at C={', '.join(map(str, nv_at))} only: "
            f"H_B,nv does not hold (it needs every primary C)"
        )
    return "neither H_B,eq nor H_B,nv: R_B is practically uncertain or inconclusive at a primary C"


class Effect(StrEnum):
    NO_EFFECT = "no effect"
    SHRINKS = "shrinks the saving"
    GROWS = "grows the saving"
    INCONCLUSIVE = "inconclusive"


class Answer(StrEnum):
    BATCH_DRIVES = "batch (GEMM size) drives the gap"
    TOKENS_DRIVE = "tokens drive it"
    BOTH = "both"
    NEITHER = "neither — inconsistent with the main run"
    INCONCLUSIVE = "inconclusive"


def classify_effect(lo: float, hi: float, margin: float) -> Effect:
    if lo >= -margin and hi <= margin:
        return Effect.NO_EFFECT
    if hi < -margin:
        return Effect.SHRINKS
    if lo > margin:
        return Effect.GROWS
    return Effect.INCONCLUSIVE


def batch_vs_tokens_answer(
    token: tuple[str, ContrastResult], batch: tuple[str, ContrastResult], rounds: float | None
) -> Answer | Incomplete:
    """§16's answer from the two arms' effects, if both have the registered rounds."""
    if rounds is None:
        return Incomplete("manifest has no valid protocol.rounds")
    (_, e_tok), (_, e_batch) = token, batch
    empty = [name for name, e in (token, batch) if not e.n_rounds]
    if empty:
        return Incomplete(f"no data for {' and '.join(empty)}")
    short = _rounds_short([e_tok.n_rounds, e_batch.n_rounds], rounds)
    if short is not None:
        return short
    answers: dict[tuple[str | None, str | None], Answer] = {
        (Effect.NO_EFFECT, Effect.SHRINKS): Answer.BATCH_DRIVES,
        (Effect.SHRINKS, Effect.NO_EFFECT): Answer.TOKENS_DRIVE,
        (Effect.SHRINKS, Effect.SHRINKS): Answer.BOTH,
        (Effect.NO_EFFECT, Effect.NO_EFFECT): Answer.NEITHER,
    }
    return answers.get((e_tok.classification, e_batch.classification), Answer.INCONCLUSIVE)


def extend_line(extension_rounds: int) -> str:
    return f"§16: extend to {extension_rounds} rounds (`--rounds {extension_rounds}`)"


def extension_note(
    g3_pass: bool,
    answer: Answer | Incomplete,
    rounds: float | None,
    replica_registered: bool,
    extension_rounds: int | None,
) -> str | None:
    """The §16 extension a run needs (G3 failed or inconclusive); None for a smoke."""
    if (
        (g3_pass and answer != Answer.INCONCLUSIVE)
        or not replica_registered
        or extension_rounds is None
    ):
        return None
    why = " and ".join(
        w
        for w, on in (
            ("G3 failed", not g3_pass),
            ("the answer is inconclusive", answer == Answer.INCONCLUSIVE),
        )
        if on
    )
    if rounds is not None and rounds >= extension_rounds:
        return (
            f"§16's extension to {extension_rounds} rounds is in this run already ({rounds} "
            f"registered) and {why}: no further remedy is pre-registered."
        )
    return f"**{extend_line(extension_rounds)}** — {why}."


@dataclass(frozen=True)
class Reproduction:
    """R(1, 1,024) against the main run's batch-1 R; reported, not gated."""

    cell: Cell
    target: float
    tolerance: float
    r: float | None
    lo: float | None
    hi: float | None
    n_rounds: int
    difference: float | None
    status: ReproStatus


def config_reproduction(r: RatioResult | None) -> Reproduction:
    if r is None:
        return Reproduction(
            REPRO_CELL, MAIN_RUN_R_BATCH1, REPRO_TOL, None, None, None, 0, None, ReproStatus.NO_DATA
        )
    diff = r.ratio - MAIN_RUN_R_BATCH1
    return Reproduction(
        REPRO_CELL,
        MAIN_RUN_R_BATCH1,
        REPRO_TOL,
        r.ratio,
        r.lo,
        r.hi,
        r.n_rounds,
        diff,
        ReproStatus.PASS if abs(diff) <= REPRO_TOL + 1e-12 else ReproStatus.FAIL,
    )


@dataclass(frozen=True)
class Eligibility:
    kernel_ok: bool | None
    failed: bool
    error: str | None
    step_c32_s: float | None
    nll: float | None
    nll_diff: float | None
    nll_ok: bool
    g5b: bool
    eligible: bool
    reasons: list[str]


@dataclass(frozen=True)
class TieBreak:
    steps: dict[Treatment, float | None]
    gap: float | None
    tie: bool | None


@dataclass(frozen=True)
class Margin:
    fastest: Treatment
    runner_up: Treatment
    step_s: dict[Treatment, float]
    gap: float
    tie_margin: float
    tie: bool
    tiebreak_c: int
    tiebreak: TieBreak | None
    decided_by: str


@dataclass(frozen=True)
class KernelScan:
    treatments: list[Treatment]
    selection_c: int
    median_step_s: dict[Treatment, dict[int, float]]
    ratio_to_nv: dict[Treatment, dict[int, float]]
    expected_kernels: dict[Treatment, LinearKernel | None]
    observed_kernels: dict[Treatment, list[str]]
    kernel_matches: dict[Treatment, bool | None]
    failed: list[Treatment]
    nv_nll: float | None
    eligibility: dict[Treatment, Eligibility]
    candidates: dict[Treatment, float]
    margin: Margin | None
    selected: Treatment | None
    selected_kernel: LinearKernel | None
    selected_server_args: tuple[str, ...] | None
    reason: str | None


def g5b_passes(g5b: Mapping | None, treatment: Treatment) -> bool:
    """Whether `treatment` on its own passes G5b, from the gate's details."""
    if not isinstance(g5b, Mapping) or g5b.get("bf16_reference_mean") is None:
        return False
    degradation = (g5b.get("degradation") or {}).get(treatment)
    return is_finite(degradation) and treatment not in (g5b.get("failed_treatments") or [])


def _rel_gap(a: float, b: float) -> float:
    return max(a, b) / min(a, b) - 1


def kernel_scan(
    step: Values[int],
    servers: Sequence[ServerRow],
    batches: Sequence[int],
    spec: KernelScanSpec,
    g5b: Mapping | None = None,
    manifest: ManifestLine | None = None,
) -> KernelScan | None:
    """The NVa rule (`spec`, METHODOLOGY.md#nv-alt); None for a run without a scan treatment.
    Each treatment is expected to run the kernel of the server args `manifest` records."""
    sessions = sessions_of(servers)
    scan = spec.treatments
    failed_by: dict[Treatment, ServerRow] = {}
    for row in failed_sessions(servers):
        failed_by.setdefault(row.treatment, row)
    if (
        not any(s.treatment in scan for s in sessions)
        and not any(t in scan for _, t, _ in step)
        and not any(t in scan for t in failed_by)
    ):
        return None
    reference = spec.reference
    treatments = [reference, *scan]
    selection_c, tiebreak_c = spec.selection_c, spec.tiebreak_c
    cs = sorted(set(batches) | {selection_c, tiebreak_c})
    median = {
        t: {c: m for c in cs if (m := median_value(step, t, c)) is not None} for t in treatments
    }
    nv_median = median[reference]
    ratio = {
        t: {c: m / nv_median[c] for c, m in median[t].items() if c in nv_median} for t in treatments
    }
    kernels = expected_kernels(manifest)
    expected = {t: kernels.get(t) for t in treatments}
    cute_dsl_kernel = LinearBackend.FLASHINFER_CUTEDSL.kernel
    observed = {
        t: sorted({k for s in sessions if s.treatment == t for k in s.linear_kernels or []})
        for t in treatments
    }
    matches: dict[Treatment, bool | None] = {}
    for t in treatments:
        mine = [s for s in sessions if s.treatment == t]
        matches[t] = (
            None
            if not mine
            else (expected[t] is not None and all(s.linear_kernels == [expected[t]] for s in mine))
        )
    nll = {t: mean_nll(sessions, t) for t in treatments}
    nv_nll = nll[reference]
    eligibility = {}
    for t in scan:
        step_c = median[t].get(selection_c)
        nll_t = nll[t]
        nll_diff = None if nll_t is None or nv_nll is None else nll_t - nv_nll
        nll_ok = nll_diff is not None and abs(nll_diff) <= spec.max_nll_diff
        is_failed = t in failed_by
        kernel_ok = matches[t] is True and expected[t] != cute_dsl_kernel
        passes_g5b = g5b_passes(g5b, t)
        reasons = []
        if is_failed:
            reasons.append(f"its session failed: {failed_by[t].error}")
        elif matches[t] is None:
            reasons.append("it has no session")
        else:
            if not kernel_ok:
                reasons.append(
                    f"it did not run its expected kernel ({expected[t]}; observed "
                    f"{', '.join(observed[t]) or 'none'}) or that kernel is CuTe-DSL"
                )
            if step_c is None:
                reasons.append(f"it has no M1 step at C={selection_c}")
            if nv_nll is None:
                reasons.append(f"{reference} has no usable mean NLL to compare with")
            elif nll_t is None:
                reasons.append("it has no usable mean NLL")
            elif not nll_ok:
                reasons.append(
                    f"its mean NLL differs from {reference}'s by {nll_diff:+.4f} "
                    f"nats/token (bound {spec.max_nll_diff})"
                )
            if not passes_g5b:
                reasons.append("it does not pass G5b")
        eligibility[t] = Eligibility(
            kernel_ok=None if matches[t] is None else kernel_ok,
            failed=is_failed,
            error=failed_by[t].error if is_failed else None,
            step_c32_s=step_c,
            nll=nll_t,
            nll_diff=nll_diff,
            nll_ok=nll_ok,
            g5b=passes_g5b,
            eligible=not reasons,
            reasons=reasons,
        )
    candidates = {t: median[t][selection_c] for t in scan if eligibility[t].eligible}
    ranked = sorted(candidates, key=lambda t: (candidates[t], scan.index(t)))
    selected = ranked[0] if ranked else None
    margin = None
    if len(ranked) >= 2:
        fastest, runner_up = ranked[:2]
        gap = candidates[runner_up] / candidates[fastest] - 1
        tie = gap < spec.tie_margin
        tiebreak, decided_by = None, f"C={selection_c}"
        if tie:
            at = {t: median[t].get(tiebreak_c) for t in (fastest, runner_up)}
            a, b = at[fastest], at[runner_up]
            gap_c = None if a is None or b is None else _rel_gap(a, b)
            tie_c = None if gap_c is None else gap_c < spec.tie_margin
            tiebreak = TieBreak(at, gap_c, tie_c)
            if tie_c is False and a is not None and b is not None:
                selected, decided_by = (fastest if a <= b else runner_up), f"C={tiebreak_c}"
            else:
                selected, decided_by = min((fastest, runner_up), key=scan.index), "order"
        margin = Margin(
            fastest,
            runner_up,
            {fastest: candidates[fastest], runner_up: candidates[runner_up]},
            gap,
            spec.tie_margin,
            tie,
            tiebreak_c,
            tiebreak,
            decided_by,
        )
    return KernelScan(
        treatments=treatments,
        selection_c=selection_c,
        median_step_s=median,
        ratio_to_nv=ratio,
        expected_kernels=expected,
        observed_kernels=observed,
        kernel_matches=matches,
        failed=sorted(failed_by),
        nv_nll=nv_nll,
        eligibility=eligibility,
        candidates=candidates,
        margin=margin,
        selected=selected,
        selected_kernel=expected[selected] if selected else None,
        selected_server_args=model.TREATMENTS[selected].server_args if selected else None,
        reason=None
        if selected
        else (
            f"no scanned kernel is eligible at C={selection_c}: a candidate needs a verified (G1) "
            f"non-CuTe-DSL kernel record, no failed session, a median M1 step at C={selection_c}, "
            f"a mean NLL within {spec.max_nll_diff} nats/token of {reference}'s and G5b (see the "
            f"eligibility table)"
        ),
    )


@dataclass(frozen=True)
class NvxRatio:
    nv_step_s: float
    nvx_step_s: float
    ratio: float


def nvx_crosscheck(
    step: Values[int], batches: Sequence[int], check: Crosscheck
) -> dict[int, NvxRatio] | None:
    """NVx's step over NV's per C, same kernel; None without NVIDIA's checkpoint."""
    if not any(t == check.treatment for _, t, _ in step):
        return None
    out = {}
    for c in sorted(batches):
        nv, nvx = (median_value(step, check.reference, c), median_value(step, check.treatment, c))
        if nv is not None and nvx is not None:
            out[c] = NvxRatio(nv, nvx, nvx / nv)
    return out
