"""Gates G1–G7 (METHODOLOGY.md#gates); `pass` is True or False."""

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fp4bench import settings
from fp4bench.analysis.cells import (
    finite_mean,
    is_pos_int,
    is_round,
    m2_values,
    none_if_nan,
    paired_rounds,
    row_cell,
    sessions_of,
    stored_prompt_len,
)
from fp4bench.analysis.compare import (
    ContrastResult,
    RatioResult,
    aa_effect_passes,
    aa_ratio_passes,
    ratio_json,
)
from fp4bench.analysis.inputs import (
    Served,
    bf16_reference,
    protocol_cells,
    registered_batches,
    registered_reps,
)
from fp4bench.core.schema import M1Row, M2Row, ManifestLine, ServerRow
from fp4bench.core.types import (
    Cell,
    CheckpointKind,
    Gate,
    LinearKernel,
    NvnfCacheStatus,
    Treatment,
    is_finite,
    is_number,
)
from fp4bench.lib_bench import VALID_AGGREGATE_SOURCE
from fp4bench.studies.base import Ratio, min_kv_tokens_for

NON_BLOCKING_GATES = frozenset({Gate.G6B})
GateReport = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class GateSpec:
    """What a run's gates check; `by_cell` lists M1 blocks by (C, P), not by batch. G6a sizes
    the KV pool with the rows' `n2`, and G1 expects the kernel and fusion each treatment was
    `served` with."""

    cells: tuple[Cell, ...]
    m2: bool
    by_cell: bool
    n2: int
    served: Served

    @property
    def names(self) -> tuple[Gate, ...]:
        return tuple(g for g in Gate if self.m2 or g != Gate.G6B)

    def wave_kv_tokens(self, cell: Cell) -> int:
        """KV tokens an M1 block of `cell` holds at the end of its N2 wave."""
        return min_kv_tokens_for(cell.batch, cell.prompt_len, self.n2)


def batch_gate_spec(
    batches: Iterable[int],
    manifest: ManifestLine | None,
    served: Served,
    *,
    prompt_len: int,
    n2: int,
) -> GateSpec:
    """The main run's gates: every batch measured or registered (protocol.concurrencies), M2."""
    expected = sorted(set(batches) | registered_batches(manifest))
    return GateSpec(
        cells=tuple(Cell(c, prompt_len) for c in expected),
        m2=True,
        by_cell=False,
        n2=n2,
        served=served,
    )


@dataclass(frozen=True)
class AaRatios:
    """G3 as the A/A ratio t_MXp / t_MX at every measured batch."""

    aa: Mapping[int, RatioResult[int]]
    batches: tuple[int, ...]


@dataclass(frozen=True)
class AaEffects:
    """G3 as the A/A test on contrasts (METHODOLOGY.md#aa)."""

    effects: Mapping[str, ContrastResult]
    margin_ms: float
    aa: Ratio


def _is_zero(v: object) -> bool:
    return is_number(v) and v == 0


def _distinct_json(values: Iterable[Any]) -> list[Any]:
    seen = {}
    for v in values:
        seen.setdefault(json.dumps(v, sort_keys=True, default=repr), v)
    return [seen[k] for k in sorted(seen)]


def _by_round(sessions: Iterable[ServerRow]) -> list[ServerRow]:
    return sorted(sessions, key=lambda s: (s.round, s.treatment))


def _compile_hashes(session: ServerRow) -> set[str]:
    hashes = session.compile_cache_hashes
    return {h for h in hashes if isinstance(h, str)} if isinstance(hashes, list) else set()


def nvnf_compile_cache_check(sessions: Sequence[ServerRow]) -> dict[str, Any]:
    """Whether an NVnf session loaded NV's compiled graph (METHODOLOGY.md#compile-cache)."""
    nv = [s for s in sessions if s.treatment == Treatment.NV]
    nvnf = sorted((s for s in sessions if s.treatment == Treatment.NVNF), key=lambda s: s.round)
    nv_hashes = set().union(*map(_compile_hashes, nv))
    nvnf_hashes = set().union(*map(_compile_hashes, nvnf))
    shared = [
        [s.round, s.treatment, sorted(_compile_hashes(s) & nv_hashes)]
        for s in nvnf
        if _compile_hashes(s) & nv_hashes
    ]
    if not nv or not nvnf:
        status = NvnfCacheStatus.NOT_APPLICABLE
    elif not nv_hashes or not nvnf_hashes:
        status = NvnfCacheStatus.UNVERIFIED
    else:
        status = NvnfCacheStatus.FAIL if shared else NvnfCacheStatus.PASS
    return {
        "status": status,
        "shared_with_nv": shared,
        "nv_hashes": sorted(nv_hashes),
        "nvnf_hashes": sorted(nvnf_hashes),
        "nvnf_sessions_without_hashes": [
            [s.round, s.treatment] for s in nvnf if not _compile_hashes(s)
        ],
    }


def g1_kernels(
    sessions: Sequence[ServerRow],
    manifest: ManifestLine | None,
    expected_kernel: Mapping[Treatment, LinearKernel],
    expected_fusion: Mapping[Treatment, bool],
) -> dict[str, Any]:
    """Kernel, fusion and compile-cache evidence of every session, and no manifest problems."""
    treatments = sorted({s.treatment for s in sessions})

    def per_treatment(name: str) -> dict[Treatment, list]:
        return {
            t: sorted({x for s in sessions if s.treatment == t for x in getattr(s, name) or []})
            for t in treatments
        }

    kernel = {t: expected_kernel.get(t) for t in treatments}
    by_round = _by_round(sessions)
    mismatched_kernels = [
        [s.round, s.treatment, s.linear_kernels, kernel[s.treatment]]
        for s in by_round
        if kernel[s.treatment] is None or s.linear_kernels != [kernel[s.treatment]]
    ]
    fusion = {t: expected_fusion.get(t) for t in treatments}

    def fusion_matches(s: ServerRow) -> bool:
        want, fusions = fusion[s.treatment], s.custom_fusions
        return (
            want is not None
            and s.fuse_act_quant is want
            and isinstance(fusions, list)
            and ("act_quant" in fusions) == want
            and s.pass_config_conflict is False
        )

    mismatched_fusion = [
        [s.round, s.treatment, s.fuse_act_quant, s.custom_fusions, fusion[s.treatment]]
        for s in by_round
        if not fusion_matches(s)
    ]
    conflicts = [
        [s.round, s.treatment, s.pass_config_conflict]
        for s in by_round
        if s.pass_config_conflict is not False
    ]
    nvnf_cache = nvnf_compile_cache_check(sessions)
    problems = None if manifest is None else manifest.problems
    return {
        "pass": (
            not mismatched_kernels
            and not mismatched_fusion
            and nvnf_cache["status"] != NvnfCacheStatus.FAIL
            and problems == []
        ),
        "expected": kernel,
        "observed": per_treatment("linear_kernels"),
        "mismatched_sessions": mismatched_kernels,
        "kernel_lines": per_treatment("linear_kernel_lines"),
        "expected_act_quant_fusion": fusion,
        "observed_fuse_act_quant": {
            t: _distinct_json(s.fuse_act_quant for s in sessions if s.treatment == t)
            for t in treatments
        },
        "fusion_mismatched_sessions": mismatched_fusion,
        "pass_config_conflict_sessions": conflicts,
        "custom_fusion_lines": per_treatment("custom_fusion_lines"),
        "nvnf_compile_cache": nvnf_cache,
        "compile_cache_hashes": per_treatment("compile_cache_hashes"),
        "backend_fallback_lines": per_treatment("linear_backend_fallback_lines"),
        "sampling_defaults_lines": per_treatment("sampling_defaults_lines"),
        "manifest_problems": problems,
    }


def bytes_ratio_ok(checkpoint: dict | None) -> bool:
    """Stored NV/MX bytes within 2% of the format-implied ratio; a malformed report fails."""
    if checkpoint is None:
        return False
    try:
        return bool(abs(checkpoint["nv_over_mx"] / checkpoint["expected_nv_over_mx"] - 1) < 0.02)
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def g2_bytes(
    sessions: Sequence[ServerRow], checkpoint: dict | None, checkpoint_source: str | None
) -> dict[str, Any]:
    treatments = sorted({s.treatment for s in sessions})
    return {
        "pass": bytes_ratio_ok(checkpoint),
        **({"reason": "no checkpoint report"} if checkpoint is None else {}),
        "checkpoint": checkpoint,
        "checkpoint_source": checkpoint_source,
        "weights_gib": {
            t: [s.weights_gib for s in sessions if s.treatment == t] for t in treatments
        },
    }


def g3_aa_ratios(g3: AaRatios) -> dict[str, Any]:
    aa = g3.aa
    return {
        "pass": bool(aa)
        and set(aa) == set(g3.batches)
        and all(aa_ratio_passes(v) for v in aa.values()),
        "aa": {c: ratio_json(v, by_cell=False) for c, v in aa.items()},
        "missing_c": sorted(set(g3.batches) - set(aa)),
    }


def g3_aa_effects(
    effects: Mapping[str, ContrastResult], has_replica: bool, margin_ms: float, aa: Ratio
) -> dict[str, Any]:
    """Fails closed: no A/A replica session, or an effect without data or a CI."""
    per = {}
    for name, e in effects.items():
        ok = has_replica and aa_effect_passes(e.lo_ms, e.hi_ms, margin_ms)
        reason = (
            None
            if ok
            else f"no A/A replica ({aa.numer})"
            if not has_replica
            else "no data"
            if not e.n_rounds
            else "needs at least 2 rounds"
            if e.lo_ms is None
            else f"the CI does not contain 0 or is not inside ±{margin_ms} ms"
        )
        per[name] = {
            "pass": ok,
            "reason": reason,
            "n_rounds": e.n_rounds,
            "mean_ms": e.mean_ms,
            "lo_ms": e.lo_ms,
            "hi_ms": e.hi_ms,
        }
    out: dict[str, Any] = {
        "pass": has_replica and bool(per) and all(v["pass"] for v in per.values()),
        "rule": (
            f"{settings.CI_LEVEL:.0%} CI of {' and of '.join(effects)} (on Δ_AA = "
            f"t_{aa.numer} − t_{aa.denom}) contains 0 and lies inside ±{margin_ms} ms (§16 "
            f"amendment)"
        ),
        "margin_ms": margin_ms,
        "effects": per,
    }
    if not has_replica:
        out["reason"] = "no A/A replica"
    return out


def _g4_by_batch(sids: set[str], m1: Sequence[M1Row], m2: Sequence[M2Row]) -> dict[str, Any]:
    throttled = sorted(
        {
            (r.round, r.treatment, r.c)
            for r in m1
            if r.session_id in sids and r.block_telemetry.env_throttle_us > 0
        }
        | {
            (r.round, r.treatment, r.c)
            for r in m2
            if r.session_id in sids and r.telemetry.env_throttle_us > 0
        }
    )
    sw_cap = max(
        [r.block_telemetry.sw_power_cap_frac for r in m1 if r.session_id in sids]
        + [r.telemetry.sw_power_cap_frac for r in m2 if r.session_id in sids],
        default=0.0,
    )
    return {"pass": not throttled, "cells": throttled, "max_sw_power_cap_frac": sw_cap}


def _block(row: M1Row) -> tuple:
    return row.round, row.treatment, row.c, stored_prompt_len(row)


def _block_order(block: Sequence[Any]) -> tuple:
    return tuple((0, x, "") if is_number(x) else (1, 0, str(x)) for x in block)


def power_cap_by_cell(m1: Iterable[M1Row], sids: set[str]) -> dict[Cell, dict[Treatment, dict]]:
    """SW power-cap fraction per cell and treatment: one value per block, max and mean."""
    per_block: dict[tuple, float] = {}
    for row in m1:
        cell, frac = row_cell(row, by_cell=True), row.block_telemetry.sw_power_cap_frac
        if row.session_id in sids and cell is not None and is_finite(frac):
            per_block.setdefault((row.session_id, row.treatment, cell), float(frac))
    grouped = defaultdict(list)
    for (_, t, cell), frac in per_block.items():
        grouped[(cell, t)].append(frac)
    out: dict[Cell, dict[Treatment, dict]] = {}
    for (cell, t), fracs in sorted(grouped.items()):
        out.setdefault(cell, {})[t] = {
            "blocks": len(fracs),
            "max": max(fracs),
            "mean": statistics.fmean(fracs),
        }
    return out


def _g4_by_cell(sids: set[str], m1: Sequence[M1Row]) -> dict[str, Any]:
    throttled, no_telemetry = {}, {}
    for r in m1:
        if r.session_id not in sids:
            continue
        env, frac = r.block_telemetry.env_throttle_us, r.block_telemetry.sw_power_cap_frac
        if not is_number(env) or not is_number(frac):
            no_telemetry.setdefault(_block(r), [*_block(r)])
        elif env != 0:
            throttled.setdefault(_block(r), [*_block(r), env])
    power = power_cap_by_cell(m1, sids)
    return {
        "pass": not throttled and not no_telemetry,
        "throttled_blocks": sorted(throttled.values(), key=_block_order),
        "blocks_without_telemetry": sorted(no_telemetry.values(), key=_block_order),
        "sw_power_cap_frac": {cell.json_key: v for cell, v in power.items()},
        "max_sw_power_cap_frac": max(
            (v["max"] for per_t in power.values() for v in per_t.values()), default=None
        ),
    }


def _structure_problems(checkpoint: dict | None) -> dict[str, list]:
    out = {}
    for kind in (CheckpointKind.MX, CheckpointKind.NV):
        found = None if checkpoint is None else checkpoint.get(f"{kind}_problems")
        if checkpoint is None:
            out[kind] = ["no checkpoint report"]
        elif isinstance(found, list):
            out[kind] = found
        else:
            out[kind] = [f"checkpoint report has no {kind}_problems"]
    return out


def g5b_nll(sessions: Sequence[ServerRow], manifest: ManifestLine | None) -> dict[str, Any]:
    """Each treatment's mean NLL at most settings.NLL_MAX_DEGRADATION above the BF16 reference."""
    treatments = sorted({s.treatment for s in sessions})
    nll = {t: finite_mean(s.nll for s in sessions if s.treatment == t) for t in treatments}
    reference, reference_problem = bf16_reference(manifest)
    ref_mean = None if reference is None else reference["mean"]
    degradation = {
        t: None if ref_mean is None or not is_finite(v) else v - ref_mean for t, v in nll.items()
    }
    failed = (
        []
        if ref_mean is None
        else [
            t for t, d in degradation.items() if d is None or not d <= settings.NLL_MAX_DEGRADATION
        ]
    )
    return {
        "pass": reference is not None and not failed,
        "nll": {t: none_if_nan(v) for t, v in nll.items()},
        "bf16_reference_mean": ref_mean,
        "reference_problem": reference_problem,
        "max_degradation": settings.NLL_MAX_DEGRADATION,
        "degradation": degradation,
        "failed_treatments": failed,
    }


def _measured_counts(
    m1: Iterable[M1Row], sids: set[str], by_cell: bool
) -> dict[tuple[str, Cell], int]:
    counts: dict[tuple[str, Cell], int] = defaultdict(int)
    for r in m1:
        if (
            r.session_id in sids
            and r.warmup is False
            and is_round(r.round)
            and (cell := row_cell(r, by_cell)) is not None
        ):
            counts[(r.session_id, cell)] += 1
    return counts


def _required_kv(spec: GateSpec) -> int | None:
    return max((spec.wave_kv_tokens(c) for c in spec.cells), default=None)


def _g6a_by_batch(
    sessions: Sequence[ServerRow],
    sids: set[str],
    m1: Sequence[M1Row],
    spec: GateSpec,
    reps: int | None,
) -> dict[str, Any]:
    cells = spec.cells
    batches = sorted({c.batch for c in cells})
    by_round = _by_round(sessions)
    bad_sizes = [
        s.session_id
        for s in sessions
        if not s.cudagraph_capture_sizes or any(c not in s.cudagraph_capture_sizes for c in batches)
    ]
    counts = _measured_counts(m1, sids, by_cell=False)
    missing = [
        [s.round, s.treatment, cell.batch, counts[(s.session_id, cell)], reps or 1]
        for s in by_round
        for cell in cells
        if (counts[(s.session_id, cell)] != reps if reps else counts[(s.session_id, cell)] < 1)
    ]
    preempted = {}
    for r in m1:
        if r.session_id in sids and not _is_zero(r.preemptions_delta):
            preempted.setdefault(
                (r.session_id, r.c), [r.round, r.treatment, r.c, r.preemptions_delta]
            )
    preempted_blocks = sorted(preempted.values(), key=lambda b: (b[0], b[1], b[2]))
    required = _required_kv(spec)
    short = [
        [s.round, s.treatment, s.kv_cache_tokens]
        for s in by_round
        if required is None or not is_pos_int(s.kv_cache_tokens) or s.kv_cache_tokens < required
    ]
    return {
        "pass": (
            not bad_sizes
            and not missing
            and not preempted_blocks
            and required is not None
            and not short
        ),
        "capture_size_sessions": bad_sizes,
        "missing_m1_rows": missing,
        "preempted_m1_blocks": preempted_blocks,
        "kv_capacity": {
            "required_tokens": required,
            "max_concurrency": max(batches, default=None),
            "short_sessions": short,
        },
        "expected_concurrencies": batches,
        "expected_m1_reps": reps,
    }


def _g6a_by_cell(
    sessions: Sequence[ServerRow],
    sids: set[str],
    m1: Sequence[M1Row],
    spec: GateSpec,
    reps: int | None,
    manifest: ManifestLine | None,
) -> dict[str, Any]:
    cells = spec.cells
    manifest_cells, cells_problem = protocol_cells(manifest)
    by_round = _by_round(sessions)
    rows = [r for r in m1 if r.session_id in sids]
    malformed = []
    for r in rows:
        why = (
            "no (c, prompt_len) cell"
            if row_cell(r, by_cell=True) is None
            else f"warmup is {r.warmup!r}"
            if not isinstance(r.warmup, bool)
            else "no round or treatment"
            if not is_round(r.round)
            else None
        )
        if why is not None:
            malformed.append([r.round, r.treatment, r.c, stored_prompt_len(r), r.set, why])
    counts = _measured_counts(rows, sids, by_cell=True)
    seen = {cell for r in rows if (cell := row_cell(r, by_cell=True)) is not None}
    outside = (
        [] if manifest_cells is None else sorted([list(c) for c in seen if c not in manifest_cells])
    )
    missing = [
        [s.round, s.treatment, c, p, counts[(s.session_id, Cell(c, p))], reps]
        for s in by_round
        for c, p in cells
        if reps is None or counts[(s.session_id, Cell(c, p))] != reps
    ]
    preempted = {}
    for r in rows:
        if not _is_zero(r.preemptions_delta):
            preempted.setdefault((r.session_id, _block(r)), [*_block(r), r.preemptions_delta])
    batches = sorted({c for c, _ in cells})
    bad_sizes = [
        [s.round, s.treatment, s.cudagraph_capture_sizes]
        for s in by_round
        if not isinstance(s.cudagraph_capture_sizes, list)
        or any(c not in s.cudagraph_capture_sizes for c in batches)
    ]
    required = _required_kv(spec)
    short = [
        [s.round, s.treatment, s.kv_cache_tokens]
        for s in by_round
        if required is None or not is_pos_int(s.kv_cache_tokens) or s.kv_cache_tokens < required
    ]
    largest = None if required is None else list(max(cells, key=spec.wave_kv_tokens))
    return {
        "pass": (
            cells_problem is None
            and reps is not None
            and bool(cells)
            and not malformed
            and not outside
            and not missing
            and not preempted
            and not bad_sizes
            and not short
        ),
        "cells_problem": cells_problem,
        "expected_cells": [list(c) for c in cells],
        "expected_m1_reps": reps,
        "malformed_rows": malformed,
        "cells_not_in_manifest": outside,
        "missing_m1_rows": missing,
        "preempted_m1_blocks": sorted(preempted.values(), key=_block_order),
        "capture_size_sessions": bad_sizes,
        "required_capture_sizes": batches,
        "kv_capacity": {
            "required_tokens": required,
            "largest_wave_cell": largest,
            "short_sessions": short,
        },
        "prompt_tokens": (
            "exact counts are enforced by the M1 client (fp4bench.wave raises "
            "on any usage mismatch), so a row exists only for an exact block; "
            "the analysis checks each row's (c, prompt_len) cell"
        ),
    }


def _g6b(sids: set[str], m2: Sequence[M2Row], batches: Sequence[int]) -> dict[str, Any]:
    valid = [r for r in m2 if r.session_id in sids and r.valid]
    invalid = [
        [r.round, r.treatment, r.c, r.invalid_reasons]
        for r in m2
        if r.session_id in sids and not r.valid
    ]
    bad_source = [
        [r.round, r.treatment, r.c, r.aggregate_source]
        for r in valid
        if r.aggregate_source != VALID_AGGREGATE_SOURCE
    ]
    preempted = [
        [r.round, r.treatment, r.c, r.preemptions_delta]
        for r in valid
        if not _is_zero(r.preemptions_delta)
    ]
    tput = m2_values(m2, sids, "aggregate_tps")
    return {
        "pass": not invalid and not bad_source and not preempted,
        "blocks_interpretation": False,
        "invalid_m2_cells": invalid,
        "bad_aggregate_source_cells": bad_source,
        "preempted_valid_m2_cells": preempted,
        "m2_paired_rounds": {
            c: len(paired_rounds(tput, Treatment.MX, Treatment.NV, c)) for c in batches
        },
    }


def g7_same_gpu(sessions: Sequence[ServerRow]) -> dict[str, Any]:
    """Each round's sessions ran on one GPU within one container start."""
    gpus = defaultdict(set)
    starts = defaultdict(set)
    for s in sessions:
        gpus[s.round].add(s.gpu_uuid)
        starts[s.round].add(s.start_id or None)
    mixed = {r: sorted(ids, key=str) for r, ids in sorted(starts.items()) if len(ids) > 1}
    no_start_id = sorted(r for r, ids in starts.items() if None in ids)
    unreadable = sorted(
        s.session_id
        for s in sessions
        if not isinstance(s.gpu_uuid, str) or s.gpu_uuid.startswith("error:")
    )
    return {
        "pass": all(len(v) == 1 for v in gpus.values())
        and not unreadable
        and not mixed
        and not no_start_id,
        "gpus": {r: sorted(v, key=str) for r, v in gpus.items()},
        "unreadable_gpu_sessions": unreadable,
        "start_ids": {r: sorted(v, key=str) for r, v in sorted(starts.items())},
        "mixed_start_rounds": mixed,
        "rounds_without_start_id": no_start_id,
    }


def gate_report(
    servers: Sequence[ServerRow],
    m1: Sequence[M1Row],
    m2: Sequence[M2Row],
    manifest: ManifestLine | None,
    checkpoint: dict | None,
    checkpoint_source: str | None,
    spec: GateSpec,
    g3: AaRatios | AaEffects,
) -> GateReport:
    """Every gate of the study; with no sessions at all, nothing passes."""
    sessions = sessions_of(servers)
    if not sessions:
        return {name: {"pass": False, "reason": "no sessions"} for name in spec.names}
    sids = {s.session_id for s in sessions}
    reps = registered_reps(manifest)
    structure = _structure_problems(checkpoint)
    if isinstance(g3, AaRatios):
        g3_gate = g3_aa_ratios(g3)
    else:
        g3_gate = g3_aa_effects(
            g3.effects, any(s.treatment == g3.aa.numer for s in sessions), g3.margin_ms, g3.aa
        )
    report: GateReport = {
        Gate.G1: g1_kernels(sessions, manifest, spec.served.kernels, spec.served.fusions),
        Gate.G2: g2_bytes(sessions, checkpoint, checkpoint_source),
        Gate.G3: g3_gate,
        Gate.G4: (_g4_by_cell(sids, m1) if spec.by_cell else _g4_by_batch(sids, m1, m2)),
        Gate.G5A: {
            "pass": not any(structure.values()),
            "problems": structure,
            "checkpoint_source": checkpoint_source,
        },
        Gate.G5B: g5b_nll(sessions, manifest),
        Gate.G6A: (
            _g6a_by_cell(sessions, sids, m1, spec, reps, manifest)
            if spec.by_cell
            else _g6a_by_batch(sessions, sids, m1, spec, reps)
        ),
    }
    if spec.m2:
        report[Gate.G6B] = _g6b(sids, m2, sorted({c.batch for c in spec.cells}))
    report[Gate.G7] = g7_same_gpu(sessions)
    return report


def _failed(gates: GateReport, blocking: bool) -> list[str]:
    return [
        name.split("_")[0]
        for name, g in gates.items()
        if g["pass"] is not None and not g["pass"] and (name in NON_BLOCKING_GATES) != blocking
    ]


def failed_gates(gates: GateReport) -> list[str]:
    """Failed gates that block interpretation (all but G6b)."""
    return _failed(gates, blocking=True)


def nonblocking_failed_gates(gates: GateReport) -> list[str]:
    return _failed(gates, blocking=False)


def gate_status_line(gates: GateReport) -> str:
    """e.g. 'gates: G1 pass · G2 n/a · G3 FAIL'."""

    def status(g: dict) -> str:
        return "n/a" if g["pass"] is None else "pass" if g["pass"] else "FAIL"

    return "gates: " + " · ".join(f"{name.split('_')[0]} {status(g)}" for name, g in gates.items())
