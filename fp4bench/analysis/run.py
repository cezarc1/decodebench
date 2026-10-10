"""analyze_run: a run directory in; verdict, gates, summary.md, JSON and figures out."""

from dataclasses import replace
from pathlib import Path

from fp4bench.analysis import plots, report, stats
from fp4bench.analysis.cells import (
    Values,
    by_batch,
    failed_sessions,
    m1_steps,
    m2_values,
    median_value,
    order_effect,
    paired_rounds,
    sessions_of,
    unusable,
)
from fp4bench.analysis.checks import (
    accuracy_note,
    design_check,
    effective_bandwidth,
    nll_crosscheck,
    reference_nll_comparison,
)
from fp4bench.analysis.compare import (
    Relabel,
    aa_ratio_passes,
    contrast,
    delta_ms,
    r_decomposition,
    ratio,
    ratio_table,
)
from fp4bench.analysis.design import arms_of, kv_tokens
from fp4bench.analysis.gates import (
    AaEffects,
    AaRatios,
    GateSpec,
    batch_gate_spec,
    failed_gates,
    gate_report,
    gate_status_line,
    nonblocking_failed_gates,
)
from fp4bench.analysis.inputs import (
    M1Shape,
    RunData,
    kv_cache_dtype,
    load_run,
    primary_batches,
    protocol,
    protocol_cells,
    registered_rounds,
)
from fp4bench.analysis.results import (
    BATCH_RATIOS,
    CELL_RATIOS,
    BatchRunResult,
    CellResult,
    CellRunResult,
    Kind,
    RunResult,
    cell_treatments,
)
from fp4bench.analysis.verdicts import (
    batch_vs_tokens_answer,
    classify_effect,
    config_reproduction,
    extension_note,
    h_b_reading,
    h_eq_verdict,
    kernel_scan,
    nvx_crosscheck,
)
from fp4bench.core.types import (
    Arm,
    Cell,
    Gate,
    KvDtype,
    RatioName,
    RecordedKvDtype,
    SkippedMetric,
    Treatment,
    VerdictLabels,
)
from fp4bench.studies.base import G3Rule, Ratio, Study, VerdictRule
from fp4bench.studies.expc import REGISTERED_CELLS, REPRO_CELL
from fp4bench.studies.registry import FP8_KV_CACHE_DTYPES, study_for

Pairs = dict[RatioName, tuple[Treatment, Treatment, Relabel | None]]
RELABELS: dict[VerdictLabels, Relabel | None] = {
    VerdictLabels.FORMAT: None,
    VerdictLabels.KERNEL: stats.relabel_for_k,
    VerdictLabels.FUSION: stats.relabel_for_phi,
}
KIND_OF_VERDICT = {
    VerdictRule.H_EQ: Kind.MAIN,
    VerdictRule.H_B: Kind.EXPB,
    VerdictRule.BATCH_VS_TOKENS: Kind.EXPC,
}


def m1_pairs(ratios: tuple[Ratio, ...]) -> Pairs:
    return {r.name: (r.numer, r.denom, RELABELS[r.labels]) for r in ratios}


def m2_pairs(ratios: tuple[Ratio, ...]) -> tuple[Pairs, Pairs]:
    """(throughput, ITL) ratio pairs; throughput is inverted, and the A/A replica has no ITL."""
    m1 = m1_pairs(ratios)
    return (
        {name: (d, n, relabel) for name, (n, d, relabel) in m1.items()},
        {name: pair for name, pair in m1.items() if name != RatioName.AA},
    )


def _require_g3(study: Study, rule: G3Rule) -> None:
    if study.g3 is not rule:
        raise ValueError(
            f"study {study.name}: its analysis implements G3 as {rule}, not {study.g3}"
        )


def _require_ratios(study: Study, names: tuple[RatioName, ...]) -> None:
    if tuple(r.name for r in study.ratios) != names:
        raise ValueError(
            f"study {study.name}: its analysis renders the ratios "
            f"{', '.join(names)}, in this order; the study has "
            f"{', '.join(r.name for r in study.ratios) or 'none'}"
        )


def _require_decomposition(study: Study) -> None:
    r, f, phi = (study.ratio(n) for n in (RatioName.R, RatioName.F, RatioName.PHI))
    if (f.denom, phi.numer, phi.denom) != (r.denom, r.numer, f.numer):
        raise ValueError(
            f"study {study.name}: R = F · Φ needs F = t_X / t_{r.denom} and "
            f"Φ = t_{r.numer} / t_X; it has F = t_{f.numer} / t_{f.denom} and "
            f"Φ = t_{phi.numer} / t_{phi.denom}"
        )


def unexpected_kv_dtype_note(kv_dtype: RecordedKvDtype) -> str | None:
    """The note for a KV dtype analysed as a main run's (neither BF16 nor FP8)."""
    if isinstance(kv_dtype, KvDtype) and (kv_dtype is KvDtype.BF16 or kv_dtype.is_fp8):
        return None
    return (
        f"**Note:** the run's KV cache dtype `{kv_dtype}` (protocol.kv_cache_dtype of the last "
        f"manifest line) is neither bfloat16 (the main run) nor an FP8 dtype "
        f"({', '.join(sorted(FP8_KV_CACHE_DTYPES))}: Experiment B, §13), so the run is "
        f"analysed as a main run. Check what it served before reading the verdict."
    )


def _evaluate_batches(data: RunData, study: Study, reference_run: Path | None) -> BatchRunResult:
    _require_g3(study, G3Rule.AA_RATIO)
    _require_ratios(study, BATCH_RATIOS)
    _require_decomposition(study)
    r = study.ratio(RatioName.R)
    manifest = data.manifest
    checkpoint, checkpoint_source = data.checkpoint
    sessions = sessions_of(data.servers)
    sids = {s.session_id for s in sessions}
    failed = failed_sessions(data.servers)
    shape, prompt_len = data.m1_shape, data.batch_prompt_len
    batches = tuple(cell.batch for cell in shape.cells)
    step = by_batch(m1_steps(data.m1, sids))
    tput = m2_values(data.m2, sids, "aggregate_tps")
    itl = m2_values(data.m2, sids, "itl_p50_ms")
    per_user = m2_values(data.m2, sids, "tps_per_user_p50")
    tput_pairs, itl_pairs = m2_pairs(study.ratios)
    m1 = {
        name: ratio_table(step, n, d, batches, relabel)
        for name, (n, d, relabel) in m1_pairs(study.ratios).items()
    }
    m2_tput = {
        name: ratio_table(tput, n, d, batches, relabel)
        for name, (n, d, relabel) in tput_pairs.items()
    }
    m2_itl = {
        name: ratio_table(itl, n, d, batches, relabel)
        for name, (n, d, relabel) in itl_pairs.items()
    }
    order = {
        t: {cell.batch: v for cell, v in per.items()}
        for t, per in order_effect(data.m1, sids).items()
    }
    primary = primary_batches(manifest)
    verdict = h_eq_verdict(manifest, step, m1[RatioName.R], r)
    gates = gate_report(
        data.servers,
        data.m1,
        data.m2,
        manifest,
        checkpoint,
        checkpoint_source,
        batch_gate_spec(batches, manifest, data.served, prompt_len=prompt_len, n2=shape.n2),
        AaRatios(m1[RatioName.AA], batches),
    )
    skipped = {
        name: found
        for name, found in (
            (SkippedMetric.M1, unusable(step)),
            (SkippedMetric.M2_THROUGHPUT, unusable(tput)),
            (SkippedMetric.M2_ITL, unusable(itl)),
            (SkippedMetric.M2_PER_USER, unusable(per_user)),
        )
        if found
    }
    accuracy = accuracy_note(sessions, manifest)
    accuracy = replace(
        accuracy,
        reference_run=None
        if reference_run is None
        else reference_nll_comparison(sessions, manifest, Path(reference_run)),
    )
    kv = kv_cache_dtype(manifest)
    context = prompt_len + shape.mean_context_extra
    return BatchRunResult(
        run_dir=data.run_dir,
        study=study,
        kind=KIND_OF_VERDICT[study.verdict],
        verdict=verdict,
        interpretable=not failed_gates(gates),
        gate_line=gate_status_line(gates),
        gates=gates,
        has_manifest=manifest is not None,
        rounds=registered_rounds(manifest),
        failed=failed,
        nonblocking_failed=nonblocking_failed_gates(gates),
        reading=h_b_reading(verdict, m1[RatioName.R], primary)
        if study.verdict is VerdictRule.H_B
        else None,
        primary=primary,
        kv_cache_dtype=kv,
        kv_cache_dtype_note=unexpected_kv_dtype_note(kv),
        context=context,
        bandwidth=effective_bandwidth(step, kv, context),
        m1=m1,
        r_f_phi=r_decomposition(m1[RatioName.R], m1[RatioName.F], m1[RatioName.PHI]),
        m2_throughput=m2_tput,
        m2_itl=m2_itl,
        accuracy=accuracy,
        scan=None
        if study.kernel_scan is None
        else kernel_scan(
            step,
            data.servers,
            batches,
            study.kernel_scan,
            served=data.served,
            g5b=gates[Gate.G5B],
        ),
        crosscheck=None
        if study.crosscheck is None
        else nvx_crosscheck(step, batches, study.crosscheck),
        order=order,
        skipped=skipped,
        paired_at_primary={c: len(paired_rounds(step, r.numer, r.denom, c)) for c in primary},
        tput=tput,
        per_user=per_user,
        served_args=data.served.args,
    )


def ordered_cells(cells: set[Cell]) -> list[Cell]:
    """Registered cells in table order first, then any other cell sorted."""
    return [c for c in REGISTERED_CELLS if c in cells] + sorted(cells - set(REGISTERED_CELLS))


def cell_result(steps: Values[Cell], cell: Cell, study: Study, shape: M1Shape) -> CellResult:
    """Δ = t_denom − t_numer of R (> 0: numerator faster), Δ_AA = t_numer − t_denom of AA."""
    r, aa_ratio = study.ratio(RatioName.R), study.ratio(RatioName.AA)
    medians = {t: median_value(steps, t, cell) for t in cell_treatments(study)}
    aa = ratio(steps, aa_ratio.numer, aa_ratio.denom, cell)
    return CellResult(
        c=cell.batch,
        p=cell.prompt_len,
        arms=arms_of(cell),
        kv_tokens_mean=kv_tokens(cell, shape.mean_context_extra),
        t_ms={t: None if v is None else v * 1000 for t, v in medians.items()},
        r=ratio(steps, r.numer, r.denom, cell),
        delta_ms=delta_ms(steps, r.denom, r.numer, cell),
        aa=aa,
        delta_aa_ms=delta_ms(steps, aa_ratio.numer, aa_ratio.denom, cell),
        aa_ratio_rule=aa_ratio_passes(aa),
    )


def _margins(study: Study) -> tuple[float, float]:
    effect, aa = study.effect_margin_ms, study.aa_effect_margin_ms
    if effect is None or aa is None:
        raise ValueError(f"study {study.name}: its contrasts have no margins")
    return effect, aa


def _evaluate_cells(data: RunData, study: Study) -> CellRunResult:
    _require_g3(study, G3Rule.AA_EFFECTS)
    _require_ratios(study, CELL_RATIOS)
    effect_margin, aa_margin = _margins(study)
    r, aa = study.ratio(RatioName.R), study.ratio(RatioName.AA)
    manifest = data.manifest
    checkpoint, checkpoint_source = data.checkpoint
    sessions = sessions_of(data.servers)
    sids = {s.session_id for s in sessions}
    steps = m1_steps(data.m1, sids, by_cell=True)
    manifest_cells, _ = protocol_cells(manifest)
    expected = ordered_cells(set(manifest_cells or ()) | {cell for (_, _, cell) in steps})
    shape = data.m1_shape
    cells = {
        cell: cell_result(steps, cell, study, shape)
        for cell in ordered_cells(set(REGISTERED_CELLS) | set(expected))
    }
    effects = {
        c.name: contrast(
            steps, c.cells, r.denom, r.numer, lambda lo, hi: classify_effect(lo, hi, effect_margin)
        )
        for c in study.contrasts
    }
    aa_effects = {
        f"{c.name},AA": contrast(steps, c.cells, aa.numer, aa.denom) for c in study.contrasts
    }
    rounds = registered_rounds(manifest)
    token, batch = (study.contrast_along(arm).name for arm in (Arm.TOKEN, Arm.BATCH))
    answer = batch_vs_tokens_answer((token, effects[token]), (batch, effects[batch]), rounds)
    repro = cells.get(REPRO_CELL)
    spec = GateSpec(cells=tuple(expected), m2=False, by_cell=True, n2=shape.n2, served=data.served)
    gates = gate_report(
        data.servers,
        data.m1,
        data.m2,
        manifest,
        checkpoint,
        checkpoint_source,
        spec,
        AaEffects(aa_effects, aa_margin, aa),
    )
    treatments = protocol(manifest).get("treatments")
    return CellRunResult(
        run_dir=data.run_dir,
        study=study,
        kind=KIND_OF_VERDICT[study.verdict],
        verdict=answer,
        interpretable=not failed_gates(gates),
        gate_line=gate_status_line(gates),
        gates=gates,
        has_manifest=manifest is not None,
        rounds=rounds,
        failed=failed_sessions(data.servers),
        effects=effects,
        aa_effects=aa_effects,
        cells=cells,
        reproduction=config_reproduction(None if repro is None else repro.r),
        design=design_check(manifest, sessions),
        nll=nll_crosscheck(sessions, manifest),
        extension=extension_note(
            gates[Gate.G3]["pass"],
            answer,
            rounds,
            isinstance(treatments, list) and aa.numer in treatments,
            study.extension_rounds,
        ),
        skipped=unusable(steps),
        paired={c: len(paired_rounds(steps, r.numer, r.denom, c)) for c in cells},
    )


def evaluate(run_dir: Path, reference_run: Path | None = None) -> RunResult:
    """Everything analyze_run reports, without writing anything."""
    data = load_run(Path(run_dir))
    study = study_for(data.manifest)
    if study.verdict is VerdictRule.BATCH_VS_TOKENS:
        if reference_run is not None:
            raise ValueError("the Experiment C analysis takes no reference run")
        return _evaluate_cells(data, study)
    return _evaluate_batches(data, study, reference_run)


def analyze_run(run_dir: Path, reference_run: Path | None = None) -> RunResult:
    """Write summary.md (and results_c.json for Experiment C) and the figures into `run_dir`."""
    result = evaluate(run_dir, reference_run)
    report.write(result)
    plots.write(result)
    return result
