"""summary.md (and results_c.json for Experiment C) of an analysed run."""

import json
from collections.abc import Mapping
from typing import NamedTuple

from fp4bench import settings
from fp4bench.analysis.cells import OrderEffect
from fp4bench.analysis.checks import (
    AccuracyNote,
    Bandwidth,
    DesignCheck,
    ReferenceComparison,
    r_ideal_or_none,
    treatment_order,
)
from fp4bench.analysis.compare import ContrastResult, Decomposition, RatioResult
from fp4bench.analysis.results import (
    BatchRunResult,
    CellResult,
    CellRunResult,
    Kind,
    RunResult,
    cell_treatments,
)
from fp4bench.analysis.verdicts import Answer, Effect, KernelScan, NvxRatio, Reproduction
from fp4bench.core.schema import ServerRow
from fp4bench.core.types import (
    Arm,
    Cell,
    ContrastName,
    DesignKind,
    Gate,
    Hypothesis,
    KernelMatch,
    KvDtype,
    RatioName,
    RecordedKvDtype,
    ReproStatus,
    SkippedMetric,
    Treatment,
    ValueKey,
)
from fp4bench.studies import model
from fp4bench.studies.base import Contrast, Ratio, Study
from fp4bench.studies.expb import EXPB_PREDICTED_R, EXPB_PREDICTION_C
from fp4bench.studies.expc import (
    HF_OVERRIDES,
    PREDICTED_DELTA_MS,
    PREDICTED_EFFECT_MS,
    PREDICTED_R,
    REGISTERED_CELLS,
    SERVER,
)
from fp4bench.studies.kernel_scan import Crosscheck, KernelScanSpec

NOT_INTERPRETABLE = "NOT INTERPRETABLE: gate failures"
FAILED_ERROR_CHARS = 300
REFERENCE_RUN_NEEDED = (
    "The comparison of this run's FP8-KV NLL with the main run's BF16-KV NLL (§13) needs "
    "--reference-run: `fp4bench analyze results/<this run> --reference-run results/full-1`."
)


def write(result: RunResult) -> None:
    if isinstance(result, CellRunResult):
        payload = result.payload()
        (result.run_dir / "summary.md").write_text(cell_summary(result, payload["gates"]))
        (result.run_dir / "results_c.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
    elif isinstance(result, BatchRunResult):
        (result.run_dir / "summary.md").write_text(batch_summary(result))


def console_lines(result: RunResult) -> list[str]:
    """What the command line prints after analyze_run."""
    run_dir = result.run_dir
    tail = "" if result.interpretable else f" — {NOT_INTERPRETABLE}"
    if isinstance(result, CellRunResult):
        return [
            f"answer: {result.verdict}{tail}",
            *([result.extension.replace("**", "")] if result.extension else []),
            result.gate_line,
            f"config reproduction: {result.reproduction.status}",
            f"see {run_dir / 'summary.md'}, {run_dir / 'results_c.json'} and "
            f"{run_dir / 'expc_delta.png'}",
        ]
    needs_reference = (
        isinstance(result, BatchRunResult)
        and result.kind == Kind.EXPB
        and result.accuracy.reference_run is None
    )
    return [
        f"verdict: {result.verdict}{tail}",
        result.gate_line,
        f"see {run_dir / 'summary.md'}",
        *([REFERENCE_RUN_NEEDED] if needs_reference else []),
    ]


def _cell(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(
    title: str,
    res: Mapping[int, RatioResult],
    ideal: bool = False,
    note: str = "",
    kv_dtype: RecordedKvDtype = KvDtype.BF16,
    context: int = settings.M1_MEAN_CONTEXT,
) -> str:
    def ideal_cell(c: int) -> str:
        value = r_ideal_or_none(c, context, kv_dtype)
        return " n/a |" if value is None else f" {value:.4f} |"

    head = f"| C | n | ratio | {settings.CI_LEVEL:.0%} CI | verdict |" + (
        " R_ideal |" if ideal else ""
    )
    sep = "|---|---|---|---|---|" + ("---|" if ideal else "")
    rows = [
        f"| {c} | {v.n_rounds} | {v.ratio:.4f} | [{v.lo:.4f}, {v.hi:.4f}] | "
        f"{v.verdict} |" + (ideal_cell(c) if ideal else "")
        for c, v in sorted(res.items())
    ]
    if not rows:
        return "\n".join(
            [
                f"### {title}",
                "",
                *([note, ""] if note else []),
                "no data: a ratio needs at least 2 paired rounds of both treatments.",
                "",
            ]
        )
    return "\n".join([f"### {title}", "", *([note, ""] if note else []), head, sep, *rows, ""])


def _decomposition_section(check: Mapping[int, Decomposition], f: Ratio) -> str:
    lines = [
        f"- C={c}: F̂ · Φ̂ = {v.f_times_phi:.4f} (F̂ {v.f:.4f} × Φ̂ {v.phi:.4f}); "
        f"R̂ = {v.r:.4f}; F̂ · Φ̂ / R̂ = {v.f_times_phi_over_r:.6f} (rounds: "
        f"R {v.n_rounds['R']}, F {v.n_rounds['F']}, Φ {v.n_rounds['Phi']})"
        for c, v in sorted(check.items())
    ]
    return "\n".join(
        [
            "### M1: R = F · Φ check",
            "",
            f"R = F · Φ holds exactly in every round, so the estimates agree exactly when R, F and "
            f"Φ are paired over the same rounds; a round without a usable {f.numer} cell drops out "
            f"of F and Φ but not R, and then they differ. Both are shown; nothing is decided here.",
            "",
            *(lines or ["no data: the check needs R, F and Φ at the same C."]),
            "",
        ]
    )


def _experiment_b_section(
    r_m1: Mapping[int, RatioResult], primary: tuple[int, ...], kv_dtype: RecordedKvDtype
) -> str:
    lo, hi = EXPB_PREDICTED_R
    c = EXPB_PREDICTION_C
    if c in r_m1:
        v = r_m1[c]
        where = (
            "inside the predicted range"
            if lo <= v.ratio <= hi
            else "below the predicted range"
            if v.ratio < lo
            else "above the predicted range"
        )
        observed = (
            f"Observed: R̂_B({c}) = {v.ratio:.4f} [{v.lo:.4f}, {v.hi:.4f}] "
            f"({v.n_rounds} rounds), {where}."
        )
    else:
        observed = f"Observed: no R̂_B({c}) (it needs at least 2 paired rounds)."
    cs = ", ".join(map(str, primary))
    return "\n".join(
        [
            "## Experiment B (EXPERIMENT.md §13)",
            "",
            f"High batch with {kv_dtype} KV, a separate run that does not feed the H_eq verdict. "
            f"Hypotheses: "
            f"H_B,eq: R_B within [{1 - settings.DELTA:.2f}, {1 + settings.DELTA:.2f}] at "
            f"C ∈ {{{cs}}}; H_B,nv: R_B < {1 - settings.DELTA:.2f} at C ∈ {{{cs}}} (NVFP4's MAMF "
            f"advantage carries over to end-to-end decode). Same paired CIs, δ and decision table "
            f"as §8; the other C are secondary (C = 128 bridges to the main run's C = 128, which "
            f"has BF16 KV).",
            "",
            f"Pre-registered prediction: R_B({c}) ≈ {lo:.3f}–{hi:.3f} (the FP4 GEMMs are a small "
            f"share of a step dominated by attention reading the KV cache), so most likely "
            f"Equivalent or practically uncertain. R_B({c}) < {lo:.3f} would point at non-GEMM "
            f"effects (e.g. the fusion); R_B > {1 + settings.DELTA:.2f} would contradict H_B,nv "
            f"outright. {observed}",
            "",
        ]
    )


def _linear_backend(treatment: Treatment) -> str | None:
    spec = model.TREATMENTS.get(treatment)
    args = () if spec is None else spec.server_args
    return args[args.index("--linear-backend") + 1] if "--linear-backend" in args else None


def _eligibility_table(scan: KernelScan, spec: KernelScanSpec) -> list[str]:
    sel_c = scan.selection_c

    def ms(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 1000:.4f}"

    def yes(ok: bool | None, none: str = "no session") -> str:
        return none if ok is None else "yes" if ok else "NO"

    rows = []
    for t, e in scan.eligibility.items():
        rows.append(
            f"| {t} | {yes(e.kernel_ok)} | {yes(not e.failed)} | {ms(e.step_c32_s)} | "
            f"{'n/a' if e.nll is None else format(e.nll, '.4f')} | "
            f"{'n/a' if e.nll_diff is None else format(e.nll_diff, '+.4f')} | "
            f"{'pass' if e.g5b else 'FAIL'} | {yes(e.eligible)} | "
            f"{_cell('; '.join(e.reasons))} |"
        )
    nv_nll, ref = scan.nv_nll, spec.reference
    return [
        "### Eligibility for NVa",
        "",
        f"A scanned kernel is a candidate only if every session logged exactly its expected "
        f"(non-CuTe-DSL) kernel, it did not fail, it has an M1 step at C={sel_c}, its mean NLL is "
        f"within {spec.max_nll_diff:.4f} nats/token of {ref}'s in this run "
        f"({ref}: {'n/a' if nv_nll is None else format(nv_nll, '.4f')}), and it passes G5b.",
        "",
        f"| treatment | kernel as expected | not failed | M1 step at C={sel_c}, ms | mean NLL | "
        f"vs {ref} | G5b | eligible | why not |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _margin_text(scan: KernelScan, spec: KernelScanSpec) -> str:
    sel_c, m = scan.selection_c, scan.margin
    if m is None:
        if scan.selected:
            return (
                f"Only one eligible candidate, so there is no margin: {scan.selected} is selected."
            )
        return ""
    fastest, runner_up, tie_margin = m.fastest, m.runner_up, m.tie_margin
    a, b = (m.step_s[t] * 1000 for t in (fastest, runner_up))
    if not m.tie:
        return (
            f"Margin at C={sel_c}: {fastest} ({a:.4f} ms) leads the runner-up {runner_up} "
            f"({b:.4f} ms) by {m.gap:.2%}, at least the {tie_margin:.1%} tie margin: not "
            f"a tie, so {fastest} is selected."
        )
    head = (
        f"Margin at C={sel_c}: {fastest} ({a:.4f} ms) and the runner-up {runner_up} "
        f"({b:.4f} ms) are {m.gap:.2%} apart, within the {tie_margin:.1%} tie margin. "
    )
    tb, tc = m.tiebreak, m.tiebreak_c
    order = ", ".join(spec.treatments)
    if tb is None or tb.tie is None:
        missing = "" if tb is None else ", ".join(t for t, v in tb.steps.items() if v is None)
        return (
            head + f"Tie-break at C={tc}: no M1 step for {missing}, so the earlier of "
            f"{order} is selected: {scan.selected}."
        )
    steps = ", ".join(f"{t} {v * 1000:.4f} ms" for t, v in tb.steps.items() if v is not None)
    if tb.tie:
        return (
            head + f"Tie-break at C={tc}: {steps} ({tb.gap:.2%} apart) is also within the "
            f"tie margin, so the earlier of {order} is selected: {scan.selected}."
        )
    return (
        head + f"Tie-break at C={tc}: {steps} ({tb.gap:.2%} apart): {scan.selected} is "
        f"faster and is selected."
    )


def _scan_section(scan: KernelScan, spec: KernelScanSpec) -> str:
    legend = []
    for t in scan.treatments:
        match = scan.kernel_matches[t]
        state = (
            KernelMatch.FAILED
            if t in scan.failed
            else KernelMatch.NO_SESSION
            if match is None
            else KernelMatch.MATCH
            if match
            else KernelMatch.MISMATCH
        )
        legend.append(
            f"| {t} | {_linear_backend(t) or 'none'} | {scan.expected_kernels[t]} | "
            f"{', '.join(scan.observed_kernels[t]) or 'none'} | {state} |"
        )
    wrong = [
        f"- {t} did not run its expected kernel ({scan.expected_kernels[t]}; observed "
        f"{', '.join(scan.observed_kernels[t]) or 'none'}): its step times are not "
        f"those of that kernel, and it cannot be selected."
        for t in scan.treatments
        if scan.kernel_matches[t] is False
    ]

    def cell(t: Treatment, c: int) -> str:
        m, r = scan.median_step_s[t].get(c), scan.ratio_to_nv[t].get(c)
        return (
            "n/a"
            if m is None
            else (f"{m * 1000:.4f}" if r is None else f"{m * 1000:.4f} ({r:.4f})")
        )

    cs = sorted({c for t in scan.treatments for c in scan.median_step_s[t]})
    rows = [f"| {c} | " + " | ".join(cell(t, c) for t in scan.treatments) + " |" for c in cs]
    sel_c = scan.selection_c
    rule = (
        f"The pre-registered rule for NVa's kernel: the fastest non-CuTe-DSL NVFP4 kernel by "
        f"median M1 step time at C={sel_c}, among the eligible candidates; if the two fastest "
        f"are within {spec.tie_margin:.1%} the one faster at C={spec.tiebreak_c} "
        f"wins, and if they are within {spec.tie_margin:.1%} there too the earlier of "
        f"{', '.join(spec.treatments)}."
    )
    if scan.selected and scan.selected_server_args:
        args = ", ".join(f'"{a}"' for a in scan.selected_server_args)
        ranked = ", ".join(
            f"{t} {v * 1000:.4f} ms"
            for t, v in sorted(scan.candidates.items(), key=lambda kv: kv[1])
        )
        verdict = [
            f"Selected: **{scan.selected}** ({scan.selected_kernel}). "
            f"Candidates at C={sel_c}: {ranked}.",
            "",
            f"To set NVa, make `({args})` its `server_args` in the `TREATMENTS` table "
            f"of fp4bench/studies/model.py.",
        ]
    else:
        verdict = [f"**No kernel selected:** {scan.reason}."]
    margin = _margin_text(scan, spec)
    return "\n".join(
        [
            "## Smoke: kernel scan (NVFP4 kernels on our NV checkpoint)",
            "",
            "| treatment | --linear-backend | expected kernel | observed kernel | G1 |",
            "|---|---|---|---|---|",
            *legend,
            "",
            *([*wrong, ""] if wrong else []),
            f"Median M1 step in ms (ratio to {spec.reference}) per C:",
            "",
            "| C | " + " | ".join(scan.treatments) + " |",
            "|---|" + "---|" * len(scan.treatments),
            *rows,
            "",
            *_eligibility_table(scan, spec),
            rule,
            "",
            *([margin, ""] if margin else []),
            *verdict,
            "",
        ]
    )


def _crosscheck_section(cross: Mapping[int, NvxRatio], check: Crosscheck) -> str:
    theirs, ours = check.treatment, check.reference
    rows = [
        f"| {c} | {v.nv_step_s * 1000:.4f} | {v.nvx_step_s * 1000:.4f} | {v.ratio:.4f} |"
        for c, v in sorted(cross.items())
    ]
    return "\n".join(
        [
            "## Smoke: cross-check, ours vs NVIDIA NVFP4",
            "",
            f"{theirs} is NVIDIA's NVFP4 checkpoint on the same pinned kernel as {ours}, with "
            f"BF16 KV. A ratio near 1 says our {ours} checkpoint is not fast or slow for a reason "
            f"of its own. It gates nothing.",
            "",
            f"| C | {ours} (ours) step, ms | {theirs} (NVIDIA) step, ms | {theirs} / {ours} |",
            "|---|---|---|---|",
            *(rows or ["| n/a | n/a | n/a | n/a |"]),
            "",
        ]
    )


def _reference_lines(cmp: ReferenceComparison | None, expb: bool) -> list[str]:
    if cmp is None:
        return [REFERENCE_RUN_NEEDED, ""] if expb else []

    def nll(x: float | None, fmt: str = ".4f") -> str:
        return "n/a" if x is None else format(x, fmt)

    this, ref = cmp.this_kv_cache_dtype, cmp.kv_cache_dtype
    rows = [
        f"| {t} | {nll(v.this)} | {nll(v.reference)} | {nll(v.this_minus_reference, '+.4f')} |"
        for t, v in cmp.per_treatment.items()
    ]
    return [
        f"### This run ({this} KV) against the reference run ({ref} KV)",
        "",
        f"Mean NLL per treatment (nats/token) of this run and of `{cmp.run}` "
        f"(--reference-run). Descriptive: a positive difference means this run has the higher "
        f"NLL.",
        "",
        f"| treatment | this run ({this} KV) | reference ({ref} KV) | this − reference |",
        "|---|---|---|---|",
        *rows,
        "",
        *([*(f"- {p}" for p in cmp.problems), ""] if cmp.problems else []),
    ]


def _accuracy_section(note: AccuracyNote, expb: bool = False) -> str:
    def nll(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    def over_bf16(x: float | None) -> str:
        return "n/a" if x is None or note.bf16 is None else f"{x - note.bf16:+.4f}"

    d = note.nv_minus_mx
    diff = (
        "NV − MX: n/a (no paired per-window NLLs)."
        if d is None
        else f"NV − MX: **{d.mean:+.4f}** nats/token, [{d.lo:+.4f}, {d.hi:+.4f}] "
        f"({d.level:.0%} paired bootstrap over {note.n_windows} windows, "
        f"{d.resamples:,} resamples, seed {d.seed}). Negative: NVFP4 has the lower NLL."
    )
    return "\n".join(
        [
            "## Accuracy note",
            "",
            "Descriptive only: no verdict and no gate other than G5b depends on it.",
            "",
            "| model | mean NLL (nats/token) | vs BF16 |",
            "|---|---|---|",
            f"| BF16 (reference) | {nll(note.bf16)} | |",
            f"| MXFP4 (MX) | {nll(note.mx)} | {over_bf16(note.mx)} |",
            f"| NVFP4 (NV) | {nll(note.nv)} | {over_bf16(note.nv)} |",
            "",
            diff,
            "",
            *([*(f"- {p}" for p in note.problems), ""] if note.problems else []),
            *_reference_lines(note.reference_run, expb),
        ]
    )


def _bandwidth_section(
    bw: Mapping[Treatment, Mapping[int, Bandwidth]], kv_dtype: RecordedKvDtype, context: int
) -> str:
    def num(x: float | None, scale: float, fmt: str = ".4f") -> str:
        return "n/a" if x is None else format(x / scale, fmt)

    rows = [
        f"| {t} | {c} | {v.rounds} | {num(v.step_s, 1e-3, '.4f')} | "
        f"{num(v.bytes, 1e9, '.2f')} | {num(v.bytes_per_s, 1e12, '.3f')} | "
        f"{num(v.fraction_of_peak, 1, '.1%')} |"
        for t in treatment_order(bw)
        for c, v in sorted(bw[t].items())
    ]
    peak = settings.HBM_PEAK_BYTES_S / 1e12
    return "\n".join(
        [
            "### Effective HBM bandwidth (bytes model / M1 step)",
            "",
            f"Interpretation aid (§8), descriptive only: the bytes model's HBM bytes per decode "
            f"step (§5: the quantized weights, the BF16 lm_head and the {kv_dtype} KV cache of C "
            f"sequences at context {context}) over the median over rounds of the M1 step, and that "
            f"as a fraction of B200's nominal {peak:g} TB/s. MX and MXp read MXFP4 weights, NV, "
            f"NVa and NVnf NVFP4 weights.",
            "",
            f"| treatment | C | rounds | median M1 step, ms | bytes per step, GB | TB/s | "
            f"of {peak:g} TB/s |",
            "|---|---|---|---|---|---|---|",
            *(rows or ["no data: no M1 step of a treatment the bytes model has a format for."]),
            "",
        ]
    )


def _order_table(eff: Mapping[Treatment, Mapping[int, OrderEffect]]) -> str:
    def ms(v: float | None) -> str:
        return "n/a" if v is None else f"{v * 1000:.4f}"

    rows = [
        f"| {t} | {c} | {ms(v.n1)} ({v.count_n1}) | {ms(v.n2)} ({v.count_n2}) | "
        f"{'n/a' if v.n2_over_n1 is None else format(v.n2_over_n1, '.4f')} |"
        for t in treatment_order(eff)
        for c, v in sorted(eff[t].items())
    ]
    return "\n".join(
        [
            "### M1 order effect (descriptive only, not part of any verdict)",
            "",
            "Median measured step in ms (number of reps) by which wave of the pair ran first.",
            "",
            "| treatment | C | first=N1 | first=N2 | N2-first / N1-first |",
            "|---|---|---|---|---|",
            *rows,
            "",
        ]
    )


def _failed_sessions_section(failed: list[ServerRow]) -> str:
    def cell(error: object) -> str:
        text = " ".join(str(error).split()).replace("|", "\\|")
        return text if len(text) <= FAILED_ERROR_CHARS else text[:FAILED_ERROR_CHARS] + "…"

    rows = [f"| {s.round} | {s.treatment} | {s.session_id} | {cell(s.error)} |" for s in failed]
    return "\n".join(
        [
            "## Failed sessions",
            "",
            "A kernel-scan session that could not be served is recorded as failed. It is excluded "
            "from every computation in this summary (cells, ratios, gates, the accuracy note, the "
            "kernel scan), and a scan kernel that failed is ineligible for NVa. Its server log is "
            "servers/<session>.log in the run directory; the full error is in errors.jsonl.",
            "",
            "| round | treatment | session | error |",
            "|---|---|---|---|",
            *rows,
            "",
        ]
    )


def _skipped_section(skipped: Mapping[SkippedMetric, list[ValueKey[int]]]) -> str:
    if not skipped:
        return ""
    lines = [
        f"- {metric}: round {r}, {t}, C={c}"
        for metric, cells in skipped.items()
        for r, t, c in cells
    ]
    return "\n".join(
        [
            "## Skipped cells",
            "",
            "Not a finite positive number, so left out of every ratio:",
            "",
            *lines,
            "",
        ]
    )


K_NOTE = (
    "A ratio above 1 means the pinned CuTe-DSL kernel is faster. Verdicts: "
    "`pinned_faster` (pinned kernel practically faster), `alt_faster` (alternative "
    "kernel practically faster)."
)
F_NOTE = (
    "NVnf is NV with vLLM's SiLU·mul + activation-quant fusion turned off, which MX never "
    "has, so F compares the formats with the fusion matched. A ratio above 1 means "
    "MXFP4 is faster (the format labels, as for R)."
)
PHI_NOTE = (
    "NV runs the fusion (vLLM's default for NVFP4), NVnf does not. A ratio above 1 "
    "means NV is slower than NVnf. Verdicts: `fusion_slower` (the fusion makes NVFP4 "
    "practically slower), `fusion_faster` (the fusion makes NVFP4 practically "
    "faster)."
)


class _Role(NamedTuple):
    m1: str
    tput: str
    itl: str
    note: str


BATCH_ROLES = {
    RatioName.R: _Role("decides H_eq", "llm-inference-bench", "", ""),
    RatioName.D: _Role("alternative kernel; does not decide H_eq", "", "", ""),
    RatioName.K: _Role(
        "kernel effect; does not decide H_eq", "pinned over alternative kernel", "", K_NOTE
    ),
    RatioName.F: _Role("fusion matched; does not decide H_eq", "F, fusion matched", "F", F_NOTE),
    RatioName.PHI: _Role(
        "fusion effect within NVFP4; does not decide H_eq", "Φ, fusion effect", "Φ", PHI_NOTE
    ),
}


def _with_role(title: str, role: str) -> str:
    return f"{title} ({role})" if role else title


SMOKE_NOTE = (
    "This run has smoke-only treatments: it is one round, so no ratio has a CI and "
    "there is no verdict, and G3 cannot pass without the A/A replica, so the run "
    "reads as not interpretable. What it is for is the kernel scan and the "
    "cross-check below."
)


def _ratio_tables(res: BatchRunResult, r_title: str) -> tuple[list[str], list[str], list[str]]:
    m1_tables, tput_tables, itl_tables = [], [], []
    for spec in res.study.ratios:
        name, n, d = spec.name, spec.numer, spec.denom
        if name == RatioName.AA:
            continue
        role = BATCH_ROLES[name]
        if name == RatioName.R:
            m1_tables.append(
                _table(
                    r_title,
                    res.m1[name],
                    ideal=True,
                    kv_dtype=res.kv_cache_dtype,
                    context=res.context,
                    note=f"R_ideal: the bytes model's upper bound (§5) at "
                    f"context {res.context} with {res.kv_cache_dtype} KV.",
                )
            )
        else:
            m1_tables.append(
                _table(
                    _with_role(f"M1: {name.symbol} = t_{n} / t_{d}", role.m1),
                    res.m1[name],
                    note=role.note,
                )
            )
        tput_tables.append(
            _table(
                _with_role(f"M2: tok/s {d} / {n}", role.tput),
                res.m2_throughput[name],
                note=role.note,
            )
        )
        itl_tables.append(
            _table(_with_role(f"M2: ITL p50 {n} / {d}", role.itl), res.m2_itl[name], note=role.note)
        )
    return m1_tables, tput_tables, itl_tables


def batch_summary(res: BatchRunResult) -> str:
    expb = res.kind == Kind.EXPB
    kv_dtype, primary, verdict = res.kv_cache_dtype, res.primary, res.verdict
    r, aa = res.study.ratio(RatioName.R), res.study.ratio(RatioName.AA)
    nonblocking_note = (
        ""
        if not res.nonblocking_failed
        else f"{', '.join(res.nonblocking_failed)} failed; reported only, does not "
        "block the M1 verdict (details under Gates)."
    )
    rounds_line = (
        "**Rounds:** no manifest"
        if not res.has_manifest
        else f"**Rounds:** {res.rounds} registered (last manifest line); paired at primary C "
        + ", ".join(f"{c}: {n}" for c, n in res.paired_at_primary.items())
    )
    if expb:
        headline = (
            f"**Experiment B verdict (M1, R_B = t_{r.numer} / t_{r.denom}, primary C "
            f"{primary}): `{verdict}`** — {res.reading}"
        )
        r_title = f"M1: R_B = t_{r.numer} / t_{r.denom} (decides H_B,eq and H_B,nv; {kv_dtype} KV)"
    else:
        headline = (
            f"**Overall verdict on H_eq (M1, R = t_{r.numer} / t_{r.denom}, primary C "
            f"{primary}): `{verdict}`**"
        )
        r_title = _with_role(f"M1: R = t_{r.numer} / t_{r.denom}", BATCH_ROLES[RatioName.R].m1)
    headline += "" if res.interpretable else f" — **{NOT_INTERPRETABLE}**"
    m1_tables, tput_tables, itl_tables = _ratio_tables(res, r_title)
    m1, tput = res.m1, res.m2_throughput
    scan, cross = res.scan, res.crosscheck
    spec, check = res.study.kernel_scan, res.study.crosscheck
    md = [
        f"# Results: {res.run_dir.name}",
        "",
        headline,
        "",
        res.gate_line,
        "",
        *([res.kv_cache_dtype_note, ""] if res.kv_cache_dtype_note else []),
        *([nonblocking_note, ""] if nonblocking_note else []),
        rounds_line,
        "",
        *([_experiment_b_section(m1[RatioName.R], primary, kv_dtype)] if expb else []),
        *([SMOKE_NOTE, ""] if scan or cross is not None else []),
        *([_failed_sessions_section(res.failed)] if res.failed else []),
        *([_scan_section(scan, spec)] if scan and spec else []),
        *([_crosscheck_section(cross, check)] if cross is not None and check else []),
        *m1_tables,
        _decomposition_section(res.r_f_phi, res.study.ratio(RatioName.F)),
        _table(f"A/A control: t_{aa.numer} / t_{aa.denom} (gate G3)", m1[RatioName.AA]),
        _bandwidth_section(res.bandwidth, kv_dtype, res.context),
        *tput_tables,
        *itl_tables,
        _table(
            f"M2 A/A control: sustained decode tok/s {aa.denom} / {aa.numer} (reported only, "
            f"not part of G3)",
            tput[RatioName.AA],
        ),
        _accuracy_section(res.accuracy, expb),
        _order_table(res.order),
        _skipped_section(res.skipped),
        "## Gates",
        "",
        "```json",
        json.dumps(res.gates, indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(md)


def _f(x: float | None, fmt: str = ".4f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _ci(lo: float | None, hi: float | None, fmt: str = ".4f") -> str:
    return "n/a" if lo is None or hi is None else f"[{format(lo, fmt)}, {format(hi, fmt)}]"


def _pred(x: float | str | None, fmt: str = "+.2f") -> str:
    return "—" if x is None else x if isinstance(x, str) else format(x, fmt)


def classification_text(e: ContrastResult) -> str:
    if not e.n_rounds:
        return "no data"
    return e.classification or f"no CI ({e.n_rounds} round)"


def _definition(contrast: Contrast, delta: str = "Δ") -> str:
    first, second = contrast.cells
    return f"{delta}{first.label} − {delta}{second.label}"


def _predicted_effect(hypothesis: Hypothesis, name: ContrastName) -> str:
    value = PREDICTED_EFFECT_MS[hypothesis].get(name)
    return "—" if value is None else f"≈ {value:+.2f}"


def _effects_section(effects: Mapping[ContrastName, ContrastResult], study: Study) -> list[str]:
    margin, r = study.effect_margin_ms, study.ratio(RatioName.R)
    contrasts = {c.name: c for c in study.contrasts}
    rows = []
    for name, e in effects.items():
        rows.append(
            f"| {name} | {_definition(contrasts[name])} | {e.n_rounds} | "
            f"{_f(e.mean_ms, '+.4f')} | {_ci(e.lo_ms, e.hi_ms, '+.4f')} | "
            f"{classification_text(e)} | {_predicted_effect(Hypothesis.H_BATCH, name)} | "
            f"{_predicted_effect(Hypothesis.H_TOKENS, name)} |"
        )
    return [
        "## Effects (§16 decision rules)",
        "",
        f"Paired per round over the rounds in which both cells have {r.denom} and {r.numer}. "
        f"Margin ±{margin} ms: CI inside → no effect; entirely below −{margin} → "
        f"shrinks the saving; entirely above +{margin} → grows the saving; otherwise "
        f"inconclusive. {settings.CI_LEVEL:.0%} t-CI.",
        "",
        f"| effect | definition | n | mean, ms | {settings.CI_LEVEL:.0%} CI | classification | "
        f"H_batch | H_tokens |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Answer rules: batch effect ({study.contrast_along(Arm.BATCH).name} {Effect.SHRINKS}) "
        f"and no token effect → "
        f"*{Answer.BATCH_DRIVES}*; token effect and no batch effect → *{Answer.TOKENS_DRIVE}*; "
        f"both → *{Answer.BOTH}*; neither → *{Answer.NEITHER}*; anything else (an inconclusive "
        f"effect, or one that grows the saving, which §16 does not name) → "
        f"*{Answer.INCONCLUSIVE}*.",
        "",
    ]


def _cells_section(cells: Mapping[Cell, CellResult], r: Ratio) -> list[str]:
    rows = []
    n_t, d_t = r.numer, r.denom
    for cell, v in cells.items():
        est, d = v.r, v.delta_ms
        head = f"| {', '.join(v.arms) or '—'} | {cell[0]} | {cell[1]:,} | {v.kv_tokens_mean:,} |"
        if est is None and d is None:
            rows.append(head + " 0 | no data | | | | | | |")
            continue
        d_mean = d.mean_ms if d is not None else None
        d_lo = d.lo_ms if d is not None else None
        d_hi = d.hi_ms if d is not None else None
        r_ratio = est.ratio if est is not None else None
        r_lo = est.lo if est is not None else None
        r_hi = est.hi if est is not None else None
        r_verdict = est.verdict if est is not None else None
        rows.append(
            head + f" {est.n_rounds if est is not None else 0} | {_f(v.t_ms[d_t])} | "
            f"{_f(v.t_ms[n_t])} | {_f(d_mean, '+.4f')} | "
            f"{_ci(d_lo, d_hi, '+.4f')} | "
            f"{_f(r_ratio)} | {_ci(r_lo, r_hi)} | "
            f"{r_verdict or 'n/a'} |"
        )
    return [
        f"## Per cell: Δ = t_{d_t} − t_{n_t} (primary) and R = t_{n_t} / t_{d_t}",
        "",
        f"t_{d_t}, t_{n_t}: median over rounds of the per-round median M1 step (ms). Δ > 0: "
        f"NVFP4 saves time per step. KV tokens: mean during the measured window, C × (P + 640). "
        f"R's verdict is §8's table (±{settings.DELTA:.0%}).",
        "",
        f"| arm | C | P | KV tokens | n | t_{d_t}, ms | t_{n_t}, ms | Δ, ms | "
        f"{settings.CI_LEVEL:.0%} CI | R | {settings.CI_LEVEL:.0%} CI | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _predictions_section(
    cells: Mapping[Cell, CellResult], effects: Mapping[ContrastName, ContrastResult]
) -> list[str]:
    rows = []
    for cell, v in cells.items():
        if cell not in REGISTERED_CELLS:
            continue
        r, d = v.r, v.delta_ms
        obs_d = "no data" if d is None else f"{d.mean_ms:+.3f} {_ci(d.lo_ms, d.hi_ms, '+.3f')}"
        obs_r = "no data" if r is None else f"{r.ratio:.3f} {_ci(r.lo, r.hi, '.3f')}"
        rows.append(
            f"| {cell.label} | {obs_d} | "
            f"{_pred(PREDICTED_DELTA_MS[Hypothesis.H_BATCH].get(cell))} | "
            f"{_pred(PREDICTED_DELTA_MS[Hypothesis.H_TOKENS].get(cell))} | {obs_r} | "
            f"{_pred(PREDICTED_R[Hypothesis.H_BATCH].get(cell), '.3f')} | "
            f"{_pred(PREDICTED_R[Hypothesis.H_TOKENS].get(cell), '.3f')} |"
        )
    for name, e in effects.items():
        obs = (
            "no data" if e.mean_ms is None else f"{e.mean_ms:+.3f} {_ci(e.lo_ms, e.hi_ms, '+.3f')}"
        )
        rows.append(
            f"| {name} | {obs} | {_pred(PREDICTED_EFFECT_MS[Hypothesis.H_BATCH].get(name))} | "
            f"{_pred(PREDICTED_EFFECT_MS[Hypothesis.H_TOKENS].get(name))} | | | |"
        )
    return [
        "## Pre-registered predictions (§16) next to the observed values",
        "",
        "Δ in ms (observed with its CI); — where §16 gives no number. H_batch: Δ depends on batch "
        "only (R shrinks in the token arm by dilution alone, assuming ~4.5 TB/s batch-1 attention; "
        "R(1, 4,096) is not listed in §16 and carries the main run's batch-1 R). H_tokens: Δ "
        "depends on KV tokens only, i.e. the main run's Δ curve read against C × 1,664; its "
        "batch-arm Δ is the same in all four cells (≈ its Δ(1, 127,360)). §16's own prediction: "
        "H_batch.",
        "",
        "| cell / effect | Δ observed | Δ H_batch | Δ H_tokens | R observed | R H_batch | "
        "R H_tokens |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _g3_section(g3: Mapping, study: Study) -> list[str]:
    aa = study.ratio(RatioName.AA)
    defs = {f"{c.name},AA": _definition(c, "Δ_AA") for c in study.contrasts}
    rows = [
        f"| {name} | {defs.get(name, '')} | {v['n_rounds']} | {_f(v['mean_ms'], '+.4f')} | "
        f"{_ci(v['lo_ms'], v['hi_ms'], '+.4f')} | {'pass' if v['pass'] else 'FAIL'} | "
        f"{v['reason'] or ''} |"
        for name, v in (g3.get("effects") or {}).items()
    ]
    return [
        "## G3: A/A test on the effects (§16 amendment)",
        "",
        f"Δ_AA = t_{aa.numer} − t_{aa.denom} per cell and round; the A/A effects are the same "
        f"differences as {' and '.join(c.name for c in study.contrasts)}, paired per round. G3 "
        f"passes if both {settings.CI_LEVEL:.0%} CIs contain 0 and lie inside "
        f"±{study.aa_effect_margin_ms} ms."
        + (f" **{g3['reason']}**: G3 fails." if g3.get("reason") else ""),
        "",
        f"| A/A effect | definition | n | mean, ms | {settings.CI_LEVEL:.0%} CI | G3 | why not |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _aa_section(cells: Mapping[Cell, CellResult], aa_ratio: Ratio) -> list[str]:
    replica, base = aa_ratio.numer, aa_ratio.denom
    title = (
        f"## A/A control per cell: t_{replica} / t_{base} and Δ_AA = t_{replica} − t_{base} "
        f"(reported only)"
    )
    if not any(v.aa for v in cells.values()):
        return [
            title,
            "",
            f"no data: no cell has an {replica}/{base} pair (no A/A replica in this run).",
            "",
        ]
    rows = []
    for cell, v in cells.items():
        aa, d = v.aa, v.delta_aa_ms
        rule = {None: "n/a", True: "yes", False: "no"}[v.aa_ratio_rule]
        if aa is None:
            rows.append(f"| {cell[0]} | {cell[1]:,} | 0 | no data | | | {rule} | | |")
            continue
        aa_mean = d.mean_ms if d is not None else None
        aa_lo = d.lo_ms if d is not None else None
        aa_hi = d.hi_ms if d is not None else None
        rows.append(
            f"| {cell[0]} | {cell[1]:,} | {aa.n_rounds} | {aa.ratio:.4f} | "
            f"{_ci(aa.lo, aa.hi)} | {aa.verdict or 'n/a'} | {rule} | "
            f"{_f(aa_mean, '+.4f')} | {_ci(aa_lo, aa_hi, '+.4f')} |"
        )
    return [
        title,
        "",
        f"Not a gate since §16's amendment (G3 is the A/A test on the effects, above). The ratio's "
        f'verdict is §8\'s table; "ratio rule" is the old per-cell G3 rule (the '
        f"{settings.CI_LEVEL:.0%} CI contains 1 and lies inside [{1 - settings.DELTA:.2f}, "
        f"{1 + settings.DELTA:.2f}]).",
        "",
        f"| C | P | n | {replica} / {base} | {settings.CI_LEVEL:.0%} CI | verdict | ratio rule | "
        f"Δ_AA, ms | {settings.CI_LEVEL:.0%} CI |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]


def _nll_section(res: CellRunResult) -> list[str]:
    nll = res.nll
    rows = [
        f"| {t} | {v.sessions} | {_f(v.this)} | {_f(v.main_run)} | "
        f"{_f(v.this_minus_main, '+.4f')} |"
        for t, v in nll.per_treatment.items()
    ]
    return [
        "## NLL cross-check against the main run (reported only)",
        "",
        f"Mean session NLL (nats/token, the 64 NLL windows) per treatment, next to the main run's "
        f"({nll.main_run}; MXp serves the MX checkpoint). The same checkpoints on the same "
        f"windows: a difference well beyond noise would say the context-window change altered "
        f"the model. G5b is the gate on NLL. NLL windows: {nll.windows}.",
        "",
        "| treatment | sessions | this run | main run | this − main |",
        "|---|---|---|---|---|",
        *(rows or ["no data: no sessions."]),
        "",
    ]


def _power_section(g4: Mapping, treatments: tuple[Treatment, ...]) -> list[str]:
    per_cell = g4.get("sw_power_cap_frac") or {}
    label = {c.json_key: c.label for c in REGISTERED_CELLS}
    rows = [
        f"| {label.get(key, key)} | "
        + " | ".join(
            "n/a"
            if t not in per_t
            else f"{per_t[t]['max']:.1%} / {per_t[t]['mean']:.1%} ({per_t[t]['blocks']})"
            for t in treatments
        )
        + " |"
        for key, per_t in per_cell.items()
    ]
    return [
        "## SW power capping per cell (reported, G4)",
        "",
        "Fraction of each M1 block's window under the 1000 W software power cap: max / mean over "
        "the blocks (number of blocks). Reported, not discarded (§9).",
        "",
        "| cell | " + " | ".join(treatments) + " |",
        "|---|" + "---|" * len(treatments),
        *(rows or ["no data: no block telemetry with a power-cap fraction."]),
        "",
    ]


def _design_lines(design: DesignCheck) -> list[str]:
    head = f"**Design check (§16, reported):** the manifest has {design.kind or 'no cells'}"
    if design.missing_registered_cells and design.kind != DesignKind.DEVIATES:
        head += (
            " (not in this run: "
            + ", ".join(c.label for c in design.missing_registered_cells)
            + ")"
        )
    head += "."
    argv = design.server_argv
    if argv is not None and not argv.problems:
        head += (
            f" Every session's served argv ({argv.sessions_checked}) has --max-model-len "
            f"{SERVER.max_model_len}, --hf-overrides {json.dumps(HF_OVERRIDES)} and no RoPE "
            f"scaling."
        )
    if design.problems:
        head += f" **{len(design.problems)} problem(s)**:"
    return [head, *(f"- {p}" for p in design.problems), ""]


def _repro_line(repro: Reproduction) -> str:
    target = f"the main run's {repro.target} ± {repro.tolerance}"
    if repro.status == ReproStatus.NO_DATA or repro.r is None or repro.difference is None:
        return f"**Config reproduction:** no data for R{repro.cell.label}."
    ci = "" if repro.lo is None else f" {_ci(repro.lo, repro.hi)}"
    n = repro.n_rounds
    return (
        f"**Config reproduction:** R{repro.cell.label} = {repro.r:.4f}{ci} "
        f"({n} round{'s' if n != 1 else ''}) vs {target}: "
        f"**{repro.status}** ({repro.difference:+.4f})"
        + (
            ""
            if repro.status == ReproStatus.PASS
            else "; the context-limit change may have moved the timing (§16)"
        )
        + "."
    )


def cell_summary(res: CellRunResult, gates_json: object) -> str:
    """`gates_json`: the gates as results_c.json holds them."""
    classes = "; ".join(f"{name}: {classification_text(e)}" for name, e in res.effects.items())
    headline = f"**Experiment C answer: `{res.verdict}`** ({classes})"
    if not res.interpretable:
        headline += f" — **{NOT_INTERPRETABLE}**"
    if res.design.problems:
        headline += " — design deviates from §16 (see the design check)"
    study = res.study
    r = study.ratio(RatioName.R)
    rounds_line = (
        "**Rounds:** no manifest"
        if not res.has_manifest
        else f"**Rounds:** {res.rounds} registered (last manifest line); paired "
        f"{r.denom}/{r.numer} rounds per cell: "
        + ", ".join(f"{c.label} {n}" for c, n in res.paired.items())
    )
    md = [
        f"# Experiment C results: {res.run_dir.name}",
        "",
        headline,
        "",
        *([res.extension, ""] if res.extension else []),
        res.gate_line,
        "",
        _repro_line(res.reproduction),
        "",
        rounds_line,
        "",
        *_design_lines(res.design),
        f"Experiment C (EXPERIMENT.md §16) asks whether batch size or the number of tokens "
        f"changes the NVFP4/MXFP4 gap. Δ = t_{r.denom} − t_{r.numer} is primary: attention "
        f"reads the same KV cache for both formats, so format-independent work leaves Δ "
        f"unchanged while it dilutes R. M1 only; separate from the main run's H_eq verdict.",
        "",
        *_effects_section(res.effects, study),
        *_cells_section(res.cells, r),
        *_predictions_section(res.cells, res.effects),
        *_g3_section(res.gates[Gate.G3], study),
        *_aa_section(res.cells, study.ratio(RatioName.AA)),
        *_power_section(res.gates[Gate.G4], cell_treatments(study)),
        *_nll_section(res),
    ]
    if res.failed:
        md += [
            "## Failed sessions",
            "",
            "Excluded from every computation:",
            "",
            *(
                f"- round {s.round}, {s.treatment}, {s.session_id}: "
                f"{_cell(s.error)[:FAILED_ERROR_CHARS]}"
                for s in res.failed
            ),
            "",
        ]
    if res.skipped:
        md += [
            "## Skipped cells",
            "",
            "Not a finite positive number, so left out of every estimate:",
            "",
            *(f"- round {r}, {t}, {c.label}" for r, t, c in res.skipped),
            "",
        ]
    md += ["## Gates", "", "```json", json.dumps(gates_json, indent=2, default=str), "```", ""]
    return "\n".join(md)
