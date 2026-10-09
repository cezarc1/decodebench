"""The figures analyze_run writes: ratio_vs_c.png and pareto_m2.png, or expc_delta.png."""

import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from fp4bench import settings
from fp4bench.analysis.cells import Values, usable
from fp4bench.analysis.checks import r_ideal_or_none
from fp4bench.analysis.compare import ContrastResult, RatioResult
from fp4bench.analysis.design import h_tokens_delta_ms, kv_tokens
from fp4bench.analysis.report import classification_text
from fp4bench.analysis.results import (
    BatchRunResult,
    CellResult,
    CellRunResult,
    RunResult,
)
from fp4bench.core.types import (
    Arm,
    Cell,
    ContrastName,
    Hypothesis,
    KvDtype,
    RatioName,
    RecordedKvDtype,
    Treatment,
    VerdictLabels,
)
from fp4bench.studies.base import Contrast, Ratio, Study
from fp4bench.studies.expc import BATCH_ARM, MEAN_CONTEXT_EXTRA, PREDICTED_DELTA_MS, TOKEN_ARM
from fp4bench.studies.main import FULL
from fp4bench.studies.smoke import SMOKE_ONLY_TREATMENTS

TREATMENT_LABELS = {
    Treatment.MX: "MX: MXFP4",
    Treatment.MXP: "MXp: MXFP4 replica",
    Treatment.NV: "NV: NVFP4, pinned CuTe-DSL",
    Treatment.NVA: "NVa: NVFP4, alternative kernel",
    Treatment.NVNF: "NVnf: NVFP4, CuTe-DSL, act-quant fusion off",
    Treatment.NVX: "NVx: NVIDIA's NVFP4 checkpoint",
    Treatment.NVC: "NVc: NVFP4, FlashInfer CUTLASS",
    Treatment.NVT: "NVt: NVFP4, FlashInfer TRT-LLM",
    Treatment.NVD: "NVd: NVFP4, FlashInfer cuDNN",
    Treatment.NVV: "NVv: NVFP4, vLLM CUTLASS",
}
PARETO_ORDER = tuple(dict.fromkeys((*FULL.treatments, *SMOKE_ONLY_TREATMENTS)))
ABOVE_ONE = {
    VerdictLabels.FORMAT: "MXFP4 faster",
    VerdictLabels.KERNEL: "pinned kernel faster",
    VerdictLabels.FUSION: "fusion slower",
}
H_BATCH_COLOR, H_TOKENS_COLOR = "#d0453b", "#8e44ad"


def pyplot() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def ratio_figure(
    ratios: Mapping[RatioName, Mapping[int, RatioResult]],
    specs: Sequence[Ratio],
    kv_dtype: RecordedKvDtype = KvDtype.BF16,
    context: int = settings.M1_MEAN_CONTEXT,
) -> Any:
    """A panel per ratio but the A/A replica's, against C. The caller closes the figure."""
    plt = pyplot()
    panels = [s for s in specs if s.name != RatioName.AA]
    aa_spec = next(s for s in specs if s.name == RatioName.AA)
    aa = ratios.get(aa_spec.name) or {}
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, spec in zip(axes, panels, strict=True):
        key = spec.name
        res = ratios.get(key) or {}
        ax.set_title(f"{key.symbol} = t_{spec.numer} / t_{spec.denom} (M1)")
        ax.axhspan(1 - settings.DELTA, 1 + settings.DELTA, color="0.9", label="±δ equivalence band")
        cs = sorted(res)
        if cs:
            ax.errorbar(
                cs,
                [res[c].ratio for c in cs],
                yerr=[
                    [res[c].ratio - res[c].ci[0] for c in cs],
                    [res[c].ci[1] - res[c].ratio for c in cs],
                ],
                fmt="o-",
                capsize=4,
                label=f"{key} ({settings.CI_LEVEL:.0%} CI)",
            )
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        if key == RatioName.R:
            acs = sorted(aa)
            if acs:
                ax.errorbar(
                    [c * 1.08 for c in acs],
                    [aa[c].ratio for c in acs],
                    yerr=[
                        [aa[c].ratio - aa[c].ci[0] for c in acs],
                        [aa[c].ci[1] - aa[c].ratio for c in acs],
                    ],
                    fmt="s--",
                    capsize=4,
                    alpha=0.6,
                    label=f"A/A: {aa_spec.numer} / {aa_spec.denom}",
                )
            ideal = [r_ideal_or_none(c, context, kv_dtype) for c in cs]
            if cs and None not in ideal:
                ax.plot(cs, ideal, "k:", label=f"R_ideal (bytes model, {kv_dtype} KV)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrency C")
        ax.set_ylabel(f"ratio (>1 = {ABOVE_ONE[spec.labels]})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def pareto_figure(
    tput: Values[int],
    per_user: Values[int],
    replica: Treatment,
) -> Any:
    """M2 tok/s/user against tok/s/GPU; None without points. The caller closes the figure."""
    plt = pyplot()
    tput = {k: v for k, v in tput.items() if usable(v)}
    per_user = {k: v for k, v in per_user.items() if usable(v)}
    series = {}
    for t in (t for t in PARETO_ORDER if t != replica):
        points = []
        for c in sorted({c for (_, tt, c) in tput if tt == t}):
            users = [v for (_, tt, cc), v in per_user.items() if tt == t and cc == c]
            rates = [v for (_, tt, cc), v in tput.items() if tt == t and cc == c]
            if users:
                points.append((c, statistics.mean(users), statistics.mean(rates)))
        if points:
            series[t] = points
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = "osD^vP*X"
    for i, (t, points) in enumerate(series.items()):
        ax.plot(
            [p[1] for p in points],
            [p[2] for p in points],
            marker=markers[i % len(markers)],
            linestyle="-",
            label=TREATMENT_LABELS.get(t, t),
        )
        for c, xi, yi in points:
            ax.annotate(f"C={c}", (xi, yi), fontsize=7)
    ax.set_xlabel("interactivity: tok/s/user (p50, llm-inference-bench)")
    ax.set_ylabel("sustained decode: tok/s/GPU")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _ci_text(lo: float | None, hi: float | None, spec: str) -> str:
    return "n/a" if lo is None or hi is None else f"[{format(lo, spec)}, {format(hi, spec)}]"


def _observed(
    ax: Any, cells: Mapping[Cell, CellResult], arm: Sequence[Cell], x_of: Callable[[Cell], int]
) -> bool:
    pts = [(x_of(c), cells[c]) for c in arm if c in cells and cells[c].delta_ms]
    deltas = [v.delta_ms for _, v in pts if v.delta_ms is not None]
    if not pts:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="0.4",
        )
        return False
    xs = [x for x, _ in pts]
    ys = [d.mean_ms for d in deltas]
    lo = [0.0 if d.lo_ms is None else y - d.lo_ms for y, d in zip(ys, deltas, strict=True)]
    hi = [0.0 if d.hi_ms is None else d.hi_ms - y for y, d in zip(ys, deltas, strict=True)]
    n = min(d.n_rounds for d in deltas)
    ax.errorbar(
        xs,
        ys,
        yerr=[lo, hi],
        fmt="-",
        marker="o",
        color="black",
        ms=7,
        lw=2.2,
        capsize=4,
        zorder=4,
        label=f"measured Δ ({n} round{'s' if n != 1 else ''}"
        + (f", {settings.CI_LEVEL:.0%} CI)" if n > 1 else ")"),
    )
    for x, y, (_, v) in zip(xs, ys, pts, strict=True):
        if v.r is not None:
            ax.annotate(
                f"R {v.r.ratio:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(7, -16),
                fontsize=9,
                color="0.2",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
            )
    return True


def _margin_band(
    ax: Any, cells: Mapping[Cell, CellResult], contrast: Contrast, margin: float
) -> None:
    cell = contrast.cells[1]
    d = cells[cell].delta_ms if cell in cells else None
    if d is not None:
        ax.axhspan(
            d.mean_ms - margin,
            d.mean_ms + margin,
            color="0.9",
            zorder=0,
            label=f"±{margin} ms around Δ{cell.label}: {contrast.name} margin",
        )


def _effect_title(name: str, e: ContrastResult) -> str:
    if not e.n_rounds:
        return f"{name}: no data"
    ci = "" if e.lo_ms is None else f" {_ci_text(e.lo_ms, e.hi_ms, '+.3f')}"
    return f"{name} = {e.mean_ms:+.3f} ms{ci}: {classification_text(e)}"


def _token_panel(
    ax: Any, cells: Mapping[Cell, CellResult], token: Contrast, e_tok: ContrastResult, margin: float
) -> None:
    import matplotlib.ticker as mticker

    ps = [p for _, p in TOKEN_ARM]
    _margin_band(ax, cells, token, margin)
    ax.axhline(0, color="0.35", lw=1)
    ax.plot(
        ps,
        [PREDICTED_DELTA_MS[Hypothesis.H_BATCH][Cell(1, p)] for p in ps],
        "--",
        marker="s",
        mfc="none",
        ms=12,
        mew=1.6,
        color=H_BATCH_COLOR,
        lw=1.8,
        label="H_batch (pre-registered): Δ ≈ 0.42 ms at every P",
    )
    grid = np.geomspace(ps[0], ps[-1], 80)
    ax.plot(
        grid,
        [h_tokens_delta_ms(kv_tokens(Cell(1, int(p)))) for p in grid],
        ":",
        color=H_TOKENS_COLOR,
        lw=2,
        label="H_tokens: main-run Δ at equal KV tokens\n(pre-registered points marked)",
    )
    pts = [
        (p, v)
        for (c, p), v in PREDICTED_DELTA_MS[Hypothesis.H_TOKENS].items()
        if (c, p) in TOKEN_ARM
    ]
    ax.plot(
        [p for p, _ in pts],
        [v for _, v in pts],
        "D",
        mfc="none",
        color=H_TOKENS_COLOR,
        ms=11,
        mew=1.6,
    )
    _observed(ax, cells, TOKEN_ARM, lambda c: c[1])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xlim(ps[0] / 1.35, ps[-1] * 1.6)
    # matplotlib may pass a numpy float, whose round() is a float before numpy 2
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(round(v)):,}"))  # noqa: RUF046
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("prompt length P (tokens), batch C = 1", fontsize=11)
    ax.set_ylabel("per-step saving Δ = t_MX − t_NV (ms)\n> 0: NVFP4 faster", fontsize=11)
    ax.set_title(
        "Token arm: batch 1, only the KV cache grows\n" + _effect_title(token.name, e_tok),
        fontsize=12,
        loc="left",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left", frameon=True)


def _batch_panel(
    ax: Any,
    cells: Mapping[Cell, CellResult],
    batch: Contrast,
    e_batch: ContrastResult,
    margin: float,
) -> None:
    import matplotlib.ticker as mticker

    cs = [c for c, _ in BATCH_ARM]
    p_of = dict(BATCH_ARM)
    _margin_band(ax, cells, batch, margin)
    ax.axhline(0, color="0.35", lw=1)
    ax.plot(
        cs,
        [PREDICTED_DELTA_MS[Hypothesis.H_BATCH][cell] for cell in BATCH_ARM],
        "--",
        marker="s",
        mfc="none",
        ms=12,
        mew=1.6,
        color=H_BATCH_COLOR,
        lw=1.8,
        label="H_batch (pre-registered): the main run's Δ at batch C",
    )
    level = h_tokens_delta_ms(kv_tokens(BATCH_ARM[0]))
    ax.plot(
        cs,
        [level] * len(cs),
        ":",
        marker="D",
        mfc="none",
        ms=11,
        mew=1.6,
        color=H_TOKENS_COLOR,
        lw=2,
        label=f"H_tokens: equal in every cell (≈ {level:.2f} ms)",
    )
    _observed(ax, cells, BATCH_ARM, lambda c: c[0])
    ax.set_xscale("log", base=2)
    ax.set_xticks(cs)
    ax.set_xlim(cs[0] / 1.5, cs[-1] * 1.9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(round(v))}\n(P {p_of.get(int(round(v)), 0):,})")  # noqa: RUF046
    )
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(
        f"batch C (prompt length P), C × (P + {MEAN_CONTEXT_EXTRA}) = 128,000 KV tokens",
        fontsize=11,
    )
    ax.set_title(
        "Batch arm: constant KV tokens, only the batch changes\n"
        + _effect_title(batch.name, e_batch),
        fontsize=12,
        loc="left",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left", frameon=True)


def delta_figure(
    cells: Mapping[Cell, CellResult],
    effects: Mapping[ContrastName, ContrastResult],
    headline: str,
    study: Study,
) -> Any:
    """Δ against P (token arm) and C (batch arm) with both predictions. The caller closes it."""
    margin, r = study.effect_margin_ms, study.ratio(RatioName.R)
    if margin is None:
        raise ValueError(f"study {study.name}: its contrasts have no margin")
    token, batch = study.contrast_along(Arm.TOKEN), study.contrast_along(Arm.BATCH)
    plt = pyplot()
    fig, (ax_t, ax_b) = plt.subplots(
        1, 2, figsize=(16, 7), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]}
    )
    _token_panel(ax_t, cells, token, effects[token.name], margin)
    _batch_panel(ax_b, cells, batch, effects[batch.name], margin)
    fig.suptitle(
        f"Experiment C (EXPERIMENT.md §16): batch size or tokens? {headline}",
        fontsize=13,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.01,
        "Qwen3-32B W4A4, vLLM v0.31.0, 1× B200, BF16 KV. Δ: mean over rounds of the paired "
        f"per-round difference of median M1 decode step times; R = t_{r.numer} / t_{r.denom} "
        "annotated. "
        "Predictions pre-registered in §16; the dotted H_tokens line reads the main run's Δ\n"
        "(0.42 / 0.49 / 0.32 / 0.14 / −0.19 ms at C = 1 / 8 / 32 / 64 / 128, context 1,664) "
        f"at equal mean KV tokens C × (P + {MEAN_CONTEXT_EXTRA}), linear interpolation. "
        f"Grey band: the ±{margin} ms margin of the §16 decision rule (decided on the paired "
        "CI).",
        fontsize=8.5,
        color="0.3",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def write(result: RunResult) -> None:
    """ratio_vs_c.png and pareto_m2.png when they have data, or expc_delta.png."""
    plt = pyplot()
    run_dir = result.run_dir
    if isinstance(result, CellRunResult):
        fig = delta_figure(
            result.cells,
            result.effects,
            f"Answer: {result.verdict}"
            + ("" if result.interpretable else " (NOT INTERPRETABLE: gate failures)"),
            result.study,
        )
        fig.savefig(run_dir / "expc_delta.png", dpi=160)
        plt.close(fig)
        return
    assert isinstance(result, BatchRunResult)
    if any(table for name, table in result.m1.items() if name != RatioName.AA):
        fig = ratio_figure(result.m1, result.study.ratios, result.kv_cache_dtype, result.context)
        fig.savefig(run_dir / "ratio_vs_c.png", dpi=150)
        plt.close(fig)
    fig = pareto_figure(result.tput, result.per_user, result.study.ratio(RatioName.AA).numer)
    if fig is not None:
        fig.savefig(run_dir / "pareto_m2.png", dpi=150)
        plt.close(fig)
