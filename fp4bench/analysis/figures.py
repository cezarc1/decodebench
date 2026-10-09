"""The shareable figures in docs/figures, from the main run, Experiment B and Experiment C."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from fp4bench import settings
from fp4bench.analysis.compare import DeltaResult
from fp4bench.analysis.plots import pyplot
from fp4bench.analysis.results import (
    BatchRunResult,
    CellResult,
    CellRunResult,
    Kind,
    RatioTable,
    RunResult,
)
from fp4bench.bytes_model import r_ideal
from fp4bench.core.types import Cell, KvDtype, RatioName, UnknownKvDtype
from fp4bench.studies.expc import BATCH_ARM, TOKEN_ARM

FIGURES = Path("docs/figures")
OVERVIEW = FIGURES / "r_vs_concurrency.png"
SHARE = FIGURES / "nvfp4_vs_mxfp4_decode_b200.png"
SHARE_SIMPLE = FIGURES / "nvfp4_vs_mxfp4_decode_b200_simple.png"
SHARE_C = FIGURES / "nvfp4_vs_mxfp4_batch_vs_tokens_b200.png"
NV_GREEN, MX_BLUE, ALT_PURPLE, BYTES_RED = "#2a9d3a", "#1f6fb4", "#8e44ad", "#d0453b"
SETUP = (
    "Qwen3-32B (dense). Both formats quantized by us from the same BF16 weights (llm-compressor, "
    "W4A4, every linear except lm_head).\nvLLM v0.31.0, FlashInfer 0.7.0.post1, 1× B200. "
    "Decode step time from synchronized batches of exactly B sequences (prefill excluded); "
    "10 repeats (5 at B ≥ 256), 90% CIs.\nControl: two identical MXFP4 servers agree within "
    "±2% at every batch size."
)
SETUP_C = (
    "Qwen3-32B (dense), W4A4, both formats quantized from the same BF16 weights. vLLM v0.31.0, "
    "1× B200, BF16 KV cache.\nDecode step time from synchronized batches (prefill excluded); "
    "5 repeats, 90% CIs (they include MXFP4's per-server autotune variation).\nControl: two "
    "identical MXFP4 servers show no token or batch effect (−0.003 and +0.02 ms)."
)
RUN_NAMES = {
    Kind.MAIN: "a main run",
    Kind.EXPB: "an Experiment B run",
    Kind.EXPC: "an Experiment C run",
}


def gain_pct(r: float) -> float:
    """A step-time ratio t_NV / t_MX as NVFP4's throughput relative to MXFP4, in %."""
    return (1 / r - 1) * 100


def _batch_run(result: RunResult, kind: Kind) -> BatchRunResult:
    if not isinstance(result, BatchRunResult) or result.kind != kind:
        raise ValueError(
            f"{result.run_dir} is analysed as {RUN_NAMES[result.kind]}; this figure "
            f"needs {RUN_NAMES[kind]}"
        )
    if not result.m1[RatioName.R]:
        raise ValueError(f"{result.run_dir}: no R with a CI (needs two paired rounds)")
    return result


def _bytes_model_kv(run: BatchRunResult) -> KvDtype:
    match run.kv_cache_dtype:
        case KvDtype() as kv:
            return kv
        case UnknownKvDtype(name):
            raise ValueError(
                f"{run.run_dir}: the bytes model has no size for its KV cache dtype {name!r}"
            )


def _cell_run(result: RunResult) -> CellRunResult:
    if not isinstance(result, CellRunResult):
        raise ValueError(
            f"{result.run_dir} is analysed as {RUN_NAMES[result.kind]}; this figure "
            f"needs {RUN_NAMES[Kind.EXPC]}"
        )
    return result


def _save(fig: Any, out: Path, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    pyplot().close(fig)


def overview_figure(main: RunResult, expb: RunResult) -> Any:
    """NVFP4's M1 advantage over MXFP4 against C. The caller closes the figure."""
    import matplotlib.ticker as mticker

    runs = (
        (_batch_run(main, Kind.MAIN), "-", "o", 0.96, "main run, BF16 KV"),
        (_batch_run(expb, Kind.EXPB), "--", "s", 1.04, "Experiment B, FP8 KV"),
    )
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhspan(
        gain_pct(1 + settings.DELTA),
        gain_pct(1 - settings.DELTA),
        color="0.88",
        zorder=0,
        label=f"equivalent: R within ±{settings.DELTA:.0%} (pre-registered δ)",
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvspan(
        200, 235, color="tab:orange", alpha=0.15, zorder=0, label="compute ridge, C ≈ 200–235 (§5)"
    )
    for run, ls, marker, shift, tag in runs:
        kv = _bytes_model_kv(run)
        r = run.m1[RatioName.R]
        cs = sorted(r)
        x = [c * shift for c in cs]
        est = [gain_pct(r[c].ratio) for c in cs]
        lo = [gain_pct(r[c].ci[1]) for c in cs]  # a larger R is a smaller advantage
        hi = [gain_pct(r[c].ci[0]) for c in cs]
        n = min(r[c].n_rounds for c in cs)
        ax.errorbar(
            x,
            est,
            yerr=[
                [e - a for e, a in zip(est, lo, strict=True)],
                [b - e for e, b in zip(est, hi, strict=True)],
            ],
            fmt=ls,
            marker=marker,
            color="k",
            capsize=4,
            ms=6,
            zorder=3,
            label=f"measured, {tag} ({n} rounds, {settings.CI_LEVEL:.0%} CI)",
        )
        for xi, e in zip(x, est, strict=True):
            ax.annotate(
                f"{e:+.1f}%",
                (xi, e),
                textcoords="offset points",
                xytext=(7, 3),
                ha="left",
                fontsize=8,
            )
        ax.plot(
            cs,
            [gain_pct(r_ideal(c, run.context, kv)) for c in cs],
            ls,
            color="tab:red",
            alpha=0.7,
            label=f"bytes model prediction, {tag}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("concurrency C (sequences decoded together = decode batch size)")
    ax.set_ylabel("NVFP4 decode throughput vs MXFP4 (%)")
    ax.set_ylim(-6.5, 9)
    ax.text(1.05, 8.2, "↑ NVFP4 faster", color="tab:green", fontsize=10, weight="bold")
    ax.text(1.05, -6.1, "↓ MXFP4 faster", color="tab:blue", fontsize=10, weight="bold")
    ax.set_title(
        "Qwen3-32B W4A4 decode, 1× B200, vLLM v0.31.0: NVFP4 leads at low C, "
        "no difference from C ≈ 64 up",
        fontsize=10.5,
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def overview(main: RunResult, expb: RunResult, out: Path) -> None:
    _save(overview_figure(main, expb), out, dpi=150)


def _series(
    ax: Any,
    res: RatioTable,
    color: str,
    marker: str,
    ls: str,
    label: str,
    shift: float = 1.0,
    below: bool = False,
) -> None:
    cs = sorted(res)
    x = [c * shift for c in cs]
    y = [gain_pct(res[c].ratio) for c in cs]
    lo = [y_ - gain_pct(res[c].ci[1]) for y_, c in zip(y, cs, strict=True)]
    hi = [gain_pct(res[c].ci[0]) - y_ for y_, c in zip(y, cs, strict=True)]
    ax.errorbar(
        x,
        y,
        yerr=[lo, hi],
        fmt=ls,
        marker=marker,
        color=color,
        ms=7,
        lw=2.2,
        capsize=4,
        label=label,
        zorder=3,
    )
    for xi, yi in zip(x, y, strict=True):
        ax.annotate(
            f"{yi:+.1f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(8, -15 if below else 5),
            fontsize=10,
            color=color,
            weight="bold",
        )


def _corner_labels(ax: Any) -> None:
    ax.text(
        0.015,
        0.975,
        "NVFP4 faster ↑",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color=NV_GREEN,
        weight="bold",
    )
    ax.text(
        0.015,
        0.025,
        "MXFP4 faster ↓",
        transform=ax.transAxes,
        va="bottom",
        fontsize=11,
        color=MX_BLUE,
        weight="bold",
    )


def _ratio_frame(
    ax: Any, xticks: list[int], ylim: tuple[float, float], corner_labels: bool = True
) -> None:
    import matplotlib.ticker as mticker

    ax.axhspan(gain_pct(1 + settings.DELTA), gain_pct(1 - settings.DELTA), color="0.9", zorder=0)
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("decode batch size (sequences per step)", fontsize=12)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    if corner_labels:
        _corner_labels(ax)


def _format_panel(ax: Any, main: BatchRunResult, expb: BatchRunResult) -> None:
    r_main, r_expb = main.m1[RatioName.R], expb.m1[RatioName.R]
    _ratio_frame(ax, [1, 8, 32, 64, 128, 256, 512], (-7, 10))
    ax.text(
        1.0,
        gain_pct(1 - settings.DELTA) - 0.25,
        "within ±2%: no meaningful difference",
        fontsize=9.5,
        color="0.35",
        va="top",
    )
    cs = sorted(r_main)
    ax.plot(
        cs,
        [gain_pct(r_ideal(c, main.context)) for c in cs],
        ":",
        color=BYTES_RED,
        lw=2.2,
        label="prediction if decode were purely\nmemory-bound (4.25 vs 4.5 bits/weight)",
    )
    _series(ax, r_main, "black", "o", "-", "measured, BF16 KV cache", shift=0.96)
    _series(
        ax, r_expb, "0.45", "s", "--", "measured, FP8 KV cache (to fit batch 256–512)", shift=1.04
    )
    ax.set_ylabel("NVFP4 decode throughput vs MXFP4", fontsize=12)
    ax.set_title(
        "NVFP4 vs MXFP4 decode speed in vLLM:\nNVFP4 ahead at small batch, no difference "
        "from batch ≈ 64 up",
        fontsize=13,
        loc="left",
    )
    ax.legend(fontsize=9.5, loc="upper right", frameon=True)


def _kernel_panel(ax: Any, main: BatchRunResult) -> None:
    _ratio_frame(ax, [1, 8, 32, 64, 128], (-11, 10), corner_labels=False)
    ax.axhline(
        0, color=MX_BLUE, lw=2.5, zorder=2, label="MXFP4 baseline (its only W4A4 kernel in vLLM)"
    )
    _series(
        ax, main.m1[RatioName.R], NV_GREEN, "o", "-", "NVFP4 on vLLM's default kernel (CuTe-DSL)"
    )
    _series(
        ax,
        main.m1[RatioName.D],
        ALT_PURPLE,
        "D",
        "-",
        "same NVFP4 weights on vLLM's cuDNN kernel",
        below=True,
    )
    ax.set_ylabel("NVFP4 throughput relative to MXFP4", fontsize=12)
    ax.set_title(
        "Same NVFP4 weights, different vLLM GEMM kernel:\nthe kernel moves the result "
        "more than the format",
        fontsize=13,
        loc="left",
    )
    ax.legend(fontsize=9.5, loc="lower right", frameon=True)


def share_figure(main: RunResult, expb: RunResult) -> Any:
    """The format panel next to the kernel panel. The caller closes the figure."""
    main_run, expb_run = _batch_run(main, Kind.MAIN), _batch_run(expb, Kind.EXPB)
    fig, (ax_l, ax_r) = pyplot().subplots(
        1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    _format_panel(ax_l, main_run, expb_run)
    _kernel_panel(ax_r, main_run)
    fig.text(0.01, 0.01, SETUP, fontsize=9, color="0.3", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def share_simple_figure(main: RunResult, expb: RunResult) -> Any:
    """The format panel alone. The caller closes the figure."""
    main_run, expb_run = _batch_run(main, Kind.MAIN), _batch_run(expb, Kind.EXPB)
    fig, ax = pyplot().subplots(figsize=(10, 6.6))
    _format_panel(ax, main_run, expb_run)
    fig.text(0.01, 0.01, SETUP, fontsize=7.5, color="0.3", va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig


def share(main: RunResult, expb: RunResult, out_dir: Path) -> tuple[Path, Path]:
    """Writes the two-panel and the format-panel-only figure into out_dir; returns their paths."""
    two, one = out_dir / SHARE.name, out_dir / SHARE_SIMPLE.name
    _save(share_figure(main, expb), two, dpi=170)
    _save(share_simple_figure(main, expb), one, dpi=170)
    return two, one


class _Measured(NamedTuple):
    cell: Cell
    delta: DeltaResult
    r: float


def _measured(cells: dict[Cell, CellResult], arm: Sequence[Cell], name: str) -> list[_Measured]:
    out = [
        _Measured(cell, res.delta_ms, res.r.ratio)
        for cell in arm
        if (res := cells.get(cell)) is not None and res.delta_ms is not None and res.r is not None
    ]
    if not out:
        raise ValueError(f"no cell of the {name} arm has a measured Δ")
    return out


def prompt_label(p: int) -> str:
    """16,384 → 16k, 127,360 → 127k, 3,360 → 3.4k."""
    if p < 1000:
        return str(p)
    if p % 1024 == 0:
        return f"{p // 1024}k"
    return f"{p / 1000:.0f}k" if p >= 10_000 else f"{p / 1000:.1f}k"


def _points(ax: Any, points: list[_Measured], x: list[int]) -> None:
    y = [m.delta.mean_ms for m in points]
    lo = [0.0 if m.delta.lo_ms is None else m.delta.mean_ms - m.delta.lo_ms for m in points]
    hi = [0.0 if m.delta.hi_ms is None else m.delta.hi_ms - m.delta.mean_ms for m in points]
    ax.errorbar(
        x,
        y,
        yerr=[lo, hi],
        fmt="-o",
        color=NV_GREEN,
        ms=8,
        lw=2.4,
        capsize=5,
        label="NVFP4 time saved per decode step",
        zorder=3,
    )
    for i, (xi, yi, m) in enumerate(zip(x, y, points, strict=True)):
        above = i % 2 == 0
        cap = m.delta.hi_ms if above else m.delta.lo_ms
        ax.annotate(
            f"{yi:+.2f} ms\n({gain_pct(m.r):+.1f}% tok/s)",
            (xi, yi if cap is None else cap),
            textcoords="offset points",
            xytext=(0, 6 if above else -6),
            ha="center",
            va="bottom" if above else "top",
            fontsize=9.5,
            color=NV_GREEN,
            weight="bold",
        )


def _delta_frame(ax: Any) -> None:
    import matplotlib.ticker as mticker

    ax.axhline(0, color=MX_BLUE, lw=2.5, zorder=2, label="MXFP4 (0 = no difference)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-0.35, 0.65)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}"))
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    _corner_labels(ax)


def share_c_figure(expc: RunResult) -> Any:
    """Experiment C's Δ in the token arm and the batch arm. The caller closes the figure."""
    import matplotlib.ticker as mticker

    cells = _cell_run(expc).cells
    tok, bat = _measured(cells, TOKEN_ARM, "token"), _measured(cells, BATCH_ARM, "batch")
    fig, (ax_l, ax_r) = pyplot().subplots(1, 2, figsize=(16, 7.2), sharey=True)

    _delta_frame(ax_l)
    x = [m.cell.prompt_len for m in tok]
    _points(ax_l, tok, x)
    ax_l.set_xticks(x)
    ax_l.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: prompt_label(int(v))))
    ax_l.set_xlim(x[0] / 1.6, x[-1] * 2.2)
    ax_l.set_xlabel("prompt length (tokens in the KV cache), batch size 1", fontsize=12)
    ax_l.set_ylabel("NVFP4 time saved per decode step vs MXFP4 (ms)", fontsize=12)
    ax_l.set_title(
        "Batch size 1, context length 1k → 127k\n(GEMM size fixed; only the KV cache grows)",
        fontsize=13,
        loc="left",
    )
    ax_l.legend(fontsize=9.5, loc="lower right")

    _delta_frame(ax_r)
    x = [m.cell.batch for m in bat]
    _points(ax_r, bat, x)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(
        [f"{m.cell.batch}\n(prompt {prompt_label(m.cell.prompt_len)})" for m in bat]
    )
    ax_r.set_xlim(x[0] / 1.6, x[-1] * 2.2)
    ax_r.set_xlabel("batch size, at a constant 128k tokens in the KV cache", fontsize=12)
    ax_r.set_title(
        "Constant 128k context tokens, batch size 1 → 128\n(KV cache fixed; only the GEMM "
        "size grows)",
        fontsize=13,
        loc="left",
    )
    ax_r.legend(fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 0.07))

    fig.suptitle(
        "NVFP4 vs. MXFP4 decode speed as context length and batch size vary",
        fontsize=14,
        weight="bold",
        x=0.01,
        ha="left",
    )
    fig.text(0.01, 0.01, SETUP_C, fontsize=9, color="0.3", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    return fig


def share_c(expc: RunResult, out: Path) -> None:
    _save(share_c_figure(expc), out, dpi=170)
