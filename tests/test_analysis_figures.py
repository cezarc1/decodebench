"""The shareable figures (fp4bench.analysis.figures) draw the run results they are given, and the
committed docs/figures are what they draw from data/runs: pixel for pixel on the golden platform,
within tests/golden/pngs.py's tolerance elsewhere."""

import dataclasses
from pathlib import Path

import pytest
from click.testing import CliRunner

from fp4bench import cli
from fp4bench.analysis import figures
from fp4bench.analysis.figures import gain_pct
from fp4bench.analysis.results import BatchRunResult, CellRunResult
from fp4bench.analysis.run import evaluate
from fp4bench.bytes_model import r_ideal
from fp4bench.core.types import KvDtype, RatioName, UnknownKvDtype
from fp4bench.studies.expc import BATCH_ARM, TOKEN_ARM
from tests import REPO, RUNS_DIR
from tests.golden.pngs import on_golden_platform, pixel_difference, pixels


@pytest.fixture(scope="module")
def runs() -> dict[str, BatchRunResult | CellRunResult]:
    out = {}
    for name in ("full-1", "expb-1", "expc-1", "smoke-nf-2", "smoke-c-1"):
        result = evaluate(RUNS_DIR / name)
        assert isinstance(result, (BatchRunResult, CellRunResult))
        out[name] = result
    return out


@pytest.fixture
def plt():
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def _errorbars(ax) -> list[tuple[str, list[float], list[float], list[float], list[float]]]:
    """(label, x, y, lower end, upper end) of each error-bar series on `ax`."""
    out = []
    for c in ax.containers:
        data_line, _, (bars, *_) = c.lines
        segments = bars.get_segments()
        out.append(
            (
                c.get_label(),
                list(data_line.get_xdata()),
                list(data_line.get_ydata()),
                [s[0][1] for s in segments],
                [s[1][1] for s in segments],
            )
        )
    return out


def _assert_ratio_series(series, table, shift: float) -> None:
    _, x, y, lower, upper = series
    cs = sorted(table)
    assert x == pytest.approx([c * shift for c in cs])
    assert y == [gain_pct(table[c].ratio) for c in cs]
    assert lower == pytest.approx([gain_pct(table[c].ci[1]) for c in cs])
    assert upper == pytest.approx([gain_pct(table[c].ci[0]) for c in cs])


def test_the_overview_draws_r_with_its_ci_as_nvfp4s_gain_for_both_runs(runs, plt):
    main, expb = runs["full-1"], runs["expb-1"]
    assert isinstance(main, BatchRunResult) and isinstance(expb, BatchRunResult)
    ax = figures.overview_figure(main, expb).axes[0]
    bars = _errorbars(ax)
    assert [label for label, *_ in bars] == [
        "measured, main run, BF16 KV (10 rounds, 90% CI)",
        "measured, Experiment B, FP8 KV (5 rounds, 90% CI)",
    ]
    _assert_ratio_series(bars[0], main.m1[RatioName.R], 0.96)
    _assert_ratio_series(bars[1], expb.m1[RatioName.R], 1.04)
    assert [t.get_text() for t in ax.texts][:5] == ["+7.1%", "+7.6%", "+4.2%", "+1.3%", "-1.2%"]
    ideal = [line for line in ax.lines if line.get_label().startswith("bytes model")]
    for line, run, kv in zip(ideal, (main, expb), (KvDtype.BF16, KvDtype.FP8), strict=True):
        cs = sorted(run.m1[RatioName.R])
        assert list(line.get_ydata()) == [gain_pct(r_ideal(c, 1664, kv)) for c in cs]


def test_the_share_figure_draws_r_for_both_runs_and_d_for_the_kernel_panel(runs, plt):
    main, expb = runs["full-1"], runs["expb-1"]
    assert isinstance(main, BatchRunResult) and isinstance(expb, BatchRunResult)
    format_ax, kernel_ax = figures.share_figure(main, expb).axes
    for ax in (format_ax, figures.share_simple_figure(main, expb).axes[0]):
        bars = _errorbars(ax)
        _assert_ratio_series(bars[0], main.m1[RatioName.R], 0.96)
        _assert_ratio_series(bars[1], expb.m1[RatioName.R], 1.04)
        memory_bound = [line for line in ax.lines if line.get_label().startswith("prediction")]
        assert list(memory_bound[0].get_ydata()) == [
            gain_pct(r_ideal(c, 1664)) for c in sorted(main.m1[RatioName.R])
        ]
    bars = _errorbars(kernel_ax)
    assert [label for label, *_ in bars] == [
        "NVFP4 on vLLM's default kernel (CuTe-DSL)",
        "same NVFP4 weights on vLLM's cuDNN kernel",
    ]
    _assert_ratio_series(bars[0], main.m1[RatioName.R], 1.0)
    _assert_ratio_series(bars[1], main.m1[RatioName.D], 1.0)


def test_the_experiment_c_figure_draws_delta_per_cell_of_each_arm(runs, plt):
    expc = runs["expc-1"]
    assert isinstance(expc, CellRunResult)
    token_ax, batch_ax = figures.share_c_figure(expc).axes
    for ax, arm, x_of in (
        (token_ax, TOKEN_ARM, lambda c: c.prompt_len),
        (batch_ax, BATCH_ARM, lambda c: c.batch),
    ):
        deltas = [d for cell in arm if (d := expc.cells[cell].delta_ms) is not None]
        ratios = [r.ratio for cell in arm if (r := expc.cells[cell].r) is not None]
        assert len(deltas) == len(ratios) == len(arm)
        ((_, x, y, lower, upper),) = _errorbars(ax)
        assert x == [x_of(cell) for cell in arm]
        assert y == [d.mean_ms for d in deltas]
        assert lower == pytest.approx([d.lo_ms for d in deltas])
        assert upper == pytest.approx([d.hi_ms for d in deltas])
        assert [t.get_text() for t in ax.texts if "ms" in t.get_text()] == [
            f"{d.mean_ms:+.2f} ms\n({gain_pct(r):+.1f}% tok/s)"
            for d, r in zip(deltas, ratios, strict=True)
        ]
    assert [t.get_text() for t in batch_ax.get_xticklabels()] == [
        "1\n(prompt 127k)",
        "8\n(prompt 15k)",
        "32\n(prompt 3.4k)",
        "128\n(prompt 360)",
    ]


def test_the_experiment_c_figure_draws_a_one_round_run_without_cis(runs, plt):
    smoke = runs["smoke-c-1"]
    assert isinstance(smoke, CellRunResult)
    token_ax, _ = figures.share_c_figure(smoke).axes
    ((_, x, y, lower, upper),) = _errorbars(token_ax)
    assert x == [1024, 32768, 127360] and lower == y and upper == y


@pytest.mark.parametrize(
    "p, label",
    [(360, "360"), (1024, "1k"), (16384, "16k"), (3360, "3.4k"), (15360, "15k"), (127360, "127k")],
)
def test_prompt_label(p, label):
    assert figures.prompt_label(p) == label


def test_the_figures_are_written_and_closed(runs, plt, tmp_path):
    main, expb, expc = runs["full-1"], runs["expb-1"], runs["expc-1"]
    figures.overview(main, expb, tmp_path / "new" / "overview.png")
    paths = figures.share(main, expb, tmp_path / "share")
    figures.share_c(expc, tmp_path / "c.png")
    assert paths == (
        tmp_path / "share" / "nvfp4_vs_mxfp4_decode_b200.png",
        tmp_path / "share" / "nvfp4_vs_mxfp4_decode_b200_simple.png",
    )
    for path in (tmp_path / "new" / "overview.png", *paths, tmp_path / "c.png"):
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not plt.get_fignums()


def test_a_figure_refuses_a_run_of_the_wrong_kind(runs, plt, tmp_path):
    main, expb, expc = runs["full-1"], runs["expb-1"], runs["expc-1"]
    with pytest.raises(
        ValueError, match="analysed as an Experiment C run; this figure needs a main run"
    ):
        figures.overview(expc, expb, tmp_path / "o.png")
    with pytest.raises(
        ValueError, match="analysed as a main run; this figure needs an Experiment B run"
    ):
        figures.share(main, main, tmp_path)
    with pytest.raises(ValueError, match="needs an Experiment C run"):
        figures.share_c(main, tmp_path / "c.png")
    with pytest.raises(ValueError, match="no R with a CI"):
        figures.overview(runs["smoke-nf-2"], expb, tmp_path / "o.png")
    assert not list(tmp_path.iterdir())


def test_the_overview_refuses_a_kv_dtype_the_bytes_model_has_no_size_for(runs, plt, tmp_path):
    main, expb = runs["full-1"], runs["expb-1"]
    assert isinstance(expb, BatchRunResult)
    auto = dataclasses.replace(expb, kv_cache_dtype=UnknownKvDtype("auto"))
    with pytest.raises(ValueError, match="no size for its KV cache dtype 'auto'"):
        figures.overview(main, auto, tmp_path / "o.png")
    assert not list(tmp_path.iterdir())


def _pixels(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return image.info.get("Software", ""), np.asarray(image.convert("RGBA"))


def plot(*args: str):
    return CliRunner().invoke(cli.main, ["plot", *args], prog_name="fp4bench")


@pytest.mark.parametrize(
    "command, args, names",
    [
        ("overview", ["full-1", "expb-1", "--out"], ["r_vs_concurrency.png"]),
        (
            "share",
            ["full-1", "expb-1", "--out-dir"],
            ["nvfp4_vs_mxfp4_decode_b200.png", "nvfp4_vs_mxfp4_decode_b200_simple.png"],
        ),
        ("share-c", ["expc-1", "--out"], ["nvfp4_vs_mxfp4_batch_vs_tokens_b200.png"]),
    ],
)
def test_the_committed_figures_are_what_plot_draws_from_data_runs(command, args, names, tmp_path):
    import matplotlib as mpl
    import numpy as np

    software, _ = _pixels(REPO / "docs" / "figures" / names[0])
    if f"version{mpl.__version__}," not in software:
        pytest.skip(f"docs/figures were drawn by {software!r}, this is {mpl.__version__}")
    out = tmp_path / names[0] if len(names) == 1 else tmp_path
    result = plot(command, *(str(RUNS_DIR / a) for a in args[:-1]), args[-1], str(out))
    assert result.exit_code == 0 and result.output.startswith("wrote "), result.output
    for name in names:
        committed_path, drawn_path = REPO / "docs" / "figures" / name, tmp_path / name
        assert pixel_difference(pixels(committed_path), pixels(drawn_path)) is None, name
        if on_golden_platform():
            _, committed = _pixels(committed_path)
            _, drawn = _pixels(drawn_path)
            assert committed.shape == drawn.shape and np.array_equal(committed, drawn), name


@pytest.mark.parametrize(
    "command, argv, code, message",
    [
        ("overview", ["nope", str(RUNS_DIR / "expb-1")], 2, "'nope' does not exist"),
        ("share", [str(RUNS_DIR / "full-1"), "nope"], 2, "'nope' does not exist"),
        ("share-c", ["nope"], 2, "'nope' does not exist"),
        ("share-c", [str(RUNS_DIR / "full-1")], 1, "needs an Experiment C run"),
    ],
)
def test_plot_exits_with_a_message(command, argv, code, message, tmp_path):
    result = plot(
        command, *argv, "--out-dir" if command == "share" else "--out", str(tmp_path / "x")
    )
    assert result.exit_code == code and message in result.output, result.output
    assert not (tmp_path / "x").exists()


def test_plot_writes_to_docs_figures_by_default(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr("fp4bench.analysis.run.evaluate", lambda run_dir: run_dir)
    monkeypatch.setattr(figures, "overview", lambda main, expb, out: drawn.append(out))
    monkeypatch.setattr(figures, "share_c", lambda expc, out: drawn.append(out))
    monkeypatch.setattr(
        figures, "share", lambda main, expb, out_dir: drawn.append(out_dir) or ("a.png", "b.png")
    )
    assert plot("overview", str(RUNS_DIR / "full-1"), str(RUNS_DIR / "expb-1")).exit_code == 0
    assert plot("share", str(RUNS_DIR / "full-1"), str(RUNS_DIR / "expb-1")).exit_code == 0
    assert plot("share-c", str(RUNS_DIR / "expc-1")).exit_code == 0
    assert drawn == [figures.OVERVIEW, figures.FIGURES, figures.SHARE_C]
