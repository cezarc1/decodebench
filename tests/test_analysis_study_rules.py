"""The analysis applies the rules of the study the run's manifest recorded (studies.registry)."""

from dataclasses import replace

import pytest

from fp4bench.analysis import plots, stats
from fp4bench.analysis import run as analysis
from fp4bench.analysis.results import BatchRunResult, CellRunResult, Kind
from fp4bench.analysis.verdicts import Answer, Effect
from fp4bench.core.types import Cell, ContrastName, Gate, RatioName, Treatment
from fp4bench.studies.base import G3Rule, Ratio, VerdictRule
from fp4bench.studies.expb import EXPB
from fp4bench.studies.expc import EXPC
from fp4bench.studies.main import FULL
from fp4bench.studies.smoke import SMOKE
from tests.analysis_runs import (  # noqa: F401 - the fixtures _no_figures and figures
    FIVE,
    _no_figures,
    c_run,
    expb,
    expb_manifest,
    figures,
    manifest,
    smoke,
    smoke_manifest,
    synthetic,
    write_run,
)

T = Treatment


def test_the_main_ratio_pairs_are_the_tables_the_analysis_had():
    assert analysis.m1_pairs(FULL.ratios) == {
        "R": (T.NV, T.MX, None),
        "D": (T.NVA, T.MX, None),
        "K": (T.NVA, T.NV, stats.relabel_for_k),
        "F": (T.NVNF, T.MX, None),
        "Phi": (T.NV, T.NVNF, stats.relabel_for_phi),
        "AA": (T.MXP, T.MX, None),
    }
    throughput, itl = analysis.m2_pairs(FULL.ratios)
    assert throughput == {
        "R": (T.MX, T.NV, None),
        "D": (T.MX, T.NVA, None),
        "K": (T.NV, T.NVA, stats.relabel_for_k),
        "F": (T.MX, T.NVNF, None),
        "Phi": (T.NVNF, T.NV, stats.relabel_for_phi),
        "AA": (T.MX, T.MXP, None),
    }
    assert itl == {
        name: pair for name, pair in analysis.m1_pairs(FULL.ratios).items() if name != "AA"
    }
    assert list(throughput) == list(analysis.m1_pairs(FULL.ratios))


def test_the_study_decides_the_kind_and_the_reading(tmp_path, monkeypatch):
    main_run = write_run(tmp_path, *synthetic(1.0), manifest(), name="main")
    assert analysis.evaluate(main_run).kind == Kind.MAIN
    expb_run = write_run(tmp_path, *expb(), expb_manifest(), name="expb")
    out = analysis.evaluate(expb_run)
    assert isinstance(out, BatchRunResult)
    assert out.kind == Kind.EXPB and out.reading is not None
    monkeypatch.setattr(analysis, "study_for", lambda manifest: FULL)
    out = analysis.evaluate(expb_run)
    assert isinstance(out, BatchRunResult)
    assert out.kind == Kind.MAIN and out.reading is None


def test_the_kernel_scan_and_the_cross_check_are_the_studys(tmp_path, monkeypatch):
    run_dir = write_run(tmp_path, *smoke(), smoke_manifest(require_published=False))
    out = analysis.evaluate(run_dir)
    assert isinstance(out, BatchRunResult) and out.study == SMOKE
    assert out.scan is not None and out.crosscheck is not None
    monkeypatch.setattr(
        analysis, "study_for", lambda manifest: replace(SMOKE, kernel_scan=None, crosscheck=None)
    )
    out = analysis.analyze_run(run_dir)
    assert isinstance(out, BatchRunResult) and out.scan is None and out.crosscheck is None
    assert "## Smoke:" not in (run_dir / "summary.md").read_text()


def test_the_verdict_rule_is_the_one_source_of_the_kind():
    assert analysis.KIND_OF_VERDICT == {
        VerdictRule.H_EQ: Kind.MAIN,
        VerdictRule.H_B: Kind.EXPB,
        VerdictRule.BATCH_VS_TOKENS: Kind.EXPC,
    }
    assert set(analysis.KIND_OF_VERDICT) == set(VerdictRule)


def test_a_study_whose_g3_rule_its_analysis_does_not_implement_is_refused(tmp_path, monkeypatch):
    main_run = write_run(tmp_path, *synthetic(1.0), manifest(), name="main")
    monkeypatch.setattr(analysis, "study_for", lambda manifest: replace(FULL, g3=G3Rule.AA_EFFECTS))
    with pytest.raises(ValueError, match="implements G3 as aa_ratio"):
        analysis.evaluate(main_run)
    monkeypatch.setattr(analysis, "study_for", lambda manifest: replace(EXPC, g3=G3Rule.AA_RATIO))
    with pytest.raises(ValueError, match="implements G3 as aa_effects"):
        analysis.evaluate(c_run(tmp_path))


def _figure_texts(out: CellRunResult) -> list[str]:
    fig = plots.delta_figure(out.cells, out.effects, "headline", out.study)
    try:
        return [str(p.get_label()) for ax in fig.axes for p in ax.patches] + [
            t.get_text() for t in fig.texts
        ]
    finally:
        plots.pyplot().close(fig)


def test_the_margins_are_the_studys_in_the_verdict_the_gates_the_summary_and_the_figure(
    tmp_path, monkeypatch
):
    run_dir = c_run(tmp_path)
    out = analysis.analyze_run(run_dir)
    assert isinstance(out, CellRunResult)
    assert list(out.effects) == ["E_tok", "E_batch"]
    assert out.effects[ContrastName.E_BATCH].classification == Effect.SHRINKS
    summary = (run_dir / "summary.md").read_text()
    assert "Margin ±0.1 ms: CI inside" in summary and "lie inside ±0.25 ms." in summary
    wide = replace(
        EXPC, effect_margin_ms=10.0, aa_effect_margin_ms=0.001, contrasts=EXPC.contrasts[::-1]
    )
    monkeypatch.setattr(analysis, "study_for", lambda manifest: wide)
    out = analysis.analyze_run(run_dir)
    assert isinstance(out, CellRunResult)
    assert list(out.effects) == ["E_batch", "E_tok"]
    assert list(out.aa_effects) == ["E_batch,AA", "E_tok,AA"]
    assert {e.classification for e in out.effects.values()} == {Effect.NO_EFFECT}
    assert out.verdict == Answer.NEITHER
    assert out.gates[Gate.G3]["margin_ms"] == 0.001 and out.gates[Gate.G3]["pass"] is False
    summary = (run_dir / "summary.md").read_text()
    assert (
        "Margin ±10.0 ms: CI inside → no effect; entirely below −10.0 → shrinks the saving; "
        "entirely above +10.0 → grows the saving"
    ) in summary
    assert "lie inside ±0.001 ms." in summary
    assert "±0.1 ms" not in summary and "±0.25 ms" not in summary
    assert summary.index("| E_batch | Δ(128, 360)") < summary.index("| E_tok | Δ(1, 127,360)")
    texts = _figure_texts(out)
    assert "±10.0 ms around Δ(1, 1,024): E_tok margin" in texts
    assert "±10.0 ms around Δ(1, 127,360): E_batch margin" in texts
    assert any("the ±10.0 ms margin of the §16 decision rule" in t for t in texts)


def test_the_contrasts_are_named_and_placed_by_the_study(tmp_path, monkeypatch):
    run_dir = c_run(tmp_path)
    token, batch = EXPC.contrasts
    study = replace(
        EXPC,
        contrasts=(
            replace(token, name="tokens"),
            replace(batch, name="batches", cells=(Cell(32, 3360), Cell(1, 127360))),
        ),
    )
    monkeypatch.setattr(analysis, "study_for", lambda manifest: study)
    out = analysis.analyze_run(run_dir)
    assert isinstance(out, CellRunResult)
    assert list(out.effects) == ["tokens", "batches"]
    assert list(out.effects.values())[1].cells == (Cell(32, 3360), Cell(1, 127360))
    assert list(out.aa_effects) == ["tokens,AA", "batches,AA"]
    assert isinstance(out.verdict, Answer)
    summary = (run_dir / "summary.md").read_text()
    assert "| tokens | Δ(1, 127,360) − Δ(1, 1,024) |" in summary
    assert "| batches | Δ(32, 3,360) − Δ(1, 127,360) |" in summary
    assert "| batches,AA | Δ_AA(32, 3,360) − Δ_AA(1, 127,360) |" in summary
    assert "batch effect (batches shrinks the saving)" in summary
    assert "same differences as tokens and batches" in summary
    assert "E_tok" not in summary
    row = next(line for line in summary.splitlines() if line.startswith("| tokens | Δ"))
    assert row.endswith("| — | — |")
    texts = _figure_texts(out)
    assert "±0.1 ms around Δ(1, 127,360): batches margin" in texts


def test_the_ratio_pairs_are_the_studys(tmp_path, monkeypatch):
    main_run = write_run(
        tmp_path, *synthetic(1.0, treatments=FIVE), manifest(treatments=FIVE), name="main"
    )
    d_on_nvnf = replace(
        FULL, ratios=tuple(replace(r, numer=T.NVNF) if r.name == "D" else r for r in FULL.ratios)
    )
    monkeypatch.setattr(analysis, "study_for", lambda manifest: d_on_nvnf)
    out = analysis.analyze_run(main_run)
    assert isinstance(out, BatchRunResult) and out.m1[RatioName.D] == out.m1[RatioName.F]
    summary = (main_run / "summary.md").read_text()
    assert "### M1: D = t_NVnf / t_MX (alternative kernel; does not decide H_eq)" in summary
    assert "### M2: tok/s MX / NVnf\n" in summary and "### M2: ITL p50 NVnf / MX\n" in summary
    run_dir = c_run(tmp_path)
    monkeypatch.undo()
    before = analysis.evaluate(run_dir)
    assert isinstance(before, CellRunResult)
    registered = before.cells[Cell(1, 1024)]
    flipped = replace(
        EXPC, ratios=(Ratio(RatioName.R, T.NV, T.MX), Ratio(RatioName.AA, T.MX, T.MXP))
    )
    monkeypatch.setattr(analysis, "study_for", lambda manifest: flipped)
    cell = analysis.analyze_run(run_dir)
    assert isinstance(cell, CellRunResult)
    one = cell.cells[Cell(1, 1024)]
    assert one.aa is not None and registered.aa is not None
    assert one.aa.ratio == pytest.approx(1 / registered.aa.ratio)
    assert one.delta_aa_ms is not None and registered.delta_aa_ms is not None
    assert one.delta_aa_ms.mean_ms == pytest.approx(-registered.delta_aa_ms.mean_ms)
    summary = (run_dir / "summary.md").read_text()
    assert "Δ_AA = t_MX − t_MXp per cell and round" in summary
    assert "## A/A control per cell: t_MX / t_MXp and Δ_AA = t_MX − t_MXp" in summary


def test_a_cell_study_without_margins_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        analysis, "study_for", lambda manifest: replace(EXPC, aa_effect_margin_ms=None)
    )
    with pytest.raises(ValueError, match="no margins"):
        analysis.evaluate(c_run(tmp_path))


def test_ratios_the_summary_does_not_render_are_refused(tmp_path, monkeypatch):
    main_run = write_run(tmp_path, *synthetic(1.0), manifest(), name="main")
    for study, refusal in (
        (
            replace(FULL, ratios=tuple(r for r in FULL.ratios if r.name != "D")),
            "renders the ratios R, D, K, F, Phi, AA, in this order; the study has R, K, F",
        ),
        (replace(FULL, ratios=FULL.ratios[::-1]), "in this order; the study has AA, Phi"),
        (
            replace(
                FULL,
                ratios=tuple(replace(r, denom=T.NV) if r.name == "F" else r for r in FULL.ratios),
            ),
            r"R = F · Φ needs F = t_X / t_MX",
        ),
    ):
        monkeypatch.setattr(analysis, "study_for", lambda manifest, study=study: study)
        with pytest.raises(ValueError, match=refusal):
            analysis.evaluate(main_run)
    run_dir = c_run(tmp_path)
    extra = replace(EXPC, ratios=(*EXPC.ratios, Ratio(RatioName.D, T.NVA, T.MX)))
    monkeypatch.setattr(analysis, "study_for", lambda manifest: extra)
    with pytest.raises(ValueError, match="renders the ratios R, AA, in this order"):
        analysis.evaluate(run_dir)


def test_a_rule_a_study_cannot_apply_is_refused_when_it_is_built():
    with pytest.raises(ValueError, match="contrasts and their margins are the batch-vs-tokens"):
        replace(FULL, contrasts=EXPC.contrasts)
    with pytest.raises(ValueError, match="contrasts and their margins are the batch-vs-tokens"):
        replace(EXPB, effect_margin_ms=0.1)
    token, batch = EXPC.contrasts
    with pytest.raises(ValueError, match="one contrast per arm"):
        replace(EXPC, contrasts=(token, replace(batch, arm=token.arm)))
    with pytest.raises(ValueError, match="one contrast per arm"):
        replace(EXPC, contrasts=(token, replace(batch, name=token.name)))
    with pytest.raises(ValueError, match="one contrast per arm"):
        replace(EXPC, contrasts=(token,))
    with pytest.raises(ValueError, match="must pair two distinct cells the study measures"):
        replace(EXPC, contrasts=(token, replace(batch, cells=(Cell(64, 1000), Cell(1, 1024)))))
    with pytest.raises(ValueError, match="no margins"):
        replace(EXPC, effect_margin_ms=0.0)
    with pytest.raises(ValueError, match="two ratios share a name"):
        replace(FULL, ratios=(*FULL.ratios, FULL.ratios[0]))


def test_the_verdict_rule_picks_the_cell_analysis(tmp_path, monkeypatch):
    run_dir = c_run(tmp_path)
    assert analysis.evaluate(run_dir).kind == Kind.EXPC
    assert EXPC.verdict is VerdictRule.BATCH_VS_TOKENS and EXPB.verdict is VerdictRule.H_B
