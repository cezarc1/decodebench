"""analyze_run end to end on synthetic Experiment C runs (§16)."""

import json

import pytest

from fp4bench.analysis import plots
from fp4bench.analysis.compare import ContrastResult
from fp4bench.analysis.results import CellResult, CellRunResult, jsonable
from fp4bench.analysis.run import analyze_run, evaluate
from fp4bench.analysis.verdicts import Answer, Effect, extend_line
from fp4bench.core.types import Cell, ContrastName, Estimate, Gate, Verdict
from fp4bench.studies import expc
from tests.analysis_runs import (  # noqa: F401 - the fixtures _no_figures and figures
    C_SMOKE_CELLS,
    H_BATCH,
    H_TOKENS,
    _no_figures,
    c_build,
    c_manifest,
    c_run,
    c_session,
    failed_row,
    figures,
    m1_row,
    mx_step,
    write_run,
)


def _analyze(tmp_path, write=True, **kw) -> tuple:
    run = c_run(tmp_path, **kw)
    out = (analyze_run if write else evaluate)(run)
    assert isinstance(out, CellRunResult)
    return run, out


def _cell(out: CellRunResult, c: int, p: int) -> CellResult:
    return out.cells[Cell(c, p)]


@pytest.mark.usefixtures("figures")
def test_h_batch_like_data_answers_batch(tmp_path):
    run, out = _analyze(tmp_path, delta=H_BATCH)
    assert out.interpretable, out.gate_line
    assert out.effects[ContrastName.E_TOK].classification == Effect.NO_EFFECT
    assert out.effects["E_batch"].classification == Effect.SHRINKS
    assert out.effects["E_batch"].mean_ms == pytest.approx(-0.61, abs=0.02)
    assert out.verdict == Answer.BATCH_DRIVES
    rs = [r.ratio for c in expc.TOKEN_ARM if (r := out.cells[c].r) is not None]
    assert len(rs) == 6 and rs == sorted(rs)
    r = _cell(out, 128, 360).r
    assert r is not None and r.ratio > 1
    summary = (run / "summary.md").read_text()
    assert f"**Experiment C answer: `{Answer.BATCH_DRIVES}`**" in summary
    assert "NOT INTERPRETABLE" not in summary
    assert (
        "gates: G1 pass · G2 pass · G3 pass · G4 pass · G5a pass · G5b pass · G6a pass · G7 pass"
    ) in summary
    assert (run / "expc_delta.png").stat().st_size > 10_000
    on_disk = json.loads((run / "results_c.json").read_text())
    assert on_disk["answer"] == Answer.BATCH_DRIVES and on_disk == json.loads(
        json.dumps(out.payload())
    )
    g3 = on_disk["gates"]["G3_aa"]
    assert g3["pass"] is True and set(g3["effects"]) == {"E_tok,AA", "E_batch,AA"}
    assert abs(g3["effects"]["E_batch,AA"]["mean_ms"]) < 0.01
    assert on_disk["extension_note"] is None and extend_line(10) not in summary
    assert "## G3: A/A test on the effects" in summary
    assert out.reproduction.status == "pass"


def test_the_observed_kv_tokens_follow_the_rows_decode_lengths(tmp_path):
    def longer(servers, m1):
        for row in m1:
            row["n1"], row["n2"] = 256, 2304

    _, out = _analyze(tmp_path, mutate=longer, write=False)
    assert _cell(out, 128, 360).kv_tokens_mean == 128 * (360 + (256 + 2304) // 2)
    required = out.gates[Gate.G6A]["kv_capacity"]["required_tokens"]
    assert required == max(c * (p + 2304) for c, p in expc.REGISTERED_CELLS)


@pytest.mark.parametrize("value", [None, "1152", True, 1152.0])
def test_a_cell_run_whose_rows_record_a_decode_length_that_is_no_positive_int_is_refused(
    tmp_path, value
):
    def odd(servers, m1):
        m1[3]["n2"] = value

    run = c_run(tmp_path, mutate=odd)
    with pytest.raises(ValueError, match="not a positive integer") as exc:
        evaluate(run)
    assert str(run) in str(exc.value) and f"n2 {value!r}" in str(exc.value)


def test_a_cell_run_whose_rows_disagree_on_the_decode_lengths_is_refused(tmp_path):
    def mixed(servers, m1):
        m1[3]["n2"] = 2304

    run = c_run(tmp_path, mutate=mixed)
    with pytest.raises(ValueError, match="several M1 decode lengths") as exc:
        evaluate(run)
    assert str(run) in str(exc.value)


def test_h_tokens_like_data_answers_tokens(tmp_path):
    _, out = _analyze(tmp_path, delta=H_TOKENS, write=False)
    assert out.effects["E_tok"].classification == Effect.SHRINKS
    assert out.effects["E_tok"].mean_ms == pytest.approx(-0.35, abs=0.02)
    assert out.effects["E_batch"].classification == Effect.NO_EFFECT
    assert out.verdict == Answer.TOKENS_DRIVE and out.interpretable


@pytest.mark.parametrize(
    "delta, answer",
    [
        (
            {
                cell: H_BATCH[cell] - (0.35 if cell[1] >= 127360 or cell[0] > 1 else 0)
                for cell in expc.REGISTERED_CELLS
            },
            Answer.BOTH,
        ),
        (dict.fromkeys(expc.REGISTERED_CELLS, 0.42), Answer.NEITHER),
        ({**H_TOKENS, Cell(128, 360): 0.8}, Answer.INCONCLUSIVE),
    ],
)
def test_both_neither_and_a_growing_saving(tmp_path, delta, answer):
    _, out = _analyze(tmp_path, delta=delta, write=False)
    assert out.verdict == answer
    if answer == Answer.INCONCLUSIVE:
        assert out.effects["E_batch"].classification == Effect.GROWS


def test_noisy_data_is_inconclusive_and_asks_for_the_extension(tmp_path):
    run, out = _analyze(tmp_path, delta=H_BATCH, noise=40.0)
    assert out.effects["E_tok"].classification == Effect.INCONCLUSIVE
    assert out.verdict == Answer.INCONCLUSIVE and out.gates[Gate.G3]["pass"]
    assert out.extension == f"**{extend_line(10)}** — the answer is inconclusive."
    assert extend_line(10) in (run / "summary.md").read_text()


@pytest.mark.usefixtures("figures")
def test_a_smoke_subset_reports_no_data_and_is_not_interpretable(tmp_path):
    run, out = _analyze(tmp_path, cells=C_SMOKE_CELLS, rounds=1, treatments=("MX", "NV"))
    assert out.verdict == "incomplete (1/1 rounds; a CI needs at least 2)"
    assert out.design.kind == "a subset of the cells (smoke)" and out.design.problems == []
    assert _cell(out, 1, 4096).delta_ms is None
    d = _cell(out, 1, 1024).delta_ms
    assert d is not None and d.mean_ms == pytest.approx(0.42, abs=0.02)
    r = _cell(out, 1, 1024).r
    assert r is not None and r.lo is None
    e = out.effects["E_batch"]
    assert e.n_rounds == 1 and e.classification is None
    assert e.mean_ms == pytest.approx(-0.61, abs=0.03)
    assert not out.gates[Gate.G3]["pass"] and out.gates[Gate.G3]["reason"] == "no A/A replica"
    assert out.extension is None
    assert out.gates[Gate.G6A]["pass"], out.gates[Gate.G6A]
    assert not out.interpretable
    summary = (run / "summary.md").read_text()
    assert "NOT INTERPRETABLE" in summary
    assert "| token | 1 | 4,096 | 4,736 | 0 | no data |" in summary
    assert out.reproduction.n_rounds == 1 and (run / "expc_delta.png").exists()


def test_missing_cells_in_a_full_run(tmp_path):
    cells = tuple(c for c in expc.REGISTERED_CELLS if c != (128, 360))
    servers, m1 = c_build(H_BATCH, cells=cells)
    run = write_run(tmp_path, servers, m1, (), c_manifest(), name="run-c")
    out = analyze_run(run)
    assert isinstance(out, CellRunResult)
    assert out.effects[ContrastName.E_BATCH].n_rounds == 0
    assert out.effects[ContrastName.E_BATCH].missing_cells == (Cell(128, 360),)
    assert out.verdict == "incomplete (no data for E_batch)"
    assert not out.gates[Gate.G6A]["pass"]
    assert [0, "MX", 128, 360, 0, 3] in out.gates[Gate.G6A]["missing_m1_rows"]
    assert "E_batch: no data" in (run / "summary.md").read_text()


def test_a_round_short_is_incomplete_but_the_effects_are_shown(tmp_path):
    servers, m1 = c_build(H_BATCH, rounds=4)
    out = evaluate(write_run(tmp_path, servers, m1, (), c_manifest(rounds=5), name="run-c"))
    assert isinstance(out, CellRunResult)
    assert out.verdict == "incomplete (4/5 rounds)"
    assert out.effects[ContrastName.E_TOK].classification == Effect.NO_EFFECT


def test_the_config_reproduction_line_in_the_summary(tmp_path):
    run, out = _analyze(tmp_path, delta={**H_BATCH, Cell(1, 1024): 0.10})
    assert out.reproduction.status == "fail"
    assert "**Config reproduction:** R(1, 1,024) = 0.98" in (run / "summary.md").read_text()


def test_a_per_cell_aa_offset_is_reported_not_gated(tmp_path):
    run, out = _analyze(tmp_path, aa={Cell(32, 3360): 1.03})
    assert out.gates[Gate.G3]["pass"]
    cell = _cell(out, 32, 3360)
    assert cell.aa is not None and cell.delta_aa_ms is not None
    assert cell.aa_ratio_rule is False and cell.aa.verdict == Verdict.MXFP4_FASTER
    assert _cell(out, 1, 1024).aa_ratio_rule is True
    assert cell.delta_aa_ms.mean_ms == pytest.approx(0.03 * mx_step((32, 3360)) * 1000, rel=0.05)
    summary = (run / "summary.md").read_text()
    assert "| 32 | 3,360 | 5 | 1.0299 |" in summary and "| mxfp4_faster | no |" in summary


def test_a_cross_cell_aa_offset_fails_g3_and_asks_for_the_extension(tmp_path):
    factor = 1 + 0.4 / (mx_step((128, 360)) * 1000)
    run, out = _analyze(tmp_path, aa={Cell(128, 360): factor})
    g3 = out.gates[Gate.G3]
    assert not g3["pass"] and g3["effects"]["E_tok,AA"]["pass"]
    assert g3["effects"]["E_batch,AA"]["mean_ms"] == pytest.approx(0.4, abs=0.02)
    assert g3["effects"]["E_batch,AA"]["reason"].startswith("the CI does not contain 0")
    assert not out.interpretable and out.verdict == Answer.BATCH_DRIVES
    summary = (run / "summary.md").read_text().splitlines()
    assert summary[4] == f"**{extend_line(10)}** — G3 failed."
    assert "G3 FAIL" in summary[6]


@pytest.mark.parametrize("aa_noise, ok", [(10.0, True), (30.0, False)])
def test_g3_needs_a_narrow_aa_ci(tmp_path, aa_noise, ok):
    _, out = _analyze(tmp_path, aa_noise=aa_noise, write=False)
    g3 = out.gates[Gate.G3]
    e = g3["effects"]["E_tok,AA"]
    assert e["lo_ms"] <= 0 <= e["hi_ms"] and g3["pass"] is ok
    assert (e["hi_ms"] - e["lo_ms"]) / 2 == pytest.approx(0.106 if ok else 0.319, abs=0.01)


def test_the_nll_crosscheck_is_in_the_summary_and_reported_only(tmp_path):
    def mutate(servers, m1):
        for s in servers:
            if s["treatment"] == "NV":
                s["nll"] = 1.9

    run, out = _analyze(tmp_path, mutate=mutate)
    assert out.interpretable and out.nll.windows == "the main run's"
    summary = (run / "summary.md").read_text()
    assert "## NLL cross-check against the main run (reported only)" in summary
    assert "| NV | 5 | 1.9000 | 1.8238 | +0.0762 |" in summary


def test_the_served_argv_check_is_in_the_summary_and_a_deviation_in_the_headline(tmp_path):
    (tmp_path / "a").mkdir()
    run, out = _analyze(tmp_path / "a")
    assert out.design.server_argv is not None and out.design.server_argv.sessions_checked == 15
    assert (
        "Every session's served argv (15) has --max-model-len 131072"
        in (run / "summary.md").read_text()
    )

    def mutate(servers, m1):
        argv = servers[4]["server_argv"]
        argv[argv.index("--max-model-len") + 1] = "4096"

    (tmp_path / "b").mkdir()
    run, out = _analyze(tmp_path / "b", mutate=mutate)
    assert "design deviates from §16" in (run / "summary.md").read_text().splitlines()[2]
    (tmp_path / "c").mkdir()
    run, _ = _analyze(tmp_path / "c", manifest=c_manifest(max_model_len=65536))
    summary = (run / "summary.md").read_text()
    assert "design deviates from §16" in summary.splitlines()[2]
    assert "protocol.max_model_len is 65536" in summary


@pytest.mark.usefixtures("figures")
def test_no_sessions_fail_every_gate_and_no_manifest_gives_no_answer(tmp_path):
    run = write_run(tmp_path, [], [], (), c_manifest(), name="empty")
    out = analyze_run(run)
    assert isinstance(out, CellRunResult)
    assert all(not g["pass"] for g in out.gates.values())
    assert out.verdict == "incomplete (no data for E_tok and E_batch)"
    assert (run / "expc_delta.png").exists()
    servers, m1 = c_build(H_BATCH, rounds=2)
    run = write_run(tmp_path, servers, m1, (), None, name="none")
    with pytest.raises(ValueError, match="several prompt lengths"):
        evaluate(run)


def test_the_results_json_is_strict_json_with_cell_keys(tmp_path):
    def mutate(servers, m1):
        m1[2]["step_s"] = float("nan")

    run, _ = _analyze(tmp_path, mutate=mutate)
    text = (run / "results_c.json").read_text()
    assert "NaN" not in text
    data = json.loads(text)
    assert data["skipped"] == [[0, "MX", [1, 1024]]]
    assert {c["c"] for c in data["cells"]} == {1, 8, 32, 128}
    assert data["predictions"]["delta_ms"]["H_batch"]["C=128,P=360"] == -0.19
    assert data["gates"]["G4_env_throttle"]["sw_power_cap_frac"]["C=1,P=1024"]["MX"]["blocks"] == 5


def test_the_summary_has_the_predictions_aa_and_power_tables(tmp_path):
    run, _ = _analyze(tmp_path)
    summary = (run / "summary.md").read_text()
    assert "## Pre-registered predictions (§16)" in summary
    assert "| (1, 127,360) |" in summary and "| +0.42 | +0.07 |" in summary
    assert "| E_batch |" in summary and "| -0.60 | +0.00 |" in summary
    assert "## A/A control per cell" in summary and "## SW power capping" in summary


def test_the_figure_renders_without_data():
    import matplotlib.pyplot as plt

    empty = ContrastResult((Cell(1, 127360), Cell(1, 1024)), 0, (), (), None, None, None, None, {})
    fig = plots.delta_figure(
        {}, {ContrastName.E_TOK: empty, ContrastName.E_BATCH: empty}, "no data", expc.EXPC
    )
    assert len(fig.axes) == 2
    plt.close(fig)


def test_a_failed_session_and_an_unregistered_cell_are_listed(tmp_path):
    def mutate(servers, m1):
        servers.append(
            failed_row(
                c_session(0, "MXp") | {"session_id": "r0_MXp-y"}, error="RuntimeError('boom')"
            )
        )
        m1.extend(
            m1_row(
                round=r,
                treatment=t,
                session_id=f"r{r}_{t}-x",
                c=1,
                prompt_len=2048,
                set=s,
                step_s=0.0061 if t == "MX" else 0.0057,
            )
            for r in range(5)
            for t in ("MX", "NV")
            for s in (1, 2, 3)
        )

    run, out = _analyze(tmp_path, mutate=mutate)
    assert [s.session_id for s in out.failed] == ["r0_MXp-y"] and Cell(1, 2048) in out.cells
    summary = (run / "summary.md").read_text()
    assert "## Failed sessions" in summary and "r0_MXp-y: RuntimeError('boom')" in summary
    assert "| (1, 2,048) |" not in summary.split("## Pre-registered predictions")[1].split("##")[0]
    assert "| — | 1 | 2,048 |" in summary
    assert json.loads((run / "results_c.json").read_text())["failed_sessions"] == ["r0_MXp-y"]


def test_jsonable_turns_cells_into_keys_tuples_into_lists_and_nan_into_null():
    assert jsonable({Cell(1, 1024): (1.0, float("nan")), "x": [float("inf")]}) == {
        "C=1,P=1024": [1.0, None],
        "x": [None],
    }


def test_jsonable_renames_only_cell_keys():
    other = Estimate(1.0, 0.5, 2.0)
    assert jsonable({other: 1, (1, 1024): 2, Cell(8, 360): 3}) == {
        other: 1,
        (1, 1024): 2,
        "C=8,P=360": 3,
    }
    assert type(next(iter(jsonable({other: 1})))) is Estimate


def test_a_session_whose_rows_lack_prompt_len_is_malformed_and_left_out(tmp_path):
    def mutate(servers, m1):
        for row in m1:
            if (row["round"], row["treatment"]) == (1, "NV"):
                del row["prompt_len"]

    _, out = _analyze(tmp_path, mutate=mutate)
    assert set(out.paired.values()) == {4} and not out.interpretable
    g6a = out.gates[Gate.G6A]
    assert {(*m[:4], m[-1]) for m in g6a["malformed_rows"]} == {
        (1, "NV", c, None, "no (c, prompt_len) cell") for c, _ in expc.REGISTERED_CELLS
    }
    assert len(g6a["missing_m1_rows"]) == len(expc.REGISTERED_CELLS)
