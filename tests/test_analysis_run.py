"""analyze_run end to end on synthetic main, Experiment B and smoke runs."""

import inspect
import json
import statistics
from typing import Any

import pytest

from fp4bench import bytes_model as bm
from fp4bench import runner, settings
from fp4bench.analysis import plots
from fp4bench.analysis.report import REFERENCE_RUN_NEEDED, _table
from fp4bench.analysis.results import BatchRunResult, CellRunResult, RunResult
from fp4bench.analysis.run import analyze_run, evaluate, unexpected_kv_dtype_note
from fp4bench.core.schema import ServerRow, load_rows
from fp4bench.core.types import (
    Format,
    Gate,
    KvDtype,
    LinearBackend,
    RatioName,
    Treatment,
    UnknownKvDtype,
    Verdict,
)
from fp4bench.studies import kernel_scan as scan_rule
from fp4bench.studies import model
from fp4bench.studies.main import FULL
from fp4bench.studies.smoke import SMOKE
from tests import RUNS_DIR
from tests.analysis_runs import (  # noqa: F401 - the fixtures _no_figures and figures
    CLEAN_CKPT,
    CUTE_DSL,
    EXPB_CS,
    FIVE,
    GATE_LINE_CLEAN,
    SCAN_FACTORS,
    SMOKE_CS,
    _no_figures,
    drop_m1,
    expb,
    expb_manifest,
    fail,
    failed_row,
    figures,
    graph_hash,
    main_and_expb,
    manifest,
    pp,
    recording_args,
    set_fields,
    set_nll,
    smoke,
    smoke_manifest,
    synthetic,
    write_run,
)

SMOKE_SCAN = ("NV", *scan_rule.NVA_SCAN.treatments)
BANDWIDTH_TITLE = "### Effective HBM bandwidth (bytes model / M1 step)"


def _run(
    tmp_path, rows=None, line: Any = "ok", name="run", reference=None, write=True
) -> tuple[Any, BatchRunResult]:
    servers, m1, m2 = synthetic(1.0, (8, 32, 128)) if rows is None else rows
    run = write_run(tmp_path, servers, m1, m2, manifest() if line == "ok" else line, name=name)
    result = (analyze_run if write else evaluate)(run, reference)
    assert isinstance(result, BatchRunResult)
    return run, result


def _md(run) -> str:
    return (run / "summary.md").read_text()


def _section(md: str, title: str) -> str:
    return md.split(title)[1].split("###", maxsplit=1)[0]


@pytest.mark.usefixtures("figures")
def test_end_to_end_clean_run(tmp_path):
    run, out = _run(tmp_path)
    assert out.verdict == "validated" and out.interpretable is True
    assert out.gate_line == GATE_LINE_CLEAN
    for name in ("summary.md", "ratio_vs_c.png", "pareto_m2.png"):
        assert (run / name).stat().st_size > 0, name
    md = _md(run)
    assert "`validated`" in md and GATE_LINE_CLEAN in md and "NOT INTERPRETABLE" not in md
    assert md.index("Overall verdict") < md.index(GATE_LINE_CLEAN) < md.index("### M1: R")
    assert "| 90% CI |" in md and "M2 A/A" in md and "order effect" in md
    assert set(out.m1[RatioName.R]) == {8, 32, 128} and set(out.m2_throughput[RatioName.AA]) == {
        8,
        32,
        128,
    }
    assert "### Accuracy note" not in md and "## Accuracy note" in md
    assert '"checkpoint_source": "manifest' in md and '"config_diff": {}' in md
    assert md.split("\n\n")[1].startswith(
        "**Overall verdict on H_eq (M1, R = t_NV / t_MX, primary C (8, 32, 128)): `validated`"
    )


def test_end_to_end_mxfp4_faster_and_an_incomplete_run(tmp_path):
    _, out = _run(tmp_path, synthetic(1.03, (8, 32, 128)))
    assert out.verdict == "nullified_mxfp4_faster" and out.gate_line == GATE_LINE_CLEAN
    run, out = _run(tmp_path, synthetic(1.0, (8, 32, 128), rounds=3), name="three")
    assert out.verdict == "incomplete (3/5 rounds)"
    assert "`incomplete (3/5 rounds)`" in _md(run)


def test_the_verdict_uses_the_last_manifest_line_and_needs_one(tmp_path):
    _, out = _run(tmp_path, line=[manifest(rounds=5), manifest(rounds=10)])
    assert out.verdict == "incomplete (5/10 rounds)"
    run, out = _run(tmp_path, line=None, name="none")
    assert out.verdict == "incomplete (no manifest)" and out.gates[Gate.G1]["pass"] is False
    assert "**Rounds:** no manifest" in _md(run)


def test_the_summary_is_identical_on_every_run(tmp_path):
    rows = synthetic(1.0, (8, 32, 128))
    texts = []
    for i in range(2):
        (tmp_path / str(i)).mkdir()
        texts.append(_md(_run(tmp_path / str(i), rows)[0]))
    assert texts[0] == texts[1]


def test_the_code_version_in_the_manifest_changes_nothing_in_the_analysis(tmp_path):
    rows = synthetic(1.0, (8, 32, 128))
    texts = []
    for i, extra in enumerate(({}, {"code_commit": "e" * 40, "code_dirty": True})):
        (tmp_path / str(i)).mkdir()
        texts.append(_md(_run(tmp_path / str(i), rows, line={**manifest(), **extra})[0]))
    assert texts[0] == texts[1]


@pytest.mark.usefixtures("figures")
def test_analyze_run_closes_its_figures(tmp_path):
    import matplotlib.pyplot as plt

    plt.close("all")
    _run(tmp_path)
    assert plt.get_fignums() == []


@pytest.mark.usefixtures("figures")
def test_an_empty_run_dir_fails_every_gate(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    out = analyze_run(run)
    assert out.verdict == "incomplete (no manifest)" and out.interpretable is False
    assert all(g["pass"] is False for g in out.gates.values())
    md = _md(run)
    assert "NOT INTERPRETABLE: gate failures" in md
    assert (
        "gates: G1 FAIL · G2 FAIL · G3 FAIL · G4 FAIL · G5a FAIL · G5b FAIL · G6a FAIL "
        "· G6b FAIL · G7 FAIL"
    ) in md
    assert not (run / "ratio_vs_c.png").exists()


def test_a_main_run_with_several_prompt_lengths_and_no_protocol_cells_is_refused(tmp_path):
    servers, m1, m2 = synthetic(1.0)
    for r in m1:
        if r["c"] == 32:
            r["prompt_len"] = 4096
    with pytest.raises(ValueError, match="several prompt lengths"):
        _run(tmp_path, (servers, m1, m2))


def _uncounted_rows(servers, m1, m2) -> None:
    """Rows the schema refuses, of sessions that do not count."""
    nv = next(i for i, s in enumerate(servers) if s["session_id"] == "r1_NV-x")
    servers.insert(nv, failed_row(servers[nv]) | {"session_id": "r1_NV-failed"})
    mx = next(s for s in servers if s["session_id"] == "r2_MX-x")
    servers.append(dict(mx, session_id="r2_MX-rerun"))
    m1.extend([dict(r, session_id="r2_MX-rerun") for r in m1 if r["session_id"] == "r2_MX-x"])
    m2.extend([dict(r, session_id="r2_MX-rerun") for r in m2 if r["session_id"] == "r2_MX-x"])
    for sid, change in (
        ("ghost", {"treatment": "XX"}),
        ("r1_NV-failed", {"block_telemetry": 1}),
        ("r2_MX-x", {"step_s": None, "treatment": "MXz"}),
    ):
        m1.append(dict(m1[0], session_id=sid, **change))
    m2.append({k: v for k, v in m2[0].items() if k != "telemetry"} | {"session_id": "ghost"})


def test_rows_of_no_counted_session_that_do_not_load_are_left_out(tmp_path):
    rows = synthetic(1.0, (8, 32, 128))
    clean = evaluate(write_run(tmp_path, *rows, manifest(), name="clean"))
    _uncounted_rows(*rows)
    out = evaluate(write_run(tmp_path, *rows, manifest(), name="run"))
    assert out.verdict == clean.verdict and out.gate_line == clean.gate_line


@pytest.mark.parametrize(
    "name, change",
    [("m1", {"treatment": "XX"}), ("m1", {"block_telemetry": None}), ("m2", {"telemetry": "x"})],
)
def test_a_row_of_a_counted_session_that_does_not_load_is_refused(tmp_path, name, change):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    rows = {"m1": m1, "m2": m2}[name]
    rows[3] = rows[3] | change
    run = write_run(tmp_path, servers, m1, m2, manifest())
    with pytest.raises(ValueError, match=rf"{name}.jsonl, line 4: "):
        evaluate(run)


@pytest.mark.parametrize("prompt_len, context", [(None, 1664), (1024, 1664), (2048, 2688)])
def test_a_main_run_reads_its_context_from_the_rows_prompt_length(tmp_path, prompt_len, context):
    assert (
        settings.M1_INPUT_LEN + (settings.M1_N1 + settings.M1_N2) // 2
        == settings.M1_MEAN_CONTEXT
        == 1664
    )
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    if prompt_len is not None:
        for row in m1:
            row["prompt_len"] = prompt_len
    run, out = _run(tmp_path, (servers, m1, m2))
    assert out.context == context
    md = _md(run)
    assert f"R_ideal: the bytes model's upper bound (§5) at context {context} " in md
    assert f"KV cache of C sequences at context {context})" in md
    r_table = _section(md, "### M1: R = t_NV / t_MX (decides H_eq)")
    assert r_table.split("\n| 8 |")[1].split("\n")[0].endswith(f" {bm.r_ideal(8, context):.4f} |")


def test_a_main_run_reads_its_context_from_the_rows_decode_lengths(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    for row in m1:
        row["n1"], row["n2"] = 256, 2304
    run, out = _run(tmp_path, (servers, m1, m2))
    assert out.context == settings.M1_INPUT_LEN + (256 + 2304) // 2 == 2304
    assert "R_ideal: the bytes model's upper bound (§5) at context 2304 " in _md(run)
    kv = out.gates[Gate.G6A]["kv_capacity"]
    assert kv["required_tokens"] == 128 * (settings.M1_INPUT_LEN + 2304)


def test_a_main_run_whose_rows_disagree_on_the_decode_lengths_is_refused(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    m1[3]["n2"] = 2304
    run = write_run(tmp_path, servers, m1, m2, manifest())
    with pytest.raises(ValueError, match=r"several M1 decode lengths") as exc:
        evaluate(run)
    assert str(run) in str(exc.value) and "(128, 1152)" in str(exc.value)


@pytest.mark.parametrize("field", ["n1", "n2"])
@pytest.mark.parametrize("value", [None, "1152", True, 1152.0])
def test_a_main_run_whose_rows_record_a_decode_length_that_is_no_positive_int_is_refused(
    tmp_path, field, value
):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    m1[3][field] = value
    run = write_run(tmp_path, servers, m1, m2, manifest())
    with pytest.raises(ValueError, match="not a positive integer") as exc:
        evaluate(run)
    assert str(run) in str(exc.value) and f"{field} {value!r}" in str(exc.value)


def test_the_committed_main_runs_are_at_context_1664():
    for run in ("full-1", "expb-1"):
        out = evaluate(RUNS_DIR / run)
        assert isinstance(out, BatchRunResult) and out.context == settings.M1_MEAN_CONTEXT


def test_a_failed_blocking_gate_marks_the_headline_but_the_numbers_stay(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    set_nll(servers, "NV", 0.6)
    run, out = _run(tmp_path, (servers, m1, m2))
    assert out.gates[Gate.G5B]["pass"] is False and out.interpretable is False
    assert "G5b FAIL" in out.gate_line and "G5a pass" in out.gate_line
    head = _md(run).split("\n\n")[1]
    assert "`validated`" in head and "NOT INTERPRETABLE: gate failures" in head
    assert '"failed_treatments"' in _md(run)


@pytest.mark.parametrize(
    "breaker",
    [
        "gpu_start",
        "capture",
        "missing_pair",
        "wrong_kernel",
        "empty_kernel",
        "nvnf_graph",
        "nvnf_fusion",
        "mx_structure",
        "no_reference",
    ],
)
def test_every_blocking_gate_failure_makes_the_run_not_interpretable(tmp_path, breaker):
    treatments = FIVE if breaker.startswith("nvnf") else ("MX", "NV", "NVa", "MXp")
    servers, m1, m2 = synthetic(1.0, (8, 32, 128), treatments=treatments)
    line = manifest(treatments=treatments)
    gate = {
        "gpu_start": "G7",
        "capture": "G6a",
        "missing_pair": "G6a",
        "wrong_kernel": "G1",
        "empty_kernel": "G1",
        "nvnf_graph": "G1",
        "nvnf_fusion": "G1",
        "mx_structure": "G5a",
        "no_reference": "G5b",
    }[breaker]
    if breaker == "gpu_start":
        servers[0]["start_id"] = "start-rerun"
    elif breaker == "capture":
        servers[0]["cudagraph_capture_sizes"] = [8, 32]
    elif breaker == "missing_pair":
        m1 = drop_m1(m1, "r3_MXp-x", 128)
    elif breaker == "wrong_kernel":
        set_fields(servers, "NV", linear_kernels=["FlashInferCutlassNvFp4LinearKernel"])
    elif breaker == "empty_kernel":
        for s in servers:
            s["linear_kernels"] = []
    elif breaker == "nvnf_graph":
        set_fields(servers, "NVnf", compile_cache_hashes=[graph_hash("NV")])
    elif breaker == "nvnf_fusion":
        set_fields(servers, "NVnf", fuse_act_quant=True, custom_fusions=["act_quant"])
    elif breaker == "mx_structure":
        line = manifest(checkpoint={**CLEAN_CKPT, "mx_problems": ["weight_scale is not E8M0"]})
    else:
        line = manifest(reference=None)
    run, out = _run(tmp_path, (servers, m1, m2), line)
    assert out.interpretable is False and f"{gate} FAIL" in out.gate_line
    assert "NOT INTERPRETABLE" in _md(run).split("\n\n")[1]
    if breaker == "empty_kernel":
        assert len(out.gates[Gate.G1]["mismatched_sessions"]) == 20
    if breaker == "nvnf_fusion":
        assert len(out.gates[Gate.G1]["fusion_mismatched_sessions"]) == 5
    if breaker == "missing_pair":
        assert out.gates[Gate.G6A]["missing_m1_rows"] == [[3, "MXp", 128, 0, 1]]
    if breaker == "no_reference":
        assert out.accuracy.bf16 is None


def test_an_invalid_m2_cell_is_reported_but_the_m1_verdict_stands(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    m2[0]["valid"], m2[0]["invalid_reasons"] = False, ["effective_concurrency"]
    run, out = _run(tmp_path, (servers, m1, m2))
    assert out.verdict == "validated" and out.interpretable is True
    assert "G6b FAIL" in out.gate_line and out.nonblocking_failed == ["G6b"]
    md = _md(run)
    note = "G6b failed; reported only, does not block the M1 verdict"
    assert "NOT INTERPRETABLE" not in md
    assert md.index(out.gate_line) < md.index(note) < md.index("### M1: R")
    assert '"invalid_m2_cells": [' in md


def test_m2_aa_is_reported_but_does_not_decide_g3(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    for row in m2:
        if row["treatment"] == "MXp":
            row["aggregate_tps"] *= 0.8
    _, out = _run(tmp_path, (servers, m1, m2))
    assert out.gates[Gate.G3]["pass"] is True and "G3 pass" in out.gate_line
    assert all(v.verdict != Verdict.EQUIVALENT for v in out.m2_throughput[RatioName.AA].values())
    assert out.m2_throughput[RatioName.AA][8].ratio == pytest.approx(1 / 0.8, rel=1e-2)


def test_unusable_cells_are_skipped_and_listed(tmp_path, capsys):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    next(r for r in m2 if (r["round"], r["treatment"], r["c"]) == (2, "NV", 32))["itl_p50_ms"] = 0.0
    for r in m1:
        if r["session_id"] == "r1_NV-x" and r["c"] == 8 and not r["warmup"]:
            r["step_s"] = None
    run, out = _run(tmp_path, (servers, m1, m2))
    assert out.verdict == "incomplete (4/5 rounds)"
    assert out.skipped == {"m1": [(1, "NV", 8)], "m2_itl": [(2, "NV", 32)]}
    assert out.m2_itl[RatioName.R][32].n_rounds == 4 and out.m2_itl[RatioName.R][8].n_rounds == 5
    assert "warning" in capsys.readouterr().err
    md = _md(run)
    assert "Skipped cells" in md and "m2_itl" in md and "round 2, NV, C=32" in md


@pytest.mark.usefixtures("figures")
def test_the_pareto_plot_tolerates_a_valid_row_without_per_user_throughput(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    set_fields(m2, "NV", tps_per_user_p50=None)
    run, _ = _run(tmp_path, (servers, m1, m2))
    assert (run / "pareto_m2.png").stat().st_size > 0


def test_the_full_run_reports_r_d_and_k_and_only_r_decides(tmp_path):
    run, out = _run(tmp_path, synthetic(1.0, (8, 32, 128), nva_factor=1.05))
    assert out.verdict == "validated" and out.gate_line == GATE_LINE_CLEAN
    for c in (8, 32, 128):
        assert out.m1[RatioName.R][c].verdict == Verdict.EQUIVALENT
        assert out.m1[RatioName.D][c].ratio == pytest.approx(1.05, abs=3e-3)
        assert out.m1[RatioName.D][c].verdict == Verdict.MXFP4_FASTER
        assert out.m1[RatioName.K][c].verdict == Verdict.PINNED_FASTER
    for table in (out.m2_throughput, out.m2_itl):
        assert all(set(table[k]) == {8, 32, 128} for k in (RatioName.R, RatioName.D, RatioName.K))
    assert out.m2_throughput[RatioName.D][32].ratio == pytest.approx(1.05, abs=3e-3)
    assert out.m2_throughput[RatioName.K][32].verdict == Verdict.PINNED_FASTER
    assert out.m2_itl[RatioName.K][32].ratio == pytest.approx(1.05, abs=3e-3)
    md = _md(run)
    for heading in (
        "### M1: R = t_NV / t_MX",
        "### M1: D = t_NVa / t_MX (alternative kernel; does not decide H_eq)",
        "### M1: K = t_NVa / t_NV",
        "### A/A control: t_MXp / t_MX",
        "### M2: tok/s MX / NV",
        "### M2: tok/s MX / NVa",
        "### M2: tok/s NV / NVa",
        "### M2: ITL p50 NV / MX",
        "### M2: ITL p50 NVa / MX",
        "### M2: ITL p50 NVa / NV",
    ):
        assert heading in md, heading
    k_section = _section(md, "### M1: K")
    assert "pinned_faster" in k_section and "mxfp4_faster" not in k_section
    assert "mxfp4_faster" in _section(md, "### M1: D")
    assert md.index("### M1: R") < md.index("### M1: D") < md.index("### M1: K")
    assert "R_ideal" in md.split("### M1: D")[0]


def test_m2_ratios_are_oriented_like_m1_so_that_above_one_means_the_second_is_faster(tmp_path):
    rows = synthetic(1.03, (8, 32), nva_factor=1.06, nvnf_factor=1.01, treatments=FIVE)
    _, out = _run(tmp_path, rows, manifest(treatments=FIVE), write=False)
    tput, itl = out.m2_throughput, out.m2_itl
    expected = {
        RatioName.R: 1.03,
        RatioName.D: 1.06,
        RatioName.K: 1.06 / 1.03,
        RatioName.F: 1.01,
        RatioName.PHI: 1.03 / 1.01,
    }
    for name, value in expected.items():
        for table in (out.m1, tput, itl):
            assert table[name][8].ratio == pytest.approx(value, abs=3e-3), name


def test_the_five_treatment_run_reports_f_and_phi(tmp_path):
    rows = synthetic(1.03, (8, 32, 128), treatments=FIVE, nvnf_factor=1.0)
    run, out = _run(tmp_path, rows, manifest(treatments=FIVE))
    assert out.verdict == "nullified_mxfp4_faster" and out.gate_line == GATE_LINE_CLEAN
    for c in (8, 32, 128):
        assert out.m1[RatioName.F][c].verdict == Verdict.EQUIVALENT
        assert out.m1[RatioName.PHI][c].ratio == pytest.approx(1.03, abs=3e-3)
        assert out.m1[RatioName.PHI][c].verdict == Verdict.FUSION_SLOWER
        assert out.m2_throughput[RatioName.F][c].ratio == pytest.approx(1.0, abs=3e-3)
        assert out.m2_throughput[RatioName.PHI][c].verdict == Verdict.FUSION_SLOWER
        assert out.m2_itl[RatioName.PHI][c].ratio == pytest.approx(1.03, abs=3e-3)
        assert out.r_f_phi[c].f_times_phi_over_r == pytest.approx(1.0, abs=1e-12)
    md = _md(run)
    for heading in (
        "### M1: F = t_NVnf / t_MX (fusion matched; does not decide H_eq)",
        "### M1: Φ = t_NV / t_NVnf (fusion effect within NVFP4; does not decide H_eq)",
        "### M1: R = F · Φ check",
        "### M2: tok/s MX / NVnf",
        "### M2: tok/s NVnf / NV",
        "### M2: ITL p50 NVnf / MX",
        "### M2: ITL p50 NV / NVnf",
    ):
        assert heading in md, heading
    assert (
        md.index("### M1: K")
        < md.index("### M1: F")
        < md.index("### M1: Φ")
        < md.index("### M1: R = F · Φ check")
        < md.index("### A/A control")
    )
    phi = _section(md, "### M1: Φ")
    assert "fusion_slower" in phi and "mxfp4_faster" not in phi
    check = _section(md, "### M1: R = F · Φ check")
    assert all(f"C={c}:" in check for c in (8, 32, 128))
    assert f"{out.m1[RatioName.R][8].ratio:.4f}" in check


def test_a_run_without_nva_or_nvnf_has_empty_tables_and_the_same_verdict(tmp_path):
    rows = synthetic(1.0, (8, 32, 128), treatments=("MX", "NV", "MXp"))
    run, out = _run(tmp_path, rows, manifest(treatments=("MX", "NV", "MXp")))
    assert out.verdict == "validated"
    assert (
        out.m1[RatioName.D]
        == out.m1[RatioName.K]
        == out.m1[RatioName.F]
        == out.m1[RatioName.PHI]
        == out.m2_itl[RatioName.PHI]
        == {}
    )
    assert out.r_f_phi == {}
    md = _md(run)
    for title in ("### M1: D", "### M1: F", "### M1: R = F · Φ check"):
        assert "no data" in _section(md, title), title
    _, alt = _run(tmp_path, synthetic(1.0, (8, 32, 128), nva_factor=0.94), name="alt")
    assert {v.verdict for v in alt.m1[RatioName.K].values()} == {Verdict.ALT_FASTER}
    assert {v.verdict for v in alt.m1[RatioName.D].values()} == {Verdict.NVFP4_FASTER}
    assert alt.verdict == "validated"


def test_the_h_eq_verdict_does_not_move_with_d_or_k(tmp_path):
    _, base = _run(tmp_path, synthetic(1.03, (8, 32, 128), nva_factor=1.0), name="a")
    _, other = _run(tmp_path, synthetic(1.03, (8, 32, 128), nva_factor=1.5), name="b")
    assert base.verdict == other.verdict == "nullified_mxfp4_faster"
    assert other.m1[RatioName.D][32].ratio > 1.45 and base.m1[RatioName.D][32].ratio < 1.05


def test_the_table_header_follows_the_configured_ci_level(tmp_path, monkeypatch):
    _, out = _run(tmp_path, write=False)
    assert "| 90% CI |" in _table("t", out.m1[RatioName.R])
    monkeypatch.setattr(settings, "CI_LEVEL", 0.95)
    assert "| 95% CI |" in _table("t", out.m1[RatioName.R])


def test_experiment_b_is_recognised_by_an_fp8_kv_cache_and_names_r_b_and_its_hypotheses(tmp_path):
    run, out = _run(tmp_path, expb(), expb_manifest())
    assert out.kind == "B" and out.verdict == "validated" and out.gate_line == GATE_LINE_CLEAN
    assert out.primary == (256, 512) and set(out.m1[RatioName.AA]) == set(EXPB_CS)
    assert out.gates[Gate.G6A]["kv_capacity"]["required_tokens"] == 1_114_112
    md = _md(run)
    head = md.split("\n\n")[1]
    assert head.startswith(
        "**Experiment B verdict (M1, R_B = t_NV / t_MX, primary C (256, 512)): `validated`"
    )
    assert "H_B,eq" in head and "H_eq" not in head and "Overall verdict on H_eq" not in md
    assert "H_B,nv" in md and "### M1: R_B = t_NV / t_MX" in md
    assert "paired at primary C 256: 5, 512: 5" in md


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_any_fp8_kv_cache_dtype_is_experiment_b_without_a_note(tmp_path, dtype):
    _, out = _run(tmp_path, expb(rounds=2), expb_manifest(kv_cache_dtype=dtype), write=False)
    assert out.kind == "B" and out.kv_cache_dtype_note is None


@pytest.mark.parametrize("dtype", ["int4", "auto", "float16", "FP8"])
def test_an_unexpected_kv_dtype_is_a_main_run_with_a_note_and_no_r_ideal(tmp_path, dtype):
    run, out = _run(tmp_path, expb(), expb_manifest(kv_cache_dtype=dtype))
    assert out.kind == "main" and out.reading is None
    note = out.kv_cache_dtype_note
    assert note is not None and f"`{dtype}`" in note and "main run" in note
    md = _md(run)
    assert md.split("\n\n")[1].startswith("**Overall verdict on H_eq") and note in md
    if dtype == "int4":
        assert "| n/a |" in _section(md, "### M1: R = ")
        bw = out.bandwidth[Treatment.NV][512]
        assert bw.step_s is not None and bw.bytes is None
        assert "| NV | 512 | 5 |" in _section(md, BANDWIDTH_TITLE)


@pytest.mark.parametrize("dtype", ["bfloat16", None])
def test_a_main_run_kv_dtype_has_no_note_and_no_experiment_b_text(tmp_path, dtype):
    assert unexpected_kv_dtype_note(KvDtype.BF16) is None
    line = manifest() if dtype is None else manifest(kv_cache_dtype=dtype)
    run, out = _run(tmp_path, line=line)
    assert out.kind == "main" and out.reading is None and out.kv_cache_dtype_note is None
    md = _md(run)
    assert "Experiment B" not in md and "R_B" not in md and "H_B" not in md
    assert "neither bfloat16" not in md and "--reference-run" not in md


@pytest.mark.parametrize(
    "nv, verdict, reading",
    [
        (1.0, "validated", "H_B,eq holds"),
        (0.95, "nullified_nvfp4_faster", "H_B,nv holds"),
        ({128: 1.0, 256: 1.0, 512: 0.95}, "nullified_nvfp4_faster", "H_B,nv does not hold"),
        (1.05, "nullified_mxfp4_faster", "contradicts H_B,nv"),
    ],
)
def test_the_experiment_b_reading_end_to_end(tmp_path, nv, verdict, reading):
    _, out = _run(tmp_path, expb(nv=nv), expb_manifest(), write=False)
    assert out.verdict == verdict and out.reading is not None and reading in out.reading
    _, short = _run(tmp_path, expb(rounds=3), expb_manifest(), name="short", write=False)
    assert short.verdict == "incomplete (3/5 rounds)" and "no reading" in str(short.reading)


def test_the_experiment_b_summary_states_the_prediction_and_uses_fp8_r_ideal(tmp_path):
    run, out = _run(tmp_path, expb(nv={128: 0.98, 256: 0.98, 512: 0.98}), expb_manifest())
    md = _md(run)
    assert "R_B(512) ≈ 0.973–0.987" in md
    assert (
        f"R̂_B(512) = {out.m1[RatioName.R][512].ratio:.4f}" in md
        and "inside the predicted range" in md
    )
    r_section = _section(md, "### M1: R_B")
    for c in EXPB_CS:
        row = next(line for line in r_section.splitlines() if line.startswith(f"| {c} |"))
        assert row.endswith(f" {bm.r_ideal(c, settings.M1_MEAN_CONTEXT, KvDtype.FP8):.4f} |")
    run, _ = _run(tmp_path, name="main")
    row = next(
        line for line in _section(_md(run), "### M1: R").splitlines() if line.startswith("| 128 |")
    )
    assert row.endswith(f" {bm.r_ideal(128, settings.M1_MEAN_CONTEXT):.4f} |")
    bw = out.bandwidth
    assert set(bw) == {"MX", "NV", "MXp"}
    assert bw[Treatment.NV][512].bytes == pytest.approx(
        bm.step_bytes(512, settings.M1_MEAN_CONTEXT, Format.NVFP4, kv_dtype=KvDtype.FP8)
    )


def test_an_experiment_b_summary_without_r_b_at_512_says_so(tmp_path):
    servers, m1, m2 = expb(rounds=1)
    run, _ = _run(tmp_path, (servers, m1, m2), expb_manifest(rounds=1))
    assert "Observed: no R̂_B(512) (it needs at least 2 paired rounds)." in _md(run)


def test_the_bandwidth_table_in_the_summary(tmp_path):
    run, out = _run(
        tmp_path, synthetic(1.0, (8, 32, 128), treatments=FIVE), manifest(treatments=FIVE)
    )
    v = out.bandwidth[Treatment.NV][128]
    section = _section(_md(run), BANDWIDTH_TITLE)
    assert v.step_s is not None and v.bytes is not None and v.bytes_per_s is not None
    assert v.fraction_of_peak is not None
    assert (
        f"| NV | 128 | 5 | {v.step_s * 1000:.4f} | {v.bytes / 1e9:.2f} | "
        f"{v.bytes_per_s / 1e12:.3f} | {v.fraction_of_peak:.1%} |"
    ) in section
    assert str(settings.M1_MEAN_CONTEXT) in section and "8 TB/s" in section


def test_the_summary_has_an_accuracy_note_with_the_interval(tmp_path):
    run, out = _run(tmp_path)
    md = _md(run)
    d = out.accuracy.nv_minus_mx
    assert d is not None and "## Accuracy note" in md
    assert f"{d.mean:+.4f}" in md and f"[{d.lo:+.4f}, {d.hi:+.4f}]" in md
    assert "95% paired bootstrap" in md and "10,000 resamples" in md and "seed 0" in md


def test_the_reference_run_goes_into_the_accuracy_note(tmp_path):
    run, ref = main_and_expb(tmp_path)
    out = analyze_run(run, ref)
    assert isinstance(out, BatchRunResult)
    cmp = out.accuracy.reference_run
    assert cmp is not None and cmp.run == str(ref) and cmp.problems == []
    accuracy = _md(run).split("## Accuracy note")[1].split("\n## ")[0]
    assert "fp8 KV" in accuracy and "bfloat16 KV" in accuracy and str(ref) in accuracy
    v = cmp.per_treatment[Treatment.NV]
    assert v.this is not None and v.reference is not None and v.this_minus_reference is not None
    assert f"| NV | {v.this:.4f} | {v.reference:.4f} | {v.this_minus_reference:+.4f} |" in accuracy
    assert "needs --reference-run" not in _md(run)


def test_an_experiment_b_summary_without_a_reference_run_says_it_needs_one(tmp_path):
    run, _ = main_and_expb(tmp_path)
    out = analyze_run(run)
    assert isinstance(out, BatchRunResult) and out.accuracy.reference_run is None
    accuracy = _md(run).split("## Accuracy note")[1].split("\n## ")[0]
    assert REFERENCE_RUN_NEEDED in accuracy and "needs --reference-run" in accuracy


def test_the_reference_nll_table_shows_a_missing_reference_treatment(tmp_path):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128), treatments=("MX", "NV", "NVa"))
    ref = write_run(tmp_path, servers, m1, m2, manifest(), name="main")
    run = write_run(tmp_path, *expb(), expb_manifest(), name="expb")
    analyze_run(run, ref)
    assert "| MXp |" in _md(run) and "n/a" in _md(run)


def _with_fallback(tmp_path, lines, fallback=None):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    run = write_run(tmp_path, servers, m1, m2, lines)
    if fallback is not None:
        (tmp_path / "checkpoint_report.json").write_text(
            fallback if isinstance(fallback, str) else json.dumps(fallback)
        )
    out = evaluate(run)
    assert isinstance(out, BatchRunResult)
    return out


def test_the_checkpoint_report_comes_from_the_last_manifest_line(tmp_path):
    bad = {**CLEAN_CKPT, "nv_problems": ["no packed FP4 (U8) expert weights"]}
    (tmp_path / "a").mkdir()
    out = _with_fallback(tmp_path / "a", [manifest(checkpoint=bad), manifest()])
    assert out.gates[Gate.G5A]["pass"] is True and out.gates[Gate.G2]["pass"] is True
    assert out.gates[Gate.G5A]["checkpoint_source"].startswith("manifest")
    (tmp_path / "b").mkdir()
    out = _with_fallback(tmp_path / "b", [manifest(), manifest(checkpoint=bad)])
    assert out.gates[Gate.G5A]["problems"]["nv"] == bad["nv_problems"]
    assert out.interpretable is False


@pytest.mark.parametrize(
    "report, fallback, ok",
    [
        ("clean", {**CLEAN_CKPT, "nv_problems": ["stale file"]}, True),
        ({**CLEAN_CKPT, "nv_problems": ["unquantized experts"]}, CLEAN_CKPT, False),
        (None, CLEAN_CKPT, True),
        (None, {**CLEAN_CKPT, "nv_problems": ["x"]}, False),
        (None, "{not json", False),
        (None, "[1, 2]", False),
        (None, "null", False),
        (None, None, False),
        ("error: ValueError: NV has no config.json", CLEAN_CKPT, False),
    ],
)
def test_the_results_directory_file_is_only_a_labelled_fallback(tmp_path, report, fallback, ok):
    out = _with_fallback(tmp_path, manifest(checkpoint=report), fallback)
    assert out.gates[Gate.G5A]["pass"] is ok
    source = out.gates[Gate.G5A]["checkpoint_source"]
    if report is None and fallback is not None:
        assert source.startswith("fallback:") and "checkpoint_report.json" in source
        assert "not run-scoped" in source
    if isinstance(report, str) and report.startswith("error"):
        assert "error: ValueError" in source and out.gates[Gate.G2]["pass"] is False
    if report is None and fallback is None:
        assert out.gates[Gate.G2]["pass"] is False and "n/a" not in out.gate_line


def test_the_bf16_reference_comes_from_the_last_manifest_line(tmp_path):
    stale = manifest(reference={"per_prompt": [], "mean": 0.5})
    (tmp_path / "a").mkdir()
    assert _with_fallback(tmp_path / "a", [stale, manifest()]).gates[Gate.G5B]["pass"] is True
    (tmp_path / "b").mkdir()
    assert _with_fallback(tmp_path / "b", [manifest(), stale]).gates[Gate.G5B]["pass"] is False


def _smoke_run(tmp_path, rows=None, name="run", **kw) -> tuple[Any, BatchRunResult]:
    return _run(tmp_path, rows or smoke(**kw), smoke_manifest(require_published=False), name=name)


@pytest.mark.usefixtures("figures")
def test_the_smoke_reports_the_scan_the_selection_and_the_crosscheck(tmp_path):
    run, out = _smoke_run(tmp_path)
    assert (
        out.verdict.startswith("incomplete") and out.m1[RatioName.R] == out.m1[RatioName.AA] == {}
    )
    assert out.scan is not None and out.scan.selected == "NVt"
    assert out.crosscheck is not None and out.crosscheck[32].ratio == pytest.approx(1.01)
    md = _md(run)
    assert "## Smoke: kernel scan" in md and "Selected: **NVt**" in md
    assert "FlashInferTrtllmNvFp4LinearKernel" in md
    assert "fastest non-CuTe-DSL NVFP4 kernel by median M1 step time at C=32" in md
    assert '("--linear-backend", "flashinfer_trtllm")' in md
    assert all(f"| {t} " in md for t in SMOKE_SCAN)
    assert "0.9700" in md and "0.8000" in md
    assert "## Smoke: cross-check, ours vs NVIDIA NVFP4" in md and "NVx / NV" in md
    assert "`incomplete" in md.split("\n\n")[1]
    assert (run / "pareto_m2.png").stat().st_size > 0 and not (run / "ratio_vs_c.png").exists()
    g1 = out.gates[Gate.G1]
    assert g1["pass"] is True and set(g1["expected"]) == set(SMOKE.treatments)
    assert out.gates[Gate.G5B]["pass"] and out.gates[Gate.G6A]["pass"]
    assert set(out.bandwidth) == {"MX", "NV"}


def test_g1_and_the_scan_expect_the_kernel_of_the_recorded_server_args(tmp_path):
    servers, m1, m2 = smoke()
    set_fields(servers, "NVc", linear_kernels=["CutlassNvFp4LinearKernel"])
    line = recording_args(
        smoke_manifest(require_published=False),
        Treatment.NVC,
        model.pinned_to(LinearBackend.CUTLASS),
    )
    run, out = _run(tmp_path, (servers, m1, m2), line)
    assert out.gates[Gate.G1]["expected"]["NVc"] == "CutlassNvFp4LinearKernel"
    assert out.gates[Gate.G1]["pass"] is True
    assert out.scan is not None and out.scan.kernel_matches[Treatment.NVC] is True
    assert "| NVc | cutlass | CutlassNvFp4LinearKernel | CutlassNvFp4LinearKernel |" in _md(run)


def test_the_selected_kernels_args_are_those_the_run_recorded(tmp_path):
    line = recording_args(
        smoke_manifest(require_published=False),
        Treatment.NVT,
        ("--linear-backend=flashinfer_trtllm",),
    )
    run, out = _run(tmp_path, smoke(), line)
    assert out.scan is not None and out.scan.selected == "NVt"
    assert out.scan.selected_server_args == ("--linear-backend=flashinfer_trtllm",)
    md = _md(run)
    assert '`("--linear-backend=flashinfer_trtllm")` its `server_args`' in md
    assert "| NVt | flashinfer_trtllm | FlashInferTrtllmNvFp4LinearKernel |" in md


def test_the_summary_prints_the_margin_and_the_eligibility_table(tmp_path):
    run, _ = _smoke_run(tmp_path)
    md = _md(run)
    table = md.split("### Eligibility for NVa")[1].split("\n\n")
    header = next(block for block in table if block.lstrip().startswith("| treatment"))
    for column in (
        "kernel as expected",
        "not failed",
        "M1 step at C=32",
        "mean NLL",
        "vs NV",
        "G5b",
        "eligible",
    ):
        assert column in header
    assert all(f"| {t} |" in "\n\n".join(table) for t in scan_rule.NVA_SCAN.treatments)
    assert "0.0100" in md and "Margin at C=32: NVt" in md and "runner-up NVd" in md
    assert "8.25%" in md and "not a tie" in md


def test_the_summary_says_why_a_kernel_is_not_eligible(tmp_path):
    servers, m1, m2 = smoke()
    set_nll(servers, "NVd", 0.06)
    run, _ = _smoke_run(tmp_path, (fail(servers, "NVc", error="RuntimeError('boom')"), m1, m2))
    eligibility = _md(run).split("### Eligibility for NVa")[1]
    line = {
        str(t): next(row for row in eligibility.splitlines() if row.startswith(f"| {t} |"))
        for t in scan_rule.NVA_SCAN.treatments
    }
    assert "failed" in line["NVc"] and line["NVc"].rstrip().endswith("|")
    assert "NLL" in line["NVd"] and "NO" in line["NVd"] and "yes" in line["NVt"]


@pytest.mark.parametrize(
    "factors, expected",
    [
        (
            {"NVc": {1: 1.0, 32: 1.000, 128: 1.00}, "NVd": {1: 1.0, 32: 1.0099, 128: 0.90}},
            ("within the 1.0% tie margin", "Tie-break at C=128", "Selected: **NVd**"),
        ),
        (
            {"NVd": {1: 1.0, 32: 1.000, 128: 1.000}, "NVt": {1: 1.0, 32: 1.005, 128: 1.004}},
            ("also within", "NVc, NVt, NVd, NVv", "Selected: **NVt**"),
        ),
    ],
)
def test_the_summary_describes_a_tie_and_how_it_was_broken(tmp_path, factors, expected):
    slow = {1: 1.3, 32: 1.3, 128: 1.3}
    rows = smoke({**SCAN_FACTORS, **dict.fromkeys(scan_rule.NVA_SCAN.treatments, slow), **factors})
    run, _ = _smoke_run(tmp_path, rows)
    md = _md(run)
    assert all(text in md for text in expected)


def test_the_summary_names_a_single_candidate_and_a_tie_without_a_step_at_the_tiebreak_c(tmp_path):
    servers, m1, m2 = smoke()
    for t in ("NVc", "NVd", "NVv"):
        set_nll(servers, t, 0.1)
    run, _ = _smoke_run(tmp_path, (servers, m1, m2), name="single")
    assert "Only one eligible candidate, so there is no margin: NVt is selected." in _md(run)
    slow = {1: 1.3, 32: 1.3, 128: 1.3}
    servers, m1, m2 = smoke(
        {
            **SCAN_FACTORS,
            **dict.fromkeys(scan_rule.NVA_SCAN.treatments, slow),
            "NVd": {1: 1.0, 32: 1.000, 128: 1.0},
            "NVt": {1: 1.0, 32: 1.005, 128: 0.5},
        }
    )
    m1 = [r for r in m1 if not (r["treatment"] == "NVd" and r["c"] == 128)]
    run, _ = _smoke_run(tmp_path, (servers, m1, m2), name="tie")
    assert (
        "Tie-break at C=128: no M1 step for NVd, so the earlier of NVc, NVt, NVd, NVv is "
        "selected: NVt."
    ) in _md(run)


def test_a_smoke_with_a_scan_kernel_on_the_wrong_kernel_fails_g1_and_says_so(tmp_path):
    servers, m1, m2 = smoke()
    set_fields(servers, "NVc", linear_kernels=[CUTE_DSL])
    run, out = _smoke_run(tmp_path, (servers, m1, m2))
    assert out.gates[Gate.G1]["pass"] is False and "G1 FAIL" in out.gate_line
    assert out.scan is not None and out.scan.kernel_matches[Treatment.NVC] is False
    assert "did not run" in _md(run)


def test_a_smoke_without_a_candidate_says_why(tmp_path):
    servers, m1, m2 = smoke()
    m1 = [r for r in m1 if not (r["c"] == 32 and r["treatment"] in scan_rule.NVA_SCAN.treatments)]
    run, out = _smoke_run(tmp_path, (servers, m1, m2))
    assert out.scan is not None and out.scan.selected is None
    assert "**No kernel selected:**" in _md(run) and "C=32" in _md(run)


def test_all_scan_kernels_failed_gives_no_selection_and_the_scan_still_reports(tmp_path):
    servers, m1, m2 = smoke()
    for t in scan_rule.NVA_SCAN.treatments:
        servers = fail(servers, t)
    run, out = _smoke_run(tmp_path, (servers, m1, m2))
    assert out.scan is not None and out.scan.selected is None
    assert [s.treatment for s in out.failed] == sorted(scan_rule.NVA_SCAN.treatments)
    assert "**No kernel selected:**" in _md(run)


def test_the_summary_lists_a_failed_session_and_analyses_the_rest(tmp_path):
    servers, m1, m2 = smoke()
    run, out = _smoke_run(tmp_path, (fail(servers, "NVc"), m1, m2))
    assert [(s.round, s.treatment, s.session_id) for s in out.failed] == [(0, "NVc", "r0_NVc-x")]
    assert all(g["pass"] for name, g in out.gates.items() if name != Gate.G3)
    md = _md(run)
    assert "NVc" in md.split("## Failed sessions")[1].split("##")[0]
    assert "r0_NVc-x" in md and "vllm serve exited with 1" in md
    assert "excluded from every computation" in md
    assert out.scan is not None and out.scan.selected == "NVt"
    assert out.payload()["failed_sessions"][0]["error"].startswith("RuntimeError")
    run, out = _smoke_run(tmp_path, name="clean")
    assert out.failed == [] and "Failed sessions" not in _md(run)


def test_the_payload_lists_a_failed_session_as_its_servers_line_holds_it(tmp_path):
    servers, m1, m2 = smoke()
    lines = {t: failed_row(next(s for s in servers if s["treatment"] == t)) for t in ("NVc", "NVt")}
    lines["NVt"] = {"schema_version": 2, **lines["NVt"]}
    servers = [lines.get(s["treatment"], s) for s in servers]
    _, out = _smoke_run(tmp_path, (servers, m1, m2))
    assert out.payload()["failed_sessions"] == [lines["NVc"], lines["NVt"]]


def test_a_session_the_runner_records_as_failed_is_left_out_and_listed(tmp_path):
    servers, m1, m2 = smoke()
    run = write_run(tmp_path, servers, m1, m2, smoke_manifest(require_published=False))
    exc = RuntimeError("vllm serve exited with 1")
    setattr(exc, runner.SESSION_ID_ATTR, "r0_NVc-x")
    runner._record_failed_session(run, 0, Treatment.NVC, "start-0", exc)
    line = json.loads((run / "servers.jsonl").read_text().splitlines()[-1])
    assert line == {
        "schema_version": 2,
        "round": 0,
        "treatment": "NVc",
        "session_id": "r0_NVc-x",
        "start_id": "start-0",
        "failed": True,
        "error": repr(exc),
    }
    assert load_rows(run / "servers.jsonl", ServerRow)[-1].failed is True
    out = analyze_run(run)
    assert isinstance(out, BatchRunResult)
    assert [(s.round, s.treatment, s.session_id) for s in out.failed] == [(0, "NVc", "r0_NVc-x")]
    assert out.payload()["failed_sessions"] == [line]
    assert out.scan is not None and out.scan.selected == "NVt"
    assert out.scan.eligibility[Treatment.NVC].reasons == [f"its session failed: {exc!r}"]
    assert out.scan.median_step_s[Treatment.NVC] == {} and Treatment.NVC not in out.order
    failed_table = _md(run).split("## Failed sessions")[1].split("##")[0]
    assert f"| 0 | NVc | r0_NVc-x | {exc!r} |" in failed_table


def test_a_full_run_has_no_smoke_sections(tmp_path):
    run, out = _run(tmp_path)
    assert out.scan is None and out.crosscheck is None and "Smoke" not in _md(run)


def _ratios(tmp_path):
    rows = synthetic(1.0, (8, 32, 128), nva_factor=1.04, nvnf_factor=1.01, treatments=FIVE)
    _, out = _run(tmp_path, rows, manifest(treatments=FIVE), write=False)
    return out.m1


def test_the_ratio_figure_has_a_panel_each_for_r_d_k_f_and_phi(tmp_path):
    import matplotlib.pyplot as plt

    ratios = _ratios(tmp_path)
    fig = plots.ratio_figure(ratios, FULL.ratios)
    try:
        assert [ax.get_title() for ax in fig.axes] == [
            "R = t_NV / t_MX (M1)",
            "D = t_NVa / t_MX (M1)",
            "K = t_NVa / t_NV (M1)",
            "F = t_NVnf / t_MX (M1)",
            "Φ = t_NV / t_NVnf (M1)",
        ]
        for ax in fig.axes:
            assert len([p for p in ax.patches if p.get_label().startswith("±δ")]) == 1
            assert not ax.texts
        labels = [[line.get_label() for line in ax.lines] for ax in fig.axes]
        assert any("R_ideal" in label for label in labels[0])
        assert not any("R_ideal" in label for panel in labels[1:] for label in panel)
        assert any("A/A" in str(c.get_label()) for c in fig.axes[0].containers)
        assert [ax.get_ylabel() for ax in fig.axes] == [
            "ratio (>1 = MXFP4 faster)",
            "ratio (>1 = MXFP4 faster)",
            "ratio (>1 = pinned kernel faster)",
            "ratio (>1 = MXFP4 faster)",
            "ratio (>1 = fusion slower)",
        ]
    finally:
        plt.close(fig)
    fig = plots.ratio_figure(
        {
            RatioName.R: ratios[RatioName.R],
            RatioName.D: {},
            RatioName.K: {},
            RatioName.AA: ratios[RatioName.AA],
        },
        FULL.ratios,
    )
    try:
        assert all([t.get_text() for t in fig.axes[i].texts] == ["no data"] for i in (1, 2, 3, 4))
        assert len(fig.axes[0].texts) == 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("dtype", [KvDtype.BF16, KvDtype.FP8, UnknownKvDtype("int4")])
def test_the_ratio_figure_draws_r_ideal_with_the_runs_kv_dtype(tmp_path, dtype):
    import matplotlib.pyplot as plt

    fig = plots.ratio_figure(_ratios(tmp_path), FULL.ratios, kv_dtype=dtype)
    try:
        lines = [ln for ln in fig.axes[0].lines if "R_ideal" in ln.get_label()]
        if isinstance(dtype, UnknownKvDtype):
            assert not lines
        else:
            assert lines[0].get_label() == f"R_ideal (bytes model, {dtype.value} KV)"
            assert list(lines[0].get_ydata()) == pytest.approx(
                [bm.r_ideal(c, settings.M1_MEAN_CONTEXT, dtype) for c in (8, 32, 128)]
            )
    finally:
        plt.close(fig)


def _pareto_labels(tmp_path, rows, line, name):
    import matplotlib.pyplot as plt

    _, out = _run(tmp_path, rows, line, name=name, write=False)
    fig = plots.pareto_figure(out.tput, out.per_user, out.study.ratio(RatioName.AA).numer)
    try:
        return [t.get_text().split(":")[0] for t in fig.axes[0].get_legend().get_texts()]
    finally:
        plt.close(fig)


def test_the_pareto_figure_shows_the_configurations_but_not_the_replica(tmp_path):
    assert {"NVnf", "MXp", "NVx"} <= set(plots.PARETO_ORDER)
    assert _pareto_labels(tmp_path, synthetic(1.0, (8, 32, 128)), manifest(), "a") == [
        "MX",
        "NV",
        "NVa",
    ]
    assert _pareto_labels(
        tmp_path, synthetic(1.0, (8, 32, 128), treatments=FIVE), manifest(treatments=FIVE), "b"
    ) == ["MX", "NV", "NVa", "NVnf"]
    assert sorted(_pareto_labels(tmp_path, smoke(), smoke_manifest(), "c")) == sorted(
        SMOKE.treatments
    )


def test_the_payload_has_the_summary_shape(tmp_path):
    _, out = _run(
        tmp_path, synthetic(1.0, (8, 32, 128), treatments=FIVE), manifest(treatments=FIVE)
    )
    payload = out.payload()
    assert payload["experiment"] == "main" and payload["primary_concurrencies"] == (8, 32, 128)
    assert payload["m1"][8] == {
        "n_rounds": 5,
        "ratio": out.m1[RatioName.R][8].ratio,
        "lo": out.m1[RatioName.R][8].lo,
        "hi": out.m1[RatioName.R][8].hi,
        "verdict": "equivalent",
        "skipped": [],
    }
    assert payload["accuracy"]["nv_minus_mx"]["seed"] == 0
    assert payload["effective_bandwidth"]["NV"][8]["format"] == "nvfp4"
    assert payload["order_effect"]["MX"][8]["count_n1"] == 5
    assert payload["gates"] is out.gates
    json.dumps(payload, default=str)
    assert statistics.fmean(pp("MX")) == pytest.approx(payload["accuracy"]["mx"])
    assert SMOKE_CS == (1, 32, 128)


def test_every_run_result_has_its_payload():
    assert inspect.isabstract(RunResult) and RunResult.__abstractmethods__ == {"payload"}
    assert not inspect.isabstract(BatchRunResult) and not inspect.isabstract(CellRunResult)
