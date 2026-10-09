"""The analysis core, on the typed rows of the committed runs (data/runs), gives the numbers the
write-up quotes. test_golden pins every other number of these runs."""

import json

from fp4bench.analysis import stats
from fp4bench.analysis.cells import by_batch, m1_steps, row_cell, session_ids
from fp4bench.analysis.compare import contrast, ratio, ratio_table
from fp4bench.analysis.inputs import load_run
from fp4bench.core.types import Cell, Treatment
from tests import GOLDEN_DIR, RUNS_DIR

MX, NV, MXP = Treatment.MX, Treatment.NV, Treatment.MXP
E_CELLS = {"E_tok": (Cell(1, 127360), Cell(1, 1024)), "E_batch": (Cell(128, 360), Cell(1, 127360))}


def _batch_steps(run: str):
    data = load_run(RUNS_DIR / run)
    sids = session_ids(data.servers)
    batches = tuple(
        sorted(
            {
                cell.batch
                for r in data.m1
                if r.session_id in sids and (cell := row_cell(r)) is not None
            }
        )
    )
    return by_batch(m1_steps(data.m1, sids)), batches


def _cell_steps(run: str):
    data = load_run(RUNS_DIR / run)
    return m1_steps(data.m1, session_ids(data.servers))


def test_r_per_batch_of_the_main_run():
    step, batches = _batch_steps("full-1")
    r = ratio_table(step, NV, MX, batches)
    assert batches == (1, 8, 32, 64, 128) and all(v.n_rounds == 10 for v in r.values())
    assert [round(r[c].ratio, 4) for c in batches] == [0.9339, 0.9297, 0.9596, 0.9869, 1.0118]
    verdicts = {c: v for c in (8, 32, 128) if (v := r[c].verdict) is not None}
    golden = json.loads((GOLDEN_DIR / "full-1.json").read_text())
    assert stats.overall_verdict(verdicts, (8, 32, 128)) == golden["verdict"]


def test_the_published_experiment_c_numbers():
    steps = _cell_steps("expc-1")
    rounded = {
        name: tuple(round(x or 0.0, 3) for x in (e.mean_ms, e.lo_ms, e.hi_ms))
        for name, cells in E_CELLS.items()
        if (e := contrast(steps, cells, MX, NV))
    }
    assert rounded == {"E_tok": (-0.031, -0.04, -0.023), "E_batch": (-0.556, -0.64, -0.472)}
    aa = [contrast(steps, cells, MXP, MX).mean_ms for cells in E_CELLS.values()]
    assert [round(x or 0.0, 3) for x in aa] == [-0.003, 0.02]
    r = ratio(steps, NV, MX, Cell(1, 1024))
    assert r is not None and round(r.ratio, 4) == 0.9354
