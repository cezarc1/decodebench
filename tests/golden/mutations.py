"""Mutated copies of the committed runs. mutations.json holds what the tag wrote for each recipe;
`refuses`: refused now, `changed`: different on purpose."""

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.analysis_runs import write_jsonl
from tests.golden.outputs import CHECKPOINT_REPORT, copy_runs, head_outputs, normalise

FILES = ("servers", "m1", "m2", "manifests")


class _Rows(dict[str, list[dict]]):
    """The rows of each of FILES by name, each file read when first looked up."""

    def __init__(self, read: Callable[[str], list[dict]]):
        super().__init__()
        self._read = read

    def __missing__(self, name: str) -> list[dict]:
        if name not in FILES:
            raise KeyError(name)
        rows = self[name] = self._read(name)
        return rows


class Run:
    """A copied run's JSONL files as lists of dicts, each read when the recipe first touches it;
    `save` writes back the files that changed."""

    def __init__(self, run_dir: Path, opened: list["Run"]):
        self.dir = run_dir
        self.root = run_dir.parent
        self._opened = opened
        opened.append(self)
        self.rows: dict[str, list[dict]] = _Rows(self._read)
        self._saved: dict[str, str] = {}

    def _read(self, name: str) -> list[dict]:
        """The rows of the copy's file, as `save` compares them."""
        path = self.path(name)
        text = path.read_text() if path.exists() else ""
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
        self._saved[name] = json.dumps(rows)
        return rows

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.jsonl"

    def other(self, run: str) -> "Run":
        """Another run copied beside this one (a reference run)."""
        return Run(self.root / run, self._opened)

    @property
    def servers(self) -> list[dict]:
        return self.rows["servers"]

    @property
    def m1(self) -> list[dict]:
        return self.rows["m1"]

    @property
    def m2(self) -> list[dict]:
        return self.rows["m2"]

    @property
    def manifest(self) -> dict:
        return self.rows["manifests"][-1]

    @property
    def protocol(self) -> dict:
        return self.manifest["protocol"]

    def session(self, rnd: int, treatment: str) -> dict:
        """The last servers row of (round, treatment)."""
        return [s for s in self.servers if (s["round"], s["treatment"]) == (rnd, treatment)][-1]

    def keep(
        self, keep: Callable[[dict], bool], files: tuple[str, ...] = ("servers", "m1", "m2")
    ) -> None:
        for name in files:
            self.rows[name] = [r for r in self.rows[name] if keep(r)]

    def m1_where(self, **match: Any) -> list[dict]:
        return [r for r in self.m1 if all(r.get(k) == v for k, v in match.items())]

    def m2_where(self, **match: Any) -> list[dict]:
        return [r for r in self.m2 if all(r.get(k) == v for k, v in match.items())]

    def remove(self, name: str) -> None:
        self.path(name).unlink()
        self.rows[name] = []
        self._saved[name] = json.dumps([])

    def save(self) -> None:
        for name, rows in self.rows.items():
            if name not in self._saved:  # replaced without being read
                self._read(name)
            if json.dumps(rows) != self._saved[name]:
                write_jsonl(self.path(name), rows)
                self._saved[name] = json.dumps(rows)


@dataclass(frozen=True)
class Recipe:
    name: str
    run: str
    mutate: Callable[[Run], None]
    others: tuple[str, ...] = ()
    reference: str | None = None
    refuses: str | None = None
    changed: str | None = None
    shows: str | None = None


RECIPES: dict[str, Recipe] = {}


def recipe(
    run: str,
    others: tuple[str, ...] = (),
    reference: str | None = None,
    refuses: str | None = None,
    changed: str | None = None,
    shows: str | None = None,
) -> Callable[[Callable[[Run], None]], Callable[[Run], None]]:
    """Register a mutation of `run`, with `others` beside it and its `reference` run."""

    def register(fn: Callable[[Run], None]) -> Callable[[Run], None]:
        name = fn.__name__
        if name in RECIPES:
            raise ValueError(f"two recipes named {name}")
        RECIPES[name] = Recipe(name, run, fn, others, reference, refuses, changed, shows)
        return fn

    return register


def build(rec: Recipe, root: Path) -> tuple[Path, Path | None]:
    """Copy and mutate the recipe's runs in `root`; (run directory, reference directory)."""
    copy_runs(root, (rec.run, *(r for r in rec.others if r != rec.run)))
    opened: list[Run] = []
    rec.mutate(Run(root / rec.run, opened))
    for run in opened:
        run.save()
    return root / rec.run, None if rec.reference is None else root / rec.reference


def head_result(name: str) -> dict[str, Any]:
    """What the current analysis writes for the recipe, and summary.md if it `shows` something."""
    rec = RECIPES[name]
    with tempfile.TemporaryDirectory(prefix="mutation-") as tmp:
        root = Path(tmp)
        run_dir, reference = build(rec, root)
        out = head_outputs(root, run_dir, reference)
        summary = run_dir / "summary.md"
        if rec.shows is not None and "files" in out and summary.exists():
            out["summary"] = normalise(summary.read_text(), root)
        return out


def scale_m1(run: Run, factor: float, **match: Any) -> None:
    for r in run.m1_where(**match):
        r["step_s"] = r["step_s"] * factor


def set_steps(run: Run, treatment: str, c: int, values: list[float]) -> None:
    for k, r in enumerate(run.m1_where(treatment=treatment, c=c)):
        r["step_s"] = values[k % len(values)]


def steps(run: Run, treatment: str, c: int) -> list[float]:
    return [r["step_s"] for r in run.m1_where(treatment=treatment, c=c)]


def _without(run: Run, rnd: int, treatment: str) -> None:
    run.keep(lambda r: (r["round"], r["treatment"]) != (rnd, treatment))


def _rerun(run: Run, rnd: int, treatment: str, session_id: str) -> tuple[dict, dict]:
    old = run.session(rnd, treatment)
    new = dict(old, session_id=session_id, nll=old["nll"] + 0.001)
    run.m1.extend(
        dict(r, session_id=session_id, step_s=r["step_s"] * 1.1)
        for r in run.m1_where(session_id=old["session_id"])
    )
    run.m2.extend(
        dict(r, session_id=session_id, aggregate_tps=r["aggregate_tps"] * 0.9)
        for r in run.m2_where(session_id=old["session_id"])
    )
    return old, new


@recipe("full-1")
def rerun_appended(run: Run) -> None:
    """A re-run of (3, NV) after the original: the later servers line counts."""
    _, new = _rerun(run, 3, "NV", "r3_NV-rerun01")
    run.servers.append(new)


@recipe("full-1")
def rerun_listed_first(run: Run) -> None:
    """The re-run's servers line before the original's: the line order decides, not the id."""
    old, new = _rerun(run, 3, "NV", "r3_NV-zzzz9999")
    run.servers.insert(run.servers.index(old), new)


@recipe("full-1")
def failed_rerun_appended(run: Run) -> None:
    run.servers.append(
        {
            "round": 3,
            "treatment": "NV",
            "session_id": "r3_NV-failed1",
            "failed": True,
            "error": "server died: CUDA OOM",
        }
    )


@recipe("full-1")
def partial_round_without_nva(run: Run) -> None:
    _without(run, 5, "NVa")


@recipe("full-1")
def partial_round_without_mx(run: Run) -> None:
    _without(run, 5, "MX")


@recipe("expc-1")
def expc_partial_round_without_mxp(run: Run) -> None:
    _without(run, 2, "MXp")


@recipe("expc-1")
def expc_partial_round_without_nv(run: Run) -> None:
    _without(run, 2, "NV")


@recipe("expb-1", others=("full-1",), reference="full-1")
def expb_partial_round_without_nv(run: Run) -> None:
    _without(run, 4, "NV")


@recipe("full-1")
def main_session_failed(run: Run) -> None:
    """(4, MX) failed and its rows gone."""
    s = run.session(4, "MX")
    run.servers[run.servers.index(s)] = {
        "round": 4,
        "treatment": "MX",
        "session_id": s["session_id"],
        "failed": True,
        "error": "boom",
    }
    run.keep(lambda r: (r["round"], r["treatment"]) != (4, "MX"), ("m1", "m2"))


@recipe("smoke-r2-2")
def scan_session_failed(run: Run) -> None:
    """NVd failed with a long error; its rows gone."""
    s = run.session(0, "NVd")
    run.servers[run.servers.index(s)] = {
        "round": 0,
        "treatment": "NVd",
        "session_id": s["session_id"],
        "failed": True,
        "error": "x" * 400,
    }
    run.keep(lambda r: r["treatment"] != "NVd", ("m1", "m2"))


@recipe("smoke-r2-2")
def scan_session_failed_rows_kept(run: Run) -> None:
    s = run.session(0, "NVd")
    s["failed"], s["error"] = True, "died"


@recipe("smoke-r2-2")
def failed_flag_not_a_bool(run: Run) -> None:
    """failed: "true" and 1 are not `true`: the sessions count."""
    run.session(0, "NVd")["failed"] = "true"
    run.session(0, "NVc")["failed"] = 1


@recipe("smoke-r2-2")
def scan_all_failed(run: Run) -> None:
    scan = {"NVc", "NVt", "NVd", "NVv"}
    run.rows["servers"] = [
        {
            "round": 0,
            "treatment": s["treatment"],
            "session_id": s["session_id"],
            "failed": True,
            "error": None,
        }
        if s["treatment"] in scan
        else s
        for s in run.servers
    ]
    run.keep(lambda r: r["treatment"] not in scan, ("m1", "m2"))


@recipe("smoke-r2-2")
def scan_failed_rerun(run: Run) -> None:
    run.servers.append(
        dict(run.session(0, "NVd"), session_id="r0_NVd-re", failed=True, error="rerun failed")
    )


@recipe("smoke-r2-2")
def scan_two_failed_sessions(run: Run) -> None:
    run.servers.append({"round": 0, "treatment": "NVv", "session_id": "x", "failed": True})
    run.servers.append(
        {"round": 0, "treatment": "NVv", "session_id": "y", "failed": True, "error": "second"}
    )


@recipe("smoke-r2-2")
def scan_tie_at_32(run: Run) -> None:
    """NVc ties NVd at C=32; C=128 breaks the tie."""
    set_steps(run, "NVc", 32, steps(run, "NVd", 32))


@recipe("smoke-r2-2")
def scan_tie_at_32_and_128(run: Run) -> None:
    set_steps(run, "NVc", 32, steps(run, "NVd", 32))
    set_steps(run, "NVc", 128, steps(run, "NVd", 128))


@recipe("smoke-r2-2")
def scan_tie_at_32_nvc_faster_at_128(run: Run) -> None:
    set_steps(run, "NVc", 32, steps(run, "NVd", 32))
    set_steps(run, "NVc", 128, [x * 0.95 for x in steps(run, "NVd", 128)])


@recipe("smoke-r2-2")
def scan_tie_without_a_tiebreak_step(run: Run) -> None:
    set_steps(run, "NVc", 32, steps(run, "NVd", 32))
    run.keep(lambda r: not (r["treatment"] == "NVc" and r["c"] == 128), ("m1",))


@recipe("smoke-r2-2")
def scan_tie_nvd_nvv(run: Run) -> None:
    """An exact tie at both C: the earlier in the scan order wins."""
    set_steps(run, "NVv", 32, steps(run, "NVd", 32))
    set_steps(run, "NVv", 128, steps(run, "NVd", 128))


def _scan_gap(run: Run, gap: float) -> None:
    """NVd and NVc the two fastest scan kernels at C=32, NVc `gap` (relative) slower."""
    set_steps(run, "NVd", 32, [0.008])
    set_steps(run, "NVc", 32, [0.008 * (1 + gap)])


@recipe("smoke-r2-2")
def scan_gap_one_percent_times_each_step(run: Run) -> None:
    set_steps(run, "NVc", 32, [x * 1.01 for x in steps(run, "NVd", 32)])


@recipe("smoke-r2-2")
def scan_gap_exactly_one_percent(run: Run) -> None:
    _scan_gap(run, 0.01)


@recipe("smoke-r2-2")
def scan_gap_just_below_one_percent(run: Run) -> None:
    _scan_gap(run, 0.0099)


@recipe("smoke-r2-2")
def scan_gap_just_above_one_percent(run: Run) -> None:
    _scan_gap(run, 0.0101)


@recipe("smoke-r2-2")
def scan_nvd_wrong_kernel(run: Run) -> None:
    run.session(0, "NVd")["linear_kernels"] = ["FlashInferCuteDslNvFp4LinearKernel"]


@recipe("smoke-r2-2")
def scan_nvd_two_kernels(run: Run) -> None:
    run.session(0, "NVd")["linear_kernels"] = [
        "FlashInferCudnnNvFp4LinearKernel",
        "CutlassNvFp4LinearKernel",
    ]


@recipe("smoke-r2-2")
def scan_nvd_nll_off(run: Run) -> None:
    run.session(0, "NVd")["nll"] = run.session(0, "NV")["nll"] + 0.02


@recipe("smoke-r2-2")
def scan_nvd_nll_null(run: Run) -> None:
    run.session(0, "NVd")["nll"] = None


@recipe("smoke-r2-2")
def scan_nv_without_nll(run: Run) -> None:
    del run.session(0, "NV")["nll"]


@recipe("smoke-r2-2")
def scan_nvd_without_c32(run: Run) -> None:
    run.keep(lambda r: not (r["treatment"] == "NVd" and r["c"] == 32), ("m1",))


@recipe("smoke-r2-2")
def smoke_without_nvx(run: Run) -> None:
    run.keep(lambda r: r["treatment"] != "NVx", ("servers", "m1"))


@recipe("smoke-r2-2")
def smoke_without_bf16_reference(run: Run) -> None:
    run.manifest["inputs"].pop("bf16_reference_nll", None)


@recipe("full-1")
def null_step_and_m2_metrics(run: Run) -> None:
    for r in run.m1_where(round=2, treatment="MX", c=32, set=2):
        r["step_s"] = None
    for r in run.m2_where(round=2, treatment="NV", c=8):
        r["aggregate_tps"] = None
    for r in run.m2_where(round=3, treatment="MX", c=128):
        r["itl_p50_ms"] = r["tps_per_user_p50"] = None


@recipe("full-1")
def nan_step_and_m2_metric(run: Run) -> None:
    """NaN tokens in the JSONL (json.loads reads them)."""
    for r in run.m1_where(round=6, treatment="NV", c=128, set=1):
        r["step_s"] = float("nan")
    for r in run.m2_where(round=6, treatment="MX", c=64):
        r["itl_p50_ms"] = float("nan")


@recipe("full-1")
def zero_and_negative_steps(run: Run) -> None:
    for r in run.m1_where(round=2, treatment="MX", c=32):
        r["step_s"] = 0.0
    for r in run.m1_where(round=4, treatment="NV", c=8, warmup=False):
        r["step_s"] = -0.001


@recipe("full-1")
def zero_and_negative_m2_metrics(run: Run) -> None:
    for r in run.m2_where(round=1, treatment="NVa", c=32):
        r["aggregate_tps"] = 0.0
    for r in run.m2_where(round=7, treatment="MX", c=1):
        r["itl_p50_ms"] = -1.0


@recipe("expc-1")
def expc_cell_missing_in_one_session(run: Run) -> None:
    run.keep(
        lambda r: (
            not (
                r["round"] == 1
                and r["treatment"] == "NV"
                and r["c"] == 8
                and r.get("prompt_len") == 15360
            )
        ),
        ("m1",),
    )


@recipe("expc-1")
def expc_null_step_in_an_effect_cell(run: Run) -> None:
    for r in run.m1_where(round=0, treatment="MX", c=1, prompt_len=127360, warmup=False):
        r["step_s"] = None
        break


@recipe("expc-1")
def expc_registered_cell_absent(run: Run) -> None:
    run.keep(lambda r: not (r["c"] == 32 and r.get("prompt_len") == 3360), ("m1",))


@recipe("expc-1")
def expc_extra_cell(run: Run) -> None:
    run.m1.extend(dict(r, c=2, prompt_len=2048) for r in run.m1_where(c=1, prompt_len=1024))


@recipe("expc-1")
def expc_aa_effect_cell_without_mxp(run: Run) -> None:
    run.keep(
        lambda r: not (r["treatment"] == "MXp" and r["c"] == 1 and r.get("prompt_len") == 127360),
        ("m1",),
    )


@recipe("expc-1")
def expc_without_prompt_len(run: Run) -> None:
    for r in run.m1:
        r.pop("prompt_len", None)


@recipe("expc-1")
def expc_session_without_prompt_len(run: Run) -> None:
    for r in run.m1_where(round=1, treatment="NV"):
        r.pop("prompt_len", None)


@recipe("expc-1")
def expc_g3_fails(run: Run) -> None:
    """MXp 2 ms slower at (1, 127,360): the A/A effects fail, §16's extension note."""
    for r in run.m1_where(treatment="MXp", c=1, prompt_len=127360):
        r["step_s"] += 0.002


@recipe("expc-1")
def expc_g3_fails_at_ten_rounds(run: Run) -> None:
    expc_g3_fails(run)
    run.protocol["rounds"] = 10


@recipe("expc-1")
def expc_answer_inconclusive(run: Run) -> None:
    """NV at (1, 127,360) alternately 3 ms faster and slower: E_tok's CI is wide."""
    for r in run.m1_where(treatment="NV", c=1, prompt_len=127360):
        r["step_s"] += 0.003 * (-1) ** r["round"]


@recipe("expc-1")
def expc_rounds_ten(run: Run) -> None:
    run.protocol["rounds"] = 10


@recipe("expc-1")
def expc_without_mxp(run: Run) -> None:
    run.protocol["treatments"] = ["MX", "NV"]
    run.keep(lambda r: r["treatment"] != "MXp", ("servers", "m1"))


@recipe("expc-1")
def expc_manifest_cell_malformed(run: Run) -> None:
    run.protocol["cells"].append([8])


@recipe("expc-1")
def expc_manifest_cell_repeated(run: Run) -> None:
    run.protocol["cells"].append([1, 1024])


@recipe("expc-1")
def expc_without_max_model_len(run: Run) -> None:
    run.protocol.pop("max_model_len")


@recipe("smoke-c-1")
def smoke_c_without_mx_round(run: Run) -> None:
    _without(run, 0, "MX")


@recipe("expb-1", others=("full-1",))
def expb_without_reference(run: Run) -> None:
    pass


@recipe("expb-1", others=("full-1",), reference="full-1")
def reference_nll_null(run: Run) -> None:
    run.other("full-1").session(2, "NV")["nll"] = None


@recipe("expb-1", others=("full-1",), reference="full-1")
def reference_without_mxp(run: Run) -> None:
    run.other("full-1").keep(lambda r: r["treatment"] != "MXp")


@recipe("expb-1", others=("full-1",), reference="full-1")
def reference_without_nll(run: Run) -> None:
    for s in run.other("full-1").servers:
        s.pop("nll", None)
        s.pop("nll_per_prompt", None)


@recipe("expb-1", reference="empty")
def reference_empty_directory(run: Run) -> None:
    (run.root / "empty").mkdir()


@recipe("expb-1", others=("expc-1",), reference="expc-1")
def reference_expc(run: Run) -> None:
    pass


@recipe("expb-1", others=("smoke-r2-2",), reference="smoke-r2-2")
def reference_smoke(run: Run) -> None:
    pass


@recipe("full-1", others=("expb-1",), reference="expb-1")
def main_with_expb_reference(run: Run) -> None:
    pass


@recipe("full-1", reference="full-1")
def main_with_itself_as_reference(run: Run) -> None:
    pass


@recipe(
    "expc-1",
    others=("full-1",),
    reference="full-1",
    refuses="the Experiment C analysis takes no reference run",
)
def expc_with_reference(run: Run) -> None:
    pass


@recipe("full-1", refuses=r"lacks required keys \['block_telemetry'\]")
def m1_row_without_block_telemetry(run: Run) -> None:
    run.m1[5].pop("block_telemetry")


@recipe("expc-1", refuses=r"lacks required keys \['block_telemetry'\]")
def expc_m1_row_without_block_telemetry(run: Run) -> None:
    """The tag failed G4 on it; every M1 row of a counted session now needs its telemetry."""
    run.m1[5].pop("block_telemetry")


@recipe("full-1")
def m2_row_without_aggregate_source(run: Run) -> None:
    run.m2[3].pop("aggregate_source")


@recipe("full-1")
def valid_m2_row_without_aggregate_source(run: Run) -> None:
    """G6b lists the valid cell with its source as null."""
    next(r for r in run.m2 if r["valid"]).pop("aggregate_source")


@recipe("full-1", refuses=r"lacks required keys \['telemetry'\]")
def m2_row_without_telemetry(run: Run) -> None:
    run.m2[3].pop("telemetry")


@recipe("full-1", refuses=r"'NVz' is not a valid Treatment")
def unknown_treatment_session(run: Run) -> None:
    """A counted session of a treatment the benchmark does not have."""
    first = run.servers[0]
    run.servers.append(dict(first, treatment="NVz", session_id="r0_NVz-1"))
    run.m1.extend(
        dict(r, treatment="NVz", session_id="r0_NVz-1")
        for r in run.m1_where(session_id=first["session_id"])
    )


@recipe("full-1")
def unknown_treatment_in_a_row_of_no_session(run: Run) -> None:
    run.m1.append(dict(run.m1[0], treatment="XX", session_id="ghost"))


@recipe("full-1")
def row_of_no_session_without_block_telemetry(run: Run) -> None:
    ghost = dict(run.m1[0], session_id="ghost")
    ghost.pop("block_telemetry")
    run.m1.append(ghost)


@recipe("full-1")
def m1_row_with_an_unknown_key(run: Run) -> None:
    run.m1[3]["foo"] = 1


@recipe("full-1")
def m1_row_without_preemptions_delta(run: Run) -> None:
    run.m1[3].pop("preemptions_delta")


@recipe("full-1")
def m1_row_without_first(run: Run) -> None:
    run.m1[3].pop("first")


@recipe("full-1")
def server_row_without_linear_kernels(run: Run) -> None:
    run.servers[4].pop("linear_kernels")


@recipe("full-1", refuses=r"unknown schema_version 3")
def m1_row_schema_version_3(run: Run) -> None:
    """The tag ignored the key; a counted row of an unknown schema version is refused now."""
    run.m1[3]["schema_version"] = 3


@recipe(
    "full-1",
    changed="a measured row needs warmup exactly false (the tag's main-run "
    "analysis took any falsy value): the session's cells count as missing",
    shows="`incomplete (9/10 rounds)`",
)
def warmup_as_int(run: Run) -> None:
    sid = run.servers[0]["session_id"]
    for r in run.m1_where(session_id=sid, warmup=False):
        r["warmup"] = 0


@recipe(
    "full-1",
    changed="an M1 row needs an int batch (the tag's main-run analysis keyed 8.0 "
    "as 8): the session's cells count as missing",
    shows="`incomplete (9/10 rounds)`",
)
def batch_as_float(run: Run) -> None:
    sid = run.servers[0]["session_id"]
    for r in run.m1_where(session_id=sid):
        r["c"] = float(r["c"])


_EXPC_WITHOUT_CELLS = (
    r"M1 cells of several prompt lengths .* but the last manifest line has "
    r"no protocol.cells"
)


@recipe("expc-1", refuses=_EXPC_WITHOUT_CELLS)
def expc_manifest_without_cells(run: Run) -> None:
    """The tag analysed it as a main run, pooling the six batch-1 cells into one."""
    run.protocol.pop("cells")


@recipe("expc-1", refuses=_EXPC_WITHOUT_CELLS)
def expc_manifest_cells_empty(run: Run) -> None:
    run.protocol["cells"] = []


@recipe("smoke-c-1", refuses=_EXPC_WITHOUT_CELLS)
def smoke_c_manifest_without_cells(run: Run) -> None:
    run.protocol.pop("cells")


@recipe("expc-1", refuses=_EXPC_WITHOUT_CELLS)
def expc_last_manifest_line_without_cells(run: Run) -> None:
    line = json.loads(json.dumps(run.manifest))
    line["protocol"].pop("cells")
    run.rows["manifests"].append(line)


@recipe("expc-1", refuses=_EXPC_WITHOUT_CELLS)
def expc_without_manifests(run: Run) -> None:
    run.remove("manifests")


@recipe("full-1")
def main_without_manifests(run: Run) -> None:
    run.remove("manifests")


@recipe("full-1")
def main_rows_with_prompt_len_1024(run: Run) -> None:
    for r in run.m1:
        r["prompt_len"] = 1024


@recipe(
    "full-1",
    changed="the bandwidth table and R_ideal use the rows' prompt length "
    "(2,048 + 640 tokens of context), where the tag assumed 1,024",
    shows="at context 2688 with bfloat16 KV",
)
def main_rows_with_prompt_len_2048(run: Run) -> None:
    for r in run.m1:
        r["prompt_len"] = 2048


@recipe("full-1")
def g1_wrong_kernel(run: Run) -> None:
    run.session(2, "NV")["linear_kernels"] = ["FlashInferCutlassNvFp4LinearKernel"]


@recipe("full-1")
def g1_no_kernel(run: Run) -> None:
    run.session(2, "NV")["linear_kernels"] = []


@recipe("full-1")
def g1_fusion_off(run: Run) -> None:
    run.session(2, "NV")["fuse_act_quant"] = False


@recipe("full-1")
def g1_nvnf_loads_nv_graph(run: Run) -> None:
    run.session(2, "NVnf")["compile_cache_hashes"] = run.session(2, "NV")["compile_cache_hashes"]


@recipe("full-1")
def g1_pass_config_conflict(run: Run) -> None:
    run.session(2, "NV")["pass_config_conflict"] = True


@recipe("full-1")
def g1_odd_moe_backend(run: Run) -> None:
    run.session(2, "MX")["moe_backends"] = ["Weird"]


@recipe("full-1")
def g1_log_facts_missing(run: Run) -> None:
    for s in run.servers:
        for key in (
            "linear_kernels",
            "fuse_act_quant",
            "compile_cache_hashes",
            "kv_cache_tokens",
            "pass_config",
            "custom_fusions",
        ):
            s.pop(key, None)


@recipe("full-1")
def g1_manifest_problem(run: Run) -> None:
    run.manifest["problems"] = ["vllm version differs"]


@recipe("full-1")
def g2_bytes_ratio_off(run: Run) -> None:
    run.manifest["checkpoint_report"]["nv_over_mx"] *= 1.05


@recipe("expb-1", others=("full-1",), reference="full-1")
def g3_fails_on_expb(run: Run) -> None:
    scale_m1(run, 1.05, treatment="MXp")


@recipe("full-1")
def g4_throttled(run: Run) -> None:
    for r in run.m1_where(round=2, treatment="NV", c=32):
        r["block_telemetry"]["env_throttle_us"] = 5


@recipe("full-1")
def g4_throttled_m2(run: Run) -> None:
    for r in run.m2_where(round=3, treatment="MX", c=64):
        r["telemetry"]["env_throttle_us"] = 9


@recipe("full-1")
def g5a_structure_problem(run: Run) -> None:
    run.manifest["checkpoint_report"]["mx_problems"] = ["missing weight_scale"]


@recipe("full-1")
def g5b_without_bf16_reference(run: Run) -> None:
    run.manifest["inputs"].pop("bf16_reference_nll")


@recipe("full-1")
def g5b_nva_degraded(run: Run) -> None:
    for s in run.servers:
        if s["treatment"] == "NVa":
            s["nll"] += 0.5


@recipe("full-1")
def g6a_kv_pool_short(run: Run) -> None:
    run.session(2, "NV")["kv_cache_tokens"] = 1000


@recipe("full-1")
def g6a_kv_tokens_null(run: Run) -> None:
    run.session(2, "NV")["kv_cache_tokens"] = None


@recipe("full-1")
def g6a_capture_sizes(run: Run) -> None:
    run.session(2, "NV")["cudagraph_capture_sizes"] = [1, 2]


@recipe("full-1")
def g6a_preempted(run: Run) -> None:
    for r in run.m1_where(round=2, treatment="NV", c=32):
        r["preemptions_delta"] = 3.0
    for r in run.m2_where(round=2, treatment="NV", c=32):
        r["preemptions_delta"] = None


@recipe("full-1")
def g6a_missing_rep(run: Run) -> None:
    run.m1.remove(run.m1_where(round=2, treatment="NV", c=32, warmup=False)[0])


@recipe("full-1")
def g6a_m1_reps_missing(run: Run) -> None:
    run.protocol.pop("m1_reps")


@recipe("full-1")
def g6b_invalid_m2(run: Run) -> None:
    for r in run.m2_where(c=8):
        r["valid"], r["invalid_reasons"] = False, ["x"]


@recipe("full-1")
def g6b_bad_aggregate_source(run: Run) -> None:
    next(r for r in run.m2 if r["valid"])["aggregate_source"] = "other"


@recipe("full-1")
def g7_other_gpu(run: Run) -> None:
    run.session(2, "NV")["gpu_uuid"] = "GPU-other"


@recipe("full-1")
def g7_unreadable_gpu(run: Run) -> None:
    run.session(2, "NV")["gpu_uuid"] = "error: nvidia-smi"


@recipe("full-1")
def g7_other_start(run: Run) -> None:
    run.session(2, "NV")["start_id"] = "other"


@recipe("full-1")
def g5_weights_gib_off(run: Run) -> None:
    run.session(2, "NV")["weights_gib"] = 25.0


@recipe("full-1")
def m2_absent(run: Run) -> None:
    run.remove("m2")


@recipe("full-1")
def m2_empty(run: Run) -> None:
    run.rows["m2"] = []


@recipe("expc-1")
def expc_g1_wrong_kernel(run: Run) -> None:
    run.session(2, "NV")["linear_kernels"] = ["X"]


@recipe("expc-1")
def expc_g2_and_g5a_checkpoint(run: Run) -> None:
    report = run.manifest["checkpoint_report"]
    report["nv_over_mx"] *= 1.05
    report["nv_problems"] = ["no input_scale"]


@recipe("expc-1")
def expc_g4_throttled(run: Run) -> None:
    for r in run.m1_where(round=2, treatment="NV", c=8):
        r["block_telemetry"]["env_throttle_us"] = 7


@recipe("expc-1")
def expc_g6a_kv_pool_short(run: Run) -> None:
    run.session(2, "NV")["kv_cache_tokens"] = 100000


@recipe("expc-1")
def expc_g6a_preempted(run: Run) -> None:
    for r in run.m1_where(round=2, treatment="NV", c=8):
        r["preemptions_delta"] = 1.0


@recipe("expc-1")
def expc_g5b_nv_degraded(run: Run) -> None:
    for s in run.servers:
        if s["treatment"] == "NV":
            s["nll"] += 0.5


@recipe("expc-1")
def expc_g5b_without_bf16_reference(run: Run) -> None:
    run.manifest["inputs"].pop("bf16_reference_nll")


@recipe("expc-1")
def expc_g7_other_gpu(run: Run) -> None:
    run.session(2, "NV")["gpu_uuid"] = "GPU-other"


@recipe("smoke-nf-2")
def nvnf_loads_nv_graph(run: Run) -> None:
    run.session(0, "NV")["compile_cache_hashes"] = ["0123abcd"]
    run.session(0, "NVnf")["compile_cache_hashes"] = ["0123abcd"]


@recipe("smoke-nf-2")
def nvnf_own_graph(run: Run) -> None:
    run.session(0, "NV")["compile_cache_hashes"] = ["0123abcd"]
    run.session(0, "NVnf")["compile_cache_hashes"] = ["4567ef01"]


@recipe("smoke-nf-2")
def compile_hashes_on_nv_only(run: Run) -> None:
    run.session(0, "NV")["compile_cache_hashes"] = ["0123abcd"]


@recipe("full-1")
def compile_hashes_missing(run: Run) -> None:
    for s in run.servers:
        s.pop("compile_cache_hashes", None)


@recipe("full-1")
def checkpoint_report_string(run: Run) -> None:
    run.manifest["checkpoint_report"] = "error: could not read"


@recipe("full-1")
def checkpoint_report_missing(run: Run) -> None:
    run.manifest.pop("checkpoint_report")
    (run.root / CHECKPOINT_REPORT).unlink()


@recipe("full-1")
def checkpoint_report_fallback(run: Run) -> None:
    run.manifest.pop("checkpoint_report")


@recipe("full-1")
def checkpoint_report_unparseable(run: Run) -> None:
    run.manifest.pop("checkpoint_report")
    (run.root / CHECKPOINT_REPORT).write_text("{bad")


@recipe("full-1")
def kv_dtype_fp8_e5m2(run: Run) -> None:
    run.protocol["kv_cache_dtype"] = "fp8_e5m2"


@recipe("full-1")
def kv_dtype_auto(run: Run) -> None:
    run.protocol["kv_cache_dtype"] = "auto"


@recipe("full-1")
def primary_concurrencies_malformed(run: Run) -> None:
    run.protocol["primary_concurrencies"] = [8, "x"]


@recipe("full-1")
def rounds_twelve(run: Run) -> None:
    run.protocol["rounds"] = 12


@recipe("full-1")
def rounds_float(run: Run) -> None:
    run.protocol["rounds"] = 10.0


@recipe("full-1")
def rounds_one(run: Run) -> None:
    run.protocol["rounds"] = 1


@recipe("full-1")
def registered_batch_not_measured(run: Run) -> None:
    run.protocol["concurrencies"].append(256)


@recipe("full-1")
def unregistered_batch_measured(run: Run) -> None:
    run.m1.extend(dict(r, c=2) for r in run.m1_where(c=1))


@recipe("full-1")
def primary_batch_not_measured(run: Run) -> None:
    run.protocol["primary_concurrencies"] = [8, 32, 256]


@recipe("expb-1", others=("full-1",), reference="full-1")
def expb_kv_dtype_fp8_e4m3(run: Run) -> None:
    run.protocol["kv_cache_dtype"] = "fp8_e4m3"
