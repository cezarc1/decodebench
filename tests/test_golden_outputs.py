"""The current analysis writes, for the committed runs, the files outputs.json pins
(tests/golden/outputs.py; PNGs as tests/golden/pngs.py compares them)."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from fp4bench.analysis.report import REFERENCE_RUN_NEEDED
from tests import GOLDEN_DIR
from tests.golden.make_golden import REFERENCE_RUNS, RUNS
from tests.golden.make_outputs import (
    DELIBERATE_UPDATES,
    DeliberateEdit,
    DeliberateUpdate,
    parse_args,
)
from tests.golden.outputs import copy_runs, head_outputs, normalise, written_hashes
from tests.golden.pngs import differences

OUTPUTS = json.loads((GOLDEN_DIR / "outputs.json").read_text())


def test_every_committed_run_is_pinned():
    assert sorted(OUTPUTS) == sorted(RUNS)


@pytest.mark.parametrize("run", RUNS)
def test_the_analysis_writes_the_tags_files(run):
    ref = REFERENCE_RUNS.get(run)
    with tempfile.TemporaryDirectory(prefix="outputs-") as tmp:
        root = Path(tmp)
        copy_runs(root, (run,) if ref is None else (run, ref))
        out = head_outputs(root, root / run, None if ref is None else root / ref)
    assert "error" not in out, out["error"]
    assert differences(OUTPUTS[run], out) == []
    assert differences(OUTPUTS[run], out, strict=False) == []


def test_normalise_removes_the_directory_of_the_copies(tmp_path):
    text = f"`{tmp_path}/full-1` and `{tmp_path.resolve()}/full-1`, not {tmp_path}x"
    assert normalise(text, tmp_path) == f"`full-1` and `full-1`, not {tmp_path}x"


TAG_NVA_LINE = (
    'To set NVa, put `TREATMENT_SERVER_ARGS["NVa"] = ("--linear-backend", '
    '"flashinfer_cudnn")` in fp4bench/config.py.'
)
TAG_REFERENCE_RUN_NEEDED = (
    "The comparison of this run's FP8-KV NLL with the main run's BF16-KV NLL (§13) needs "
    "--reference-run: `python -m analysis.analyze results/<this run> --reference-run "
    "results/full-1`."
)


def test_the_deliberate_updates_turn_the_tags_lines_into_what_the_analysis_writes_now():
    edit = DeliberateEdit()
    with tempfile.TemporaryDirectory(prefix="outputs-") as tmp:
        root = Path(tmp)
        copy_runs(root, ("smoke-r2-2",))
        head_outputs(root, root / "smoke-r2-2", None)
        summary = (root / "smoke-r2-2" / "summary.md").read_text().splitlines()
    assert TAG_NVA_LINE not in summary and edit("summary.md", TAG_NVA_LINE) in summary
    assert edit("summary.md", TAG_REFERENCE_RUN_NEEDED) == REFERENCE_RUN_NEEDED
    assert all(u.commit == "c3438c0" and u.file == "summary.md" for u in DELIBERATE_UPDATES)


def test_a_deliberate_update_edits_its_file_kind_only_and_counts_the_files_it_changed():
    update = DeliberateUpdate("abc1234", "summary.md", "old words", "new words")
    edit = DeliberateEdit((update,))
    assert edit("summary.md", "a old words b old words") == "a new words b new words"
    assert edit("results_c.json", "old words") == "old words"
    assert edit("summary.md", "nothing to change") == "nothing to change"
    assert edit.counts == {update: 1}
    edit.check_used()
    with pytest.raises(SystemExit, match="abc1234"):
        DeliberateEdit((update,)).check_used()


def test_the_edit_applies_to_text_files_before_hashing(tmp_path):
    (tmp_path / "summary.md").write_text(f"`{tmp_path}/full-1`: old")
    (tmp_path / "ratio_vs_c.png").write_bytes(b"old")
    found = written_hashes(tmp_path, set(), tmp_path, lambda name, text: text.replace("old", "new"))
    assert found == {
        "summary.md": hashlib.sha256(b"`full-1`: new").hexdigest(),
        "ratio_vs_c.png": hashlib.sha256(b"old").hexdigest(),
    }


def test_make_outputs_writes_into_the_golden_directory_unless_told_otherwise(tmp_path):
    assert parse_args(["tag"]) == (Path("tag"), GOLDEN_DIR)
    assert parse_args(["tag", "--out", str(tmp_path)]) == (Path("tag"), tmp_path)
    with pytest.raises(SystemExit):
        parse_args([])
