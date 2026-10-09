"""fp4bench.cli: what each command validates before any executor is asked and what it hands
the (fake) executor; analyze on synthetic runs."""

import json
import subprocess
import sys
from pathlib import Path
from typing import override

import click
import pytest
from click.testing import CliRunner

from fp4bench import cli
from fp4bench.analysis.report import REFERENCE_RUN_NEEDED
from fp4bench.analysis.verdicts import Answer, extend_line
from fp4bench.core.types import Cell, CheckpointKind, FetchKind, Format
from fp4bench.manifest import CodeVersion
from fp4bench.studies.registry import STUDIES
from tests import REPO
from tests.analysis_runs import (
    GATE_LINE_CLEAN,
    c_run,
    main_and_expb,
    manifest,
    mx_step,
    set_nll,
    synthetic,
    write_run,
)

CODE = CodeVersion("a" * 40, False)
CALL_ID = "fc-FAKE123"
SHA = "0123456789abcdef" * 2 + "01234567"


class FakeModal:
    """The ModalExecutor interface, recording each call; `results` overrides what one returns."""

    def __init__(self, results: dict | None = None):
        self.calls: list[tuple] = []
        self.results = results or {}

    def _record(self, name, *args, default=None):
        self.calls.append((name, *args))
        return self.results.get(name, default)

    def env(self):
        return self._record("env", default={"gpu": {"name": "NVIDIA B200"}, "problems": []})

    def prepare(self, expc, force, spawned):
        spawned(CALL_ID)
        return self._record("prepare", expc, force, default={"m1_prompts_sha256": "ab"})

    def quantize(self, fmt, spawned):
        spawned(CALL_ID)
        return self._record("quantize", fmt, default={"format": fmt})

    def check(self):
        return self._record("check", default={"mx_problems": [], "nv_problems": []})

    def publish(self, kind, repo_id, private):
        return self._record(
            "publish", kind, repo_id, private, default="https://huggingface.co/u/n/commit/abc"
        )

    def fetch(self, kind, repo_id, revision):
        return self._record("fetch", kind, repo_id, revision, default={"kind": kind})

    def run(self, study, run_id, rounds, rerun_rounds, code):
        return self._record("run", study, run_id, rounds, rerun_rounds, code, default=CALL_ID)

    def microbench(self, run_id):
        return self._record("microbench", run_id, default=CALL_ID)

    def profile(self, run_id):
        return self._record("profile", run_id, default=CALL_ID)

    def download(self, run_id, dest, force):
        return self._record("download", run_id, dest, force)


class FakeLocal:
    def __init__(self, paths, fail: int = 0):
        self.paths, self.fail, self.calls = paths, fail, []

    def run(self, study, run_id, rounds, rerun_rounds, code):
        self.calls.append((study, run_id, rounds, rerun_rounds, code))
        if self.fail:
            raise subprocess.CalledProcessError(self.fail, ["python"])
        return self.paths[1] / run_id


@pytest.fixture
def fake(monkeypatch):
    executor = FakeModal()
    monkeypatch.setattr(cli, "modal_executor", lambda: executor)
    monkeypatch.setattr(cli, "code_version", lambda: CODE)
    return executor


@pytest.fixture
def local(monkeypatch):
    made = []

    def make(data_dir, results_dir, local_dir):
        made.append(FakeLocal((data_dir, results_dir, local_dir)))
        return made[-1]

    monkeypatch.setattr(cli, "local_executor", make)
    return made


def invoke(*args: str):
    return CliRunner().invoke(cli.main, list(args), prog_name="fp4bench")


def refused(result, *needles: str) -> None:
    assert result.exit_code == 2, result.output
    for needle in needles:
        assert needle in result.output, (needle, result.output)


def test_parse_rerun_rounds_accepts_a_comma_separated_list():
    assert cli.parse_rerun_rounds("1,3", 5) == (1, 3)
    assert cli.parse_rerun_rounds(" 3 , 1 ", 5) == (1, 3)
    assert cli.parse_rerun_rounds("2,2,0", 5) == (0, 2)
    assert cli.parse_rerun_rounds("", 5) == () == cli.parse_rerun_rounds("  ", 5)


@pytest.mark.parametrize(
    "text, message",
    [
        ("a", "not an integer"),
        ("1,,3", "empty"),
        ("1.5", "not an integer"),
        ("-1", "outside the study's rounds 0..4"),
        ("1;3", "not an integer"),
        ("1,5", "outside the study's rounds 0..4"),
    ],
)
def test_parse_rerun_rounds_rejects_bad_input_clearly(text, message):
    with pytest.raises(click.BadParameter, match=message):
        cli.parse_rerun_rounds(text, 5)


def test_run_checks_rerun_rounds_against_the_rounds_it_will_run(fake):
    assert (
        invoke(
            "run", "full", "--run-id", "full-1", "--rounds", "10", "--rerun-rounds", "9,7"
        ).exit_code
        == 0
    )
    refused(invoke("run", "full", "--run-id", "full-1", "--rerun-rounds", "7"), "0..4")
    refused(invoke("run", "smoke", "--run-id", "s-1", "--rerun-rounds", "1"), "0..0")
    refused(invoke("run", "full", "--run-id", "full-1", "--rerun-rounds", "x"), "--rerun-rounds")
    assert [c[4] for c in fake.calls] == [(7, 9)]


@pytest.mark.parametrize("name", list(STUDIES))
def test_a_study_with_a_registered_extension_may_only_gain_rounds(fake, name):
    study = STUDIES[name]
    below = invoke("run", name, "--run-id", "x-1", "--rounds", str(study.rounds - 1))
    if study.extension_rounds is not None:
        refused(below, "--rounds", "pre-registered", f"(to {study.extension_rounds})")
    else:
        assert below.exit_code == 0, below.output
    for more in (study.rounds, study.rounds + 1, 10, 12):
        assert invoke("run", name, "--run-id", "x-1", "--rounds", str(more)).exit_code == 0
    assert all(c[0] == "run" for c in fake.calls)
    assert {n for n, s in STUDIES.items() if s.extension_rounds} == {"full", "expb", "expc"}


def test_rounds_cannot_be_negative(fake):
    refused(invoke("run", "smoke", "--run-id", "s-1", "--rounds", "-1"), "--rounds")
    assert fake.calls == []


def test_run_spawns_the_study_with_the_code_version_and_says_how_to_get_the_results(fake):
    result = invoke("run", "full", "--run-id", "full-1", "--rounds", "10", "--rerun-rounds", "7,9")
    assert result.exit_code == 0, result.output
    assert fake.calls == [("run", "full", "full-1", 10, (7, 9), CODE)]
    assert f"code: {'a' * 40}\n" in result.output
    assert f"spawned {CALL_ID} in a detached app" in result.output
    assert "when done: fp4bench download full-1" in result.output


@pytest.mark.parametrize("name", list(STUDIES))
def test_every_study_runs_under_its_name(fake, name):
    assert invoke("run", name, "--run-id", f"{name}-1").exit_code == 0
    assert fake.calls == [("run", name, f"{name}-1", 0, (), CODE)]


def test_an_unknown_study_is_refused_naming_every_study(fake):
    result = invoke("run", "smoke_nf", "--run-id", "x")
    refused(result, *(f"'{name}'" for name in STUDIES))
    assert fake.calls == []


@pytest.mark.parametrize("name", list(STUDIES))
def test_every_study_needs_an_explicit_run_id(fake, name):
    refused(invoke("run", name), "Missing option '--run-id'")
    assert fake.calls == []


@pytest.mark.parametrize("command", [["run", "smoke"], ["microbench"], ["profile"]])
@pytest.mark.parametrize("bad", ["../x", "a/b", "a b", ".", "-x", "x\n"])
def test_a_run_id_must_be_a_plain_directory_name(fake, command, bad):
    refused(invoke(*command, "--run-id", bad), "--run-id", "plain directory name")
    refused(invoke("download", bad), "RUN_ID")
    assert fake.calls == []


def test_the_code_version_says_when_the_checkout_has_changes(fake, monkeypatch):
    monkeypatch.setattr(cli, "code_version", lambda: CodeVersion("b" * 40, True))
    assert (
        f"code: {'b' * 40} with uncommitted changes"
        in invoke("run", "smoke", "--run-id", "s-1").output
    )
    monkeypatch.setattr(cli, "code_version", lambda: CodeVersion(None, None))
    assert (
        "code: unknown commit (unknown changes)" in invoke("run", "smoke", "--run-id", "s-1").output
    )
    assert [c[5] for c in fake.calls] == [CodeVersion("b" * 40, True), CodeVersion(None, None)]


def test_run_local_runs_the_study_against_the_local_directories(fake, local, tmp_path):
    result = invoke(
        "run",
        "smoke-nf",
        "--run-id",
        "nf-1",
        "--executor",
        "local",
        "--data-dir",
        str(tmp_path / "data"),
        "--results-dir",
        str(tmp_path / "res"),
        "--local-dir",
        str(tmp_path / "scratch"),
        "--rounds",
        "2",
        "--rerun-rounds",
        "1",
    )
    assert result.exit_code == 0, result.output
    (executor,) = local
    assert executor.paths == (tmp_path / "data", tmp_path / "res", tmp_path / "scratch")
    assert executor.calls == [("smoke-nf", "nf-1", 2, (1,), CODE)]
    assert f"done: {tmp_path / 'res' / 'nf-1'}" in result.output and fake.calls == []


def test_run_local_defaults_to_results_and_local_in_the_working_directory(fake, local, tmp_path):
    assert (
        invoke(
            "run", "smoke", "--run-id", "s-1", "--executor", "local", "--data-dir", str(tmp_path)
        ).exit_code
        == 0
    )
    assert local[0].paths == (tmp_path, Path("results"), Path("local"))


def test_the_local_directories_are_for_the_local_executor_only(fake, local, tmp_path):
    for option in ("--data-dir", "--results-dir", "--local-dir"):
        result = invoke("run", "smoke", "--run-id", "s-1", option, str(tmp_path))
        assert result.exit_code == 2 and "are for --executor local" in result.output
    result = invoke("run", "smoke", "--run-id", "s-1", "--executor", "local")
    assert result.exit_code == 2 and "--executor local needs --data-dir" in result.output
    refused(
        invoke(
            "run",
            "smoke",
            "--run-id",
            "s-1",
            "--executor",
            "local",
            "--data-dir",
            str(tmp_path),
            "--local-dir",
            str(tmp_path / "."),
        ),
        "--local-dir",
    )
    assert fake.calls == [] and all(e.calls == [] for e in local)


def test_a_local_run_that_fails_exits_with_its_status(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "local_executor", lambda *paths: FakeLocal(paths, fail=3))
    result = invoke(
        "run", "smoke", "--run-id", "s-1", "--executor", "local", "--data-dir", str(tmp_path)
    )
    assert result.exit_code == 1 and "the run exited with status 3" in result.output


@pytest.mark.parametrize("command", ["microbench", "profile"])
def test_microbench_and_profile_spawn_into_a_fresh_run_and_say_where_it_lands(fake, command):
    result = invoke(command, "--run-id", f"{command}-2")
    assert result.exit_code == 0, result.output
    assert fake.calls == [(command, f"{command}-2")]
    assert CALL_ID in result.output and f"fp4bench download {command}-2" in result.output
    refused(invoke(command), "Missing option '--run-id'")


def test_download_copies_the_run_directory_into_dest(fake, tmp_path):
    assert invoke("download", "full-1").exit_code == 0
    assert invoke("download", "full-1", "--dest", str(tmp_path), "--force").exit_code == 0
    assert fake.calls == [
        ("download", "full-1", Path("results"), False),
        ("download", "full-1", tmp_path, True),
    ]


def test_a_failed_download_exits_with_modals_status(monkeypatch):
    class Failing(FakeModal):
        @override
        def download(self, run_id, dest, force):
            raise subprocess.CalledProcessError(4, ["modal"])

    monkeypatch.setattr(cli, "modal_executor", Failing)
    result = invoke("download", "full-1")
    assert result.exit_code == 1 and "modal volume get exited with status 4" in result.output


def test_env_prints_the_manifest(fake):
    result = invoke("env")
    assert result.exit_code == 0 and json.loads(result.output)["gpu"]["name"] == "NVIDIA B200"


@pytest.mark.parametrize(
    "flags, expc, force",
    [
        ([], False, False),
        (["--force"], False, True),
        (["--expc"], True, False),
        (["--expc", "--force"], True, True),
    ],
)
def test_prepare_builds_the_m1_or_the_cell_prompts(fake, flags, expc, force):
    result = invoke("prepare", *flags)
    assert result.exit_code == 0, result.output
    assert fake.calls == [("prepare", expc, force)]
    waiting, summary = result.output.split("\n", 1)
    assert waiting.startswith(f"spawned {CALL_ID}; waiting") and "goes on" in waiting
    assert json.loads(summary) == {"m1_prompts_sha256": "ab"}


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_quantize_forwards_the_format(fake, fmt):
    result = invoke("quantize", "--fmt", fmt)
    assert result.exit_code == 0 and fake.calls == [("quantize", Format(fmt))]
    assert type(fake.calls[0][1]) is Format
    assert json.loads(result.output.split("\n", 1)[1]) == {"format": fmt}


@pytest.mark.parametrize("bad", ["", "NVFP4", "mx", "nvfp4 ", "fp8"])
def test_quantize_needs_one_of_the_two_formats(fake, bad):
    refused(invoke("quantize", "--fmt", bad), "--fmt")
    refused(invoke("quantize"), "Missing option '--fmt'")
    assert fake.calls == []


def test_check_writes_the_report_next_to_the_results(fake, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = invoke("check")
    assert result.exit_code == 0 and fake.calls == [("check",)]
    report = json.loads((tmp_path / "results" / "checkpoint_report.json").read_text())
    assert report == json.loads(result.output) == {"mx_problems": [], "nv_problems": []}


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_publish_is_private_unless_public_is_given(fake, kind):
    assert invoke("publish", "--kind", kind, "--repo-id", "u/n").exit_code == 0
    result = invoke("publish", "--kind", kind, "--repo-id", "u/n", "--public")
    assert fake.calls == [("publish", kind, "u/n", True), ("publish", kind, "u/n", False)]
    assert result.output == "https://huggingface.co/u/n/commit/abc\n"


@pytest.mark.parametrize(
    "args, needle",
    [
        (["--kind", "nvidia", "--repo-id", "u/n"], "--kind"),
        (["--kind", "MX", "--repo-id", "u/n"], "--kind"),
        (["--repo-id", "u/n"], "Missing option '--kind'"),
        (["--kind", "mx"], "Missing option '--repo-id'"),
        (["--kind", "mx", "--repo-id", "no-slash"], "--repo-id"),
    ],
)
def test_publish_validates_before_asking_modal(fake, args, needle):
    refused(invoke("publish", *args), needle)
    assert fake.calls == []


@pytest.mark.parametrize("kind", ["mx", "nv", "nvidia"])
def test_fetch_forwards_kind_repo_and_revision(fake, kind):
    result = invoke("fetch", "--kind", kind, "--repo-id", "u/n", "--revision", SHA)
    assert result.exit_code == 0 and fake.calls == [("fetch", kind, "u/n", SHA)]
    assert json.loads(result.output) == {"kind": kind}


@pytest.mark.parametrize(
    "args, needle",
    [
        (["--kind", "mxfp4", "--repo-id", "u/n", "--revision", SHA], "--kind"),
        (["--kind", "mx", "--repo-id", "u", "--revision", SHA], "--repo-id"),
        (["--kind", "mx", "--repo-id", "u/n"], "Missing option '--revision'"),
        (["--kind", "mx", "--repo-id", "u/n", "--revision", "main"], "--revision"),
        (["--kind", "mx", "--repo-id", "u/n", "--revision", "b" * 39], "--revision"),
        (["--kind", "mx", "--repo-id", "u/n", "--revision", "B" * 40], "--revision"),
    ],
)
def test_fetch_validates_before_asking_modal(fake, args, needle):
    refused(invoke("fetch", *args), needle)
    assert fake.calls == []


@pytest.mark.parametrize("repo_id", ["u/n", "some-org/Qwen3-32B-MXFP4", "a.b/c_d-e", "A1/b2"])
def test_a_repo_id_is_user_slash_name(fake, repo_id):
    assert invoke("publish", "--kind", "mx", "--repo-id", repo_id).exit_code == 0


@pytest.mark.parametrize(
    "repo_id",
    [
        "name",
        "u/",
        "/n",
        "a/b/c",
        "u/n m",
        "../x/y",
        "u/..",
        "u/.hidden",
        "-u/n",
        "u/n\n",
        "u /n",
        "https://x/y",
    ],
)
def test_anything_else_is_not_a_repo_id(fake, repo_id):
    refused(invoke("publish", "--kind", "mx", "--repo-id", repo_id), "--repo-id")


@pytest.mark.parametrize("revision", ["", " " + "a" * 39, "a" * 39 + "\n", "g" * 40, "a" * 41])
def test_a_revision_is_a_full_lowercase_hex_sha(fake, revision):
    refused(
        invoke("fetch", "--kind", "mx", "--repo-id", "u/n", "--revision", revision), "--revision"
    )


def test_the_kinds_are_what_prep_and_publish_support():
    from fp4bench import publish

    assert {FetchKind(kind).checkpoint for kind in cli.FETCH_KINDS} == set(CheckpointKind)
    assert set(cli.PUBLISH_KINDS) == set(publish.KINDS)


@pytest.mark.parametrize(
    "command, result",
    [
        (["env"], "env"),
        (["check"], "check"),
        (["prepare"], "prepare"),
        (["fetch", "--kind", "mx", "--repo-id", "u/n", "--revision", SHA], "fetch"),
        (["run", "smoke", "--run-id", "s-1"], "run"),
    ],
)
def test_an_interrupted_call_aborts(monkeypatch, command, result):
    monkeypatch.setattr(cli, "modal_executor", lambda: FakeModal({result: None}))
    monkeypatch.setattr(cli, "code_version", lambda: CODE)
    out = invoke(*command)
    assert out.exit_code == 1 and "Aborted!" in out.output


def test_the_help_lists_the_steps_of_a_study_and_the_secret_modal_needs():
    text = invoke("--help").output
    for command in (
        "env",
        "prepare",
        "quantize",
        "check",
        "publish",
        "fetch",
        "run",
        "download",
        "analyze",
        "plot",
        "microbench",
        "profile",
    ):
        assert command in text, command
    assert "huggingface-secret" in text
    assert all(name in invoke("run", "--help").output for name in STUDIES)


@pytest.mark.parametrize("module", ["fp4bench", "fp4bench.cli"])
def test_the_command_line_runs_as_a_module(module):
    done = subprocess.run(
        [sys.executable, "-m", module, "run", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert f"Usage: python -m {module} run [OPTIONS] STUDY" in done.stdout


def test_the_command_line_imports_modal_only_for_a_modal_step():
    code = (
        "import sys, fp4bench.cli; print('modal' in sys.modules, 'fp4bench.runner' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert done.stdout.split() == ["False", "False"]


def _main_run(tmp_path, nv_degradation=None):
    servers, m1, m2 = synthetic(1.0, (8, 32, 128))
    if nv_degradation is not None:
        set_nll(servers, "NV", nv_degradation)
    return write_run(tmp_path, servers, m1, m2, manifest())


def test_analyze_prints_the_verdict_the_gate_line_and_the_summary(tmp_path):
    run = _main_run(tmp_path)
    result = invoke("analyze", str(run))
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "verdict: validated",
        GATE_LINE_CLEAN,
        f"see {run / 'summary.md'}",
    ]
    assert (run / "summary.md").exists() and (run / "ratio_vs_c.png").exists()


def test_analyze_says_when_the_run_is_not_interpretable(tmp_path):
    out = invoke("analyze", str(_main_run(tmp_path, nv_degradation=0.6))).output.splitlines()
    assert (
        out[0] == "verdict: validated — NOT INTERPRETABLE: gate failures" and "G5b FAIL" in out[1]
    )


@pytest.mark.parametrize("args", [[], ["a", "b"], ["--reference-run"], ["--bogus", "x"]])
def test_analyze_with_bad_arguments_is_a_usage_error(tmp_path, args):
    result = invoke("analyze", *[str(_main_run(tmp_path)) if a == "a" else a for a in args])
    assert result.exit_code == 2 and "Error: " in result.output
    assert not list(tmp_path.glob("*/summary.md"))


def test_analyze_passes_the_reference_run_to_the_accuracy_note(tmp_path):
    run, ref = main_and_expb(tmp_path)
    out = invoke("analyze", str(run), "--reference-run", str(ref)).output.splitlines()
    assert out[0].startswith("verdict: ") and out[2] == f"see {run / 'summary.md'}"
    assert len(out) == 3
    assert "| NV |" in (run / "summary.md").read_text().split("## Accuracy note")[1]


def test_analyze_of_experiment_b_without_a_reference_says_how_to_pass_one(tmp_path):
    run, _ = main_and_expb(tmp_path)
    out = invoke("analyze", str(run)).output.splitlines()
    assert out[3] == REFERENCE_RUN_NEEDED and "`fp4bench analyze results/" in out[3]
    assert REFERENCE_RUN_NEEDED in (run / "summary.md").read_text()


def test_analyze_refuses_a_directory_that_does_not_exist(tmp_path):
    run, _ = main_and_expb(tmp_path)
    result = invoke("analyze", str(run), "--reference-run", str(tmp_path / "missing"))
    assert result.exit_code == 2 and "--reference-run" in result.output
    assert "does not exist" in result.output
    result = invoke("analyze", str(tmp_path / "nope"))
    assert result.exit_code == 2 and "does not exist" in result.output


def test_analyze_recognises_an_experiment_c_run_by_its_manifest(tmp_path):
    run = c_run(tmp_path)
    out = invoke("analyze", str(run)).output.splitlines()
    assert out[0] == f"answer: {Answer.BATCH_DRIVES}" and out[2] == "config reproduction: pass"
    assert out[3] == (
        f"see {run / 'summary.md'}, {run / 'results_c.json'} and {run / 'expc_delta.png'}"
    )
    assert (run / "results_c.json").exists()


def test_analyze_prints_the_extension_note_without_markdown(tmp_path):
    factor = 1 + 0.4 / (mx_step((128, 360)) * 1000)
    out = invoke("analyze", str(c_run(tmp_path, aa={Cell(128, 360): factor}))).output.splitlines()
    assert out[0].endswith("NOT INTERPRETABLE: gate failures")
    assert out[1] == f"{extend_line(10)} — G3 failed."


def test_an_experiment_c_run_takes_no_reference_run(tmp_path):
    run = c_run(tmp_path)
    result = invoke("analyze", str(run), "--reference-run", str(run))
    assert result.exit_code == 1 and "takes no reference run" in result.output


def test_a_row_that_does_not_load_is_printed_with_its_file_and_line(tmp_path):
    run = _main_run(tmp_path)
    lines = (run / "m1.jsonl").read_text().splitlines()
    lines[4] = lines[4].replace('"warmup"', '"warmup_"')
    (run / "m1.jsonl").write_text("\n".join(lines) + "\n")
    result = invoke("analyze", str(run))
    assert result.exit_code == 1
    assert (
        f"Error: {run / 'm1.jsonl'}, line 5: M1Row row lacks required keys ['warmup']"
        in result.output
    )


def test_a_manifest_naming_an_unregistered_study_is_refused(tmp_path):
    run = write_run(tmp_path, *synthetic(1.0, (8, 32, 128)), {**manifest(), "study": "nope"})
    result = invoke("analyze", str(run))
    assert result.exit_code == 1 and "names study 'nope', which is not registered" in result.output


def test_the_cli_choices_are_the_enum_values():
    from fp4bench.core.types import ExecutorKind, Format

    assert cli.FORMATS == ("mxfp4", "nvfp4") == tuple(Format)
    assert cli.PUBLISH_KINDS == ("mx", "nv") == (CheckpointKind.MX, CheckpointKind.NV)
    assert cli.FETCH_KINDS == ("mx", "nv", "nvidia") == tuple(FetchKind)
    assert cli.EXECUTORS == ("modal", "local") == tuple(ExecutorKind)
    assert all(type(c) is str for c in (*cli.FORMATS, *cli.FETCH_KINDS, *cli.EXECUTORS))
