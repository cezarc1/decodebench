"""The executors: the Modal app imported locally (it contacts nothing) with fakes for the runner and
the volumes, and the local executor."""

import contextlib
import dataclasses
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import modal
import pytest

from fp4bench import settings
from fp4bench.core.schema import ManifestLine, load_rows
from fp4bench.core.types import Format
from fp4bench.executors import local
from fp4bench.executors import modal as mx
from fp4bench.manifest import CODE_COMMIT_ENV, CODE_DIRTY_ENV, CodeVersion
from fp4bench.studies.registry import STUDIES
from tests import REPO

GIB = 1024


def test_registered_functions_are_the_format_and_kind_parametrised_set():
    assert sorted(mx.app.registered_functions) == [
        "check_checkpoints",
        "env_check",
        "experiment",
        "fetch_checkpoint",
        "kernel_microbench",
        "prepare_data",
        "prepare_expc_data",
        "profile_sessions",
        "publish_checkpoint",
        "quantize_checkpoint",
    ]


@pytest.fixture(scope="module")
def probe():
    rec = types.SimpleNamespace(functions={}, commands=[], module=None)
    with pytest.MonkeyPatch.context() as mp:
        function, run_commands = modal.App.function, modal.Image.run_commands

        def recording_function(self, *args, **kwargs):
            decorator = function(self, *args, **kwargs)

            def apply(fn):
                rec.functions[fn.__name__] = kwargs
                return decorator(fn)

            return apply

        def recording_run_commands(self, *commands, **kwargs):
            rec.commands.extend(c for c in commands if isinstance(c, str))
            return run_commands(self, *commands, **kwargs)

        mp.setattr(modal.App, "function", recording_function)
        mp.setattr(modal.Image, "run_commands", recording_run_commands)
        spec = importlib.util.spec_from_file_location("modal_executor_probe", mx.__file__)
        assert spec is not None and spec.loader is not None
        rec.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rec.module)
    return rec


def test_the_probe_sees_every_registered_function(probe):
    assert sorted(probe.functions) == sorted(mx.app.registered_functions)


def test_prepare_has_a_timeout_for_the_64_gb_download(probe):
    kwargs = probe.functions["prepare_data"]
    assert kwargs["timeout"] >= 2 * 3600
    assert "gpu" not in kwargs and kwargs["image"] is probe.module.vllm_image


def test_prepare_c_runs_on_cpu_with_the_volumes_and_downloads_nothing(probe):
    kwargs = probe.functions["prepare_expc_data"]
    assert "gpu" not in kwargs and "secrets" not in kwargs
    assert kwargs["volumes"] == probe.module.VOLUMES and kwargs["image"] is probe.module.vllm_image
    assert kwargs["memory"] >= 16 * GIB and kwargs["timeout"] >= 1800


def test_the_experiment_timeout_covers_experiment_c(probe):
    assert probe.functions["experiment"]["timeout"] >= 26 * 60 * 15 * 1.2


def test_quantize_gets_a_b200_256_gib_of_memory_and_three_hours(probe):
    kwargs = probe.functions["quantize_checkpoint"]
    assert kwargs["gpu"] == "B200"
    assert kwargs["memory"] >= 256 * GIB
    assert kwargs["timeout"] == 3 * 3600
    assert kwargs["image"] is probe.module.quant_image
    assert kwargs["ephemeral_disk"] >= 200 * GIB
    assert kwargs["volumes"] == probe.module.VOLUMES


def test_experiment_gets_at_least_200_gib_of_ephemeral_disk_for_the_local_copies(probe):
    kwargs = probe.functions["experiment"]
    assert kwargs["ephemeral_disk"] >= 200 * 1024
    assert kwargs["gpu"] == "B200" and kwargs["cpu"] == 16 and kwargs["memory"] == 64 * GIB
    assert kwargs["timeout"] == 10 * 3600
    assert kwargs["image"] is probe.module.vllm_image
    assert (mx.experiment.spec.ephemeral_disk or 0) >= 200 * 1024
    assert mx.experiment.spec.memory == 64 * GIB and mx.experiment.spec.gpus == "B200"


def test_microbench_gets_the_scratch_runs_b200_and_writes_only_to_the_results_volume(probe):
    kwargs = probe.functions["kernel_microbench"]
    assert (kwargs["gpu"], kwargs["cpu"], kwargs["memory"], kwargs["timeout"]) == (
        "B200",
        8,
        64 * GIB,
        3600,
    )
    assert kwargs["image"] is probe.module.vllm_image
    assert kwargs["volumes"] == {settings.RESULTS_DIR: probe.module.results}
    assert "ephemeral_disk" not in kwargs and "secrets" not in kwargs


def test_profile_gets_the_experiments_b200_volumes_and_local_disk_for_three_hours(probe):
    kwargs = probe.functions["profile_sessions"]
    assert (kwargs["gpu"], kwargs["cpu"], kwargs["memory"], kwargs["timeout"]) == (
        "B200",
        16,
        64 * GIB,
        3 * 3600,
    )
    assert kwargs["ephemeral_disk"] == probe.module.LOCAL_DISK
    assert kwargs["volumes"] == probe.module.VOLUMES
    assert kwargs["image"] is probe.module.vllm_image
    assert "secrets" not in kwargs


MODAL_DISK_BOUNDS_MIB = (524288, 3145728)


@pytest.mark.parametrize("fn", ["quantize_checkpoint", "experiment", "profile_sessions"])
def test_ephemeral_disk_is_within_modal_server_bounds(probe, fn):
    lo, hi = MODAL_DISK_BOUNDS_MIB
    assert lo <= probe.functions[fn]["ephemeral_disk"] <= hi


def test_fetch_and_publish_get_the_hugging_face_secret_and_no_gpu(probe):
    for name in ("fetch_checkpoint", "publish_checkpoint"):
        kwargs = probe.functions[name]
        assert kwargs["secrets"] == [probe.module.hf_secret], name
        assert "gpu" not in kwargs and kwargs["volumes"] == probe.module.VOLUMES, name
    assert probe.functions["fetch_checkpoint"]["timeout"] >= 3600


def test_check_and_env_are_unchanged(probe):
    check = probe.functions["check_checkpoints"]
    assert (check["cpu"], check["memory"], check["timeout"]) == (4, 16384, 600)
    assert "gpu" not in check and "secrets" not in check
    env = probe.functions["env_check"]
    assert (env["gpu"], env["cpu"], env["memory"], env["timeout"]) == ("B200", 4, 16384, 600)


def test_the_quant_image_installs_llmcompressor_then_checks_that_torch_is_unchanged(probe):
    commands = probe.commands
    (install,) = [i for i, c in enumerate(commands) if "llmcompressor" in c]
    assert f"llmcompressor=={settings.LLMCOMPRESSOR_VERSION}" in commands[install]
    assert commands[install + 1] == mx.torch_pin_check(settings.EXPECTED_VERSIONS["torch"])
    assert "2.13.0" in commands[install + 1]


def _run_torch_check(tmp_path, torch_source: str) -> subprocess.CompletedProcess:
    """Run the shell command the image build runs, with a stand-in `torch` module."""
    (tmp_path / "torch.py").write_text(torch_source)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(  # noqa: S602 - the image build runs this shell snippet
        mx.torch_pin_check("2.13.0"),
        shell=True,
        check=False,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "installed, ok",
    [
        ("2.13.0+cu130", True),
        ("2.13.0", True),
        ("2.14.0+cu130", False),
        ("2.12.1+cu130", False),
        ("2.13.01", False),
        ("12.13.0", False),
    ],
)
def test_the_torch_check_exits_nonzero_unless_torch_is_the_pinned_version(tmp_path, installed, ok):
    done = _run_torch_check(tmp_path, f"__version__ = {installed!r}\n")
    assert (done.returncode == 0) is ok, done.stderr
    if not ok:
        assert installed in done.stderr and "2.13.0" in done.stderr


def test_the_torch_check_fails_when_torch_cannot_be_imported(tmp_path):
    done = _run_torch_check(tmp_path, "raise ImportError('no torch')\n")
    assert done.returncode != 0 and "no torch" in done.stderr


def test_safe_commit_swallows_and_logs_exceptions(capsys):
    calls = []

    def flaky():
        calls.append(1)
        raise RuntimeError("volume busy")

    mx.safe_commit(flaky)()
    assert calls == [1]
    err = capsys.readouterr().err
    assert "volume busy" in err and "commit" in err


def test_safe_commit_passes_through_success_and_never_swallows_interrupts():
    calls = []
    mx.safe_commit(lambda: calls.append("ok"))()
    assert calls == ["ok"]

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        mx.safe_commit(interrupted)()


class FakeVolume:
    def __init__(self, fail=False):
        self.commits, self.fail = 0, fail

    def commit(self):
        self.commits += 1
        if self.fail:
            raise RuntimeError("commit failed")


@pytest.fixture
def runner_calls(monkeypatch):
    calls = []

    def fake_run_experiment(run_dir, study, commit, rerun_rounds=()):
        calls.append(
            {"run_dir": run_dir, "study": study, "commit": commit, "rerun_rounds": rerun_rounds}
        )

    monkeypatch.setattr("fp4bench.runner.run_experiment", fake_run_experiment)
    return calls


def _experiment(*args, **kwargs):
    return mx.experiment.get_raw_f()(*args, **kwargs)


def test_experiment_runs_the_named_study(runner_calls, monkeypatch):
    monkeypatch.setattr(mx, "results", FakeVolume())
    _experiment("full", "full-1")
    _experiment("smoke", "smoke-1")
    assert runner_calls[0]["study"] == STUDIES["full"]
    assert runner_calls[1]["study"] == STUDIES["smoke"]
    assert str(runner_calls[0]["run_dir"]) == f"{settings.RESULTS_DIR}/full-1"
    assert runner_calls[0]["rerun_rounds"] == ()


@pytest.mark.parametrize(
    "mode, study",
    [
        ("smoke-nf", STUDIES["smoke-nf"]),
        ("expb", STUDIES["expb"]),
        ("smoke-c", STUDIES["smoke-c"]),
        ("expc", STUDIES["expc"]),
    ],
)
def test_experiment_runs_every_study_with_its_rounds(runner_calls, monkeypatch, mode, study):
    monkeypatch.setattr(mx, "results", FakeVolume())
    _experiment(mode, f"{mode}-1")
    _experiment(mode, f"{mode}-1", rounds=10)
    assert runner_calls[0]["study"] == study == STUDIES[mode]
    assert runner_calls[1]["study"] == dataclasses.replace(study, rounds=10)
    assert str(runner_calls[0]["run_dir"]) == f"{settings.RESULTS_DIR}/{mode}-1"


def test_experiment_rounds_override_only_rounds(runner_calls, monkeypatch):
    monkeypatch.setattr(mx, "results", FakeVolume())
    _experiment("full", "full-1", rounds=10)
    _experiment("full", "full-1", rounds=0)
    assert runner_calls[0]["study"] == dataclasses.replace(STUDIES["full"], rounds=10)
    assert runner_calls[1]["study"] == STUDIES["full"]


def test_experiment_passes_rerun_rounds_as_a_tuple(runner_calls, monkeypatch):
    monkeypatch.setattr(mx, "results", FakeVolume())
    _experiment("full", "full-1", rounds=10, rerun_rounds=[2, 7])
    assert runner_calls[0]["rerun_rounds"] == (2, 7)


def test_experiment_hands_the_code_version_to_the_manifest(monkeypatch):
    from fp4bench import manifest

    seen = []
    monkeypatch.setattr(
        "fp4bench.runner.run_experiment",
        lambda *a, **k: seen.append(manifest.CodeVersion.from_env()),
    )
    monkeypatch.setattr(mx, "results", FakeVolume())
    monkeypatch.setenv(manifest.CODE_COMMIT_ENV, "stale")
    monkeypatch.setenv(manifest.CODE_DIRTY_ENV, "1")
    _experiment("full", "full-1", code_commit="d" * 40, code_dirty=False)
    _experiment("full", "full-1")
    assert seen == [manifest.CodeVersion("d" * 40, False), manifest.CodeVersion(None, None)]


def test_experiment_commit_failures_never_abort_the_run(runner_calls, monkeypatch, capsys):
    volume = FakeVolume(fail=True)
    monkeypatch.setattr(mx, "results", volume)
    _experiment("smoke", "smoke-1")
    runner_calls[0]["commit"]()
    assert volume.commits == 1
    assert "commit failed" in capsys.readouterr().err


def test_prepare_data_passes_force_through(monkeypatch):
    seen = []
    monkeypatch.setattr("fp4bench.prep.prepare_all", lambda force=False: seen.append(force) or {})
    monkeypatch.setattr(mx, "data", FakeVolume())
    raw = mx.prepare_data.get_raw_f()
    raw()
    raw(force=True)
    assert seen == [False, True]


def test_prepare_expc_data_passes_force_through_and_then_commits(monkeypatch):
    events = []
    monkeypatch.setattr(
        "fp4bench.prep.prepare_expc",
        lambda force=False: events.append(("prepare_expc", force)) or {"ok": 1},
    )
    monkeypatch.setattr(mx, "data", RecordingVolume(events))
    raw = mx.prepare_expc_data.get_raw_f()
    assert raw() == {"ok": 1}
    raw(force=True)
    assert events == [("prepare_expc", False), "commit", ("prepare_expc", True), "commit"]


class RecordingVolume:
    def __init__(self, events):
        self.events = events

    def commit(self):
        self.events.append("commit")


@pytest.mark.parametrize("fmt", list(Format))
def test_quantize_checkpoint_quantizes_the_format_and_only_then_commits_the_volume(
    monkeypatch, fmt
):
    events, summary = [], {"format": fmt}
    monkeypatch.setattr(
        "fp4bench.quantize.quantize", lambda f: events.append(("quantize", f)) or summary
    )
    monkeypatch.setattr(mx, "data", RecordingVolume(events))
    assert mx.quantize_checkpoint.get_raw_f()(fmt) == summary
    assert events == [("quantize", fmt), "commit"] and events[0][1] is fmt


@pytest.mark.parametrize("kind", ["mx", "nv", "nvidia"])
def test_fetch_checkpoint_mirrors_the_kind_and_only_then_commits_the_volume(monkeypatch, kind):
    events, summary = [], {"kind": kind}
    monkeypatch.setattr(
        "fp4bench.prep.fetch_checkpoint",
        lambda k, repo, rev: events.append(("fetch", k, repo, rev)) or summary,
    )
    monkeypatch.setattr(mx, "data", RecordingVolume(events))
    assert mx.fetch_checkpoint.get_raw_f()(kind, "someone/name", "a" * 40) == summary
    assert events == [("fetch", kind, "someone/name", "a" * 40), "commit"]


@pytest.mark.parametrize("kind, private", [("mx", True), ("nv", True), ("nv", False)])
def test_publish_checkpoint_publishes_the_kind_and_then_commits_the_readme(
    monkeypatch, kind, private
):
    events = []
    monkeypatch.setattr(
        "fp4bench.publish.publish",
        lambda k, repo, priv: (
            events.append(("publish", k, repo, priv)) or "https://huggingface.co/x/commit/abc"
        ),
    )
    monkeypatch.setattr(mx, "data", RecordingVolume(events))
    url = mx.publish_checkpoint.get_raw_f()(kind, "someone/name", private)
    assert url == "https://huggingface.co/x/commit/abc"
    assert events == [("publish", kind, "someone/name", private), "commit"]


@pytest.fixture
def results_dir(monkeypatch, tmp_path):
    """The fp4-results mount point, here a temporary directory."""
    monkeypatch.setattr(settings, "RESULTS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_microbench(monkeypatch):
    """fp4bench.gpu.microbench, which imports the GPU stack, replaced by one with a fake run."""
    calls, results = [], {"cells": [{"impl": "vllm_cutedsl"}], "wall_s": 410.5}

    def run(out_path: str, **kwargs):
        calls.append((out_path, kwargs))
        Path(out_path).write_text(json.dumps(results, indent=1))
        return results

    gpu_microbench = types.ModuleType("fp4bench.gpu.microbench")
    gpu_microbench.__dict__["run"] = run
    monkeypatch.setitem(sys.modules, "fp4bench.gpu.microbench", gpu_microbench)
    monkeypatch.setattr(
        "fp4bench.microbench.summarize", lambda r: f"# summary of {len(r['cells'])} cells\n"
    )
    return types.SimpleNamespace(calls=calls, results=results)


def test_kernel_microbench_runs_the_preregistered_set_and_commits_both_files(
    results_dir, fake_microbench, monkeypatch
):
    events = []
    monkeypatch.setattr(mx, "results", RecordingVolume(events))
    mx.kernel_microbench.get_raw_f()("microbench-1")
    run_dir = results_dir / "microbench-1"
    assert fake_microbench.calls == [(str(run_dir / "results.json"), {})]
    assert json.loads((run_dir / "results.json").read_text()) == fake_microbench.results
    assert (run_dir / "summary.md").read_text() == "# summary of 1 cells\n"
    assert events == ["commit"]


def test_kernel_microbench_commits_only_after_both_files_are_written(
    results_dir, fake_microbench, monkeypatch
):
    seen = []

    class CheckingVolume:
        def commit(self):
            seen.append(sorted(p.name for p in (results_dir / "mb-2").iterdir()))

    monkeypatch.setattr(mx, "results", CheckingVolume())
    mx.kernel_microbench.get_raw_f()("mb-2")
    assert seen == [["results.json", "summary.md"]]


@pytest.mark.parametrize("fn", ["kernel_microbench", "profile_sessions"])
def test_microbench_and_profile_refuse_a_run_directory_that_already_has_files(
    results_dir, fake_microbench, monkeypatch, fn
):
    profiled = []
    monkeypatch.setattr("fp4bench.profiling.run_profiles", lambda *a, **k: profiled.append(a))
    volume = FakeVolume()
    monkeypatch.setattr(mx, "results", volume)
    (results_dir / "used-1").mkdir()
    (results_dir / "used-1" / "results.json").write_text("{}")
    with pytest.raises(FileExistsError, match="used-1"):
        getattr(mx, fn).get_raw_f()("used-1")
    assert fake_microbench.calls == [] and profiled == [] and volume.commits == 0
    assert (results_dir / "used-1" / "results.json").read_text() == "{}"


def test_kernel_microbench_accepts_an_existing_empty_run_directory(
    results_dir, fake_microbench, monkeypatch
):
    monkeypatch.setattr(mx, "results", FakeVolume())
    (results_dir / "mb-3").mkdir()
    mx.kernel_microbench.get_raw_f()("mb-3")
    assert len(fake_microbench.calls) == 1


@pytest.fixture
def profile_calls(monkeypatch):
    calls = []

    def fake_run_profiles(out_dir, sessions=None, commit=None, work_dir=None):
        calls.append(
            {"out_dir": out_dir, "sessions": sessions, "commit": commit, "work_dir": work_dir}
        )
        return {"sessions": {}}

    monkeypatch.setattr("fp4bench.profiling.run_profiles", fake_run_profiles)
    return calls


def test_profile_sessions_profiles_every_default_session_into_the_run_directory(
    results_dir, profile_calls, monkeypatch
):
    monkeypatch.setattr(mx, "results", FakeVolume())
    assert mx.profile_sessions.get_raw_f()("profile-2") is None
    (call,) = profile_calls
    assert call["out_dir"] == results_dir / "profile-2"
    assert call["sessions"] is None and call["work_dir"] is None


def test_profile_sessions_commit_failures_never_abort_the_run(
    results_dir, profile_calls, monkeypatch, capsys
):
    volume = FakeVolume(fail=True)
    monkeypatch.setattr(mx, "results", volume)
    mx.profile_sessions.get_raw_f()("profile-2")
    profile_calls[0]["commit"]()
    assert volume.commits == 1
    assert "commit failed" in capsys.readouterr().err


def test_run_profiles_takes_what_profile_sessions_passes():
    import inspect

    from fp4bench import profiling

    params = inspect.signature(profiling.run_profiles).parameters
    assert next(iter(params)) == "out_dir" and "commit" in params
    assert params["sessions"].default is profiling.DEFAULT_SESSIONS


class FakeApp:
    """App.run as a context manager; `events` records what ran inside which app."""

    def __init__(self, interruptible: bool = False):
        self.events: list[tuple] = []
        self.detached: bool | None = None
        self.interruptible = interruptible

    @contextlib.contextmanager
    def run(self, detach: bool = False):
        self.events.append(("enter", detach))
        self.detached = detach
        try:
            yield self
        except KeyboardInterrupt:
            if not self.interruptible:
                raise
        finally:
            self.detached = None
            self.events.append(("exit", detach))


class FakeCall:
    object_id = "fc-FAKE123"

    def __init__(self, fn: "FakeFunction"):
        self.fn = fn

    def get(self):
        self.fn.app.events.append(("get", self.fn.name, self.fn.app.detached))
        return self.fn.result


class FakeFunction:
    def __init__(self, app: FakeApp, name: str, result=None, interrupt: bool = False):
        self.app, self.name, self.result, self.interrupt = app, name, result, interrupt

    def remote(self, **kwargs):
        self.app.events.append(("remote", self.name, self.app.detached, kwargs))
        if self.interrupt:
            raise KeyboardInterrupt
        return self.result

    def spawn(self, **kwargs):
        self.app.events.append(("spawn", self.name, self.app.detached, kwargs))
        return FakeCall(self)


FUNCTIONS = (
    "env_check",
    "prepare_data",
    "prepare_expc_data",
    "quantize_checkpoint",
    "fetch_checkpoint",
    "check_checkpoints",
    "publish_checkpoint",
    "experiment",
    "kernel_microbench",
    "profile_sessions",
)


@pytest.fixture
def fake_app(monkeypatch):
    app = FakeApp()
    monkeypatch.setattr(mx, "app", app)
    for name in FUNCTIONS:
        monkeypatch.setattr(mx, name, FakeFunction(app, name, result={"from": name}))
    return app


def test_a_run_is_spawned_in_a_detached_app_with_the_code_version(fake_app):
    code = CodeVersion("a" * 40, True)
    call_id = mx.ModalExecutor().run("full", "full-1", 10, (7, 9), code)
    assert call_id == "fc-FAKE123"
    assert fake_app.events == [
        ("enter", True),
        (
            "spawn",
            "experiment",
            True,
            {
                "study": "full",
                "run_id": "full-1",
                "rounds": 10,
                "rerun_rounds": (7, 9),
                "code_commit": "a" * 40,
                "code_dirty": True,
            },
        ),
        ("exit", True),
    ]


@pytest.mark.parametrize(
    "method, fn", [("microbench", "kernel_microbench"), ("profile", "profile_sessions")]
)
def test_microbench_and_profile_are_spawned_in_a_detached_app(fake_app, method, fn):
    assert getattr(mx.ModalExecutor(), method)("x-2") == "fc-FAKE123"
    assert fake_app.events == [
        ("enter", True),
        ("spawn", fn, True, {"run_id": "x-2"}),
        ("exit", True),
    ]


@pytest.mark.parametrize(
    "args, fn, kwargs",
    [
        ((False, True), "prepare_data", {"force": True}),
        ((True, False), "prepare_expc_data", {"force": False}),
    ],
)
def test_prepare_is_spawned_detached_and_waited_for(fake_app, args, fn, kwargs):
    spawned = []
    expc, force = args
    assert mx.ModalExecutor().prepare(expc, force, spawned.append) == {"from": fn}
    assert spawned == ["fc-FAKE123"]
    assert fake_app.events == [
        ("enter", True),
        ("spawn", fn, True, kwargs),
        ("get", fn, True),
        ("exit", True),
    ]


def test_quantize_is_spawned_detached_and_waited_for(fake_app):
    spawned = []
    assert mx.ModalExecutor().quantize(Format.NVFP4, spawned.append) == {
        "from": "quantize_checkpoint"
    }
    assert spawned == ["fc-FAKE123"] and [e[0] for e in fake_app.events] == [
        "enter",
        "spawn",
        "get",
        "exit",
    ]
    assert fake_app.events[1] == ("spawn", "quantize_checkpoint", True, {"fmt": "nvfp4"})


@pytest.mark.parametrize(
    "method, args, fn, kwargs",
    [
        ("env", (), "env_check", {}),
        ("check", (), "check_checkpoints", {}),
        (
            "publish",
            ("nv", "u/n", True),
            "publish_checkpoint",
            {"kind": "nv", "repo_id": "u/n", "private": True},
        ),
        (
            "fetch",
            ("nvidia", "u/n", "b" * 40),
            "fetch_checkpoint",
            {"kind": "nvidia", "repo_id": "u/n", "revision": "b" * 40},
        ),
    ],
)
def test_the_short_steps_wait_for_their_result_in_an_ordinary_app(
    fake_app, method, args, fn, kwargs
):
    assert getattr(mx.ModalExecutor(), method)(*args) == {"from": fn}
    assert fake_app.events == [("enter", False), ("remote", fn, False, kwargs), ("exit", False)]


def test_an_interrupted_step_returns_none(monkeypatch):
    app = FakeApp(interruptible=True)
    monkeypatch.setattr(mx, "app", app)
    monkeypatch.setattr(mx, "env_check", FakeFunction(app, "env_check", interrupt=True))
    assert mx.ModalExecutor().env() is None


def test_download_is_modal_volume_get_into_dest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mx.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    mx.ModalExecutor().download("full-1", tmp_path / "results", False)
    mx.ModalExecutor().download("full-1", tmp_path / "results", True)
    assert (tmp_path / "results").is_dir()
    base = [sys.executable, "-m", "modal", "volume", "get"]
    assert calls == [
        ([*base, "fp4-results", "full-1", str(tmp_path / "results")], {"check": True}),
        ([*base, "--force", "fp4-results", "full-1", str(tmp_path / "results")], {"check": True}),
    ]


def test_app_run_hands_detach_to_the_same_run_app_as_modal_run_detach(monkeypatch):
    import modal.cli.run
    import modal.runner

    assert inspect.signature(modal.App.run).parameters["detach"].default is False
    assert getattr(modal.runner.run_app, "__wrapped__", None) is modal.runner._run_app
    assert 'detach=ctx.obj["detach"]' in inspect.getsource(modal.cli.run)
    assert "APP_STATE_DETACHED if detach else api_pb2.APP_STATE_EPHEMERAL" in inspect.getsource(
        modal.runner._run_app
    )
    seen = []

    @contextlib.asynccontextmanager
    async def run_app(app, *, client=None, detach=False, interactive=False, environment_name=None):
        seen.append(detach)
        yield app

    monkeypatch.setattr(modal.runner, "_run_app", run_app)
    with mx.app.run(detach=True):
        pass
    with mx.app.run():
        pass
    assert seen == [True, False]


def test_the_local_executor_runs_the_runner_in_a_child_with_the_local_directories(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(local.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    monkeypatch.setenv(CODE_COMMIT_ENV, "stale")
    executor = local.LocalExecutor(tmp_path / "data", tmp_path / "res", tmp_path / "scratch")
    run_dir = executor.run("smoke-nf", "nf-1", 2, (1,), CodeVersion("c" * 40, False))
    assert run_dir == (tmp_path / "res" / "nf-1").resolve()
    ((argv, kwargs),) = calls
    assert argv[:3] == [sys.executable, "-m", "fp4bench.executors.local"]
    assert json.loads(argv[3]) == {
        "study": "smoke-nf",
        "run_dir": str(run_dir),
        "rounds": 2,
        "rerun_rounds": [1],
    }
    env = kwargs.pop("env")
    assert kwargs == {"check": True}
    assert env[settings.DATA_DIR_ENV] == str((tmp_path / "data").resolve())
    assert env[settings.LOCAL_DIR_ENV] == str((tmp_path / "scratch").resolve())
    assert (env[CODE_COMMIT_ENV], env[CODE_DIRTY_ENV]) == ("c" * 40, "0")
    assert env["PATH"] == os.environ["PATH"]
    assert CODE_COMMIT_ENV not in executor.env(CodeVersion(None, None))


def test_the_child_runs_the_study_it_is_given(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "fp4bench.runner.run_experiment",
        lambda run_dir, study, commit, rerun_rounds: seen.append(
            (run_dir, study, commit(), rerun_rounds)
        ),
    )
    spec = {"study": "expc", "run_dir": str(tmp_path / "c-1"), "rounds": 10, "rerun_rounds": [3]}
    local.main([json.dumps(spec)])
    local.main([json.dumps({**spec, "rounds": 0, "rerun_rounds": []})])
    assert seen == [
        (tmp_path / "c-1", dataclasses.replace(STUDIES["expc"], rounds=10), None, (3,)),
        (tmp_path / "c-1", STUDIES["expc"], None, ()),
    ]


def _in_a_child(code: str, env: dict[str, str]) -> str:
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_the_childs_settings_and_model_paths_are_the_local_directories(tmp_path):
    code = (
        "from fp4bench import settings; from fp4bench.studies import model; "
        "print(settings.DATA_DIR, settings.M1_PROMPTS_PATH, settings.C_PROMPTS_PATH, "
        "model.MX_MODEL_DIR, model.TREATMENTS['NVa'].volume_dir, "
        "model.TREATMENTS['NVa'].model_dir, settings.RESULTS_DIR)"
    )
    out = _in_a_child(code, {settings.DATA_DIR_ENV: "/d", settings.LOCAL_DIR_ENV: "/s"})
    assert out.split() == [
        "/d",
        "/d/m1_prompts.json",
        "/d/c_prompts.json",
        "/d/qwen3-32b-mxfp4",
        "/d/qwen3-32b-nvfp4",
        "/s/qwen3-32b-nvfp4",
        "/results",
    ]
    assert _in_a_child(code, {}).split()[:4] == [
        "/data",
        "/data/m1_prompts.json",
        "/data/c_prompts.json",
        "/data/qwen3-32b-mxfp4",
    ]


def test_modal_mounts_the_data_volume_where_the_container_reads_it_whatever_the_local_env():
    code = "from fp4bench.executors import modal; print(sorted(map(str, modal.VOLUMES)))"
    out = _in_a_child(code, {settings.DATA_DIR_ENV: "/elsewhere"})
    assert json.loads(out.replace("'", '"')) == sorted(
        [settings.HF_CACHE, settings.VLLM_CACHE, "/data", "/results"]
    )


def test_a_local_run_writes_its_manifest_line_from_the_local_directories(tmp_path):
    executor = local.LocalExecutor(tmp_path / "data", tmp_path / "res", tmp_path / "scratch")
    with pytest.raises(subprocess.CalledProcessError):
        executor.run("smoke-nf", "nf-1", 0, (), CodeVersion("e" * 40, True))
    (line,) = load_rows(tmp_path / "res" / "nf-1" / "manifests.jsonl", ManifestLine)
    assert (line.study, line.code_commit, line.code_dirty) == ("smoke-nf", "e" * 40, True)
    data = str((tmp_path / "data").resolve())
    assert line.inputs is not None and data in str(line.inputs["m1_prompts_sha256"])
    assert line.problems and not (tmp_path / "scratch").exists()
