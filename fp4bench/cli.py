import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from fp4bench.core.types import CheckpointKind, ExecutorKind, FetchKind, Format
from fp4bench.manifest import CodeVersion, code_version
from fp4bench.studies.base import Study
from fp4bench.studies.registry import STUDIES, study_to_run

if TYPE_CHECKING:
    from fp4bench.executors.local import LocalExecutor
    from fp4bench.executors.modal import ModalExecutor

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
REVISION = re.compile(r"[0-9a-f]{40}")
FORMATS = tuple(fmt.value for fmt in Format)
PUBLISH_KINDS = (CheckpointKind.MX.value, CheckpointKind.NV.value)
FETCH_KINDS = tuple(kind.value for kind in FetchKind)
EXECUTORS = tuple(executor.value for executor in ExecutorKind)


def modal_executor() -> "ModalExecutor":
    from fp4bench.executors.modal import ModalExecutor

    return ModalExecutor()


def local_executor(data_dir: Path, results_dir: Path, local_dir: Path) -> "LocalExecutor":
    from fp4bench.executors.local import LocalExecutor

    return LocalExecutor(data_dir, results_dir, local_dir)


def _matching(pattern: re.Pattern, what: str):
    def check(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
        if value is not None and not pattern.fullmatch(value):
            raise click.BadParameter(f"{value!r} is not {what}")
        return value

    return check


check_run_id = _matching(
    RUN_ID,
    "a plain directory name (letters, digits, '.', '_', '-'; starting with a letter or digit)",
)


def as_format(_ctx: click.Context, _param: click.Parameter, value: str) -> Format:
    return Format(value)


check_repo_id = _matching(REPO_ID, "<user-or-org>/<name>")
check_revision = _matching(REVISION, "a full 40-character lowercase hex commit sha")


def parse_rerun_rounds(text: str, n_rounds: int) -> tuple[int, ...]:
    """'1,3' -> (1, 3): sorted, de-duplicated rounds of 0..n_rounds-1."""
    if not text.strip():
        return ()
    parsed = set()
    for part in (p.strip() for p in text.split(",")):
        if not part:
            raise click.BadParameter(
                f"{text!r} has an empty entry (expected e.g. 1,3)", param_hint="'--rerun-rounds'"
            )
        try:
            value = int(part)
        except ValueError:
            raise click.BadParameter(
                f"{part!r} in {text!r} is not an integer", param_hint="'--rerun-rounds'"
            ) from None
        if not 0 <= value < n_rounds:
            raise click.BadParameter(
                f"round {value} is outside the study's rounds 0..{n_rounds - 1}",
                param_hint="'--rerun-rounds'",
            )
        parsed.add(value)
    return tuple(sorted(parsed))


def check_rounds(name: str, rounds: int) -> Study:
    """The study a run runs; the executors run it through the same study_to_run."""
    try:
        return study_to_run(name, rounds)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--rounds'") from None


def _program() -> str:
    return click.get_current_context().find_root().info_name or "fp4bench"


def _result(value: Any) -> Any:
    if value is None:
        raise click.Abort
    return value


def _print_json(value: Any) -> None:
    click.echo(json.dumps(_result(value), indent=2))


def _waiting(call_id: str) -> None:
    click.echo(
        f"spawned {call_id}; waiting for its result (if this command stops, the call "
        "goes on in its detached app)"
    )


def _spawned(call_id: str | None, run_id: str) -> None:
    click.echo(
        f"spawned {_result(call_id)} in a detached app: it keeps running after this command exits"
    )
    click.echo(f"when done: {_program()} download {run_id}")


def _describe(code: CodeVersion) -> str:
    dirty = {True: " with uncommitted changes", False: "", None: " (unknown changes)"}
    return f"code: {code.commit or 'unknown commit'}{dirty[code.dirty]}"


@click.group()
def main() -> None:
    """FP4 decode benchmark: NVFP4 against MXFP4 on Qwen3-32B (EXPERIMENT.md).

    \b
    A study, in order:
      env                         GPU and package versions in the experiment's image
      prepare [--expc]            model, ShareGPT and prompt files on the fp4-data volume
      quantize --fmt FMT          the MXFP4 and the NVFP4 checkpoint
      check                       the checkpoint report (G2, G5a)
      publish / fetch             the published checkpoints, which runs serve
      run STUDY --run-id ID       a study, on Modal or here (--executor local)
      download ID                 its run directory, from the fp4-results volume
      analyze results/ID          verdict, gates, summary.md and figures

    Modal resolves every function's secrets on each start: a secret named huggingface-secret
    must exist (an HF token, or UNUSED=1 when every repo involved is public).
    """


@main.command()
def env() -> None:
    """GPU, driver and package versions in the experiment's image (gate G1)."""
    _print_json(modal_executor().env())


@main.command()
@click.option(
    "--expc",
    is_flag=True,
    help="Build Experiment C's cell prompts from what prepare put on the volume.",
)
@click.option("--force", is_flag=True, help="Rebuild the prompt files even if they exist.")
def prepare(expc: bool, force: bool) -> None:
    """Download the pinned BF16 model and ShareGPT and build the prompt files (fp4-data)."""
    _print_json(modal_executor().prepare(expc, force, _waiting))


@main.command()
@click.option("--fmt", type=click.Choice(FORMATS), required=True, callback=as_format)
def quantize(fmt: Format) -> None:
    """Quantize the BF16 model with llm-compressor (the first one also stores the BF16 NLL)."""
    _print_json(modal_executor().quantize(fmt, _waiting))


@main.command()
def check() -> None:
    """The checkpoint report (G2, G5a), also written to results/checkpoint_report.json."""
    report = _result(modal_executor().check())
    Path("results").mkdir(exist_ok=True)
    Path("results/checkpoint_report.json").write_text(json.dumps(report, indent=2))
    _print_json(report)


@main.command()
@click.option("--kind", type=click.Choice(PUBLISH_KINDS), required=True)
@click.option("--repo-id", required=True, callback=check_repo_id, metavar="USER/NAME")
@click.option("--public", is_flag=True, help="Publish a public repo (private by default).")
def publish(kind: str, repo_id: str, public: bool) -> None:
    """Upload a checkpoint to the Hub; prints the commit URL whose revision fetch takes."""
    click.echo(_result(modal_executor().publish(kind, repo_id, not public)))


@main.command()
@click.option(
    "--kind",
    type=click.Choice(FETCH_KINDS),
    required=True,
    help="nvidia: NVIDIA's NVFP4 checkpoint, for the kernel-scan smoke.",
)
@click.option("--repo-id", required=True, callback=check_repo_id, metavar="USER/NAME")
@click.option("--revision", required=True, callback=check_revision, metavar="SHA")
def fetch(kind: str, repo_id: str, revision: str) -> None:
    """Make the volume's checkpoint directory an exact mirror of a published revision."""
    _print_json(modal_executor().fetch(kind, repo_id, revision))


@main.command(epilog=f"STUDY: {', '.join(STUDIES)}.")
@click.argument("study", type=click.Choice(list(STUDIES)), metavar="STUDY")
@click.option(
    "--run-id",
    required=True,
    callback=check_run_id,
    help="The run directory; a run resumes from it.",
)
@click.option(
    "--rounds",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Rounds to run; 0 keeps the study's. Any other value must be its pre-registered "
    "count or its registered extension's; pass the same value on every restart.",
)
@click.option(
    "--rerun-rounds",
    default="",
    metavar="2,4",
    help="Complete rounds to run again, whole (e.g. after a G4 throttling flag).",
)
@click.option(
    "--executor",
    type=click.Choice(EXECUTORS),
    default=ExecutorKind.MODAL.value,
    show_default=True,
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="local: the data directory (prompt files, model directories).",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="local: where the run directory goes [default: results].",
)
@click.option(
    "--local-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="local: scratch the run copies the checkpoints to [default: local].",
)
def run(
    study: str,
    run_id: str,
    rounds: int,
    rerun_rounds: str,
    executor: str,
    data_dir: Path | None,
    results_dir: Path | None,
    local_dir: Path | None,
) -> None:
    """Run STUDY: its rounds that are not complete, plus --rerun-rounds.

    On Modal the run is spawned in a detached app and this command returns at once."""
    rerun = parse_rerun_rounds(rerun_rounds, check_rounds(study, rounds).rounds)
    if executor == ExecutorKind.MODAL:
        if (data_dir, results_dir, local_dir) != (None, None, None):
            raise click.UsageError(
                "--data-dir, --results-dir and --local-dir are for --executor local"
            )
        code = code_version()
        click.echo(_describe(code))
        _spawned(modal_executor().run(study, run_id, rounds, rerun, code), run_id)
        return
    if data_dir is None:
        raise click.UsageError("--executor local needs --data-dir")
    local_dir = local_dir or Path("local")
    if local_dir.resolve() == data_dir.resolve():
        raise click.BadParameter("must not be the data directory", param_hint="'--local-dir'")
    code = code_version()
    click.echo(_describe(code))
    local = local_executor(data_dir, results_dir or Path("results"), local_dir)
    try:
        run_dir = local.run(study, run_id, rounds, rerun, code)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"the run exited with status {exc.returncode}") from None
    click.echo(f"done: {run_dir}")


@main.command()
@click.option(
    "--run-id",
    required=True,
    callback=check_run_id,
    help="A fresh run directory: one with files is refused.",
)
def microbench(run_id: str) -> None:
    """The FP4 GEMM and activation-quant kernel microbenchmark (§14), on one B200."""
    _spawned(modal_executor().microbench(run_id), run_id)


@main.command()
@click.option(
    "--run-id",
    required=True,
    callback=check_run_id,
    help="A fresh run directory: one with files is refused.",
)
def profile(run_id: str) -> None:
    """Torch-profiler sessions of MX, NV and NV-nf (§15), from the fetched checkpoints."""
    _spawned(modal_executor().profile(run_id), run_id)


@main.command()
@click.argument("run_id", callback=check_run_id)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
)
@click.option("--force", is_flag=True, help="Overwrite files downloaded before.")
def download(run_id: str, dest: Path, force: bool) -> None:
    """Copy RUN_ID's directory from the fp4-results volume into --dest."""
    try:
        modal_executor().download(run_id, dest, force)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"modal volume get exited with status {exc.returncode}"
        ) from None


RUN_DIR = click.Path(exists=True, file_okay=False, path_type=Path)


def _analysed(action: Any) -> Any:
    try:
        return action()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None


@main.command()
@click.argument("run_dir", type=RUN_DIR)
@click.option(
    "--reference-run",
    type=RUN_DIR,
    help="Another run (for Experiment B, the main run) whose mean NLL per treatment "
    "the accuracy note puts next to this run's.",
)
def analyze(run_dir: Path, reference_run: Path | None) -> None:
    """Verdict, gates, summary.md and figures of the run in RUN_DIR, written into it."""
    from fp4bench.analysis.report import console_lines
    from fp4bench.analysis.run import analyze_run

    result = _analysed(lambda: analyze_run(run_dir, reference_run))
    click.echo("\n".join(console_lines(result)))


@main.group()
def plot() -> None:
    """The shareable figures (docs/figures), drawn from the runs' data."""


@plot.command()
@click.argument("main_run", type=RUN_DIR)
@click.argument("expb_run", type=RUN_DIR)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="[default: docs/figures/r_vs_concurrency.png]",
)
def overview(main_run: Path, expb_run: Path, out: Path | None) -> None:
    """NVFP4's decode-throughput advantage over MXFP4 against concurrency (main run and
    Experiment B)."""
    from fp4bench.analysis import figures
    from fp4bench.analysis.run import evaluate

    out = out or figures.OVERVIEW
    _analysed(lambda: figures.overview(evaluate(main_run), evaluate(expb_run), out))
    click.echo(f"wrote {out}")


@plot.command()
@click.argument("main_run", type=RUN_DIR)
@click.argument("expb_run", type=RUN_DIR)
@click.option(
    "--out-dir", type=click.Path(file_okay=False, path_type=Path), help="[default: docs/figures]"
)
def share(main_run: Path, expb_run: Path, out_dir: Path | None) -> None:
    """NVFP4 against MXFP4: the format and kernel panels, and the format panel alone."""
    from fp4bench.analysis import figures
    from fp4bench.analysis.run import evaluate

    two, one = _analysed(
        lambda: figures.share(evaluate(main_run), evaluate(expb_run), out_dir or figures.FIGURES)
    )
    click.echo(f"wrote {two} and {one}")


@plot.command("share-c")
@click.argument("expc_run", type=RUN_DIR)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="[default: docs/figures/nvfp4_vs_mxfp4_batch_vs_tokens_b200.png]",
)
def share_c(expc_run: Path, out: Path | None) -> None:
    """Experiment C: batch size against KV tokens."""
    from fp4bench.analysis import figures
    from fp4bench.analysis.run import evaluate

    out = out or figures.SHARE_C
    _analysed(lambda: figures.share_c(evaluate(expc_run), out))
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()
