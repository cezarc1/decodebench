"""The Modal app: images, volumes, functions (METHODOLOGY.md#modal). Importing contacts nothing."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from fp4bench import settings
from fp4bench.core.types import Format
from fp4bench.manifest import CodeVersion
from fp4bench.studies import model
from fp4bench.studies.registry import study_to_run

Mounts = dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]

app = modal.App("fp4-decode-bench")

hf_cache = modal.Volume.from_name("fp4-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("fp4-vllm-cache", create_if_missing=True)
data = modal.Volume.from_name("fp4-data", create_if_missing=True)
RESULTS_VOLUME = "fp4-results"
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
VOLUMES: Mounts = {
    settings.HF_CACHE: hf_cache,
    settings.VLLM_CACHE: vllm_cache,
    settings.CONTAINER_DATA_DIR: data,
    settings.RESULTS_DIR: results,
}
RESULTS_ONLY: Mounts = {settings.RESULTS_DIR: results}
hf_secret = modal.Secret.from_name("huggingface-secret")

base_image = (
    modal.Image.from_registry(settings.VLLM_IMAGE)
    .entrypoint([])
    .run_commands('ln -sf "$(command -v python3)" /usr/local/bin/python')
    .apt_install("git")
    .run_commands(
        f"git clone {settings.LIB_REPO} {settings.LIB_DIR} "
        f"&& git -C {settings.LIB_DIR} checkout {settings.LIB_COMMIT}",
        "python -m pip install httpx rich psutil",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": settings.HF_CACHE,
            "LLM_BENCH_NO_UPDATE_CHECK": "1",
        }
    )
)
vllm_image = base_image.add_local_python_source("fp4bench")


def torch_pin_check(version: str) -> str:
    """A shell command failing the image build unless torch is `version` (METHODOLOGY.md#image)."""
    code = (
        'import sys, torch; v = torch.__version__.split("+")[0]; '
        f'sys.exit(0 if v == "{version}" else '
        f'"torch " + torch.__version__ + " is installed, expected {version}")'
    )
    return f"python -c '{code}'"


quant_image = (
    base_image.run_commands(
        f"python -m pip install 'llmcompressor=={settings.LLMCOMPRESSOR_VERSION}'"
    )
    .run_commands(torch_pin_check(settings.EXPECTED_VERSIONS["torch"]))
    .add_local_python_source("fp4bench")
)

GIB = 1024
LOCAL_DISK = 512 * GIB  # METHODOLOGY.md#modal-disk


def safe_commit(commit: Callable[[], None]) -> Callable[[], None]:
    """A volume commit whose failure is logged, not raised (METHODOLOGY.md#modal-volumes)."""

    def wrapper() -> None:
        try:
            commit()
        except Exception as exc:
            print(f"warning: volume commit failed, continuing: {exc!r}", file=sys.stderr)

    return wrapper


def fresh_run_dir(run_id: str) -> Path:
    """The run directory of a step that does not resume (microbench, profile): it must be empty."""
    run_dir = Path(settings.RESULTS_DIR) / run_id
    if run_dir.is_dir() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already has files: pass a fresh --run-id")
    return run_dir


@app.function(image=vllm_image, gpu="B200", cpu=4, memory=16384, timeout=600)
def env_check() -> dict:
    from fp4bench.manifest import collect_manifest, verify_manifest

    manifest = collect_manifest()
    return {**manifest, "problems": verify_manifest(manifest)}


@app.function(image=vllm_image, cpu=8, memory=32 * GIB, timeout=3 * 3600, volumes=VOLUMES)
def prepare_data(force: bool = False) -> dict:
    from fp4bench.prep import prepare_all

    summary = prepare_all(force=force)
    data.commit()
    return summary


@app.function(image=vllm_image, cpu=8, memory=32 * GIB, timeout=3600, volumes=VOLUMES)
def prepare_expc_data(force: bool = False) -> dict:
    from fp4bench.prep import prepare_expc

    summary = prepare_expc(force=force)
    data.commit()
    return summary


@app.function(
    image=quant_image,
    gpu="B200",
    cpu=16,
    memory=256 * GIB,
    ephemeral_disk=LOCAL_DISK,
    timeout=3 * 3600,
    volumes=VOLUMES,
)
def quantize_checkpoint(fmt: Format) -> dict:
    from fp4bench.quantize import quantize

    summary = quantize(fmt)
    data.commit()
    return summary


@app.function(
    image=vllm_image, cpu=8, memory=32 * GIB, timeout=3600, volumes=VOLUMES, secrets=[hf_secret]
)
def fetch_checkpoint(kind: str, repo_id: str, revision: str) -> dict:
    from fp4bench import prep

    summary = prep.fetch_checkpoint(kind, repo_id, revision)
    data.commit()
    return summary


@app.function(image=vllm_image, cpu=4, memory=16 * GIB, timeout=600, volumes=VOLUMES)
def check_checkpoints() -> dict:
    from fp4bench.sanity import checkpoint_report

    return checkpoint_report(model.MX_MODEL_DIR, model.NV_MODEL_DIR)


@app.function(
    image=vllm_image, cpu=4, memory=16 * GIB, timeout=3600, volumes=VOLUMES, secrets=[hf_secret]
)
def publish_checkpoint(kind: str, repo_id: str, private: bool) -> str:
    from fp4bench import publish

    url = publish.publish(kind, repo_id, private)
    data.commit()
    return url


@app.function(
    image=vllm_image,
    gpu="B200",
    cpu=16,
    memory=64 * GIB,
    ephemeral_disk=LOCAL_DISK,
    timeout=10 * 3600,
    volumes=VOLUMES,
)
def experiment(
    study: str,
    run_id: str,
    rounds: int = 0,
    rerun_rounds: tuple[int, ...] = (),
    code_commit: str | None = None,
    code_dirty: bool | None = None,
) -> None:
    from fp4bench.runner import run_experiment

    CodeVersion(code_commit, code_dirty).export()
    run_experiment(
        Path(settings.RESULTS_DIR) / run_id,
        study_to_run(study, rounds),
        commit=safe_commit(results.commit),
        rerun_rounds=tuple(rerun_rounds),
    )


@app.function(
    image=vllm_image, gpu="B200", cpu=8, memory=64 * GIB, timeout=3600, volumes=RESULTS_ONLY
)
def kernel_microbench(run_id: str) -> None:
    from fp4bench.gpu.microbench import run
    from fp4bench.microbench import summarize

    run_dir = fresh_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run(str(run_dir / "results.json"))
    (run_dir / "summary.md").write_text(summarize(out))
    results.commit()


@app.function(
    image=vllm_image,
    gpu="B200",
    cpu=16,
    memory=64 * GIB,
    ephemeral_disk=LOCAL_DISK,
    timeout=3 * 3600,
    volumes=VOLUMES,
)
def profile_sessions(run_id: str) -> None:
    from fp4bench.profiling import run_profiles

    run_profiles(fresh_run_dir(run_id), commit=safe_commit(results.commit))


def _remote(fn: modal.Function, **kwargs: Any) -> Any:
    with modal.enable_output(), app.run():
        return fn.remote(**kwargs)
    return None


def _spawn(fn: modal.Function, **kwargs: Any) -> str | None:
    with modal.enable_output(), app.run(detach=True):
        return fn.spawn(**kwargs).object_id
    return None


def _spawn_and_wait(fn: modal.Function, spawned: Callable[[str], None], **kwargs: Any) -> Any:
    with modal.enable_output(), app.run(detach=True):
        call = fn.spawn(**kwargs)
        spawned(call.object_id)
        return call.get()
    return None


class ModalExecutor:
    """What the command line asks of Modal. A result of None: this process was interrupted."""

    def env(self) -> dict | None:
        return _remote(env_check)

    def prepare(self, expc: bool, force: bool, spawned: Callable[[str], None]) -> dict | None:
        return _spawn_and_wait(prepare_expc_data if expc else prepare_data, spawned, force=force)

    def quantize(self, fmt: Format, spawned: Callable[[str], None]) -> dict | None:
        return _spawn_and_wait(quantize_checkpoint, spawned, fmt=fmt)

    def check(self) -> dict | None:
        return _remote(check_checkpoints)

    def publish(self, kind: str, repo_id: str, private: bool) -> str | None:
        return _remote(publish_checkpoint, kind=kind, repo_id=repo_id, private=private)

    def fetch(self, kind: str, repo_id: str, revision: str) -> dict | None:
        return _remote(fetch_checkpoint, kind=kind, repo_id=repo_id, revision=revision)

    def run(
        self, study: str, run_id: str, rounds: int, rerun_rounds: tuple[int, ...], code: CodeVersion
    ) -> str | None:
        return _spawn(
            experiment,
            study=study,
            run_id=run_id,
            rounds=rounds,
            rerun_rounds=rerun_rounds,
            code_commit=code.commit,
            code_dirty=code.dirty,
        )

    def microbench(self, run_id: str) -> str | None:
        return _spawn(kernel_microbench, run_id=run_id)

    def profile(self, run_id: str) -> str | None:
        return _spawn(profile_sessions, run_id=run_id)

    def download(self, run_id: str, dest: Path, force: bool) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                *(["--force"] if force else []),
                RESULTS_VOLUME,
                run_id,
                str(dest),
            ],
            check=True,
        )
