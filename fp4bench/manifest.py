# METHODOLOGY.md#stack
import importlib.metadata
import os
import subprocess
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from fp4bench import settings
from fp4bench.core.types import ProbeError, error_as_object, error_as_text, probe

GPU_FIELDS = ("name", "uuid", "compute_cap", "driver_version", "memory.total", "power.limit")
PACKAGES = (
    "vllm",
    "flashinfer-python",
    "flashinfer-cubin",
    "torch",
    "nvidia-cutlass-dsl",
    "transformers",
    "compressed-tensors",
    "httpx",
    "rich",
    "psutil",
)


def gpu_info() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = out.strip().splitlines()
    fields = [x.strip() for x in lines[0].split(",")] if lines else []
    if len(fields) != len(GPU_FIELDS):
        raise ValueError(
            f"nvidia-smi returned {len(fields)} fields, expected {len(GPU_FIELDS)} "
            f"({','.join(GPU_FIELDS)}): {out.strip()!r}"
        )
    return dict(zip(GPU_FIELDS, fields, strict=True))


def package_versions() -> dict[str, str | None]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def lib_commit() -> str:
    out = subprocess.run(
        ["git", "-C", settings.LIB_DIR, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(
            f"git rev-parse HEAD failed in {settings.LIB_DIR} "
            f"(exit {out.returncode}): {out.stderr.strip()}"
        )
    return out.stdout.strip()


def _torch_cuda() -> str | None:
    import torch

    return torch.version.cuda


def cuda_version() -> str | None:
    cuda = probe(_torch_cuda)
    return None if isinstance(cuda, ProbeError) else cuda


CODE_COMMIT_ENV = "FP4BENCH_CODE_COMMIT"
CODE_DIRTY_ENV = "FP4BENCH_CODE_DIRTY"
REPO = Path(__file__).resolve().parents[1]


class CodeVersion(NamedTuple):
    """The fp4bench checkout a run was started from; None where git could not tell."""

    commit: str | None = None
    dirty: bool | None = None

    def env(self) -> dict[str, str]:
        out = {}
        if self.commit is not None:
            out[CODE_COMMIT_ENV] = self.commit
        if self.dirty is not None:
            out[CODE_DIRTY_ENV] = "1" if self.dirty else "0"
        return out

    def export(self, environ: MutableMapping[str, str] = os.environ) -> None:
        for key in (CODE_COMMIT_ENV, CODE_DIRTY_ENV):
            environ.pop(key, None)
        environ.update(self.env())

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "CodeVersion":
        dirty = environ.get(CODE_DIRTY_ENV)
        return cls(environ.get(CODE_COMMIT_ENV), None if dirty is None else dirty == "1")


def code_version(repo: Path = REPO) -> CodeVersion:
    """`git rev-parse HEAD`, and whether `git status --porcelain` lists anything."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return out.stdout if out.returncode == 0 else None

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return CodeVersion(
        (commit or "").strip() or None, None if status is None else bool(status.strip())
    )


def collect_manifest() -> dict:
    """Never raises: a failed probe is recorded as an error value that verify_manifest flags."""
    gpu, commit = probe(gpu_info), probe(lib_commit)
    code = CodeVersion.from_env()
    return {
        "utc": datetime.now(UTC).isoformat(),
        "image": settings.VLLM_IMAGE,
        "vllm_build_commit": os.environ.get("VLLM_BUILD_COMMIT"),
        "llm_inference_bench_commit": error_as_text(commit),
        "gpu": error_as_object(gpu),
        "packages": package_versions(),
        "cuda": cuda_version(),
        "code_commit": code.commit,
        "code_dirty": code.dirty,
    }


def _version_matches(got: str | None, want: str) -> bool:
    return got is not None and got.split("+", 1)[0] == want


def _gpu_problems(gpu: dict) -> list[str]:
    if not gpu or "error" in gpu:
        return [f"GPU query (nvidia-smi) failed: {gpu.get('error') if gpu else 'no GPU info'}"]
    problems = []
    if gpu.get("name") != "NVIDIA B200":
        problems.append(f"expected an NVIDIA B200, got {gpu.get('name')}")
    if gpu.get("compute_cap") != "10.0":
        problems.append(f"expected compute capability 10.0, got {gpu.get('compute_cap')}")
    return problems


def verify_manifest(manifest: dict) -> list[str]:
    problems = _gpu_problems(manifest.get("gpu") or {})
    packages = manifest.get("packages") or {}
    for pkg, want in settings.EXPECTED_VERSIONS.items():
        got = packages.get(pkg)
        if not _version_matches(got, want):
            problems.append(f"{pkg}: expected {want}, got {got}")
    if manifest.get("vllm_build_commit") != settings.VLLM_COMMIT:
        problems.append(
            f"vllm build commit: expected {settings.VLLM_COMMIT}, "
            f"got {manifest.get('vllm_build_commit')}"
        )
    if manifest.get("llm_inference_bench_commit") != settings.LIB_COMMIT:
        problems.append(
            f"llm-inference-bench: expected {settings.LIB_COMMIT}, "
            f"got {manifest.get('llm_inference_bench_commit')}"
        )
    return problems
