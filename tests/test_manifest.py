import subprocess
import sys
import types

import pytest

from fp4bench import manifest as mf
from fp4bench import settings
from fp4bench.manifest import verify_manifest
from tests import REPO

GOOD = {
    "gpu": {"name": "NVIDIA B200", "compute_cap": "10.0", "uuid": "GPU-1"},
    "packages": {
        "vllm": "0.31.0",
        "flashinfer-python": "0.7.0.post1",
        "flashinfer-cubin": "0.7.0.post1",
        "torch": "2.13.0+cu130",
    },
    "llm_inference_bench_commit": settings.LIB_COMMIT,
    "vllm_build_commit": settings.VLLM_COMMIT,
    "cuda": "13.0",
}


def with_package(name: str, version: str | None) -> dict:
    return {**GOOD, "packages": {**GOOD["packages"], name: version}}


def test_verify_manifest_accepts_pinned_environment():
    assert verify_manifest(GOOD) == []


def test_verify_manifest_flags_wrong_gpu_versions_and_bench_commit():
    bad = {
        **GOOD,
        "gpu": {"name": "NVIDIA B300", "compute_cap": "10.3", "uuid": "GPU-2"},
        "packages": {**GOOD["packages"], "vllm": "0.32.0", "flashinfer-python": None},
        "llm_inference_bench_commit": "deadbeef",
    }
    problems = verify_manifest(bad)
    assert len(problems) == 5
    assert any("B200" in p for p in problems) and any("flashinfer-python" in p for p in problems)
    assert any("llm-inference-bench" in p for p in problems)


@pytest.mark.parametrize(
    "pkg,version",
    [
        ("vllm", "0.31.0rc1"),
        ("vllm", "0.31.0.post1"),
        ("flashinfer-python", "0.7.0.post10"),
        ("flashinfer-python", "0.7.0.post1.dev1"),
        ("flashinfer-cubin", "0.7.0.post10"),
        ("torch", "2.13.0.dev20260901+cu130"),
        ("torch", "2.130.0"),
    ],
)
def test_verify_manifest_rejects_version_prefix_lookalikes(pkg, version):
    problems = verify_manifest(with_package(pkg, version))
    assert len(problems) == 1 and pkg in problems[0]


@pytest.mark.parametrize("pkg,version", [("torch", "2.13.0+cu130"), ("vllm", "0.31.0+cu130")])
def test_verify_manifest_accepts_local_version_suffix(pkg, version):
    assert verify_manifest(with_package(pkg, version)) == []


def test_verify_manifest_pins_flashinfer_cubin():
    problems = verify_manifest(with_package("flashinfer-cubin", None))
    assert len(problems) == 1 and "flashinfer-cubin" in problems[0]


def test_verify_manifest_rejects_gb200_which_shares_compute_capability():
    problems = verify_manifest({**GOOD, "gpu": {**GOOD["gpu"], "name": "NVIDIA GB200"}})
    assert len(problems) == 1 and "B200" in problems[0] and "GB200" in problems[0]


@pytest.mark.parametrize("commit", [None, "db9527a", "0" * 40])
def test_verify_manifest_checks_vllm_build_commit(commit):
    problems = verify_manifest({**GOOD, "vllm_build_commit": commit})
    assert len(problems) == 1 and "vllm" in problems[0].lower() and "commit" in problems[0]


def test_verify_manifest_flags_collection_errors_without_raising():
    broken = {
        **GOOD,
        "gpu": {"error": "FileNotFoundError: nvidia-smi"},
        "llm_inference_bench_commit": "error: RuntimeError: not a git repository",
    }
    problems = verify_manifest(broken)
    assert any("nvidia-smi" in p for p in problems)
    assert any("not a git repository" in p for p in problems)


def test_verify_manifest_tolerates_missing_sections():
    assert verify_manifest({}) != []


NVIDIA_SMI_OUT = "NVIDIA B200, GPU-0a1b2c, 10.0, 580.126.09, 183359 MiB, 1000.00 W\n"


def fake_run(stdout="", returncode=0, stderr="", seen=None):
    def run(cmd, **kwargs):
        if seen is not None:
            seen.append(cmd)
        if returncode and kwargs.get("check"):
            raise subprocess.CalledProcessError(returncode, cmd, stdout, stderr)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return run


def test_gpu_info_parses_nvidia_smi_csv(monkeypatch):
    seen = []
    monkeypatch.setattr(mf.subprocess, "run", fake_run(NVIDIA_SMI_OUT, seen=seen))
    info = mf.gpu_info()
    assert info == {
        "name": "NVIDIA B200",
        "uuid": "GPU-0a1b2c",
        "compute_cap": "10.0",
        "driver_version": "580.126.09",
        "memory.total": "183359 MiB",
        "power.limit": "1000.00 W",
    }
    assert seen[0][0] == "nvidia-smi" and "--query-gpu=" + ",".join(mf.GPU_FIELDS) in seen[0]


@pytest.mark.parametrize(
    "out",
    ["NVIDIA B200, GPU-1, 10.0, 580.1, 183359 MiB\n", NVIDIA_SMI_OUT.strip() + ", extra\n", "\n"],
)
def test_gpu_info_detects_field_count_mismatch(monkeypatch, out):
    monkeypatch.setattr(mf.subprocess, "run", fake_run(out))
    with pytest.raises(ValueError, match="nvidia-smi"):
        mf.gpu_info()


def test_lib_commit_returns_head(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", fake_run(settings.LIB_COMMIT + "\n"))
    assert mf.lib_commit() == settings.LIB_COMMIT


def test_lib_commit_error_includes_git_stderr(monkeypatch):
    monkeypatch.setattr(
        mf.subprocess, "run", fake_run(returncode=128, stderr="fatal: not a git repository\n")
    )
    with pytest.raises(RuntimeError, match="fatal: not a git repository"):
        mf.lib_commit()


def test_collect_manifest_survives_missing_nvidia_smi_and_git(monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(mf.subprocess, "run", missing)
    manifest = mf.collect_manifest()
    assert manifest["gpu"]["error"].startswith("FileNotFoundError")
    assert manifest["llm_inference_bench_commit"].startswith("error: FileNotFoundError")
    problems = verify_manifest(manifest)
    assert any("nvidia-smi" in p for p in problems)
    assert any("llm-inference-bench" in p for p in problems)


def test_collect_manifest_records_gpu_commit_and_cuda(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", fake_run(NVIDIA_SMI_OUT))
    monkeypatch.setattr(mf, "package_versions", lambda: dict(GOOD["packages"]))
    monkeypatch.setenv("VLLM_BUILD_COMMIT", settings.VLLM_COMMIT)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(cuda="13.0"))
    )
    manifest = mf.collect_manifest()
    assert manifest["cuda"] == "13.0"
    assert manifest["vllm_build_commit"] == settings.VLLM_COMMIT
    assert manifest["gpu"]["name"] == "NVIDIA B200"


def test_cuda_version_is_none_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert mf.cuda_version() is None


def test_package_versions_records_the_bench_runtime_deps_without_verifying_them():
    for pkg in ("httpx", "rich", "psutil"):
        assert pkg in mf.PACKAGES
        assert pkg not in settings.EXPECTED_VERSIONS
    assert set(mf.package_versions()) == set(mf.PACKAGES)


def test_compressed_tensors_is_recorded_but_not_verified():
    assert "compressed-tensors" in mf.PACKAGES
    assert "compressed-tensors" not in settings.EXPECTED_VERSIONS
    assert "compressed-tensors" in mf.package_versions()


@pytest.mark.parametrize("version", [None, "0.19.0", "0.99.0"])
def test_whatever_version_of_compressed_tensors_the_image_has_passes_verification(version):
    assert verify_manifest(with_package("compressed-tensors", version)) == []


def test_package_versions_reads_the_installed_compressed_tensors(monkeypatch):
    real = mf.importlib.metadata.version
    monkeypatch.setattr(
        mf.importlib.metadata,
        "version",
        lambda pkg: "0.19.0" if pkg == "compressed-tensors" else real(pkg),
    )
    assert mf.package_versions()["compressed-tensors"] == "0.19.0"


def git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def test_code_version_is_the_commit_and_whether_the_checkout_has_changes(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "commit", "-q", "-m", "a")
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert mf.code_version(tmp_path) == mf.CodeVersion(head, False)
    (tmp_path / "a.py").write_text("x = 2\n")
    assert mf.code_version(tmp_path) == mf.CodeVersion(head, True)
    git(tmp_path, "checkout", "-q", "a.py")
    (tmp_path / "new.py").write_text("")
    assert mf.code_version(tmp_path) == mf.CodeVersion(head, True)


def test_code_version_is_unknown_outside_a_checkout_or_without_git(tmp_path, monkeypatch):
    assert mf.code_version(tmp_path / "missing") == mf.CodeVersion(None, None)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(mf.subprocess, "run", no_git)
    assert mf.code_version() == mf.CodeVersion(None, None)


def test_the_code_version_is_the_checkout_fp4bench_is_imported_from():
    assert mf.REPO == REPO
    assert (mf.REPO / "fp4bench" / "manifest.py").is_file()


@pytest.mark.parametrize(
    "version, env",
    [
        (mf.CodeVersion("a" * 40, False), {mf.CODE_COMMIT_ENV: "a" * 40, mf.CODE_DIRTY_ENV: "0"}),
        (mf.CodeVersion("b" * 40, True), {mf.CODE_COMMIT_ENV: "b" * 40, mf.CODE_DIRTY_ENV: "1"}),
        (mf.CodeVersion(None, None), {}),
    ],
)
def test_the_code_version_travels_as_environment_variables(version, env):
    assert version.env() == env and mf.CodeVersion.from_env(env) == version
    environ = {"OTHER": "x", mf.CODE_COMMIT_ENV: "stale", mf.CODE_DIRTY_ENV: "1"}
    version.export(environ)
    assert environ == {"OTHER": "x", **env}


def test_collect_manifest_records_the_code_version_it_was_given(monkeypatch):
    monkeypatch.setattr(mf.subprocess, "run", fake_run(NVIDIA_SMI_OUT))
    monkeypatch.setattr(mf, "package_versions", lambda: dict(GOOD["packages"]))
    monkeypatch.setenv(mf.CODE_COMMIT_ENV, "c" * 40)
    monkeypatch.setenv(mf.CODE_DIRTY_ENV, "1")
    manifest = mf.collect_manifest()
    assert (manifest["code_commit"], manifest["code_dirty"]) == ("c" * 40, True)
    assert (
        verify_manifest({**GOOD, **{k: manifest[k] for k in ("code_commit", "code_dirty")}}) == []
    )
    monkeypatch.delenv(mf.CODE_COMMIT_ENV)
    monkeypatch.delenv(mf.CODE_DIRTY_ENV)
    manifest = mf.collect_manifest()
    assert (manifest["code_commit"], manifest["code_dirty"]) == (None, None)
