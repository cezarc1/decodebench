"""Files an analysis writes into a run directory, as sha256 per name (text without the copies'
directory) and the pixels of each PNG (tests/golden/pngs.py), for the tag's analysis and the
current one."""

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests import REPO, RUNS_DIR, env_without_git
from tests.golden.pngs import pixels

CHECKPOINT_REPORT = "checkpoint_report.json"
OLD_TAG = "results-2026-10"
TEXT_SUFFIXES = frozenset({".md", ".json"})
Edit = Callable[[str, str], str]


def copy_runs(root: Path, runs: tuple[str, ...]) -> None:
    """data/runs/<run> for each of `runs`, with the checkpoint report beside them."""
    shutil.copy(RUNS_DIR / CHECKPOINT_REPORT, root / CHECKPOINT_REPORT)
    for run in runs:
        shutil.copytree(RUNS_DIR / run, root / run)


def root_prefixes(root: Path) -> list[str]:
    """`root/` as given and resolved (once if the same): what the copies' paths start with."""
    return list(dict.fromkeys(f"{p}/" for p in (str(root), str(root.resolve()))))


def normalise(text: str, root: Path) -> str:
    for prefix in root_prefixes(root):
        text = text.replace(prefix, "")
    return text


def file_hash(path: Path, root: Path, edit: Edit | None = None) -> str:
    """sha256 of the file; a text file without the copies' directory and edited by `edit`."""
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES:
        text = normalise(data.decode(), root)
        data = (text if edit is None else edit(path.name, text)).encode()
    return hashlib.sha256(data).hexdigest()


def _files(run_dir: Path) -> set[str]:
    return {p.name for p in run_dir.iterdir() if p.is_file()} if run_dir.is_dir() else set()


def written_hashes(
    run_dir: Path, before: set[str], root: Path, edit: Edit | None = None
) -> dict[str, str]:
    """sha256 of every file the analysis added to `run_dir`."""
    return {
        name: file_hash(run_dir / name, root, edit) for name in sorted(_files(run_dir) - before)
    }


def written_pngs(run_dir: Path, before: set[str]) -> dict[str, dict[str, Any]]:
    """The pixels of every PNG the analysis added to `run_dir`."""
    return {
        name: pixels(run_dir / name)
        for name in sorted(_files(run_dir) - before)
        if name.endswith(".png")
    }


def head_outputs(root: Path, run_dir: Path, reference: Path | None) -> dict[str, Any]:
    """{"files": {name: sha256}, "pngs": {name: pixels}} of analyze_run, or {"error": message} if
    it refuses the run."""
    import matplotlib as mpl

    mpl.use("Agg")
    from fp4bench.analysis.run import analyze_run

    before = _files(run_dir)
    try:
        analyze_run(run_dir, reference)
    except ValueError as exc:
        return {"error": normalise(str(exc), root)}
    return {"files": written_hashes(run_dir, before, root), "pngs": written_pngs(run_dir, before)}


def _old_module(run_dir: Path) -> str:
    """The tag's entry point for the run: analyze_c for a protocol with cells, else analyze."""
    import json

    path = run_dir / "manifests.jsonl"
    try:
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        protocol = lines[-1].get("protocol") or {}
        cells = protocol.get("cells")
    except (OSError, ValueError, IndexError, AttributeError):
        cells = None
    return "analysis.analyze_c" if cells else "analysis.analyze"


def old_outputs(
    worktree: Path, root: Path, run_dir: Path, reference: Path | None, edit: Edit | None = None
) -> dict[str, Any]:
    """{"files": ..., "pngs": ...} of the tag's command line, its text files edited by `edit`, or
    {"error": its last output line} (and the files it wrote)."""
    before = _files(run_dir)
    argv = [sys.executable, "-m", _old_module(run_dir), str(run_dir)]
    if reference is not None:
        argv += ["--reference-run", str(reference)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"MPLBACKEND": "Agg"}
    proc = subprocess.run(argv, cwd=worktree, env=env, capture_output=True, text=True, check=False)
    files = written_hashes(run_dir, before, root, edit)
    written = {"files": files, "pngs": written_pngs(run_dir, before)}
    if proc.returncode == 0:
        return written
    lines = [x for x in (proc.stderr or proc.stdout).splitlines() if x.strip()]
    out: dict[str, Any] = {
        "error": normalise(lines[-1] if lines else f"exit {proc.returncode}", root)
    }
    return out | (written if files else {})


def check_worktree(worktree: Path) -> None:
    """`worktree` is a checkout of OLD_TAG whose fp4bench the subprocess imports."""

    def rev(cwd: Path, ref: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", ref],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            env=env_without_git(),
        ).stdout.strip()

    if rev(worktree, "HEAD") != rev(REPO, f"{OLD_TAG}^{{commit}}"):
        raise SystemExit(f"{worktree} is not a checkout of {OLD_TAG}")
    found = subprocess.run(
        [sys.executable, "-c", "import fp4bench; print(fp4bench.__file__)"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not Path(found.strip()).resolve().is_relative_to(worktree.resolve()):
        raise SystemExit(f"the subprocess imports fp4bench from {found.strip()}, not {worktree}")
