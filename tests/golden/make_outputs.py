"""Write outputs.json and mutations.json: what the pre-release analysis (tag results-2026-10, which
is not part of the public history) writes for the committed runs and the mutation recipes, with
DELIBERATE_UPDATES applied, so that regenerating reproduces the committed files exactly; and
pngs.json: this platform and the pixels of every PNG they pin (tests/golden/pngs.py). Run it on
the golden platform, which pngs.json then records.
Usage: python -m tests.golden.make_outputs <worktree of the tag> [--out <directory>]"""

import argparse
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests import GOLDEN_DIR
from tests.golden.make_golden import REFERENCE_RUNS, RUNS
from tests.golden.mutations import RECIPES, build
from tests.golden.outputs import Edit, check_worktree, copy_runs, old_outputs
from tests.golden.pngs import this_platform
from tests.golden.view import dumps

OUTPUTS = "outputs.json"
MUTATIONS = "mutations.json"
PNGS = "pngs.json"


@dataclass(frozen=True)
class DeliberateUpdate:
    """A golden change made on purpose after the tag: `old` -> `new` in every `file` of that name
    the tag's analysis writes, applied before hashing. `commit` changed the code and the goldens
    together, and its message says why and which entries changed. To change a written file on
    purpose, change the code and the committed JSON in one commit and add its updates here."""

    commit: str
    file: str
    old: str
    new: str


DELIBERATE_UPDATES: tuple[DeliberateUpdate, ...] = (
    DeliberateUpdate(
        "c3438c0",
        "summary.md",
        "--reference-run: `python -m analysis.analyze results/<this run> --reference-run "
        "results/full-1`.",
        "--reference-run: `fp4bench analyze results/<this run> --reference-run results/full-1`.",
    ),
    DeliberateUpdate(
        "c3438c0",
        "summary.md",
        'To set NVa, put `TREATMENT_SERVER_ARGS["NVa"] = (',
        "To set NVa, make `(",
    ),
    DeliberateUpdate(
        "c3438c0",
        "summary.md",
        ")` in fp4bench/config.py.",
        ")` its `server_args` in the `TREATMENTS` table of fp4bench/studies/model.py.",
    ),
)


@dataclass
class DeliberateEdit:
    """Applies `updates` to a written file's text, counting the files each one changed."""

    updates: tuple[DeliberateUpdate, ...] = DELIBERATE_UPDATES
    counts: Counter[DeliberateUpdate] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __call__(self, name: str, text: str) -> str:
        for update in self.updates:
            if update.file == name and update.old in text:
                text = text.replace(update.old, update.new)
                with self._lock:
                    self.counts[update] += 1
        return text

    def check_used(self) -> None:
        """Refuse an update that changed no file: its `old` is not what the tag writes."""
        unused = [u for u in self.updates if not self.counts[u]]
        if unused:
            raise SystemExit(
                "deliberate updates that changed no file: "
                + "; ".join(f"{u.commit} {u.file}: {u.old!r}" for u in unused)
            )


def run_outputs(worktree: Path, run: str, edit: Edit) -> dict[str, Any]:
    ref = REFERENCE_RUNS.get(run)
    with tempfile.TemporaryDirectory(prefix="outputs-") as tmp:
        root = Path(tmp)
        copy_runs(root, (run,) if ref is None else (run, ref))
        out = old_outputs(worktree, root, root / run, None if ref is None else root / ref, edit)
    if "error" in out:
        raise SystemExit(f"{run}: the tag's analysis failed: {out['error']}")
    return out


def recipe_outputs(worktree: Path, name: str, edit: Edit) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mutation-") as tmp:
        root = Path(tmp)
        run_dir, reference = build(RECIPES[name], root)
        return old_outputs(worktree, root, run_dir, reference, edit)


def parse_args(argv: list[str]) -> tuple[Path, Path]:
    """(worktree of the tag, directory to write the JSON into)."""
    parser = argparse.ArgumentParser(prog="python -m tests.golden.make_outputs")
    parser.add_argument("worktree", type=Path)
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR)
    args = parser.parse_args(argv)
    return args.worktree, args.out


def main(argv: list[str] | None = None) -> None:
    worktree, out = parse_args(sys.argv[1:] if argv is None else argv)
    check_worktree(worktree)
    edit = DeliberateEdit()
    outputs = {run: run_outputs(worktree, run, edit) for run in RUNS}
    with ThreadPoolExecutor(8) as pool:
        found = dict(
            zip(
                RECIPES,
                pool.map(lambda name: recipe_outputs(worktree, name, edit), RECIPES),
                strict=True,
            )
        )
    edit.check_used()
    pngs = {
        sha: o["pngs"][name]
        for o in (*outputs.values(), *found.values())
        for name, sha in o.get("files", {}).items()
        if name.endswith(".png")
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / OUTPUTS).write_text(dumps({run: o["files"] for run, o in outputs.items()}))
    print(f"wrote {out / OUTPUTS}")
    (out / MUTATIONS).write_text(
        dumps({name: {k: v for k, v in o.items() if k != "pngs"} for name, o in found.items()})
    )
    print(f"wrote {out / MUTATIONS} ({len(found)} recipes)")
    (out / PNGS).write_text(dumps({"platform": this_platform(), "pngs": pngs}))
    print(f"wrote {out / PNGS} ({len(pngs)} PNGs drawn on {this_platform()})")
    for update in edit.updates:
        print(
            f"{update.commit} {update.file}: {update.old!r} -> {update.new!r} "
            f"in {edit.counts[update]} files"
        )


if __name__ == "__main__":
    main()
