"""Write tests/golden/<run>.json, golden views of data/runs: python -m tests.golden.make_golden."""

from pathlib import Path

from tests import GOLDEN_DIR, REPO, RUNS_DIR
from tests.golden.view import dumps, golden_view

RUNS = ("full-1", "expb-1", "expc-1", "smoke-r2-2", "smoke-nf-2", "smoke-c-1")
REFERENCE_RUNS = {"expb-1": "full-1"}


def golden_path(run: str) -> Path:
    return GOLDEN_DIR / f"{run}.json"


def run_view(run: str) -> dict:
    ref = REFERENCE_RUNS.get(run)
    return golden_view(RUNS_DIR / run, None if ref is None else RUNS_DIR / ref)


def main() -> None:
    for run in RUNS:
        path = golden_path(run)
        path.write_text(dumps(run_view(run)))
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
