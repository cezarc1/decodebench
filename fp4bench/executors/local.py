"""Runs a study on this machine: the same runner, with data and scratch directories from the
environment."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fp4bench import settings
from fp4bench.manifest import CODE_COMMIT_ENV, CODE_DIRTY_ENV, CodeVersion


@dataclass(frozen=True)
class LocalExecutor:
    data_dir: Path
    results_dir: Path
    local_dir: Path

    def env(self, code: CodeVersion) -> dict[str, str]:
        inherited = {
            k: v for k, v in os.environ.items() if k not in (CODE_COMMIT_ENV, CODE_DIRTY_ENV)
        }
        return {
            **inherited,
            settings.DATA_DIR_ENV: str(self.data_dir.resolve()),
            settings.LOCAL_DIR_ENV: str(self.local_dir.resolve()),
            **code.env(),
        }

    def run(
        self, study: str, run_id: str, rounds: int, rerun_rounds: tuple[int, ...], code: CodeVersion
    ) -> Path:
        run_dir = (self.results_dir / run_id).resolve()
        spec = {
            "study": study,
            "run_dir": str(run_dir),
            "rounds": rounds,
            "rerun_rounds": list(rerun_rounds),
        }
        subprocess.run(
            [sys.executable, "-m", "fp4bench.executors.local", json.dumps(spec)],
            env=self.env(code),
            check=True,
        )
        return run_dir


def main(argv: list[str] | None = None) -> None:
    from fp4bench.runner import run_experiment
    from fp4bench.studies.registry import study_to_run

    spec = json.loads((sys.argv[1:] if argv is None else argv)[0])
    run_experiment(
        Path(spec["run_dir"]),
        study_to_run(spec["study"], spec["rounds"]),
        commit=lambda: None,
        rerun_rounds=tuple(spec["rerun_rounds"]),
    )


if __name__ == "__main__":
    main()
