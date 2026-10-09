"""ty's ratchet (typing milestone): `ty check` may report no diagnostic that tests/ty_baseline.txt
does not list, matched by (file, rule, message), and the baseline only shrinks; phase 10 empties
it. ty's messages carry no line numbers, so moving code does not disturb the match.

    uv run python -m tests.ty_baseline                      # check (CI, test_ty_baseline.py)
    uv run python -m tests.ty_baseline --update             # after a fix; never adds a diagnostic
    uv run python -m tests.ty_baseline --update --reworded  # also new wording, no count rising
"""

import argparse
import re
import subprocess
import sys
from collections import Counter

from tests import REPO

BASELINE = REPO / "tests" / "ty_baseline.txt"
DIAGNOSTIC = re.compile(
    r"(?P<path>[^:\s]+):\d+:\d+: (?:error|warning)\[(?P<rule>[\w-]+)\] (?P<message>.+)"
)
SUMMARY = re.compile(r"Found (\d+) diagnostics?|All checks passed!")

type Key = tuple[str, str, str]  # (file, rule, message)


def run_ty() -> list[str]:
    """ty's diagnostics on the project, one concise line each. `python -m ty` finds the binary
    beside the ty package, also from an environment layered on the project's (`uv run --with`)."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--output-format",
            "concise",
            "--no-progress",
            "--color",
            "never",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = done.stdout.splitlines()
    found = [line for line in lines if DIAGNOSTIC.match(line)]
    summary = next((m for line in lines if (m := SUMMARY.fullmatch(line))), None)
    if done.returncode not in (0, 1) or summary is None or int(summary[1] or 0) != len(found):
        raise RuntimeError(f"ty check (exit {done.returncode}):\n{done.stdout}{done.stderr}")
    return found


def counts(lines: list[str]) -> Counter[Key]:
    return Counter(
        (m["path"], m["rule"], m["message"]) for line in lines if (m := DIAGNOSTIC.match(line))
    )


def per_rule(diagnostics: Counter[Key]) -> Counter[tuple[str, str]]:
    totals: Counter[tuple[str, str]] = Counter()
    for (path, rule, _), n in diagnostics.items():
        totals[path, rule] += n
    return totals


def read_baseline() -> Counter[Key]:
    baseline: Counter[Key] = Counter()
    for line in BASELINE.read_text().splitlines():
        if line and not line.startswith("#"):
            path, rule, n, message = line.split(" ", 3)
            baseline[path, rule, message] = int(n)
    return baseline


def write_baseline(now: Counter[Key]) -> None:
    header = [
        "# ty diagnostics allowed, as `file rule count message`; the counts only fall. Written by",
        "# `uv run python -m tests.ty_baseline --update`.",
        f"# {now.total()} diagnostics",
    ]
    rows = [f"{path} {rule} {n} {message}" for (path, rule, message), n in sorted(now.items())]
    BASELINE.write_text("\n".join([*header, *rows]) + "\n")


def compare(now: Counter[Key], baseline: Counter[Key]) -> tuple[list[Key], list[Key]]:
    """(the keys with more diagnostics than the baseline allows, the keys with fewer)."""
    keys = now.keys() | baseline.keys()
    return (
        sorted(k for k in keys if now[k] > baseline[k]),
        sorted(k for k in keys if now[k] < baseline[k]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.ty_baseline")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline to fewer")
    parser.add_argument(
        "--reworded", action="store_true", help="with --update: accept changed messages too"
    )
    args = parser.parse_args(argv)
    if args.reworded and not args.update:
        parser.error("--reworded only goes with --update")
    lines = run_ty()
    now, baseline = counts(lines), read_baseline()
    more, fewer = compare(now, baseline)
    now_rule, baseline_rule = per_rule(now), per_rule(baseline)
    risen = sorted(k for k in now_rule if now_rule[k] > baseline_rule[k])
    if more and not (args.reworded and not risen):
        print("ty reports diagnostics tests/ty_baseline.txt does not allow:", file=sys.stderr)
        print(*(x for x in lines if counts([x]).keys() & set(more)), sep="\n", file=sys.stderr)
        for path, rule in risen:
            print(
                f"  {path} {rule}: {now_rule[path, rule]} > {baseline_rule[path, rule]}",
                file=sys.stderr,
            )
        if args.update:
            print("--update never adds a diagnostic; fix it", file=sys.stderr)
        return 1
    if args.update:
        write_baseline(now)
        print(f"ty baseline: {baseline.total()} -> {now.total()} diagnostics")
        return 0
    if fewer:
        print(
            f"ty reports {baseline.total() - now.total()} fewer diagnostics than"
            " tests/ty_baseline.txt; lower it: uv run python -m tests.ty_baseline --update",
            file=sys.stderr,
        )
        return 1
    print(f"ty: {now.total()} diagnostics, as tests/ty_baseline.txt allows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
