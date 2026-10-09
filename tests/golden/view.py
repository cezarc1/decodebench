"""The golden view of a run: the analysis payload as canonical JSON. golden_view is test_golden's
adapter. dumps writes every snapshot file in tests/golden; first_difference says where a value
departs from its snapshot."""

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

from tests.golden.outputs import root_prefixes

mpl.use("Agg")


def dumps(view: Any, *, key_order: bool = False) -> str:
    """A snapshot file's text: keys sorted, unless `key_order` keeps them and non-ASCII text as
    it is (pins.json)."""
    text = json.dumps(
        view, sort_keys=not key_order, ensure_ascii=not key_order, indent=1, allow_nan=False
    )
    return text + "\n"


def first_difference(
    want: Any,
    got: Any,
    path: str = "$",
    *,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
    key_order: bool = False,
) -> str | None:
    """The JSON path of the first place `got` departs from `want`, or None. Types count (1 is not
    1.0, True is not 1); floats within `rel_tol` or `abs_tol` are equal; with `key_order`, a dict
    whose keys differ in order or membership departs at the dict, else at the first key."""

    def inner(w: Any, g: Any, where: str) -> str | None:
        return first_difference(w, g, where, rel_tol=rel_tol, abs_tol=abs_tol, key_order=key_order)

    if type(want) is not type(got):
        return f"{path}: {type(want).__name__} {want!r:.200} != {type(got).__name__} {got!r:.200}"
    if isinstance(want, dict):
        if key_order and list(want) != list(got):
            return f"{path}: keys {list(want)} != {list(got)}"
        for key in sorted(want.keys() | got.keys()):
            if key not in got:
                return f"{path}.{key}: missing now"
            if key not in want:
                return f"{path}.{key}: not in the snapshot"
            if (diff := inner(want[key], got[key], f"{path}.{key}")) is not None:
                return diff
        return None
    if isinstance(want, list):
        for i, (w, g) in enumerate(zip(want, got, strict=False)):
            if (diff := inner(w, g, f"{path}[{i}]")) is not None:
                return diff
        return None if len(want) == len(got) else f"{path}: {len(want)} items != {len(got)}"
    if isinstance(want, float):
        same = math.isclose(want, got, rel_tol=rel_tol, abs_tol=abs_tol)
    else:
        same = want == got
    return None if same else f"{path}: {want!r:.200} != {got!r:.200}"


def canonical(obj: Any, strip_prefixes: tuple[str, ...] = ()) -> Any:
    """`obj` as JSON-native values, with each of `strip_prefixes` removed from every string."""
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = canonical(k, strip_prefixes) if isinstance(k, str) else str(k)
            if key in out:
                raise ValueError(f"two keys canonicalise to {key!r}")
            out[key] = canonical(v, strip_prefixes)
        return out
    if isinstance(obj, (list, tuple)):
        return [canonical(v, strip_prefixes) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [canonical(v, strip_prefixes) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, str):
        for prefix in strip_prefixes:
            obj = obj.replace(prefix, "")
        return obj
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, Path):
        return canonical(str(obj), strip_prefixes)
    raise TypeError(f"no canonical form for {type(obj).__name__}: {obj!r}")


def _analyze(run_dir: Path, reference_run: Path | None) -> dict:
    """The current analysis's payload for `run_dir`. It writes nothing: the files analyze_run
    writes are test_golden_outputs' (tests/golden/outputs.json)."""
    from fp4bench.analysis.run import evaluate

    return evaluate(run_dir, reference_run).payload()


def golden_view(run_dir: Path, reference_run: Path | None = None) -> dict:
    """The analysis payload of a temporary copy of `run_dir`, gates as {gate: passed}, canonical."""
    run_dir = Path(run_dir)
    with tempfile.TemporaryDirectory(prefix="golden-") as tmp:
        root = Path(tmp)
        copy = root / run_dir.name
        shutil.copytree(run_dir, copy)
        ref_copy = None
        if reference_run is not None:
            reference_run = Path(reference_run)
            if reference_run.name == run_dir.name:
                raise ValueError(f"reference run and run share the name {run_dir.name!r}")
            ref_copy = root / reference_run.name
            shutil.copytree(reference_run, ref_copy)
        payload = dict(_analyze(copy, ref_copy))
        payload["gates"] = {name: bool(gate["pass"]) for name, gate in payload["gates"].items()}
        prefixes = tuple(root_prefixes(root))
        view = canonical(payload, prefixes)
        leaked = [p for p in prefixes if p.rstrip("/") in json.dumps(view)]
        if leaked:
            raise AssertionError(f"the temporary directory leaks into the view: {leaked}")
        return view
