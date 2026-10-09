"""PNGs across platforms. pngs.json records the platform that drew the golden PNGs (those
outputs.json and mutations.json pin, and docs/figures): sys.platform, machine and matplotlib
version; and, by each golden PNG's sha256, its pixels: size, sha256 of the RGBA pixels and the grey
means of its 16x16-pixel blocks. On that platform a PNG must be byte-identical. Elsewhere only PNG
rendering and compression may differ: a PNG must have the same size and either the same RGBA pixels
or block means within MAX_BLOCK_DIFF grey levels each and MAX_MEAN_DIFF on average (noise of up to
32 levels on every edge pixel, or the whole figure shifted half a pixel, passes; a five-character
label does not). Other files are byte-exact everywhere. make_outputs writes pngs.json with
outputs.json and mutations.json."""

import base64
import functools
import hashlib
import json
import platform
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tests import GOLDEN_DIR

PNGS_PATH = GOLDEN_DIR / "pngs.json"
BLOCK = 16
MAX_BLOCK_DIFF = 10
MAX_MEAN_DIFF = 0.5
Pixels = dict[str, Any]


def this_platform() -> dict[str, str]:
    import matplotlib as mpl

    return {
        "sys.platform": sys.platform,
        "machine": platform.machine(),
        "matplotlib": mpl.__version__,
    }


@functools.cache
def _golden() -> dict[str, Any]:
    return json.loads(PNGS_PATH.read_text())


def golden_platform() -> dict[str, str]:
    return dict(_golden()["platform"])


def golden_pngs() -> dict[str, Pixels]:
    """The pixels of every golden PNG, by the PNG's sha256."""
    return _golden()["pngs"]


def on_golden_platform() -> bool:
    """Whether PNGs are compared byte for byte here."""
    return this_platform() == golden_platform()


def pixels(path: Path) -> Pixels:
    """The PNG's size, the sha256 of its RGBA pixels and its grey block means (zlib, base64)."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    grey = np.asarray(rgba.convert("L"), dtype=np.int64)
    rows, cols = -(-grey.shape[0] // BLOCK), -(-grey.shape[1] // BLOCK)
    padded = np.pad(
        grey, ((0, rows * BLOCK - grey.shape[0]), (0, cols * BLOCK - grey.shape[1])), mode="edge"
    )
    sums = padded.reshape(rows, BLOCK, cols, BLOCK).sum(axis=(1, 3))
    means = ((sums + BLOCK * BLOCK // 2) // (BLOCK * BLOCK)).astype(np.uint8)
    return {
        "size": list(rgba.size),
        "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
        "grey_blocks": base64.b64encode(zlib.compress(means.tobytes(), 9)).decode(),
    }


def grey_blocks(found: Pixels) -> np.ndarray:
    width, height = found["size"]
    data = zlib.decompress(base64.b64decode(found["grey_blocks"]))
    return np.frombuffer(data, dtype=np.uint8).reshape(-(-height // BLOCK), -(-width // BLOCK))


def same_pixels(want: Pixels, got: Pixels) -> bool:
    """Equal pixels, however zlib compressed the grey blocks."""
    return (
        want["size"] == got["size"]
        and want["rgba_sha256"] == got["rgba_sha256"]
        and np.array_equal(grey_blocks(want), grey_blocks(got))
    )


def pixel_difference(want: Pixels, got: Pixels) -> str | None:
    """Why `got` is not `want` within the tolerance, or None."""
    if want["size"] != got["size"]:
        return f"size {got['size']} != {want['size']}"
    if want["rgba_sha256"] == got["rgba_sha256"]:
        return None
    diff = np.abs(grey_blocks(want).astype(int) - grey_blocks(got).astype(int))
    if diff.max() > MAX_BLOCK_DIFF:
        row, col = np.unravel_index(int(diff.argmax()), diff.shape)
        return (
            f"a block's grey moved by {diff.max()} > {MAX_BLOCK_DIFF} levels (at x={col * BLOCK}, "
            f"y={row * BLOCK})"
        )
    if diff.mean() > MAX_MEAN_DIFF:
        return f"the grey moved by {diff.mean():.3f} > {MAX_MEAN_DIFF} levels on average"
    return None


def differences(
    want: dict[str, str],
    got: dict[str, Any],
    golden: dict[str, Pixels] | None = None,
    strict: bool | None = None,
) -> list[str]:
    """How `got` ({"files": {name: sha256}, "pngs": {name: pixels}}) departs from the pinned
    `want` ({name: sha256}): PNGs byte for byte if `strict` (by default, on the golden platform),
    else by pixels within the tolerance; every other file byte for byte."""
    golden = golden_pngs() if golden is None else golden
    strict = on_golden_platform() if strict is None else strict
    files, found = got.get("files", {}), got.get("pngs", {})
    out = []
    for name in sorted(want.keys() | files.keys()):
        if name not in files:
            out.append(f"{name}: pinned, not written")
        elif name not in want:
            out.append(f"{name}: written, not pinned")
        elif (not name.endswith(".png") or strict) and files[name] != want[name]:
            out.append(f"{name}: sha256 {files[name]} != {want[name]}")
        elif not name.endswith(".png"):
            continue
        elif want[name] not in golden:
            out.append(f"{name}: no pixels recorded for {want[name]}")
        elif strict and not (name in found and same_pixels(golden[want[name]], found[name])):
            out.append(f"{name}: its pixels are not the ones recorded for {want[name]}")
        elif not strict and (reason := pixel_difference(golden[want[name]], found[name])):
            out.append(f"{name}: {reason}")
    return out
