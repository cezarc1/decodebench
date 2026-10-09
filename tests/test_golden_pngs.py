"""PNGs are compared byte for byte on the platform that drew the goldens and by pixels, within a
tolerance, elsewhere (tests/golden/pngs.py); every other written file byte for byte everywhere."""

import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, PngImagePlugin

from tests import GOLDEN_DIR, REPO
from tests.golden import pngs
from tests.golden.pngs import differences, golden_platform, golden_pngs, pixel_difference, pixels

FIGURE = REPO / "docs" / "figures" / "r_vs_concurrency.png"


def _rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def _save(array: np.ndarray, path: Path, **kwargs) -> Path:
    Image.fromarray(array).save(path, **kwargs)
    return path


def _edge_noise(array: np.ndarray, amplitude: int) -> np.ndarray:
    """Every pixel beside a change of colour moved by up to `amplitude` grey levels."""
    grey = array[..., :3].astype(int).sum(axis=-1)
    edge = np.zeros(grey.shape, dtype=bool)
    edge[:, 1:] |= grey[:, 1:] != grey[:, :-1]
    edge[1:, :] |= grey[1:, :] != grey[:-1, :]
    noise = np.random.default_rng(0).integers(-amplitude, amplitude + 1, size=grey.shape)
    out = array.astype(int)
    out[..., :3] += np.where(edge, noise, 0)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _half_pixel_shift(array: np.ndarray) -> np.ndarray:
    out = array.astype(float)
    out[:, 1:] = (out[:, 1:] + out[:, :-1]) / 2
    return np.round(out).astype(np.uint8)


def _with_text(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    ImageDraw.Draw(image).text(
        (array.shape[1] // 2, array.shape[0] // 3), "+7.2%", fill=(0, 0, 0, 255)
    )
    return np.asarray(image)


def test_the_pixels_of_a_png(tmp_path):
    array = np.full((20, 40, 4), 255, dtype=np.uint8)
    array[:16, :16, :3] = 0
    found = pixels(_save(array, tmp_path / "a.png"))
    assert found["size"] == [40, 20]
    assert found["rgba_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    assert pngs.grey_blocks(found).tolist() == [[0, 255, 255], [255, 255, 255]]


def test_the_same_pixels_in_another_encoding_are_the_same_pixels(tmp_path):
    array = _rgba(FIGURE)
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "another build")
    other = _save(array, tmp_path / "other.png", compress_level=1, pnginfo=info)
    assert other.read_bytes() != FIGURE.read_bytes()
    assert pixels(other) == pixels(FIGURE)


@pytest.mark.parametrize("change", [lambda a: _edge_noise(a, 32), _half_pixel_shift])
def test_rendering_noise_is_within_the_tolerance(change, tmp_path):
    changed = pixels(_save(change(_rgba(FIGURE)), tmp_path / "noisy.png"))
    assert changed != pixels(FIGURE)
    assert pixel_difference(pixels(FIGURE), changed) is None


@pytest.mark.parametrize(
    "change, reason",
    [
        (_with_text, "a block's grey moved by"),
        (lambda a: np.clip(a.astype(int) - 2, 0, 255).astype(np.uint8), "the grey moved by"),
        (lambda a: a[1:], "size"),
    ],
)
def test_a_drawn_change_is_not_within_the_tolerance(change, reason, tmp_path):
    changed = pixels(_save(change(_rgba(FIGURE)), tmp_path / "changed.png"))
    difference = pixel_difference(pixels(FIGURE), changed)
    assert difference is not None and reason in difference, difference


def test_strict_comparison_applies_on_the_platform_that_drew_the_goldens(monkeypatch):
    golden = golden_platform()
    assert sorted(golden) == ["machine", "matplotlib", "sys.platform"]
    monkeypatch.setattr(pngs, "this_platform", lambda: dict(golden))
    assert pngs.on_golden_platform()
    for key in golden:
        monkeypatch.setattr(pngs, "this_platform", lambda key=key: {**golden, key: "other"})
        assert not pngs.on_golden_platform()


def _outputs(tmp_path: Path, array: np.ndarray, summary: str) -> tuple[dict[str, str], dict]:
    """What outputs.json pins for the run, and what head_outputs returns for it."""
    tmp_path.mkdir(exist_ok=True)
    png = _save(array, tmp_path / "ratio_vs_c.png")
    files = {
        "summary.md": hashlib.sha256(summary.encode()).hexdigest(),
        "ratio_vs_c.png": hashlib.sha256(png.read_bytes()).hexdigest(),
    }
    return files, {"files": files, "pngs": {"ratio_vs_c.png": pixels(png)}}


def test_differences_compare_pngs_by_bytes_on_the_golden_platform_and_by_pixels_elsewhere(tmp_path):
    array = _rgba(FIGURE)
    want, golden = _outputs(tmp_path / "golden", array, "# summary")
    want_pixels = {want["ratio_vs_c.png"]: golden["pngs"]["ratio_vs_c.png"]}
    _, noisy = _outputs(tmp_path / "noisy", _edge_noise(array, 32), "# summary")
    _, label = _outputs(tmp_path / "label", _with_text(array), "# summary")
    _, text = _outputs(tmp_path / "text", array, "# summary\n")
    for strict in (True, False):
        assert differences(want, golden, want_pixels, strict) == []
        assert differences(want, text, want_pixels, strict) == [
            f"summary.md: sha256 {text['files']['summary.md']} != {want['summary.md']}"
        ]
        assert [d.split(":")[0] for d in differences(want, label, want_pixels, strict)] == [
            "ratio_vs_c.png"
        ]
    assert differences(want, noisy, want_pixels, strict=False) == []
    assert differences(want, noisy, want_pixels, strict=True) == [
        f"ratio_vs_c.png: sha256 {noisy['files']['ratio_vs_c.png']} != {want['ratio_vs_c.png']}"
    ]


def test_recorded_pixels_are_compared_by_content_not_by_how_zlib_compressed_them(tmp_path):
    want, got = _outputs(tmp_path, _rgba(FIGURE), "# summary")
    found = got["pngs"]["ratio_vs_c.png"]
    blocks = pngs.grey_blocks(found).tobytes()
    recompressed = {**found, "grey_blocks": base64.b64encode(zlib.compress(blocks, 1)).decode()}
    assert recompressed["grey_blocks"] != found["grey_blocks"]
    golden = {want["ratio_vs_c.png"]: recompressed}
    assert differences(want, got, golden, strict=True) == []
    other = {**found, "grey_blocks": base64.b64encode(zlib.compress(blocks[::-1], 9)).decode()}
    assert differences(want, got, {want["ratio_vs_c.png"]: other}, strict=True) == [
        f"ratio_vs_c.png: its pixels are not the ones recorded for {want['ratio_vs_c.png']}"
    ]


def test_differences_name_missing_and_extra_files_and_unrecorded_pixels(tmp_path):
    want, got = _outputs(tmp_path, _rgba(FIGURE), "# summary")
    extra = {"files": {**got["files"], "results_c.json": "0" * 64}, "pngs": got["pngs"]}
    missing = {"files": {"summary.md": want["summary.md"]}, "pngs": {}}
    for strict in (True, False):
        assert differences(want, extra, {}, strict) == [
            "ratio_vs_c.png: no pixels recorded for " + want["ratio_vs_c.png"],
            "results_c.json: written, not pinned",
        ]
        assert differences(want, missing, {}, strict) == ["ratio_vs_c.png: pinned, not written"]


def test_every_pinned_png_has_its_pixels_recorded():
    outputs = json.loads((GOLDEN_DIR / "outputs.json").read_text())
    mutations = json.loads((GOLDEN_DIR / "mutations.json").read_text())
    pinned = [*outputs.values(), *(m.get("files", {}) for m in mutations.values())]
    shas = {sha for files in pinned for name, sha in files.items() if name.endswith(".png")}
    assert sorted(golden_pngs()) == sorted(shas)
