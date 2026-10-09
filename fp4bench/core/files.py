"""Digests and atomic writes of the files a run reads and writes."""

import hashlib
from pathlib import Path

from fp4bench.core.types import Sha256


def file_sha256(path: Path | str) -> Sha256:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return Sha256(digest.hexdigest())


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
