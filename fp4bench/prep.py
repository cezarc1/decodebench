# METHODOLOGY.md#prompts
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from fp4bench import settings
from fp4bench.core.files import file_sha256, write_text_atomic
from fp4bench.core.types import ChatFrame, FetchKind, PromptSet
from fp4bench.prompts import (
    build_cell_prompt_sets,
    build_m1_prompt_sets,
    build_nll_prompts,
    chat_frame,
    check_cell_prompt_sets,
    load_sharegpt_user_texts,
)
from fp4bench.studies import expc, model

FETCH_FILE = "fp4bench_fetch.json"
SRC_IGNORE_PATTERNS = ["*.pt", "*.bin"]


class CellPromptFile(TypedDict):
    cells: dict[str, list[PromptSet]]
    seed: int
    n_sets: int
    m1_prompts_sha256: str
    reused_m1_cells: NotRequired[list[str]]
    sharegpt_revision: str
    tokenizer_revision: str


def _require_split_special_tokens(tokenizer) -> None:
    probes = [
        t
        for t in dict.fromkeys(tokenizer.all_special_tokens)
        if len(tokenizer.encode(t, add_special_tokens=False)) == 1
    ]
    if not probes:
        raise RuntimeError(
            "the tokenizer has no special token that encodes to a single id, so there is nothing "
            "to probe split_special_tokens with; its behaviour is unverified, and so are any "
            "prompt files built with it"
        )
    unsplit = {
        t: len(ids)
        for t in probes
        if len(ids := tokenizer.encode(t, add_special_tokens=False, split_special_tokens=True)) <= 1
    }
    if unsplit:
        raise RuntimeError(
            f"the tokenizer ignores split_special_tokens (with it, special tokens "
            f"{', '.join(f'{t!r} ({n} id)' for t, n in unsplit.items())} still encode to a "
            "single id); user text containing special-token literals would become control tokens "
            "in the prompts. Prompts built with this tokenizer, including the existing prompt "
            "files that prepare would reuse, are suspect"
        )


def _prompts_stale(m1, frame: ChatFrame) -> str | None:
    try:
        first = m1[0][0]
    except (IndexError, KeyError, TypeError):
        return "the M1 prompt file has no prompts"
    if not frame.frames(first):
        return "the M1 prompts do not carry this tokenizer's chat-template frame"
    return None


def prepare_all(force: bool = False) -> dict:
    """Download the pinned inputs and build the M1 and NLL prompt files; their sha256s."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    snapshot_download(
        model.SRC_MODEL_ID,
        revision=model.SRC_MODEL_REVISION,
        local_dir=model.BF16_MODEL_DIR,
        ignore_patterns=SRC_IGNORE_PATTERNS,
    )
    hf_hub_download(
        settings.SHAREGPT_REPO,
        settings.SHAREGPT_FILE,
        repo_type="dataset",
        revision=settings.SHAREGPT_REVISION,
        local_dir=settings.DATA_DIR,
    )

    tok = AutoTokenizer.from_pretrained(model.BF16_MODEL_DIR)
    _require_split_special_tokens(tok)
    frame = chat_frame(tok)
    m1_path, nll_path = Path(settings.M1_PROMPTS_PATH), Path(settings.NLL_PROMPTS_PATH)
    rebuild = force or not (m1_path.exists() and nll_path.exists())
    if not rebuild:
        rebuild = _prompts_stale(json.loads(m1_path.read_text()), frame) is not None
    n_texts = None
    if rebuild:
        texts = load_sharegpt_user_texts(settings.SHAREGPT_PATH)
        n_texts = len(texts)
        m1 = build_m1_prompt_sets(
            tok, texts, settings.M1_INPUT_LEN, settings.M1_SETS, settings.M1_SET_SIZE, settings.SEED
        )
        nll = build_nll_prompts(tok, texts, settings.NLL_PROMPTS, settings.NLL_LEN, settings.SEED)
        m1_path.unlink(missing_ok=True)
        nll_path.unlink(missing_ok=True)
        write_text_atomic(m1_path, json.dumps(m1))
        write_text_atomic(nll_path, json.dumps(nll))
    else:
        m1, nll = json.loads(m1_path.read_text()), json.loads(nll_path.read_text())
    return {
        "sharegpt_texts": n_texts,
        "prompts_rebuilt": rebuild,
        "m1_sets": len(m1),
        "m1_set_size": len(m1[0]),
        "m1_prompt_len": len(m1[0][0]),
        "chat_frame_tokens": frame.n_tokens,
        "nll_prompts": len(nll),
        "m1_prompts_sha256": file_sha256(m1_path),
        "nll_prompts_sha256": file_sha256(nll_path),
    }


def _c_prompts_stale(record, cells, m1_sha: str, frame: ChatFrame) -> str | None:
    if not (isinstance(record, dict) and isinstance(record.get("cells"), dict)):
        return "the existing cell prompt file is not readable as one"
    if record.get("m1_prompts_sha256") != m1_sha:
        return (
            f"the existing cell prompt file was built from other M1 prompts "
            f"({record.get('m1_prompts_sha256')!r}, the volume's are {m1_sha!r})"
        )
    keys = [cell.prompt_file_key for cell in cells]
    if list(record["cells"]) != keys:
        return f"the existing cell prompt file holds the cells {list(record['cells'])}, not {keys}"
    try:
        check_cell_prompt_sets(record["cells"], cells, settings.M1_SETS)
    except ValueError as exc:
        return f"the existing cell prompt file cannot serve its cells: {exc}"
    for key, sets in record["cells"].items():
        if not frame.frames(sets[0][0]):
            return (
                f"cell {key} of the existing file does not carry this tokenizer's "
                "chat-template frame"
            )
    return None


def prepare_expc(force: bool = False) -> dict:
    """Build Experiment C's cell prompt file from what `prepare` put on the volume."""
    from transformers import AutoTokenizer

    for path in (settings.SHAREGPT_PATH, settings.M1_PROMPTS_PATH, model.BF16_MODEL_DIR):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} is missing: Experiment C's prompts are built from what `fp4bench "
                "prepare` puts on the volume (ShareGPT, the Qwen3 tokenizer, "
                "the M1 prompts its (1, 1024) cell reuses); run that first"
            )
    tok = AutoTokenizer.from_pretrained(model.BF16_MODEL_DIR)
    _require_split_special_tokens(tok)
    frame = chat_frame(tok)
    m1_path, c_path = Path(settings.M1_PROMPTS_PATH), Path(settings.C_PROMPTS_PATH)
    m1 = json.loads(m1_path.read_text())
    if (stale := _prompts_stale(m1, frame)) is not None:
        raise RuntimeError(
            f"{stale}: the (1, {settings.M1_INPUT_LEN}) cell reuses them, so run "
            "`fp4bench prepare` (with --force if needed) first"
        )
    m1_sha = file_sha256(m1_path)
    cells = expc.C_CELLS
    record: CellPromptFile | None = None
    if force:
        reason = "forced"
    elif not c_path.exists():
        reason = "no cell prompt file"
    else:
        try:
            record = json.loads(c_path.read_text())
        except ValueError:
            record = None
        reason = _c_prompts_stale(record, cells, m1_sha, frame)
    n_texts = None
    if reason is not None:
        texts = load_sharegpt_user_texts(settings.SHAREGPT_PATH)
        n_texts = len(texts)
        try:
            built = build_cell_prompt_sets(
                tok,
                texts,
                cells,
                settings.M1_SETS,
                settings.SEED,
                m1_prompt_sets=m1,
                m1_prompt_len=settings.M1_INPUT_LEN,
            )
        except ValueError as exc:
            exc.add_note(
                f"the texts are the {n_texts} first human turns of {settings.SHAREGPT_PATH}"
            )
            raise
        record = {
            "cells": {cell.prompt_file_key: sets for cell, sets in built.items()},
            "seed": settings.SEED,
            "n_sets": settings.M1_SETS,
            "m1_prompts_sha256": m1_sha,
            "reused_m1_cells": [
                cell.prompt_file_key for cell in cells if cell.prompt_len == settings.M1_INPUT_LEN
            ],
            "sharegpt_revision": settings.SHAREGPT_REVISION,
            "tokenizer_revision": model.SRC_MODEL_REVISION,
        }
        check_cell_prompt_sets(record["cells"], cells, settings.M1_SETS)
        write_text_atomic(c_path, json.dumps(record))
    assert record is not None
    reused = set(record.get("reused_m1_cells", []))
    return {
        "prompts_rebuilt": reason is not None,
        "rebuild_reason": reason,
        "sharegpt_texts": n_texts,
        "chat_frame_tokens": frame.n_tokens,
        "cells": {
            key: {
                "sets": len(sets),
                "set_size": len(sets[0]),
                "prompt_len": len(sets[0][0]),
                "reused_m1": key in reused,
            }
            for key, sets in record["cells"].items()
        },
        "total_prompt_tokens": sum(
            len(q) for sets in record["cells"].values() for s in sets for q in s
        ),
        "m1_prompts_sha256": m1_sha,
        "c_prompts_sha256": file_sha256(c_path),
    }


def fetch_checkpoint(kind: str, repo_id: str, revision: str) -> dict:
    """Mirror a published revision into the kind's volume directory (METHODOLOGY.md#fetch)."""
    from huggingface_hub import snapshot_download

    if kind not in FetchKind:
        known = sorted(k.value for k in FetchKind)
        raise ValueError(f"unknown checkpoint kind {kind!r}; expected one of {known}")
    target = model.volume_dir(FetchKind(kind).checkpoint)
    shutil.rmtree(target, ignore_errors=True)
    path = snapshot_download(repo_id, revision=revision, local_dir=target)
    record = {
        "repo_id": repo_id,
        "revision": revision,
        "utc": datetime.now(UTC).isoformat(),
    }
    (Path(target) / FETCH_FILE).write_text(json.dumps(record))
    return {"kind": kind, "repo_id": repo_id, "revision": revision, "local_dir": path}
