# METHODOLOGY.md#prompts
import json
import random
from pathlib import Path

from fp4bench.core.types import Cell, ChatFrame, Prompt, PromptSet

SENTINEL = "@@FP4BENCH_CONTENT@@"
CELL_SEED_OFFSET = 3


def load_sharegpt_user_texts(path: str) -> list[str]:
    """First human turn of each ShareGPT conversation; an empty one skips the conversation."""
    with Path(path).open() as f:
        conversations = json.load(f)
    texts = []
    for conv in conversations:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human":
                value = (turn.get("value") or "").strip()
                if value:
                    texts.append(value)
                break
    return texts


def chat_frame(tokenizer) -> ChatFrame:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": SENTINEL}], add_generation_prompt=True, tokenize=False
    )
    if text.count(SENTINEL) != 1:
        raise ValueError(
            f"chat template must render the user content verbatim exactly once; "
            f"found {text.count(SENTINEL)} occurrences of the sentinel in {text!r}"
        )
    pre, post = text.split(SENTINEL)
    return ChatFrame(
        tokenizer.encode(pre, add_special_tokens=False),
        tokenizer.encode(post, add_special_tokens=False),
    )


def _encode_content(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)


def fixed_length_ids(frame: ChatFrame, content_ids: list[int], length: int) -> Prompt:
    room = frame.room(length)
    if room <= 0:
        raise ValueError("chat frame is longer than the target length")
    if len(content_ids) < room:
        raise ValueError(f"need {room} content tokens, got {len(content_ids)}")
    return frame.pre + content_ids[:room] + frame.post


def build_m1_prompt_sets(
    tokenizer, texts: list[str], length: int, n_sets: int, set_size: int, seed: int
) -> list[PromptSet]:
    """n_sets × set_size templated prompts of exactly `length` tokens of real text."""
    frame = chat_frame(tokenizer)
    room = frame.room(length)
    pool = list(texts)
    random.Random(seed).shuffle(pool)
    stream = iter(pool)
    consumed = 0
    buffer: list[int] = []
    sets = []
    for _ in range(n_sets):
        prompts = []
        for _ in range(set_size):
            while len(buffer) < room:
                text = next(stream, None)
                if text is None:
                    raise ValueError(
                        f"ran out of texts after {consumed} while building {n_sets} sets x "
                        f"{set_size} prompts of {length} tokens; supply more texts"
                    )
                consumed += 1
                buffer.extend(_encode_content(tokenizer, text + "\n\n"))
            prompts.append(fixed_length_ids(frame, buffer, length))
            buffer = buffer[room:]
        sets.append(prompts)
    return sets


def build_nll_prompts(tokenizer, texts: list[str], n: int, length: int, seed: int) -> list[Prompt]:
    """Plain-text token windows for the G5b log-likelihood check."""
    pool = list(texts)
    random.Random(seed + 2).shuffle(pool)
    out = []
    for text in pool:
        ids = _encode_content(tokenizer, text)
        if len(ids) >= length:
            out.append(ids[:length])
            if len(out) == n:
                return out
    raise ValueError(f"only {len(out)} texts have {length} tokens")


def _reused_m1_sets(m1_prompt_sets: list[PromptSet], cell: Cell, n_sets: int) -> list[PromptSet]:
    c, p = cell
    if len(m1_prompt_sets) < n_sets:
        raise ValueError(
            f"cell {cell.prompt_file_key} reuses the M1 prompts, which have "
            f"{len(m1_prompt_sets)} sets, not the {n_sets} sets it needs"
        )
    out = []
    for s in range(n_sets):
        prompts = [list(q) for q in m1_prompt_sets[s][:c]]
        if len(prompts) != c:
            raise ValueError(
                f"cell {cell.prompt_file_key} reuses the M1 prompts, whose set {s} has "
                f"{len(prompts)} prompts, not the {c} prompts it needs"
            )
        if any(len(q) != p for q in prompts):
            lengths = sorted({len(q) for q in prompts})
            raise ValueError(
                f"cell {cell.prompt_file_key} reuses the M1 prompts, but set {s} has a "
                f"prompt that is not {p} tokens long: {lengths}"
            )
        out.append(prompts)
    return out


def build_cell_prompt_sets(
    tokenizer,
    texts: list[str],
    cells,
    n_sets: int,
    seed: int,
    m1_prompt_sets: list[PromptSet] | None = None,
    m1_prompt_len: int | None = None,
) -> dict[Cell, list[PromptSet]]:
    """Per cell (C, P), n_sets sets of C distinct prompts of exactly P tokens."""
    if (m1_prompt_sets is None) != (m1_prompt_len is None):
        raise ValueError(
            "m1_prompt_sets and m1_prompt_len go together: the M1 prompts are reused "
            "for the cells of their length"
        )
    cells = [tuple(cell) for cell in cells]
    if len(set(cells)) != len(cells):
        raise ValueError(f"cells {cells} repeat a cell")
    frame = chat_frame(tokenizer)
    for c, p in cells:
        if c < 1 or frame.room(p) <= 0:
            raise ValueError(
                f"cell {Cell(c, p).prompt_file_key}: need C >= 1 and P longer than the "
                f"{frame.n_tokens}-token chat frame"
            )
    pool = list(dict.fromkeys(texts))
    random.Random(seed + CELL_SEED_OFFSET).shuffle(pool)
    stream = iter(pool)
    consumed = 0
    seen: set[tuple[int, ...]] = set()
    out: dict[Cell, list[PromptSet]] = {}
    for c, p in cells:
        cell = Cell(c, p)
        if m1_prompt_sets is not None and p == m1_prompt_len:
            out[cell] = _reused_m1_sets(m1_prompt_sets, cell, n_sets)
            continue
        room = frame.room(p)
        sets = []
        for s in range(n_sets):
            prompts: PromptSet = []
            while len(prompts) < c:
                content: list[int] = []
                while len(content) < room:
                    text = next(stream, None)
                    if text is None:
                        raise ValueError(
                            f"ran out of texts after {consumed} of {len(pool)} distinct ones while "
                            f"building prompt {len(prompts)} of set {s} of cell "
                            f"{cell.prompt_file_key} ({room} content tokens per prompt) for "
                            f"cells {[Cell(*key).prompt_file_key for key in cells]} x {n_sets} "
                            "sets; supply more texts"
                        )
                    consumed += 1
                    content.extend(_encode_content(tokenizer, text + "\n\n"))
                prompt = fixed_length_ids(frame, content, p)
                if tuple(prompt) in seen:
                    continue
                seen.add(tuple(prompt))
                prompts.append(prompt)
            sets.append(prompts)
        out[cell] = sets
    return out


def check_cell_prompt_sets(by_key, cells, n_sets: int) -> None:
    """Raise ValueError unless `by_key` serves every cell: n_sets sets of C prompts of P ids."""
    if not isinstance(by_key, dict):
        raise ValueError(
            f"the cell prompt sets are not an object keyed by <C>x<P>: {type(by_key).__name__}"
        )
    for c, p in cells:
        key = Cell(c, p).prompt_file_key
        if key not in by_key:
            raise ValueError(
                f"cell {key} is missing from the cell prompt sets (they have {sorted(by_key)})"
            )
        sets = by_key[key]
        if not isinstance(sets, list) or len(sets) < n_sets:
            got = f"{len(sets)} sets" if isinstance(sets, list) else f"a {type(sets).__name__}"
            raise ValueError(f"cell {key} needs a list of {n_sets} sets, got {got}")
        for s, prompts in enumerate(sets[:n_sets]):
            if not isinstance(prompts, list) or len(prompts) < c:
                got = (
                    f"{len(prompts)} prompts"
                    if isinstance(prompts, list)
                    else f"a {type(prompts).__name__}"
                )
                raise ValueError(f"cell {key}, set {s}: needs {c} prompts, has {got}")
            for i, prompt in enumerate(prompts[:c]):
                if not (isinstance(prompt, list) and all(type(t) is int for t in prompt)):
                    raise ValueError(f"cell {key}, set {s}, prompt {i} is not a list of token ids")
                if len(prompt) != p:
                    raise ValueError(
                        f"cell {key}, set {s}, prompt {i} has {len(prompt)} tokens, not {p}"
                    )
