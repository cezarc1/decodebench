import hashlib
import json
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import override

import pytest

from fp4bench import prep, settings
from fp4bench.core.types import Cell, ChatFrame
from fp4bench.studies import expc as expc_study
from fp4bench.studies import model
from tests.golden.make_identity import QWEN3_SPECIAL, QWEN3_TEMPLATE


class FakeTokenizer:
    def __init__(
        self,
        honours_split: bool = True,
        template: str = QWEN3_TEMPLATE,
        special: dict[str, int] | None = None,
        extra_special_tokens: tuple = (),
    ):
        self.honours_split = honours_split
        self.template = template
        self.special = dict(QWEN3_SPECIAL if special is None else special)
        self.all_special_tokens = [*self.special, *extra_special_tokens]
        self.encode_calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        assert tokenize is False and add_generation_prompt is True
        return self.template.format(content=messages[0]["content"])

    def encode(self, text, add_special_tokens=True, split_special_tokens=False):
        self.encode_calls.append((text, add_special_tokens, split_special_tokens))
        if text in self.special and not (split_special_tokens and self.honours_split):
            return [self.special[text]]
        return [ord(c) for c in text]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Fake huggingface_hub and transformers modules, and settings paths inside tmp_path."""
    state = types.SimpleNamespace(
        downloads=[], tokenizer=FakeTokenizer(), dir_at_download=None, ignore_patterns=[]
    )

    def snapshot_download(repo_id, revision=None, *, local_dir, ignore_patterns=None):
        local = Path(local_dir)
        state.dir_at_download = sorted(p.name for p in local.iterdir()) if local.is_dir() else None
        state.downloads.append(("snapshot", repo_id, revision, local_dir))
        state.ignore_patterns.append(ignore_patterns)
        local.mkdir(parents=True, exist_ok=True)
        (local / "model.safetensors").write_text("fresh")
        return local_dir

    def hf_hub_download(repo, filename, repo_type=None, revision=None, *, local_dir):
        state.downloads.append(("file", repo, revision, filename))
        return str(Path(local_dir) / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download, hf_hub_download=hf_hub_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: state.tokenizer)
        ),
    )
    for name, value in {
        "BF16_MODEL_DIR": tmp_path / "bf16",
        "MX_MODEL_DIR": tmp_path / "mx",
        "NV_MODEL_DIR": tmp_path / "nv",
        "NVX_MODEL_DIR": tmp_path / "nvx",
    }.items():
        monkeypatch.setattr(model, name, str(value))
    for name, value in {
        "DATA_DIR": tmp_path / "data",
        "SHAREGPT_PATH": tmp_path / "data" / "sharegpt.json",
        "M1_PROMPTS_PATH": tmp_path / "data" / "m1.json",
        "NLL_PROMPTS_PATH": tmp_path / "data" / "nll.json",
        "C_PROMPTS_PATH": tmp_path / "data" / "c.json",
    }.items():
        monkeypatch.setattr(settings, name, str(value))
    (tmp_path / "data").mkdir()
    return state


@pytest.fixture
def builders(monkeypatch):
    """Stub the prompt builders so the tests need no real tokenizer or ShareGPT."""
    calls = types.SimpleNamespace(m1=0, nll=0)

    def build_m1(tok, texts, length, n_sets, set_size, seed):
        calls.m1 += 1
        return [[[1, 2, calls.m1, 3] for _ in range(2)] for _ in range(3)]

    def build_nll(tok, texts, n, length, seed):
        calls.nll += 1
        return [[calls.nll] * 3 for _ in range(5)]

    monkeypatch.setattr(prep, "load_sharegpt_user_texts", lambda path: ["a", "b", "c"])
    monkeypatch.setattr(prep, "build_m1_prompt_sets", build_m1)
    monkeypatch.setattr(prep, "build_nll_prompts", build_nll)
    monkeypatch.setattr(prep, "chat_frame", lambda tok: ChatFrame([1, 2], [3]))
    return calls


def _append(path: str, text: str) -> None:
    with Path(path).open("a") as f:
        f.write(text)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_prepare_all_builds_prompts_and_returns_their_sha256(hub, builders):
    summary = prep.prepare_all()
    assert (builders.m1, builders.nll) == (1, 1)
    assert summary["m1_prompts_sha256"] == sha(settings.M1_PROMPTS_PATH)
    assert summary["nll_prompts_sha256"] == sha(settings.NLL_PROMPTS_PATH)
    assert summary["prompts_rebuilt"] is True
    assert (
        summary["sharegpt_texts"],
        summary["m1_sets"],
        summary["m1_set_size"],
        summary["m1_prompt_len"],
        summary["nll_prompts"],
        summary["chat_frame_tokens"],
    ) == (3, 3, 2, 4, 5, 3)
    assert not Path(settings.M1_PROMPTS_PATH + ".tmp").exists()


def test_prepare_all_does_not_regenerate_existing_prompts(hub, builders):
    prep.prepare_all()
    before = (sha(settings.M1_PROMPTS_PATH), sha(settings.NLL_PROMPTS_PATH))
    summary = prep.prepare_all()
    assert (builders.m1, builders.nll) == (1, 1)
    assert (sha(settings.M1_PROMPTS_PATH), sha(settings.NLL_PROMPTS_PATH)) == before
    assert (summary["m1_prompts_sha256"], summary["nll_prompts_sha256"]) == before
    assert summary["prompts_rebuilt"] is False
    assert (
        summary["m1_sets"],
        summary["m1_set_size"],
        summary["m1_prompt_len"],
        summary["nll_prompts"],
    ) == (3, 2, 4, 5)


def test_prepare_all_force_regenerates_existing_prompts(hub, builders):
    first = prep.prepare_all()
    summary = prep.prepare_all(force=True)
    assert (builders.m1, builders.nll) == (2, 2)
    assert summary["prompts_rebuilt"] is True
    assert summary["m1_prompts_sha256"] != first["m1_prompts_sha256"]


def test_prepare_all_rebuilds_both_when_only_one_prompt_file_exists(hub, builders):
    prep.prepare_all()
    Path(settings.NLL_PROMPTS_PATH).unlink()
    summary = prep.prepare_all()
    assert (builders.m1, builders.nll) == (2, 2)
    assert summary["prompts_rebuilt"] is True
    assert Path(settings.NLL_PROMPTS_PATH).exists()


def test_prepare_all_downloads_pinned_inputs_even_when_prompts_exist(hub, builders):
    prep.prepare_all()
    hub.downloads.clear()
    prep.prepare_all()
    assert [d[:3] for d in hub.downloads] == [
        ("snapshot", model.SRC_MODEL_ID, model.SRC_MODEL_REVISION),
        ("file", settings.SHAREGPT_REPO, settings.SHAREGPT_REVISION),
    ]


def test_prepare_all_checks_that_the_tokenizer_honours_split_special_tokens(hub, builders):
    prep.prepare_all()
    assert ("<|im_end|>", False, True) in hub.tokenizer.encode_calls
    assert all(call[0] != "<|end|>" for call in hub.tokenizer.encode_calls)


def test_prepare_all_rejects_a_tokenizer_that_ignores_split_special_tokens(hub, builders):
    hub.tokenizer = FakeTokenizer(honours_split=False)
    with pytest.raises(RuntimeError, match="split_special_tokens"):
        prep.prepare_all()
    assert (builders.m1, builders.nll) == (0, 0)
    assert not Path(settings.M1_PROMPTS_PATH).exists()


def test_the_old_probe_was_vacuous_on_qwen3_and_the_new_one_is_not(hub):
    tok = FakeTokenizer(honours_split=False)
    assert len(tok.encode("<|end|>", add_special_tokens=False, split_special_tokens=True)) > 1
    with pytest.raises(RuntimeError, match="split_special_tokens"):
        prep._require_split_special_tokens(tok)
    prep._require_split_special_tokens(FakeTokenizer(honours_split=True))


def test_the_probe_runs_when_the_prompts_exist_and_are_kept(hub, builders):
    prep.prepare_all()
    hub.tokenizer.encode_calls.clear()
    summary = prep.prepare_all()
    assert summary["prompts_rebuilt"] is False
    assert ("<|im_end|>", False, True) in hub.tokenizer.encode_calls


def test_a_failed_probe_on_the_skip_path_raises_and_leaves_the_existing_prompts_alone(
    hub, builders
):
    prep.prepare_all()
    before = (sha(settings.M1_PROMPTS_PATH), sha(settings.NLL_PROMPTS_PATH))
    hub.tokenizer = FakeTokenizer(honours_split=False)
    with pytest.raises(RuntimeError, match="split_special_tokens") as exc:
        prep.prepare_all()
    assert "existing prompt" in str(exc.value) and "suspect" in str(exc.value)
    assert (sha(settings.M1_PROMPTS_PATH), sha(settings.NLL_PROMPTS_PATH)) == before
    assert (builders.m1, builders.nll) == (1, 1)


def test_the_probe_names_the_token_that_did_not_split(hub):
    tok = FakeTokenizer(honours_split=False)
    with pytest.raises(RuntimeError) as exc:
        prep._require_split_special_tokens(tok)
    assert "<|im_end|>" in str(exc.value) or "<|endoftext|>" in str(exc.value)


def test_a_special_token_that_is_already_several_ids_cannot_show_the_flag_and_is_skipped(hub):
    tok = FakeTokenizer(special={"<|im_end|>": 151645}, extra_special_tokens=("[not-in-vocab]",))
    prep._require_split_special_tokens(tok)
    assert ("[not-in-vocab]", False, True) not in tok.encode_calls
    assert ("<|im_end|>", False, True) in tok.encode_calls


def test_every_usable_special_token_must_split(hub):
    class PartlyBroken(FakeTokenizer):
        @override
        def encode(self, text, add_special_tokens=True, split_special_tokens=False):
            if text == "<|im_start|>":
                return [self.special[text]]
            return super().encode(text, add_special_tokens, split_special_tokens)

    with pytest.raises(RuntimeError, match="<\\|im_start\\|>"):
        prep._require_split_special_tokens(PartlyBroken())


def test_a_tokenizer_without_a_usable_special_token_is_refused_not_trusted(hub):
    for tok in (
        FakeTokenizer(special={}),
        FakeTokenizer(special={}, extra_special_tokens=("[x]",)),
    ):
        with pytest.raises(RuntimeError, match="no special token"):
            prep._require_split_special_tokens(tok)


def test_duplicate_special_tokens_are_probed_once(hub):
    tok = FakeTokenizer(special={"<|im_end|>": 151645}, extra_special_tokens=("<|im_end|>",))
    prep._require_split_special_tokens(tok)
    assert tok.encode_calls.count(("<|im_end|>", False, True)) == 1


def test_prepare_all_downloads_qwen3_into_the_bf16_dir_ignoring_only_pickles(hub, builders):
    prep.prepare_all()
    snapshot = [d for d in hub.downloads if d[0] == "snapshot"]
    assert snapshot == [
        ("snapshot", "Qwen/Qwen3-32B", model.SRC_MODEL_REVISION, model.BF16_MODEL_DIR)
    ]
    assert hub.ignore_patterns == [["*.pt", "*.bin"]]


def test_prepare_all_replaces_prompts_built_with_another_chat_template(hub, builders):
    prep.prepare_all()
    old = [[[200006, 17360, 200008, 7, 200007] for _ in range(2)] for _ in range(3)]
    with Path(settings.M1_PROMPTS_PATH).open("w") as f:
        json.dump(old, f)
    summary = prep.prepare_all()
    assert (builders.m1, builders.nll) == (2, 2)
    assert summary["prompts_rebuilt"] is True
    assert json.loads(Path(settings.M1_PROMPTS_PATH).read_text())[0][0][:2] == [1, 2]


def test_prepare_all_replaces_an_empty_m1_prompt_file(hub, builders):
    prep.prepare_all()
    with Path(settings.M1_PROMPTS_PATH).open("w") as f:
        json.dump([], f)
    assert prep.prepare_all()["prompts_rebuilt"] is True


def test_prepare_all_keeps_prompts_that_carry_the_current_chat_frame(hub, builders):
    prep.prepare_all()
    assert prep.prepare_all()["prompts_rebuilt"] is False
    assert (builders.m1, builders.nll) == (1, 1)


@pytest.mark.parametrize(
    "template",
    [
        "<|im_start|>user\n{content}{content}<|im_end|>\n",
        "<|im_start|>user\nfixed<|im_end|>\n",
    ],
)
def test_prepare_all_refuses_a_template_that_does_not_render_the_content_exactly_once(
    hub, template, monkeypatch
):
    hub.tokenizer = FakeTokenizer(template=template)
    monkeypatch.setattr(prep, "load_sharegpt_user_texts", lambda path: ["a"])
    with pytest.raises(ValueError, match="exactly once"):
        prep.prepare_all()
    assert not Path(settings.M1_PROMPTS_PATH).exists()


def test_prepare_all_builds_with_the_qwen3_chat_frame(hub, monkeypatch):
    seen = {}

    def build_m1(tok, texts, length, n_sets, set_size, seed):
        pre, post = prep.chat_frame(tok)
        seen["frame"] = ("".join(map(chr, pre)), "".join(map(chr, post)))
        return [[[*pre, 9, *post]]]

    monkeypatch.setattr(prep, "load_sharegpt_user_texts", lambda path: ["a"])
    monkeypatch.setattr(prep, "build_m1_prompt_sets", build_m1)
    monkeypatch.setattr(prep, "build_nll_prompts", lambda *a: [[1]])
    summary = prep.prepare_all()
    assert seen["frame"] == ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
    assert summary["chat_frame_tokens"] == len(seen["frame"][0]) + len(seen["frame"][1])
    assert prep.prepare_all()["prompts_rebuilt"] is False


KINDS = {"mx": "mx", "nv": "nv", "nvidia": "nvx"}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_fetch_checkpoint_mirrors_the_published_revision_exactly(hub, tmp_path, kind):
    target = tmp_path / KINDS[kind]
    target.mkdir()
    (target / "stale.safetensors").write_text("from an earlier quantization")
    (target / "hf_quant_config.json").write_text("{}")
    out = prep.fetch_checkpoint(kind, "someone/Qwen3-32B-X", "a" * 40)
    assert hub.dir_at_download is None or hub.dir_at_download == []
    assert sorted(p.name for p in target.iterdir()) == ["fp4bench_fetch.json", "model.safetensors"]
    assert (target / "model.safetensors").read_text() == "fresh"
    assert out == {
        "kind": kind,
        "repo_id": "someone/Qwen3-32B-X",
        "revision": "a" * 40,
        "local_dir": str(target),
    }
    assert hub.downloads[-1] == ("snapshot", "someone/Qwen3-32B-X", "a" * 40, str(target))


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_fetch_checkpoint_touches_only_its_own_dir(hub, tmp_path, kind):
    others = [tmp_path / d for k, d in KINDS.items() if k != kind]
    for d in others:
        d.mkdir()
        (d / "keep.txt").write_text("another checkpoint")
    prep.fetch_checkpoint(kind, "someone/name", "b" * 40)
    assert all((d / "keep.txt").read_text() == "another checkpoint" for d in others)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_fetch_checkpoint_records_where_the_checkpoint_came_from(hub, tmp_path, kind):
    prep.fetch_checkpoint(kind, "someone/name", "b" * 40)
    record = json.loads((tmp_path / KINDS[kind] / prep.FETCH_FILE).read_text())
    assert prep.FETCH_FILE == "fp4bench_fetch.json"
    assert set(record) == {"repo_id", "revision", "utc"}
    assert (record["repo_id"], record["revision"]) == ("someone/name", "b" * 40)
    assert datetime.fromisoformat(record["utc"]).tzinfo is not None


def test_fetch_checkpoint_works_when_nothing_was_there_before(hub, tmp_path):
    assert not (tmp_path / "mx").exists()
    prep.fetch_checkpoint("mx", "someone/name", "c" * 40)
    assert (tmp_path / "mx" / "model.safetensors").exists()


def test_fetch_checkpoint_rejects_an_unknown_kind_before_touching_anything(hub, tmp_path):
    (tmp_path / "nv").mkdir()
    (tmp_path / "nv" / "keep.txt").write_text("x")
    expected = "unknown checkpoint kind 'bf16'; expected one of ['mx', 'nv', 'nvidia']"
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        prep.fetch_checkpoint("bf16", "someone/name", "d" * 40)
    assert hub.downloads == [] and (tmp_path / "nv" / "keep.txt").exists()


FRAME = len("<|im_start|>user\n") + len("<|im_end|>\n<|im_start|>assistant\n")


def _sharegpt_text(i: int) -> str:
    """About 80 distinct characters, so that even a 10-id cut of each is a new prompt."""
    return (f"{i:05d} conversation " + "lorem ipsum dolor sit amet "[i % 7 :] * 3).strip()


@pytest.fixture
def expc(hub, monkeypatch):
    """What `fp4bench prepare` leaves on the volume, at small sizes."""
    from fp4bench.prompts import build_m1_prompt_sets

    monkeypatch.setattr(settings, "M1_INPUT_LEN", 70)
    monkeypatch.setattr(settings, "M1_SET_SIZE", 4)
    cells = (Cell(1, 70), Cell(1, 300), Cell(2, 120), Cell(3, 70), Cell(4, 60))
    monkeypatch.setattr(expc_study, "C_CELLS", cells)
    texts = [_sharegpt_text(i) for i in range(400)]
    conversations = [
        {"conversations": [{"from": "human", "value": t}, {"from": "gpt", "value": "ok"}]}
        for t in texts
    ]
    with Path(settings.SHAREGPT_PATH).open("w") as f:
        json.dump(conversations, f)
    Path(model.BF16_MODEL_DIR).mkdir(parents=True)
    m1 = build_m1_prompt_sets(
        hub.tokenizer,
        texts,
        settings.M1_INPUT_LEN,
        settings.M1_SETS,
        settings.M1_SET_SIZE,
        settings.SEED,
    )
    with Path(settings.M1_PROMPTS_PATH).open("w") as f:
        json.dump(m1, f)
    calls = []
    real = prep.build_cell_prompt_sets
    monkeypatch.setattr(
        prep, "build_cell_prompt_sets", lambda *a, **k: calls.append((a, k)) or real(*a, **k)
    )
    return types.SimpleNamespace(m1=m1, texts=texts, builds=calls)


def c_record():
    return json.loads(Path(settings.C_PROMPTS_PATH).read_text())


def test_prepare_expc_builds_every_cell_from_the_volumes_sharegpt_and_tokenizer(hub, expc):
    summary = prep.prepare_expc()
    record = c_record()
    assert list(record["cells"]) == ["1x70", "1x300", "2x120", "3x70", "4x60"]
    for key, sets in record["cells"].items():
        c, p = map(int, key.split("x"))
        assert len(sets) == settings.M1_SETS == 6
        assert all(len(s) == c and all(len(q) == p for q in s) for s in sets)
    assert record["cells"]["1x70"] == [[expc.m1[s][0]] for s in range(6)]
    assert record["cells"]["3x70"] == [expc.m1[s][:3] for s in range(6)]
    assert record["seed"] == settings.SEED and record["n_sets"] == 6
    assert record["m1_prompts_sha256"] == sha(settings.M1_PROMPTS_PATH)
    assert record["reused_m1_cells"] == ["1x70", "3x70"]
    assert summary["c_prompts_sha256"] == sha(settings.C_PROMPTS_PATH)
    assert summary["m1_prompts_sha256"] == sha(settings.M1_PROMPTS_PATH)
    assert summary["prompts_rebuilt"] is True and summary["sharegpt_texts"] == 400
    assert summary["cells"]["2x120"] == {
        "sets": 6,
        "set_size": 2,
        "prompt_len": 120,
        "reused_m1": False,
    }
    assert summary["cells"]["1x70"]["reused_m1"] is True
    assert summary["total_prompt_tokens"] == 6 * (70 + 300 + 240 + 210 + 240)
    assert summary["chat_frame_tokens"] == FRAME
    assert not Path(settings.C_PROMPTS_PATH + ".tmp").exists()
    ((args, kwargs),) = expc.builds
    assert args[1] == expc.texts
    assert kwargs == {"m1_prompt_sets": expc.m1, "m1_prompt_len": 70}


def test_prepare_expc_downloads_nothing(hub, expc):
    prep.prepare_expc()
    assert hub.downloads == []


def test_prepare_expc_prompts_carry_the_chat_frame_of_the_tokenizer(hub, expc):
    prep.prepare_expc()
    pre = [ord(ch) for ch in "<|im_start|>user\n"]
    post = [ord(ch) for ch in "<|im_end|>\n<|im_start|>assistant\n"]
    for sets in c_record()["cells"].values():
        for q in (q for s in sets for q in s):
            assert q[: len(pre)] == pre and q[-len(post) :] == post


def test_prepare_expc_keeps_an_existing_file_that_still_fits(hub, expc):
    first = prep.prepare_expc()
    summary = prep.prepare_expc()
    assert len(expc.builds) == 1
    assert summary["prompts_rebuilt"] is False and summary["rebuild_reason"] is None
    assert summary["c_prompts_sha256"] == first["c_prompts_sha256"]
    assert summary["cells"] == first["cells"] and summary["sharegpt_texts"] is None


def test_prepare_expc_force_rebuilds_the_same_bytes(hub, expc):
    first = prep.prepare_expc()
    summary = prep.prepare_expc(force=True)
    assert len(expc.builds) == 2 and summary["prompts_rebuilt"] is True
    assert summary["c_prompts_sha256"] == first["c_prompts_sha256"]


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda: _append(settings.M1_PROMPTS_PATH, " "), "M1 prompts"),
        (lambda: setattr(expc_study, "C_CELLS", (Cell(1, 70), Cell(2, 120))), "cells"),
        (lambda: Path(settings.C_PROMPTS_PATH).write_text('{"cells": '), "not readable"),
        (lambda: Path(settings.C_PROMPTS_PATH).write_text("[]"), "not readable"),
    ],
)
def test_prepare_expc_rebuilds_a_file_that_no_longer_fits(hub, expc, change, reason):
    prep.prepare_expc()
    change()
    summary = prep.prepare_expc()
    assert summary["prompts_rebuilt"] is True and reason in summary["rebuild_reason"]
    assert len(expc.builds) == 2
    assert c_record()["m1_prompts_sha256"] == sha(settings.M1_PROMPTS_PATH)


def test_prepare_expc_rebuilds_prompts_made_with_another_chat_template(hub, expc):
    prep.prepare_expc()
    record = c_record()
    record["cells"]["2x120"][0][0][0] += 1
    with Path(settings.C_PROMPTS_PATH).open("w") as f:
        json.dump(record, f)
    summary = prep.prepare_expc()
    assert summary["prompts_rebuilt"] is True and "chat-template" in summary["rebuild_reason"]


def test_prepare_expc_rebuilds_a_file_that_cannot_serve_its_cells(hub, expc):
    prep.prepare_expc()
    record = c_record()
    record["cells"]["4x60"][2].pop()
    with Path(settings.C_PROMPTS_PATH).open("w") as f:
        json.dump(record, f)
    summary = prep.prepare_expc()
    assert summary["prompts_rebuilt"] is True and "4x60" in summary["rebuild_reason"]


@pytest.mark.parametrize(
    "home, missing",
    [(settings, "SHAREGPT_PATH"), (settings, "M1_PROMPTS_PATH"), (model, "BF16_MODEL_DIR")],
)
def test_prepare_expc_needs_what_prepare_puts_on_the_volume(hub, expc, home, missing):
    path = getattr(home, missing)
    if Path(path).is_dir():
        Path(path).rmdir()
    else:
        Path(path).unlink()
    with pytest.raises(FileNotFoundError, match="fp4bench prepare") as exc:
        prep.prepare_expc()
    assert path in str(exc.value)
    assert not Path(settings.C_PROMPTS_PATH).exists()


def test_prepare_expc_refuses_m1_prompts_of_another_chat_template(hub, expc):
    with Path(settings.M1_PROMPTS_PATH).open("w") as f:
        json.dump([[[200006, 1, 200007]]], f)
    with pytest.raises(RuntimeError, match=r"chat-template frame.*fp4bench prepare"):
        prep.prepare_expc()
    assert not Path(settings.C_PROMPTS_PATH).exists()


def test_prepare_expc_refuses_a_tokenizer_that_ignores_split_special_tokens(hub, expc):
    prep.prepare_expc()
    before = sha(settings.C_PROMPTS_PATH)
    hub.tokenizer = FakeTokenizer(honours_split=False)
    with pytest.raises(RuntimeError, match="split_special_tokens"):
        prep.prepare_expc()
    assert sha(settings.C_PROMPTS_PATH) == before


def test_prepare_expc_says_so_when_sharegpt_has_too_little_text(hub, expc, monkeypatch):
    monkeypatch.setattr(expc_study, "C_CELLS", (Cell(1, 70), Cell(64, 300)))
    with pytest.raises(ValueError, match="ran out of texts") as exc:
        prep.prepare_expc()
    assert any(settings.SHAREGPT_PATH in note for note in exc.value.__notes__)
    assert not Path(settings.C_PROMPTS_PATH).exists()
