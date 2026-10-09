import json
from typing import Any, override

import pytest

from fp4bench import prompts as pr
from fp4bench.core.types import Cell, ChatFrame


class FakeTokenizer:
    """Whitespace tokenizer: each word becomes its length. The chat frame is 2 + 2 tokens."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize is False
        return f"<s> <user> {messages[0]['content']} </user> <assistant>"

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [len(w) for w in text.split()]


class SpecialTokenizer(FakeTokenizer):
    """Like a real tokenizer: "<|end|>" becomes special id 999 unless special tokens are split."""

    def __init__(self, template="<s> <user> {content} </user> <assistant>"):
        self.template = template

    @override
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize is False
        return self.template.format(content=messages[0]["content"])

    @override
    def encode(self, text, add_special_tokens=False, split_special_tokens=False, **kwargs):
        return [
            999 if w == "<|end|>" and not split_special_tokens else len(w) for w in text.split()
        ]


TEXTS = [" ".join("x" * ((i * 7 + j) % 9 + 1) for j in range(40)) for i in range(200)]


def test_load_sharegpt_user_texts(tmp_path):
    data = [
        {"conversations": [{"from": "human", "value": " hello "}, {"from": "gpt", "value": "hi"}]},
        {"conversations": [{"from": "gpt", "value": "x"}, {"from": "human", "value": "second"}]},
        {"conversations": [{"from": "human", "value": "   "}]},
        {
            "conversations": [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": "hm"},
                {"from": "human", "value": "later"},
            ]
        },
        {"conversations": []},
    ]
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(data))
    assert pr.load_sharegpt_user_texts(str(path)) == ["hello", "second"]


def test_chat_frame_splits_around_content():
    pre, post = pr.chat_frame(FakeTokenizer())
    assert (len(pre), len(post)) == (2, 2)


def test_chat_frame_keeps_real_special_tokens():
    tok = SpecialTokenizer("<s> <user> {content} <|end|> <assistant>")
    _, post = pr.chat_frame(tok)
    assert post[0] == 999


@pytest.mark.parametrize(
    "template",
    [
        "<s> <user> </user> <assistant>",
        "<s> <user> {content} </user> {content} <assistant>",
    ],
)
def test_chat_frame_requires_content_exactly_once(template):
    with pytest.raises(ValueError, match="exactly once"):
        pr.chat_frame(SpecialTokenizer(template))


def test_fixed_length_ids_exact_length():
    ids = pr.fixed_length_ids(ChatFrame([1, 1], [2, 2]), list(range(100)), 50)
    assert len(ids) == 50 and ids[:2] == [1, 1] and ids[-2:] == [2, 2] and ids[2] == 0


def test_fixed_length_ids_rejects_short_content():
    with pytest.raises(ValueError, match="need 46 content tokens, got 10"):
        pr.fixed_length_ids(ChatFrame([1, 1], [2, 2]), [0] * 10, 50)


def test_a_chat_frame_measures_and_recognises_its_frame():
    frame = ChatFrame([1, 1], [2, 2, 2])
    assert (frame.n_tokens, frame.room(50)) == (5, 45)
    assert frame.frames([1, 1, 7, 2, 2, 2]) and frame.frames([1, 1, 2, 2, 2])
    assert not frame.frames([1, 7, 2, 2, 2]) and not frame.frames([1, 1, 7, 2, 2])


def test_m1_prompt_sets_shapes_and_determinism():
    tok = FakeTokenizer()
    a = pr.build_m1_prompt_sets(tok, TEXTS, length=20, n_sets=3, set_size=4, seed=0)
    b = pr.build_m1_prompt_sets(tok, TEXTS, length=20, n_sets=3, set_size=4, seed=0)
    c = pr.build_m1_prompt_sets(tok, TEXTS, length=20, n_sets=3, set_size=4, seed=1)
    assert len(a) == 3 and all(len(s) == 4 for s in a)
    assert all(len(p) == 20 for s in a for p in s)
    assert a == b and a != c
    assert a[0][0] != a[0][1]


def test_m1_prompt_sets_do_not_turn_user_text_into_special_tokens():
    texts = [" ".join(["<|end|>", "ab", "cde"] * 20) for _ in range(50)]
    sets = pr.build_m1_prompt_sets(
        SpecialTokenizer(), texts, length=20, n_sets=2, set_size=3, seed=0
    )
    assert all(999 not in p for s in sets for p in s)


def test_m1_prompt_sets_raise_clear_error_when_texts_run_out():
    with pytest.raises(ValueError, match="ran out of texts"):
        pr.build_m1_prompt_sets(FakeTokenizer(), TEXTS[:2], length=20, n_sets=3, set_size=4, seed=0)


def test_nll_prompts_do_not_turn_user_text_into_special_tokens():
    texts = [" ".join(["<|end|>", "ab", "cde"] * 20) for _ in range(10)]
    out = pr.build_nll_prompts(SpecialTokenizer(), texts, n=5, length=30, seed=0)
    assert all(999 not in ids for ids in out)


def test_nll_prompts_are_fixed_windows():
    out = pr.build_nll_prompts(FakeTokenizer(), TEXTS, n=5, length=30, seed=0)
    assert len(out) == 5 and all(len(ids) == 30 for ids in out)


def _c_text(i: int) -> str:
    """25 words, the first three spelling i in base 9, so that the 400 texts are distinct."""
    head = [(i // 9**k) % 9 + 1 for k in range(3)]
    return " ".join("y" * n for n in head + [(i * 5 + j * 3) % 9 + 1 for j in range(22)])


C_TEXTS = [_c_text(i) for i in range(400)]
CELLS = ((1, 20), (1, 60), (4, 12), (3, 30))


def m1_sets(n_sets=6, set_size=4, length=20):
    return pr.build_m1_prompt_sets(
        FakeTokenizer(), TEXTS, length=length, n_sets=n_sets, set_size=set_size, seed=0
    )


def test_cell_prompt_sets_have_n_sets_of_c_prompts_of_exactly_p_tokens_in_the_frame():
    out = pr.build_cell_prompt_sets(FakeTokenizer(), C_TEXTS, CELLS, n_sets=6, seed=0)
    assert list(out) == list(CELLS)
    pre, post = pr.chat_frame(FakeTokenizer())
    for (c, p), sets in out.items():
        assert len(sets) == 6 and all(len(s) == c for s in sets)
        for prompt in (q for s in sets for q in s):
            assert len(prompt) == p
            assert prompt[: len(pre)] == pre and prompt[len(prompt) - len(post) :] == post


def test_cell_prompt_content_is_consecutive_texts_joined_by_a_blank_line_then_cut():
    tok = FakeTokenizer()
    pre, post = pr.chat_frame(tok)
    (sets,) = pr.build_cell_prompt_sets(tok, C_TEXTS, ((1, 60),), n_sets=1, seed=0).values()
    room = 60 - len(pre) - len(post)
    pool = list(dict.fromkeys(C_TEXTS))
    pr.random.Random(pr.CELL_SEED_OFFSET).shuffle(pool)
    expected = [i for text in pool[:3] for i in tok.encode(text + "\n\n")][:room]
    assert sets[0][0] == pre + expected + post


def test_cell_prompts_are_distinct_within_a_set_across_sets_and_across_cells():
    out = pr.build_cell_prompt_sets(FakeTokenizer(), C_TEXTS, CELLS, n_sets=6, seed=0)
    prompts = [tuple(q) for sets in out.values() for s in sets for q in s]
    assert len(set(prompts)) == len(prompts)


def test_cell_prompts_skip_duplicate_texts_and_prompts_with_the_same_content():
    same_start = [C_TEXTS[0] + " tail" * (i + 1) for i in range(5)]
    texts = [C_TEXTS[0]] * 5 + same_start + C_TEXTS[1:40]
    out = pr.build_cell_prompt_sets(FakeTokenizer(), texts, ((3, 14),), n_sets=6, seed=0)
    prompts = [tuple(q) for s in out[Cell(3, 14)] for q in s]
    assert len(set(prompts)) == len(prompts) == 18


def test_cell_prompt_sets_are_deterministic_for_a_seed():
    tok = FakeTokenizer()
    a = pr.build_cell_prompt_sets(tok, C_TEXTS, CELLS, n_sets=6, seed=0)
    assert a == pr.build_cell_prompt_sets(tok, C_TEXTS, CELLS, n_sets=6, seed=0)
    assert a != pr.build_cell_prompt_sets(tok, C_TEXTS, CELLS, n_sets=6, seed=1)


def test_cell_prompts_do_not_share_the_m1_prompts_shuffle():
    tok = FakeTokenizer()
    m1 = pr.build_m1_prompt_sets(tok, C_TEXTS, length=20, n_sets=1, set_size=1, seed=0)
    out = pr.build_cell_prompt_sets(tok, C_TEXTS, ((1, 20),), n_sets=1, seed=0)
    assert out[Cell(1, 20)][0][0] != m1[0][0]


def test_a_cell_with_the_m1_prompt_length_reuses_the_main_runs_m1_prompts():
    m1 = m1_sets()
    out = pr.build_cell_prompt_sets(
        FakeTokenizer(),
        C_TEXTS,
        ((1, 20), (3, 20), (1, 60)),
        n_sets=6,
        seed=0,
        m1_prompt_sets=m1,
        m1_prompt_len=20,
    )
    assert out[Cell(1, 20)] == [[m1[s][0]] for s in range(6)]
    assert out[Cell(3, 20)] == [m1[s][:3] for s in range(6)]
    assert all(len(q) == 60 for s in out[Cell(1, 60)] for q in s)


def test_the_reused_cells_do_not_consume_texts():
    tok = FakeTokenizer()
    alone = pr.build_cell_prompt_sets(tok, C_TEXTS, ((1, 60),), n_sets=6, seed=0)
    after = pr.build_cell_prompt_sets(
        tok,
        C_TEXTS,
        ((1, 20), (1, 60)),
        n_sets=6,
        seed=0,
        m1_prompt_sets=m1_sets(),
        m1_prompt_len=20,
    )
    assert after[Cell(1, 60)] == alone[Cell(1, 60)]


def _short_prompt(sets, s, i):
    sets = [[list(q) for q in prompts] for prompts in sets]
    sets[s][i] = sets[s][i][:-1]
    return sets


@pytest.mark.parametrize(
    "m1, match",
    [
        (m1_sets(n_sets=5), "6 sets"),
        (m1_sets(set_size=2), "3 prompts"),
        (m1_sets(length=19), "20 tokens"),
        (_short_prompt(m1_sets(), 5, 2), "20 tokens"),
    ],
)
def test_m1_prompts_that_cannot_serve_a_reused_cell_are_refused_not_replaced(m1, match):
    with pytest.raises(ValueError, match=match):
        pr.build_cell_prompt_sets(
            FakeTokenizer(),
            C_TEXTS,
            ((3, 20),),
            n_sets=6,
            seed=0,
            m1_prompt_sets=m1,
            m1_prompt_len=20,
        )


def test_the_m1_prompts_and_their_length_come_together():
    alone: list[dict[str, Any]] = [{"m1_prompt_sets": m1_sets()}, {"m1_prompt_len": 20}]
    for kwargs in alone:
        with pytest.raises(ValueError, match="m1_prompt"):
            pr.build_cell_prompt_sets(
                FakeTokenizer(), C_TEXTS, ((1, 20),), n_sets=6, seed=0, **kwargs
            )


def test_cell_prompts_do_not_turn_user_text_into_special_tokens():
    texts = [
        " ".join(["<|end|>", "ab", "c" * (i % 7 + 1), "d" * (i // 7 + 1)] * 20) for i in range(60)
    ]
    out = pr.build_cell_prompt_sets(SpecialTokenizer(), texts, ((2, 30),), n_sets=3, seed=0)
    assert all(999 not in q for s in out[Cell(2, 30)] for q in s)


def test_cell_prompts_raise_a_clear_error_when_the_texts_run_out():
    with pytest.raises(ValueError, match="ran out of texts") as exc:
        pr.build_cell_prompt_sets(FakeTokenizer(), C_TEXTS[:10], ((2, 60),), n_sets=6, seed=0)
    assert "2x60" in str(exc.value)


@pytest.mark.parametrize(
    ("cells", "problem"),
    [
        (((1, 4),), "longer than the 4-token chat frame"),
        (((0, 20),), "need C >= 1"),
        (((1, 20), (1, 20)), "repeat a cell"),
    ],
)
def test_cells_that_cannot_be_built_are_refused(cells, problem):
    with pytest.raises(ValueError, match=problem):
        pr.build_cell_prompt_sets(FakeTokenizer(), C_TEXTS, cells, n_sets=1, seed=0)


def test_cell_keys_are_c_x_p():
    assert Cell(128, 360).prompt_file_key == "128x360"


def by_key(cells=CELLS, n_sets=6):
    built = pr.build_cell_prompt_sets(FakeTokenizer(), C_TEXTS, cells, n_sets=n_sets, seed=0)
    return {cell.prompt_file_key: sets for cell, sets in built.items()}


def test_built_cell_prompt_sets_pass_the_check_also_for_fewer_sets_and_a_subset_of_cells():
    pr.check_cell_prompt_sets(by_key(), CELLS, 6)
    pr.check_cell_prompt_sets(by_key(), CELLS[1:3], 4)


def _broken(change):
    data = json.loads(json.dumps(by_key()))
    change(data)
    return data


@pytest.mark.parametrize(
    "data, match",
    [
        ([], "not an object"),
        (_broken(lambda d: d.pop("4x12")), "4x12.*missing"),
        (_broken(lambda d: d.__setitem__("4x12", d["4x12"][:4])), "4x12.*5 sets, got 4 sets"),
        (_broken(lambda d: d.__setitem__("4x12", {"0": []})), "4x12.*5 sets|4x12.*list"),
        (_broken(lambda d: d["4x12"][3].pop()), "4x12.*set 3.*3 prompts"),
        (_broken(lambda d: d["1x60"][4][0].pop()), "1x60.*set 4.*59 tokens"),
        (_broken(lambda d: d["3x30"][0][2].__setitem__(4, 1.5)), "3x30.*set 0.*not .*token ids"),
        (_broken(lambda d: d["3x30"][0][2].__setitem__(4, True)), "3x30.*set 0.*not .*token ids"),
        (_broken(lambda d: d["3x30"][1].__setitem__(1, "abc")), "3x30.*set 1.*not .*token ids"),
    ],
)
def test_cell_prompt_sets_that_cannot_serve_every_cell_are_refused(data, match):
    with pytest.raises(ValueError, match=match):
        pr.check_cell_prompt_sets(data, CELLS, 5)


def test_only_the_sets_and_prompts_a_block_sends_are_checked():
    data = _broken(lambda d: (d["1x60"][5][0].pop(), d["4x12"][0].append([1])))
    pr.check_cell_prompt_sets(data, CELLS, 5)
    with pytest.raises(ValueError, match=r"1x60.*set 5"):
        pr.check_cell_prompt_sets(data, CELLS, 6)
