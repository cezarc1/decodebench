"""quantize(fmt) against fakes of torch, transformers, datasets and llm-compressor."""

import contextlib
import hashlib
import importlib.metadata
import json
import math
import random
import re
import sys
import types
from itertools import pairwise
from pathlib import Path
from typing import override

import numpy as np
import pytest

from fp4bench import quantize as q
from fp4bench import sanity, settings
from fp4bench.core.types import Format
from fp4bench.studies import model as qwen3
from tests.ct_checkpoints import ct_tensors, qwen3_config, unquantize, write_checkpoint

TOKENIZER_BYTES = {
    "tokenizer.json": b'{"model": "\xff\x00 not utf-8 safe"}\n',
    "tokenizer_config.json": b'{"chat_template": "x"}',
    "vocab.json": b'{"a": 0}',
    "merges.txt": b"#version: 0.2\r\na b\n",
    "generation_config.json": b'{"do_sample": true, "temperature": 0.6}',
    "LICENSE": b"Apache License\n",
}
VERSIONS = {"llmcompressor": "0.14.0", "compressed-tensors": "0.19.0"}

V = 5
BIGRAM = np.random.default_rng(7).normal(size=(V, V))
WINDOWS = [[0, 1, 2, 3, 4, 0], [4, 3, 2, 1, 0, 1], [2, 2, 3, 3, 1, 4]]


def expected_nll(window: list[int]) -> float:
    """-log p(token_i | token_{i-1}) averaged over positions 1..L-1, in plain Python."""
    total = 0.0
    for prev, tok in pairwise(window):
        row = BIGRAM[prev]
        total -= row[tok] - math.log(sum(math.exp(x) for x in row))
    return total / (len(window) - 1)


def word_id(word: str) -> int:
    return int.from_bytes(hashlib.md5(word.encode()).digest()[:3], "big")  # noqa: S324 - not security


def all_texts() -> list[str]:
    """300 prompts of 1-7 words; every 50th is 3000 words (longer than the 1024-token cap)."""
    return [f"q{i} " + "w " * (3000 if i % 50 == 0 else i % 7 + 1) for i in range(300)]


class FakeTensor(np.ndarray):
    def float(self):
        return self

    def to(self, *args, **kwargs):
        return self


def cross_entropy(logits, target, reduction="mean"):
    z = np.asarray(logits, dtype=float)
    logp = z - z.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    picked = -logp[np.arange(len(target)), np.asarray(target)]
    assert reduction == "mean"
    return picked.mean()


class FakeLinear:
    quantization_scheme: types.SimpleNamespace


class FakeModel:
    device = "cuda:0"

    def __init__(self, st):
        self.st = st
        self.scheme: str | None = None
        self.modules = [("model.embed_tokens", object()), ("model.norm", object())]
        for layer in range(2):
            for mod in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ):
                self.modules.append((f"model.layers.{layer}.{mod}", FakeLinear()))
        self.modules.append(("lm_head", FakeLinear()))

    def named_modules(self):
        return iter([("", self), *self.modules])

    def __call__(self, input_ids=None, use_cache=True, **kwargs):
        self.st.events.append("forward")
        self.st.forward_kwargs.append({"use_cache": use_cache, **kwargs})
        self.st.contexts_at_forward.append(list(self.st.open_contexts))
        ids = np.asarray(input_ids)
        return types.SimpleNamespace(logits=BIGRAM[ids].view(FakeTensor))

    def save_pretrained(self, path, save_compressed=None, **kwargs):
        p = Path(path)
        st = self.st
        st.events.append("save")
        st.saves.append((str(path), save_compressed, kwargs))
        st.listing_at_save = sorted(x.name for x in p.iterdir()) if p.exists() else None
        p.mkdir(parents=True, exist_ok=True)
        assert self.scheme is not None
        fmt = self.scheme.lower()
        write_checkpoint(p, st.tensors(fmt), st.config(fmt))
        (p / "generation_config.json").write_text('{"from": "save_pretrained"}')
        (p / "recipe.yaml").write_text("recipe")


@pytest.fixture
def env(monkeypatch, tmp_path):
    st = types.SimpleNamespace(
        events=[],
        forward_kwargs=[],
        open_contexts=[],
        contexts_at_forward=[],
        loads=[],
        oneshots=[],
        saves=[],
        datasets=[],
        tokenizer_loads=[],
        listing_at_save=None,
        skip_quantize=set(),
        also_quantize=set(),
        empty_cache=0,
        tensors=ct_tensors,
        config=qwen3_config,
        tmp=tmp_path,
    )
    bf16, mx, nv, data = (tmp_path / d for d in ("bf16", "mx", "nv", "data"))
    for d in (bf16, data):
        d.mkdir()
    for name, blob in TOKENIZER_BYTES.items():
        (bf16 / name).write_bytes(blob)
    (bf16 / "config.json").write_text(json.dumps(qwen3_config()))
    (data / "nll.json").write_text(json.dumps(WINDOWS))
    for name, value in {"BF16_MODEL_DIR": bf16, "MX_MODEL_DIR": mx, "NV_MODEL_DIR": nv}.items():
        monkeypatch.setattr(qwen3, name, str(value))
    for name, value in {
        "DATA_DIR": data,
        "NLL_PROMPTS_PATH": data / "nll.json",
        "BF16_REF_NLL_PATH": data / "ref_nll.json",
        "SHAREGPT_PATH": data / "sharegpt.json",
    }.items():
        monkeypatch.setattr(settings, name, str(value))
    st.bf16, st.mx, st.nv, st.ref = bf16, mx, nv, data / "ref_nll.json"
    monkeypatch.setattr(q, "load_sharegpt_user_texts", lambda path: all_texts())

    def version(name):
        if name not in st.versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return st.versions[name]

    st.versions = dict(VERSIONS)
    monkeypatch.setattr(importlib.metadata, "version", version)

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class FakeTok:
        def encode(self, text, add_special_tokens=True, split_special_tokens=False):
            assert add_special_tokens is False and split_special_tokens is True
            return [word_id(w) for w in text.split()]

        def save_pretrained(self, path):
            raise AssertionError("tokenizer files must be copied byte-for-byte, not re-saved")

    class Tok:
        @staticmethod
        def from_pretrained(path):
            st.tokenizer_loads.append(path)
            return FakeTok()

    class Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            st.events.append("load")
            st.loads.append((path, kwargs))
            model = FakeModel(st)
            st.models = [*getattr(st, "models", []), model]
            return model

    class Dataset:
        @staticmethod
        def from_dict(data):
            ds = types.SimpleNamespace(data=data, column_names=list(data))
            st.datasets.append(ds)
            return ds

    def modifier(**kwargs):
        return types.SimpleNamespace(**kwargs)

    def oneshot(**kwargs):
        st.events.append("oneshot")
        st.oneshots.append(kwargs)
        model, recipe = kwargs["model"], kwargs["recipe"]
        model.scheme = recipe.scheme
        for name, mod in model.named_modules():
            if isinstance(mod, FakeLinear) and (
                (name not in recipe.ignore and name not in st.skip_quantize)
                or name in st.also_quantize
            ):
                mod.quantization_scheme = types.SimpleNamespace(
                    weights=object(), input_activations=object()
                )
        return model

    @contextlib.contextmanager
    def context(name):
        st.open_contexts.append(name)
        try:
            yield
        finally:
            st.open_contexts.remove(name)

    def empty_cache():
        st.empty_cache += 1

    module(
        "torch",
        bfloat16="bf16",
        __version__="2.13.0+cu130",
        no_grad=lambda: context("no_grad"),
        inference_mode=lambda: context("inference_mode"),
        tensor=lambda data, device=None: np.array(data).view(FakeTensor),
        nn=types.SimpleNamespace(
            Linear=FakeLinear, functional=types.SimpleNamespace(cross_entropy=cross_entropy)
        ),
        cuda=types.SimpleNamespace(empty_cache=empty_cache),
    )
    module("transformers", __version__="5.17.0", AutoTokenizer=Tok, AutoModelForCausalLM=Auto)
    module("datasets", Dataset=Dataset)
    quant_mod = module("llmcompressor.modifiers.quantization", QuantizationModifier=modifier)
    mods = module("llmcompressor.modifiers", quantization=quant_mod)
    module("llmcompressor", oneshot=oneshot, modifiers=mods)
    return st


def read_json(path):
    return json.loads(Path(path).read_text())


def expected_calibration(n=256, max_len=1024):
    texts = all_texts()
    random.Random(settings.SEED + 3).shuffle(texts)
    return [[word_id(w) for w in t.split()][:max_len] for t in texts[:n]]


def test_window_nll_is_the_mean_over_positions_1_to_l_minus_1(env):
    model = FakeModel(env)
    for window in WINDOWS:
        assert q.window_nll(model, window) == pytest.approx(expected_nll(window), abs=1e-12)


def test_window_nll_matches_what_the_server_side_probe_computes(env):
    model = FakeModel(env)
    for window in WINDOWS:
        plp: list[dict | None] = [None]
        for prev, tok in pairwise(window):
            row = BIGRAM[prev]
            plp.append({str(tok): {"logprob": float(row[tok] - math.log(np.exp(row).sum()))}})
        body = {"choices": [{"prompt_logprobs": plp}]}
        assert q.window_nll(model, window) == pytest.approx(
            sanity.prompt_nll_from_response(body, window), abs=1e-12
        )


def test_window_nll_feeds_the_window_as_is_without_cache_or_extra_tokens(env):
    seen = []
    model = FakeModel(env)
    real_call = model.__call__.__func__

    class Spy(FakeModel):
        @override
        def __call__(self, input_ids=None, use_cache=True, **kwargs):
            seen.append(np.asarray(input_ids).tolist())
            return real_call(self, input_ids=input_ids, use_cache=use_cache, **kwargs)

    q.window_nll(Spy(env), WINDOWS[0])
    assert seen == [[WINDOWS[0]]]
    assert env.forward_kwargs == [{"use_cache": False}]


def test_reference_is_computed_before_oneshot_and_stored_per_window(env):
    q.quantize(Format.NVFP4)
    assert env.events.index("forward") < env.events.index("oneshot")
    assert env.events.count("forward") == len(WINDOWS)
    ref = read_json(env.ref)
    assert ref["per_prompt"] == pytest.approx([expected_nll(w) for w in WINDOWS], abs=1e-12)
    assert ref["mean"] == pytest.approx(sum(ref["per_prompt"]) / len(WINDOWS), abs=1e-12)
    assert ref["source_revision"] == qwen3.SRC_MODEL_REVISION
    assert {"per_prompt", "mean", "source_revision"} <= set(ref)
    assert (
        ref["nll_prompts_sha256"]
        == hashlib.sha256(Path(settings.NLL_PROMPTS_PATH).read_bytes()).hexdigest()
    )
    assert not Path(str(env.ref) + ".tmp").exists()


def test_the_nll_pass_runs_under_no_grad_never_inference_mode(env):
    q.quantize(Format.MXFP4)
    assert env.contexts_at_forward and all(c == ["no_grad"] for c in env.contexts_at_forward)


def test_an_existing_reference_is_reused_and_not_recomputed(env):
    q.quantize(Format.MXFP4)
    before = env.ref.read_bytes()
    env.events.clear()
    prov = q.quantize(Format.NVFP4)
    assert "forward" not in env.events
    assert env.ref.read_bytes() == before
    assert prov["bf16_reference_nll"]["computed_by_this_run"] is False


def test_the_first_run_records_that_it_computed_the_reference(env):
    prov = q.quantize(Format.MXFP4)
    assert prov["bf16_reference_nll"] == {
        "path": str(env.ref),
        "mean": read_json(env.ref)["mean"],
        "computed_by_this_run": True,
        "nll_prompts_sha256": read_json(env.ref)["nll_prompts_sha256"],
    }


@pytest.mark.parametrize(
    "mutate, problem",
    [
        (lambda r: r.update(nll_prompts_sha256="0" * 64), "nll_prompts_sha256"),
        (lambda r: r.update(source_revision="deadbeef"), "source_revision"),
        (lambda r: r.update(per_prompt=r["per_prompt"][:-1]), "per_prompt"),
        (lambda r: r.update(per_prompt=None), "per_prompt"),
        (lambda r: r.pop("mean"), "mean"),
    ],
)
def test_a_stale_or_malformed_reference_is_refused_before_the_model_is_loaded(env, mutate, problem):
    q.quantize(Format.MXFP4)
    ref = read_json(env.ref)
    mutate(ref)
    env.ref.write_text(json.dumps(ref))
    env.loads.clear()
    env.events.clear()
    with pytest.raises(RuntimeError, match=problem) as exc:
        q.quantize(Format.NVFP4)
    assert str(env.ref) in str(exc.value) and "delete" in str(exc.value)
    assert env.loads == [] and "oneshot" not in env.events


def test_missing_nll_prompts_fail_before_the_model_is_loaded(env):
    Path(settings.NLL_PROMPTS_PATH).unlink()
    with pytest.raises(FileNotFoundError, match=r"nll\.json"):
        q.quantize(Format.MXFP4)
    assert env.loads == []


def test_each_format_loads_the_bf16_model_fresh(env):
    q.quantize(Format.MXFP4)
    q.quantize(Format.NVFP4)
    assert [path for path, _ in env.loads] == [str(env.bf16)] * 2
    assert all(kwargs == {"dtype": "bf16", "device_map": "cuda"} for _, kwargs in env.loads)
    assert len(env.models) == 2 and env.models[0] is not env.models[1]
    assert env.oneshots[0]["model"] is env.models[0] and env.oneshots[1]["model"] is env.models[1]


def test_an_unexpected_llmcompressor_version_is_refused_before_loading(env):
    env.versions["llmcompressor"] = "0.13.0"
    with pytest.raises(RuntimeError, match=r"llmcompressor.*0\.13\.0.*0\.14\.0"):
        q.quantize(Format.MXFP4)
    assert env.loads == []


@pytest.mark.parametrize(
    "fmt, scheme, out", [(Format.MXFP4, "MXFP4", "mx"), (Format.NVFP4, "NVFP4", "nv")]
)
def test_oneshot_runs_the_scheme_preset_on_every_linear_except_lm_head(env, fmt, scheme, out):
    q.quantize(fmt)
    (call,) = env.oneshots
    recipe = call["recipe"]
    assert (recipe.targets, recipe.scheme, recipe.ignore) == ("Linear", scheme, ["lm_head"])
    assert call["max_seq_length"] == 1024 and call["num_calibration_samples"] == 256
    assert call["shuffle_calibration_samples"] is False
    assert "output_dir" not in call
    assert env.saves == [(str(getattr(env, out)), True, {})]


def test_calibration_is_256_seeded_sharegpt_prompts_of_at_most_1024_tokens(env):
    q.quantize(Format.NVFP4)
    (ds,) = env.datasets
    assert env.oneshots[0]["dataset"] is ds
    ids = ds.data["input_ids"]
    assert ids == expected_calibration()
    assert len(ids) == 256 and max(len(x) for x in ids) == 1024
    assert ds.data["attention_mask"] == [[1] * len(x) for x in ids]
    assert ds.column_names == ["input_ids", "attention_mask"]


def test_the_calibration_set_is_identical_for_both_formats(env):
    mx, nv = q.quantize(Format.MXFP4), q.quantize(Format.NVFP4)
    assert env.datasets[0].data == env.datasets[1].data
    assert mx["calib_sha256"] == nv["calib_sha256"]
    assert (
        mx["calib_sha256"]
        == hashlib.sha256(json.dumps(expected_calibration()).encode()).hexdigest()
    )


def test_calibration_does_not_depend_on_global_random_state(env):
    random.seed(1)
    q.quantize(Format.MXFP4)
    random.seed(999)
    q.quantize(Format.NVFP4)
    assert env.datasets[0].data == env.datasets[1].data


def test_too_few_prompts_for_the_calibration_set_fail_before_loading(env, monkeypatch):
    monkeypatch.setattr(q, "load_sharegpt_user_texts", lambda path: all_texts()[:255])
    with pytest.raises(ValueError, match=r"255.*256"):
        q.quantize(Format.NVFP4)
    assert env.loads == []


def test_quantize_refuses_to_export_when_a_linear_was_not_quantized(env):
    env.skip_quantize = {"model.layers.1.self_attn.k_proj", "model.layers.0.mlp.up_proj"}
    env.nv.mkdir()
    (env.nv / "keep.txt").write_text("the previous checkpoint is not wiped for a refused export")
    with pytest.raises(RuntimeError) as exc:
        q.quantize(Format.NVFP4)
    assert "model.layers.1.self_attn.k_proj" in str(exc.value)
    assert "model.layers.0.mlp.up_proj" in str(exc.value)
    assert "model.layers.0.self_attn.q_proj" not in str(exc.value)
    assert env.saves == [] and (env.nv / "keep.txt").exists()
    assert not (env.nv / settings.NV_PROVENANCE_FILE).exists()


def test_quantize_refuses_to_export_when_lm_head_got_quantized(env):
    env.also_quantize = {"lm_head"}
    with pytest.raises(RuntimeError, match="lm_head"):
        q.quantize(Format.MXFP4)
    assert env.saves == []


def test_quantize_refuses_to_export_when_nothing_is_quantized(env):
    env.skip_quantize = {name for name, m in FakeModel(env).modules if isinstance(m, FakeLinear)}
    with pytest.raises(RuntimeError, match="nothing was quantized"):
        q.quantize(Format.MXFP4)
    assert env.saves == []


def test_a_weight_only_scheme_does_not_count_as_quantized_for_a_w4a4_checkpoint(env):
    def oneshot_leaving_k_proj_weight_only(**kwargs):
        model = kwargs["model"]
        model.scheme = "MXFP4"
        for name, mod in model.named_modules():
            if isinstance(mod, FakeLinear) and name != "lm_head":
                acts = None if name.endswith("k_proj") else object()
                mod.quantization_scheme = types.SimpleNamespace(
                    weights=object(), input_activations=acts
                )
        return model

    setattr(sys.modules["llmcompressor"], "oneshot", oneshot_leaving_k_proj_weight_only)  # noqa: B010
    with pytest.raises(RuntimeError, match="not quantized") as exc:
        q.quantize(Format.MXFP4)
    assert "model.layers.0.self_attn.k_proj" in str(exc.value)
    assert "q_proj" not in str(exc.value)
    assert env.saves == []


def test_the_target_dir_is_emptied_before_the_export(env):
    env.nv.mkdir()
    (env.nv / "model-00003-of-00003.safetensors").write_bytes(b"stale shard from an earlier run")
    (env.nv / "README.md").write_text("stale card")
    q.quantize(Format.NVFP4)
    assert env.listing_at_save is None
    assert not (env.nv / "model-00003-of-00003.safetensors").exists()
    assert not (env.nv / "README.md").exists()


def test_the_bf16_dir_is_never_written_to(env):
    before = {p.name: p.read_bytes() for p in env.bf16.iterdir()}
    q.quantize(Format.MXFP4)
    assert {p.name: p.read_bytes() for p in env.bf16.iterdir()} == before


def test_the_tokenizer_files_are_copied_byte_for_byte_from_the_bf16_dir(env):
    q.quantize(Format.NVFP4)
    for name, blob in TOKENIZER_BYTES.items():
        assert (env.nv / name).read_bytes() == blob
    assert not (env.nv / "chat_template.jinja").exists()


def test_copy_tokenizer_files_copies_only_what_exists(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "tokenizer.json").write_bytes(b"\x00\x01")
    (src / "unrelated.bin").write_bytes(b"x")
    assert q.copy_tokenizer_files(src, dst) == ["tokenizer.json"]
    assert sorted(p.name for p in dst.iterdir()) == ["tokenizer.json"]
    assert q.TOKENIZER_FILES == (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
    )


@pytest.mark.parametrize("fmt, out", [(Format.MXFP4, "mx"), (Format.NVFP4, "nv")])
def test_both_formats_pass_the_real_structure_and_config_checks(env, fmt, out):
    prov = q.quantize(fmt)
    directory = getattr(env, out)
    assert sanity.check_structure(sanity.read_safetensors_headers(directory), fmt) == []
    assert sanity.quant_config_problems(directory, fmt) == []
    assert read_json(directory / settings.NV_PROVENANCE_FILE) == prov


def test_a_checkpoint_with_an_unquantized_linear_on_disk_fails_fast_without_provenance(env):
    env.tensors = lambda fmt: unquantize(ct_tensors(fmt), "layers.3.mlp.down_proj")
    with pytest.raises(RuntimeError, match="structure") as exc:
        q.quantize(Format.MXFP4)
    assert "layers.3.mlp.down_proj" in str(exc.value)
    assert not (env.mx / settings.NV_PROVENANCE_FILE).exists()


def test_a_wrong_quantization_config_fails_fast_without_provenance(env):
    env.config = lambda fmt: qwen3_config(Format.MXFP4)
    with pytest.raises(RuntimeError, match="quantization config") as exc:
        q.quantize(Format.NVFP4)
    assert "nvfp4-pack-quantized" in str(exc.value)
    assert not (env.nv / settings.NV_PROVENANCE_FILE).exists()


def test_a_kv_cache_scheme_in_the_saved_config_fails_fast(env):
    def config(fmt):
        out = qwen3_config(fmt)
        out["quantization_config"]["kv_cache_scheme"] = {"num_bits": 8, "type": "float"}
        return out

    env.config = config
    with pytest.raises(RuntimeError, match="kv_cache_scheme"):
        q.quantize(Format.MXFP4)


def test_the_cuda_cache_is_emptied_even_when_the_export_is_refused(env):
    env.also_quantize = {"lm_head"}
    with pytest.raises(RuntimeError):
        q.quantize(Format.MXFP4)
    assert env.empty_cache == 1


def test_the_cuda_cache_is_emptied_after_a_successful_run(env):
    q.quantize(Format.MXFP4)
    assert env.empty_cache == 1


def test_provenance_records_tools_recipe_calibration_and_quantized_modules(env):
    prov = q.quantize(Format.NVFP4)
    assert read_json(env.nv / settings.NV_PROVENANCE_FILE) == prov
    assert (prov["source_model"], prov["source_revision"]) == (
        qwen3.SRC_MODEL_ID,
        qwen3.SRC_MODEL_REVISION,
    )
    assert prov["format"] == "nvfp4" and prov["scheme"] == "NVFP4"
    assert (prov["llmcompressor_version"], prov["compressed_tensors_version"]) == (
        "0.14.0",
        "0.19.0",
    )
    assert (prov["transformers_version"], prov["torch_version"]) == ("5.17.0", "2.13.0+cu130")
    assert prov["recipe"] == {
        "modifier": "QuantizationModifier",
        "targets": "Linear",
        "scheme": "NVFP4",
        "ignore": ["lm_head"],
    }
    assert prov["ignore"] == ["lm_head"]
    calib = expected_calibration()
    assert prov["calib_dataset"] == f"{settings.SHAREGPT_REPO}@{settings.SHAREGPT_REVISION}"
    assert (prov["calib_samples"], prov["calib_max_len"], prov["calib_seed"]) == (
        256,
        1024,
        settings.SEED + 3,
    )
    assert prov["calib_tokens"] == sum(len(x) for x in calib)
    assert prov["calib_sha256"] == hashlib.sha256(json.dumps(calib).encode()).hexdigest()
    assert prov["quantized_modules"] == sorted(
        name
        for name, m in FakeModel(env).modules
        if isinstance(m, FakeLinear) and name != "lm_head"
    )
    assert len(prov["quantized_modules"]) == 14 and "lm_head" not in prov["quantized_modules"]
    assert prov["tokenizer_files"] == sorted(TOKENIZER_BYTES)
    assert datetime_ok(prov["utc"])


def datetime_ok(text: str) -> bool:
    from datetime import datetime

    return datetime.fromisoformat(text).tzinfo is not None


def test_a_first_checkpoint_has_nothing_to_compare_with(env):
    prov = q.quantize(Format.NVFP4)
    assert prov["config_parity"] == {"compared_with": None, "differing_keys": []}


def test_identical_configs_are_left_alone(env):
    q.quantize(Format.MXFP4)
    nv_prov = q.quantize(Format.NVFP4)
    assert nv_prov["config_parity"] == {"compared_with": str(env.mx), "differing_keys": []}
    assert read_json(env.nv / "config.json") == qwen3_config(Format.NVFP4)
    report = sanity.checkpoint_report(str(env.mx), str(env.nv))
    assert (
        report["config_diff"] == {} and report["mx_problems"] == [] and report["nv_problems"] == []
    )


RE_SERIALISED = {"bos_token_id": None, "rope_parameters": {"rope_theta": 1000000.0}}


def test_a_config_difference_is_recorded_and_left_for_g5a_to_refuse(env):
    env.config = lambda fmt: {**qwen3_config(fmt), **(RE_SERIALISED if fmt == "nvfp4" else {})}
    q.quantize(Format.MXFP4)
    prov = q.quantize(Format.NVFP4)
    assert prov["config_parity"] == {
        "compared_with": str(env.mx),
        "differing_keys": ["bos_token_id", "rope_parameters"],
    }
    assert read_json(env.nv / "config.json") == {**qwen3_config(Format.NVFP4), **RE_SERIALISED}
    report = sanity.checkpoint_report(str(env.mx), str(env.nv))
    assert sorted(report["config_diff"]) == ["bos_token_id", "rope_parameters"]
    assert report["nv_problems"] != []


def test_mx_saved_after_nv_records_the_difference_and_never_rewrites_the_nv_checkpoint(env):
    env.config = lambda fmt: {**qwen3_config(fmt), **(RE_SERIALISED if fmt == "nvfp4" else {})}
    q.quantize(Format.NVFP4)
    nv_config_before = (env.nv / "config.json").read_bytes()
    prov = q.quantize(Format.MXFP4)
    assert prov["config_parity"] == {
        "compared_with": str(env.nv),
        "differing_keys": ["bos_token_id", "rope_parameters"],
    }
    assert (env.nv / "config.json").read_bytes() == nv_config_before


def test_an_other_dir_without_a_config_json_is_not_compared(env):
    env.nv.mkdir()
    (env.nv / "stray.txt").write_text("x")
    assert q.quantize(Format.MXFP4)["config_parity"]["compared_with"] is None


@pytest.mark.parametrize(("fmt", "other"), [(Format.NVFP4, "mx"), (Format.MXFP4, "nv")])
def test_the_other_formats_config_is_read_as_that_format(env, fmt, other):
    """NV is compared with MX's checkpoint and MX with NV's, each read under its own label."""
    other_dir = getattr(env, other)
    other_dir.mkdir()
    (other_dir / "config.json").write_text("{")
    message = rf"^{other.upper()} checkpoint {re.escape(str(other_dir))}: config.json is not valid"
    with pytest.raises(ValueError, match=message):
        q.quantize(fmt)
