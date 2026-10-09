import ast
import json
import sys
import types
from pathlib import Path

import pytest

from fp4bench import publish as pub
from fp4bench import quantize as q
from fp4bench import sanity, settings
from fp4bench.core.types import Format
from fp4bench.publish import assert_publishable, model_card
from fp4bench.studies import model
from tests.test_quantize import env  # noqa: F401  (the fake llm-compressor stack, for the e2e test)

MX_BYTES, NV_BYTES = 18_000_000_000, 19_062_000_000
REPORT = {
    "mx_quantized_bytes": MX_BYTES,
    "nv_quantized_bytes": NV_BYTES,
    "nv_over_mx": NV_BYTES / MX_BYTES,
    "expected_nv_over_mx": 4.5 / 4.25,
    "mx_problems": [],
    "nv_problems": [],
    "mx_quant_format": "mxfp4-pack-quantized",
    "nv_quant_format": "nvfp4-pack-quantized",
}
FORMAT = {"mx": "mxfp4", "nv": "nvfp4"}
LABEL = {"mx": "MXFP4", "nv": "NVFP4"}
REPO = {"mx": "someone/Qwen3-32B-MXFP4", "nv": "someone/Qwen3-32B-NVFP4"}


def prov(kind: str) -> dict:
    """Provenance as quantize.quantize writes it (the keys the card reads)."""
    fmt = FORMAT[kind]
    return {
        "source_model": model.SRC_MODEL_ID,
        "source_revision": model.SRC_MODEL_REVISION,
        "format": fmt,
        "scheme": fmt.upper(),
        "llmcompressor_version": "9.1.2",
        "compressed_tensors_version": "8.3.4",
        "transformers_version": "5.17.0",
        "torch_version": "2.13.0+cu130",
        "recipe": {
            "modifier": "QuantizationModifier",
            "targets": "Linear",
            "scheme": fmt.upper(),
            "ignore": ["lm_head"],
        },
        "ignore": ["lm_head"],
        "calib_dataset": f"{settings.SHAREGPT_REPO}@{settings.SHAREGPT_REVISION}",
        "calib_samples": 256,
        "calib_tokens": 98765,
        "calib_max_len": 1024,
        "calib_seed": 3,
        "calib_sha256": "c" * 64,
        "quantized_modules": [f"model.layers.{i}.mlp.up_proj" for i in range(448)],
        "tokenizer_files": ["tokenizer.json"],
        "config_parity": {},
        "bf16_reference_nll": {
            "path": "/data/bf16_reference_nll.json",
            "mean": 1.23456,
            "computed_by_this_run": True,
            "nll_prompts_sha256": "d" * 64,
        },
        "utc": "2026-10-05T01:02:03+00:00",
    }


@pytest.fixture(autouse=True)
def no_repo_url(monkeypatch):
    monkeypatch.setattr(pub, "REPO_URL", None)


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_has_front_matter_and_names_the_source(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert card.startswith("---\n")
    for needle in (
        "license: apache-2.0",
        f"base_model: {model.SRC_MODEL_ID}",
        "base_model_relation: quantized",
        "pipeline_tag: text-generation",
        model.SRC_MODEL_REVISION,
        f"# {REPO[kind].split('/')[-1]}",
    ):
        assert needle in card
    assert "library_name" not in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_states_w4a4_coverage_and_the_tool_versions(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert f"{LABEL[kind]} W4A4" in card
    assert "all `Linear` layers except `lm_head`" in card
    assert "llm-compressor 9.1.2" in card and "compressed-tensors 8.3.4" in card
    assert f'scheme="{LABEL[kind]}"' in card and 'ignore=["lm_head"]' in card
    assert "448" in card


def test_model_card_describes_each_format_by_its_own_layout():
    mx = model_card("mx", REPO["mx"], REPORT, prov("mx"))
    nv = model_card("nv", REPO["nv"], REPORT, prov("nv"))
    for needle in ("32-element blocks", "E8M0", "4.25 bits"):
        assert needle in mx and needle not in nv
    for needle in ("16-element blocks", "E4M3", "FP32", "4.5 bits"):
        assert needle in nv and needle not in mx


def test_only_the_nv_card_claims_a_calibrated_activation_scale():
    mx = model_card("mx", REPO["mx"], REPORT, prov("mx"))
    nv = model_card("nv", REPO["nv"], REPORT, prov("nv"))
    assert "input_global_scale" in nv and "calibrat" in nv
    assert "input_global_scale" not in mx
    assert "no static activation scales" in mx.replace("\n", " ")


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_states_the_calibration_set_at_the_pinned_revision(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    for needle in (
        "256 prompts",
        settings.SHAREGPT_REPO,
        settings.SHAREGPT_REVISION,
        "1024",
        "seed 3",
        "98765",
    ):
        assert needle in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_says_the_kv_cache_is_bf16(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert "BF16" in card and "--kv-cache-dtype bfloat16" in card
    assert "kv_cache_scheme" in card


def test_model_card_reports_the_measured_bytes_of_its_own_format_and_the_ratio():
    mx = model_card("mx", REPO["mx"], REPORT, prov("mx"))
    nv = model_card("nv", REPO["nv"], REPORT, prov("nv"))
    assert f"{MX_BYTES:,}" in mx and f"{NV_BYTES:,}" in nv
    for card in (mx, nv):
        assert "1.0590" in card and "1.0588" in card and "4.5 / 4.25" in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_states_the_bf16_reference_nll(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert "1.2346" in card
    assert str(settings.NLL_PROMPTS) in card and str(settings.NLL_LEN) in card
    assert "d" * 64 in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_says_it_was_made_for_this_benchmark(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert "made for" in card and "benchmark" in card
    assert "EXPERIMENT.md" not in card and "github.com" not in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_links_the_design_when_a_repo_url_is_configured(kind, monkeypatch):
    monkeypatch.setattr(pub, "REPO_URL", "https://github.com/someone/decode_benchmark/")
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert "https://github.com/someone/decode_benchmark/blob/main/EXPERIMENT.md" in card
    assert "//blob" not in card.replace("https://", "")


def test_repo_url_is_left_unset_in_the_source():
    tree = ast.parse(Path(pub.__file__).read_text())
    (assign,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "REPO_URL" for t in n.targets)
    ]
    assert isinstance(assign.value, ast.Constant) and assign.value.value is None


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_serving_command_uses_the_benchmarks_pinned_kernel(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind))
    assert (
        f"vllm serve {REPO[kind]} --kv-cache-dtype bfloat16 --linear-backend flashinfer_cutedsl"
    ) in card


def test_model_card_derives_the_vllm_version_from_the_config(monkeypatch):
    assert f"vLLM v{settings.EXPECTED_VERSIONS['vllm']} on B200" in model_card(
        "mx", REPO["mx"], REPORT, prov("mx")
    )
    monkeypatch.setattr(
        settings, "EXPECTED_VERSIONS", {**settings.EXPECTED_VERSIONS, "vllm": "9.8.7"}
    )
    card = model_card("mx", REPO["mx"], REPORT, prov("mx"))
    assert "vLLM v9.8.7 on B200" in card and "v0.31.0" not in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_has_no_revision_1_fields(kind):
    card = model_card(kind, REPO[kind], REPORT, prov(kind)).lower()
    for gone in ("gpt-oss", "gpt_oss", "modelopt", "experts", "disabled_patterns"):
        assert gone not in card


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_model_card_reads_the_values_from_the_provenance_not_from_constants(kind):
    changed = {
        **prov(kind),
        "calib_samples": 111,
        "calib_max_len": 777,
        "calib_seed": 42,
        "llmcompressor_version": "0.0.1",
    }
    card = model_card(kind, REPO[kind], REPORT, changed)
    assert "111 prompts" in card and "777" in card and "seed 42" in card and "0.0.1" in card
    assert "256 prompts" not in card


def test_model_card_refuses_a_report_that_is_not_publishable():
    with pytest.raises(ValueError, match="structure"):
        model_card("mx", REPO["mx"], {**REPORT, "mx_problems": ["bad"]}, prov("mx"))


def test_model_card_refuses_provenance_of_another_format_or_with_gaps():
    with pytest.raises(ValueError, match="mxfp4"):
        model_card("mx", REPO["mx"], REPORT, prov("nv"))
    gappy = {k: v for k, v in prov("mx").items() if k not in ("bf16_reference_nll", "utc")}
    with pytest.raises(ValueError, match="bf16_reference_nll"):
        model_card("mx", REPO["mx"], REPORT, gappy)
    with pytest.raises(ValueError, match="calib_dataset"):
        model_card("mx", REPO["mx"], REPORT, {**prov("mx"), "calib_dataset": "no-revision"})
    with pytest.raises(ValueError, match="mean"):
        model_card("mx", REPO["mx"], REPORT, {**prov("mx"), "bf16_reference_nll": {}})


def test_model_card_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        model_card("nvidia", "x/y", REPORT, prov("nv"))


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_assert_publishable_blocks_a_checkpoint_with_structure_problems(kind):
    assert_publishable(REPORT, kind)
    with pytest.raises(ValueError, match="structure"):
        assert_publishable({**REPORT, f"{kind}_problems": ["no packed FP4 (U8) weights"]}, kind)


def test_assert_publishable_looks_only_at_the_publishing_sides_problems():
    assert_publishable({**REPORT, "nv_problems": ["x"]}, "mx")
    assert_publishable({**REPORT, "mx_problems": ["x"]}, "nv")
    with pytest.raises(ValueError, match="structure"):
        assert_publishable({**REPORT, "mx_problems": ["x"]}, "mx")
    with pytest.raises(ValueError, match="structure"):
        assert_publishable({**REPORT, "nv_problems": ["x"]}, "nv")


@pytest.mark.parametrize("kind", ["mx", "nv"])
@pytest.mark.parametrize("ratio", [1.3, 1.0, 0.9, float("nan"), float("inf")])
def test_assert_publishable_blocks_a_bytes_ratio_that_is_off_the_format_layout(kind, ratio):
    with pytest.raises(ValueError, match="bytes"):
        assert_publishable({**REPORT, "nv_over_mx": ratio}, kind)


def test_assert_publishable_fails_closed_without_a_problems_list():
    for report in (
        {k: v for k, v in REPORT.items() if k != "mx_problems"},
        {**REPORT, "mx_problems": None},
        {**REPORT, "mx_problems": "ok"},
    ):
        with pytest.raises(ValueError, match="mx_problems"):
            assert_publishable(report, "mx")


def test_assert_publishable_rejects_an_unknown_kind():
    for kind in ("nvidia", "", "NV"):
        with pytest.raises(ValueError, match="kind"):
            assert_publishable(REPORT, kind)


def install_fake_hub(monkeypatch):
    """Fake huggingface_hub.HfApi that records calls; `remote_private` is the repo's visibility."""
    st = types.SimpleNamespace(calls=[], remote_private=True, report=REPORT)

    class HfApi:
        def create_repo(self, repo_id, private=None, exist_ok=False):
            st.calls.append(("create_repo", repo_id, private, exist_ok))

        def repo_info(self, repo_id):
            st.calls.append(("repo_info", repo_id))
            return types.SimpleNamespace(private=st.remote_private)

        def upload_folder(self, **kwargs):
            st.calls.append(("upload_folder", kwargs))
            return types.SimpleNamespace(commit_url="https://huggingface.co/x/commit/abc")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return st


@pytest.fixture
def hub(monkeypatch, tmp_path):
    st = install_fake_hub(monkeypatch)
    for kind, attr in (("mx", "MX_MODEL_DIR"), ("nv", "NV_MODEL_DIR")):
        d = tmp_path / kind
        d.mkdir()
        (d / settings.NV_PROVENANCE_FILE).write_text(json.dumps(prov(kind)))
        monkeypatch.setattr(model, attr, str(d))
        setattr(st, kind, d)
    monkeypatch.setattr(sanity, "checkpoint_report", lambda mx, nv: st.report)
    return st


def names(st):
    return [c[0] for c in st.calls]


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_publish_uploads_to_a_private_repo_and_returns_the_commit_url(hub, kind):
    assert pub.publish(kind, REPO[kind], private=True) == "https://huggingface.co/x/commit/abc"
    assert names(hub) == ["create_repo", "repo_info", "upload_folder"]
    assert hub.calls[0] == ("create_repo", REPO[kind], True, True)
    readme = (getattr(hub, kind) / "README.md").read_text()
    assert readme.startswith("---\n") and f"{LABEL[kind]} W4A4" in readme
    other = hub.nv if kind == "mx" else hub.mx
    assert not (other / "README.md").exists()


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_publish_uploads_the_published_formats_directory_only(hub, kind):
    pub.publish(kind, REPO[kind], private=True)
    (kwargs,) = [c[1] for c in hub.calls if c[0] == "upload_folder"]
    assert kwargs["repo_id"] == REPO[kind]
    assert kwargs["folder_path"] == str(getattr(hub, kind))
    assert LABEL[kind] in kwargs["commit_message"] and "Qwen3-32B" in kwargs["commit_message"]


def test_publish_builds_each_card_from_its_own_formats_provenance(hub):
    (hub.mx / settings.NV_PROVENANCE_FILE).write_text(
        json.dumps({**prov("mx"), "llmcompressor_version": "1.1.1"})
    )
    (hub.nv / settings.NV_PROVENANCE_FILE).write_text(
        json.dumps({**prov("nv"), "llmcompressor_version": "2.2.2"})
    )
    pub.publish("mx", REPO["mx"], private=True)
    pub.publish("nv", REPO["nv"], private=True)
    assert "llm-compressor 1.1.1" in (hub.mx / "README.md").read_text()
    assert "llm-compressor 2.2.2" in (hub.nv / "README.md").read_text()


def test_publish_refuses_to_upload_into_an_existing_public_repo_when_private_was_requested(hub):
    hub.remote_private = False
    with pytest.raises(RuntimeError, match="public") as exc:
        pub.publish("mx", REPO["mx"], private=True)
    assert REPO["mx"] in str(exc.value)
    assert "upload_folder" not in names(hub)
    assert not (hub.mx / "README.md").exists()


def test_publish_does_not_assume_privacy_when_the_hub_does_not_say(hub):
    hub.remote_private = None
    with pytest.raises(RuntimeError, match="private"):
        pub.publish("nv", REPO["nv"], private=True)
    assert "upload_folder" not in names(hub)


def test_publish_to_a_public_repo_when_explicitly_requested(hub):
    hub.remote_private = False
    pub.publish("nv", REPO["nv"], private=False)
    assert hub.calls[0] == ("create_repo", REPO["nv"], False, True)
    assert names(hub) == ["create_repo", "upload_folder"]


@pytest.mark.parametrize("kind", ["mx", "nv"])
def test_publish_mirrors_the_local_folder_by_deleting_stale_remote_files(hub, kind):
    pub.publish(kind, REPO[kind], private=True)
    (kwargs,) = [c[1] for c in hub.calls if c[0] == "upload_folder"]
    assert kwargs["delete_patterns"] == ["*.safetensors", "*.json", "*.md", "*.jinja"]
    assert kwargs["delete_patterns"] == pub.DELETE_PATTERNS


def test_publish_needs_a_token_and_a_publishable_checkpoint_before_touching_the_hub(
    hub, monkeypatch
):
    hub.report = {**REPORT, "mx_problems": ["no packed FP4 (U8) weights"]}
    with pytest.raises(ValueError, match="structure"):
        pub.publish("mx", REPO["mx"], private=True)
    assert hub.calls == [] and not (hub.mx / "README.md").exists()
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="token"):
        pub.publish("mx", REPO["mx"], private=True)
    assert hub.calls == []


def test_publish_accepts_the_alternative_token_variable(hub, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_other")
    pub.publish("mx", REPO["mx"], private=True)
    assert "upload_folder" in names(hub)


def test_the_publishing_sides_problems_decide_not_the_other_sides(hub):
    hub.report = {**REPORT, "nv_problems": ["something wrong with NV"]}
    pub.publish("mx", REPO["mx"], private=True)
    assert names(hub).count("upload_folder") == 1
    with pytest.raises(ValueError, match="structure"):
        pub.publish("nv", REPO["nv"], private=True)
    assert names(hub).count("upload_folder") == 1


def test_publish_rejects_an_unknown_kind_before_anything_else(hub, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    for kind in ("nvidia", "", "mxfp4"):
        with pytest.raises(ValueError, match="kind"):
            pub.publish(kind, "someone/x", private=True)
    assert hub.calls == []


def test_publish_without_provenance_stops_before_the_hub(hub):
    (hub.mx / settings.NV_PROVENANCE_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=settings.NV_PROVENANCE_FILE):
        pub.publish("mx", REPO["mx"], private=True)
    assert hub.calls == []


def test_publish_refuses_provenance_that_belongs_to_the_other_format(hub):
    (hub.mx / settings.NV_PROVENANCE_FILE).write_text(json.dumps(prov("nv")))
    with pytest.raises(ValueError, match="mxfp4"):
        pub.publish("mx", REPO["mx"], private=True)
    assert hub.calls == [] and not (hub.mx / "README.md").exists()


@pytest.mark.parametrize("fmt, kind", [("mxfp4", "mx"), ("nvfp4", "nv")])
def test_a_card_can_be_built_from_the_provenance_quantize_really_writes(
    env,  # noqa: F811
    monkeypatch,
    fmt,
    kind,
):
    hub = install_fake_hub(monkeypatch)
    q.quantize(Format.MXFP4)
    q.quantize(Format.NVFP4)
    assert pub.publish(kind, REPO[kind], private=True) == "https://huggingface.co/x/commit/abc"
    folder = env.mx if kind == "mx" else env.nv
    readme = (folder / "README.md").read_text()
    written = json.loads((folder / settings.NV_PROVENANCE_FILE).read_text())
    assert written["format"] == fmt
    assert f"{LABEL[kind]} W4A4" in readme and settings.SHAREGPT_REVISION in readme
    assert f"{written['bf16_reference_nll']['mean']:.4f}" in readme
    assert names(hub) == ["create_repo", "repo_info", "upload_folder"]
