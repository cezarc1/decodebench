import json
import subprocess
import sys
from dataclasses import replace

import pytest

from fp4bench import settings
from fp4bench.core.schema import ManifestLine, load_rows
from fp4bench.core.types import Cell, KvDtype, Treatment
from fp4bench.studies import base, model
from fp4bench.studies.base import (
    Prompts,
    ServerSettings,
    Study,
    VerdictRule,
    cells_at,
    min_kv_tokens_for,
)
from fp4bench.studies.expb import EXPB
from fp4bench.studies.expc import C_CELLS, EXPC, SMOKE_C
from fp4bench.studies.kernel_scan import NVA_SCAN, NVX_CROSSCHECK, Crosscheck, KernelScanSpec
from fp4bench.studies.main import FULL
from fp4bench.studies.registry import (
    LEGACY_DEFAULTS,
    STUDIES,
    check_distinct,
    fallback_study,
    protocol_shape,
    registered_study,
    study_for,
)
from fp4bench.studies.smoke import SMOKE, SMOKE_NF
from tests import REPO, RUNS_DIR
from tests.golden.make_identity import IDENTITY_PATH

IDENTITY = json.loads(IDENTITY_PATH.read_text())
PROTOCOL_KEYS = [
    "rounds",
    "treatments",
    "concurrencies",
    "m1_reps",
    "m2_duration_s",
    "require_published",
    "kv_cache_dtype",
    "gpu_memory_utilization",
    "primary_concurrencies",
    "cells",
    "max_model_len",
    "hf_overrides",
]


def test_the_registry_has_todays_modes_in_order():
    assert list(STUDIES) == ["smoke", "smoke-nf", "full", "expb", "smoke-c", "expc"]
    assert STUDIES == {
        "smoke": SMOKE,
        "smoke-nf": SMOKE_NF,
        "full": FULL,
        "expb": EXPB,
        "smoke-c": SMOKE_C,
        "expc": EXPC,
    }
    assert all(study.name == name for name, study in STUDIES.items())


@pytest.mark.parametrize("name", list(STUDIES))
def test_a_studys_protocol_dict_is_the_one_its_manifests_record(name):
    protocol = STUDIES[name].to_protocol_dict()
    assert list(protocol) == PROTOCOL_KEYS
    assert json.loads(json.dumps(protocol)) == IDENTITY["protocols"][name]["protocol"]
    assert type(protocol["kv_cache_dtype"]) is KvDtype
    assert all(type(t) is Treatment for t in protocol["treatments"])
    assert all(type(cell) is tuple for cell in protocol["cells"])


@pytest.mark.parametrize("run", ["expc-1", "smoke-c-1"])
def test_the_protocol_dict_is_byte_for_byte_the_line_a_cell_run_recorded(run):
    line = json.loads((RUNS_DIR / run / "manifests.jsonl").read_text().splitlines()[-1])
    study = STUDIES[IDENTITY["committed_runs"][run]["protocol"][0]]
    assert json.dumps(study.to_protocol_dict()) == json.dumps(line["protocol"])


@pytest.mark.parametrize("name", list(STUDIES))
def test_cells_m2_batches_and_the_largest_wave_are_the_schedule_the_snapshot_pins(name):
    study, pinned = STUDIES[name], IDENTITY["protocols"][name]
    assert [list(cell) for cell in study.cell_list()] == pinned["cells"]
    assert all(type(cell) is Cell for cell in study.cell_list())
    assert sorted(study.m2_batches()) == sorted(map(int, pinned["m2_argv"]))
    assert list(study.largest_wave()) == pinned["m1_largest_wave"]
    assert study.min_kv_tokens() == pinned["min_kv_tokens"]
    assert (study.prompts is Prompts.M1) == pinned["cells_are_the_legacy_expansion"]


def test_cells_served_from_the_m1_prompts_must_be_one_1024_token_cell_per_batch():
    with pytest.raises(ValueError, match="one \\(1024-token\\) cell per batch"):
        replace(FULL, cells=(Cell(1, 1024), Cell(8, 2048)))
    with pytest.raises(ValueError, match="cells must be distinct"):
        replace(FULL, cells=(Cell(8, 1024), Cell(32, 1024), Cell(128, 1024), Cell(8, 1024)))
    study = replace(FULL, cells=cells_at(1024, (128, 32, 8, 1)))
    assert study.to_protocol_dict()["concurrencies"] == (128, 32, 8, 1)
    assert study.to_protocol_dict()["cells"] == ()


def test_a_cell_study_records_its_cells_and_their_batches():
    protocol = EXPC.to_protocol_dict()
    assert protocol["cells"] == tuple(tuple(cell) for cell in EXPC.cells)
    assert protocol["concurrencies"] == (1, 8, 32, 128) == EXPC.batches
    assert SMOKE_C.batches == (1, 128)


def manifest(protocol: dict | None) -> ManifestLine:
    return ManifestLine(protocol=protocol)


def recorded(name: str) -> dict:
    return json.loads(json.dumps(STUDIES[name].to_protocol_dict()))


def test_every_committed_run_is_its_registered_study_with_the_rounds_it_registered():
    found = {}
    for run in sorted(p for p in RUNS_DIR.iterdir() if (p / "manifests.jsonl").exists()):
        line = load_rows(run / "manifests.jsonl", ManifestLine)[-1]
        study = registered_study(line)
        assert study is not None and study_for(line) == study, run.name
        found[run.name] = (study.name, study.rounds)
    assert found == {
        "expb-1": ("expb", 5),
        "expc-1": ("expc", 5),
        "full-1": ("full", 10),
        "smoke-c-1": ("smoke-c", 1),
        "smoke-nf-2": ("smoke-nf", 1),
        "smoke-r2-2": ("smoke", 1),
    }
    assert {run: [name] for run, (name, _) in found.items()} == {
        run: r["protocol"] for run, r in IDENTITY["committed_runs"].items()
    }


OLDER_RUNS = {
    "smoke-1": (
        {
            "concurrencies": [1, 32, 512],
            "m1_reps": 1,
            "m2_duration_s": 10,
            "require_published_nv": False,
            "rounds": 1,
            "treatments": ["MX", "NV"],
        },
        None,
    ),
    "smoke-r2-1": (
        {
            "concurrencies": [1, 32, 128],
            "m1_reps": 3,
            "m2_duration_s": 10,
            "require_published": False,
            "rounds": 1,
            "treatments": ["MX", "NV", "NVx", "NVc", "NVt", "NVd", "NVv"],
        },
        "smoke",
    ),
    "full-1-5rounds-snapshot": (
        {
            "concurrencies": [1, 8, 32, 64, 128],
            "gpu_memory_utilization": 0.9,
            "kv_cache_dtype": "bfloat16",
            "m1_reps": 5,
            "m2_duration_s": 30,
            "primary_concurrencies": [8, 32, 128],
            "require_published": True,
            "rounds": 5,
            "treatments": ["MX", "NV", "NVa", "MXp", "NVnf"],
        },
        "full",
    ),
}


@pytest.mark.parametrize("run", list(OLDER_RUNS))
def test_older_runs_match_their_study_in_the_shape_they_recorded(run):
    protocol, name = OLDER_RUNS[run]
    study = registered_study(manifest(protocol))
    assert (None if study is None else study.name) == name
    assert study_for(manifest(protocol)) == (FULL if name is None else STUDIES[name])


def test_legacy_defaults_are_the_values_the_main_run_had_before_the_keys_existed():
    assert {k: v for k, v in recorded("full").items() if k in LEGACY_DEFAULTS} == LEGACY_DEFAULTS
    assert sorted(LEGACY_DEFAULTS) == sorted(set(PROTOCOL_KEYS) - set(OLDER_RUNS["smoke-r2-1"][0]))


@pytest.mark.parametrize("name", list(STUDIES))
def test_any_change_but_rounds_is_not_the_registered_study(name):
    protocol = recorded(name)
    assert registered_study(manifest(protocol)) == STUDIES[name]
    assert registered_study(manifest({**protocol, "rounds": 10})) == replace(
        STUDIES[name], rounds=10
    )
    for key, value in protocol.items():
        if key == "rounds":
            continue
        other = (value[:-1] or [[1, 1024]]) if isinstance(value, list) else [value]
        assert registered_study(manifest({**protocol, key: other})) is None, key
    assert registered_study(manifest({**protocol, "extra": 1})) is None
    assert registered_study(manifest({**protocol, "m1_reps": float(protocol["m1_reps"])})) is None


@pytest.mark.parametrize("rounds", [10.0, 0, -1, True, "5", None])
def test_a_malformed_round_count_keeps_the_registered_one(rounds):
    assert registered_study(manifest({**recorded("full"), "rounds": rounds})) == FULL


@pytest.mark.parametrize(
    "protocol, expected",
    [
        ({"cells": [[1, 1024]]}, "expc"),
        ({"cells": [[8]], "kv_cache_dtype": "fp8"}, "expc"),
        ({"cells": "anything"}, "expc"),
        ({"cells": []}, "full"),
        ({"kv_cache_dtype": "fp8"}, "expb"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "expb"),
        ({"kv_cache_dtype": "fp8_e5m2"}, "expb"),
        ({"kv_cache_dtype": "auto"}, "full"),
        ({"kv_cache_dtype": None}, "full"),
        ({"kv_cache_dtype": "bfloat16"}, "full"),
        ({}, "full"),
    ],
)
def test_an_unregistered_protocol_gets_the_study_todays_detection_rule_names(protocol, expected):
    assert fallback_study(protocol) == STUDIES[expected]
    assert registered_study(manifest(protocol)) is None
    assert study_for(manifest(protocol)) == STUDIES[expected]


def test_a_run_without_a_manifest_or_protocol_is_analysed_as_a_main_run():
    assert study_for(None) == FULL
    assert study_for(manifest(None)) == FULL
    assert study_for(ManifestLine.from_json({"protocol": "not a dict"})) == FULL


@pytest.mark.parametrize("name", list(STUDIES))
def test_a_named_study_is_the_study_with_the_rounds_its_protocol_registered(name):
    protocol = recorded(name)
    assert study_for(ManifestLine(study=name, protocol=protocol)) == STUDIES[name]
    extended = ManifestLine(study=name, protocol={**protocol, "rounds": 12})
    assert study_for(extended) == replace(STUDIES[name], rounds=12)


def test_the_name_wins_over_the_protocol_rule():
    line = ManifestLine(study="smoke-c", protocol={**recorded("expc"), "treatments": ["MX", "NV"]})
    assert study_for(line).name == "smoke-c"
    assert study_for(ManifestLine.from_json({"study": "expb"})) == EXPB


@pytest.mark.parametrize("name", ["bogus", "", None, 3, "FULL"])
def test_a_name_that_is_not_registered_is_refused(name):
    with pytest.raises(ValueError, match=r"not registered.*smoke, smoke-nf, full"):
        study_for(ManifestLine.from_json({"study": name, "protocol": recorded("full")}))


def test_every_registered_study_records_its_own_protocol():
    assert len({protocol_shape(s) for s in STUDIES.values()}) == len(STUDIES)
    check_distinct(STUDIES)
    with pytest.raises(ValueError, match="studies full and full-again record the same protocol"):
        check_distinct({"full": FULL, "full-again": replace(FULL, name="full-again", rounds=7)})


def test_the_registry_refuses_to_load_two_studies_with_one_protocol():
    code = (
        "import sys, dataclasses; import fp4bench.studies.smoke as s; "
        "assert 'fp4bench.studies.registry' not in sys.modules; "
        "s.SMOKE_NF = dataclasses.replace(s.SMOKE, name='smoke-nf'); "
        "import fp4bench.studies.registry"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert done.returncode != 0
    assert "studies smoke and smoke-nf record the same protocol" in done.stderr


def test_the_studies_are_data_that_imports_no_analysis_library():
    code = (
        "import sys; import fp4bench.studies.registry, fp4bench.studies.model; "
        "print(sorted(m for m in ('numpy', 'scipy', 'matplotlib', 'fp4bench.analysis') "
        "if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    )
    assert done.stdout.strip() == "[]"


def test_a_mutated_registered_protocol_falls_back_by_the_rule():
    expc_without_mxp = {**recorded("expc"), "treatments": ["MX", "NV"]}
    assert study_for(manifest(expc_without_mxp)) == EXPC
    expb_e4m3 = {**recorded("expb"), "kv_cache_dtype": "fp8_e4m3"}
    assert study_for(manifest(expb_e4m3)) == EXPB
    assert study_for(manifest({**recorded("smoke"), "m1_reps": 2})) == SMOKE
    assert study_for(manifest({**recorded("smoke"), "treatments": ["MX", "NV", "NVt"]})) == SMOKE
    assert study_for(manifest({**recorded("smoke"), "treatments": ["MX", "NV", "NVx"]})) == FULL


def test_every_study_of_a_kind_shares_its_analysis_rules():
    rules = {
        s.name: (s.ratios, s.contrasts, s.effect_margin_ms, s.aa_effect_margin_ms, s.g3, s.verdict)
        for s in STUDIES.values()
    }
    assert rules["smoke"] == rules["smoke-nf"] == rules["full"]
    assert rules["smoke-c"] == rules["expc"]
    assert rules["expb"][:-1] == rules["full"][:-1] and rules["expb"] != rules["full"]
    assert [r.name for r in FULL.ratios] == ["R", "D", "K", "F", "Phi", "AA"]
    assert [c.name for c in EXPC.contrasts] == ["E_tok", "E_batch"]


def test_the_verdict_rule_and_the_smokes_scans_are_each_studys_own():
    assert {s.name: s.verdict for s in STUDIES.values()} == {
        "smoke": VerdictRule.H_EQ,
        "smoke-nf": VerdictRule.H_EQ,
        "full": VerdictRule.H_EQ,
        "expb": VerdictRule.H_B,
        "smoke-c": VerdictRule.BATCH_VS_TOKENS,
        "expc": VerdictRule.BATCH_VS_TOKENS,
    }
    assert {
        s.name: (s.kernel_scan, s.crosscheck)
        for s in STUDIES.values()
        if s.kernel_scan or s.crosscheck
    } == {"smoke": (NVA_SCAN, NVX_CROSSCHECK)}
    assert [s.name for s in STUDIES.values() if s.prompts is Prompts.CELLS] == ["smoke-c", "expc"]


def test_a_kernel_scan_reads_treatments_and_batches_its_study_has():
    with pytest.raises(
        ValueError, match="study smoke: its kernel scan reads NVd, which it does not serve"
    ):
        replace(SMOKE, treatments=tuple(t for t in SMOKE.treatments if t != T.NVD))
    with pytest.raises(ValueError, match="study smoke: its kernel scan reads NVnf, which"):
        replace(SMOKE, kernel_scan=replace(NVA_SCAN, reference=T.NVNF))
    with pytest.raises(ValueError, match="decides at C=8, which it does not measure"):
        replace(SMOKE, kernel_scan=replace(NVA_SCAN, tiebreak_c=8))
    with pytest.raises(ValueError, match="study smoke-nf: its cross-check reads NVx, which"):
        replace(SMOKE_NF, crosscheck=NVX_CROSSCHECK)
    with pytest.raises(ValueError, match="the batch-vs-tokens analysis has neither"):
        replace(EXPC, crosscheck=NVX_CROSSCHECK)


def test_a_kernel_scan_spec_is_consistent_on_its_own():
    with pytest.raises(ValueError, match="distinct treatments"):
        replace(NVA_SCAN, treatments=(T.NVC, T.NVC))
    with pytest.raises(ValueError, match="distinct treatments"):
        replace(NVA_SCAN, treatments=())
    with pytest.raises(ValueError, match="reference NV is not scanned"):
        replace(NVA_SCAN, treatments=(T.NV, T.NVC))
    with pytest.raises(ValueError, match="a kernel of their own"):
        replace(NVA_SCAN, treatments=(T.NVD, T.NVA))
    with pytest.raises(ValueError, match="must be positive"):
        replace(NVA_SCAN, tie_margin=0.0)
    assert isinstance(NVA_SCAN, KernelScanSpec)


def test_the_cross_check_is_nvidias_checkpoint_against_ours_on_one_kernel():
    assert Crosscheck(T.NVX, T.NV) == NVX_CROSSCHECK
    with pytest.raises(ValueError, match="NVIDIA's checkpoint against ours"):
        Crosscheck(T.NVA, T.NV)
    with pytest.raises(ValueError, match="the same kernel and fusion"):
        Crosscheck(T.NVX, T.NVNF)


def test_the_runs_with_a_preregistered_round_count_extend_to_ten_and_the_smokes_do_not():
    assert {s.name: s.extension_rounds for s in STUDIES.values()} == {
        "smoke": None,
        "smoke-nf": None,
        "full": 10,
        "expb": 10,
        "smoke-c": None,
        "expc": 10,
    }
    assert all(
        s.extension_rounds is None or s.extension_rounds > s.rounds for s in STUDIES.values()
    )
    assert "extension_rounds" not in FULL.to_protocol_dict()


def test_a_study_is_frozen_and_hashable():
    with pytest.raises(AttributeError):
        FULL.rounds = 10  # type: ignore[misc]  # ty: ignore[invalid-assignment]
    assert len({FULL, replace(FULL), EXPC}) == 2
    assert (
        Study(
            name="x", treatments=(Treatment.MX,), cells=(Cell(1, 1024),), primary_batches=(1,)
        ).server
        == ServerSettings()
    )


def test_a_study_serves_distinct_treatments_with_a_spec_over_distinct_cells():
    with pytest.raises(ValueError, match="study full: its treatments must be distinct"):
        replace(FULL, treatments=(*FULL.treatments, T.MX))
    with pytest.raises(ValueError, match="its treatments must be distinct and at least one"):
        replace(FULL, treatments=())
    with pytest.raises(ValueError, match="its cells must be distinct and at least one"):
        replace(FULL, cells=())
    with pytest.raises(ValueError, match=r"no treatment spec .* for NVz"):
        replace(FULL, treatments=(*FULL.treatments, "NVz"))  # type: ignore[arg-type]
    assert all(t in model.TREATMENTS for s in STUDIES.values() for t in s.treatments)


def test_a_study_that_decides_measures_its_primary_batches_and_a_smoke_need_not():
    with pytest.raises(
        ValueError,
        match=r"primary batches \(8, 256\), not all among its "
        r"batches \(1, 8, 32, 64, 128\)",
    ):
        replace(FULL, primary_batches=(8, 256))
    for study in STUDIES.values():
        assert study.smoke or set(study.primary_batches) <= set(study.batches), study.name
    assert not set(SMOKE.primary_batches) <= set(SMOKE.batches)
    with pytest.raises(ValueError, match="study smoke: its verdict reads primary batches"):
        replace(SMOKE, smoke=False)
    with pytest.raises(ValueError, match="a smoke decides nothing, so it has no extension"):
        replace(SMOKE, extension_rounds=10)


def test_the_smokes_are_the_one_round_studies_without_an_extension():
    smokes = {s.name for s in STUDIES.values() if s.smoke}
    assert smokes == {"smoke", "smoke-nf", "smoke-c"}
    assert smokes == {s.name for s in STUDIES.values() if s.rounds == 1}
    assert smokes == {s.name for s in STUDIES.values() if s.extension_rounds is None}
    assert all("smoke" not in s.to_protocol_dict() for s in STUDIES.values())


def test_a_study_requires_the_published_checkpoints_unless_it_says_otherwise():
    study = Study(
        name="x", treatments=(T.MX, T.NV), cells=cells_at(1024, (8,)), primary_batches=(8,)
    )
    assert study.require_published is True
    assert {s.name for s in STUDIES.values() if not s.require_published} == {"smoke"}


T = Treatment


def test_the_main_run_matches_experiment_md():
    assert FULL.rounds == 5 and FULL.m1_reps == 5 and FULL.m2_duration_s == 30
    assert FULL.batches == (1, 8, 32, 64, 128)
    assert FULL.cells == cells_at(1024, (1, 8, 32, 64, 128))
    assert FULL.treatments == ("MX", "NV", "NVa", "MXp", "NVnf")
    assert FULL.require_published is True
    assert FULL.primary_batches == (8, 32, 128)
    assert set(FULL.primary_batches) <= set(FULL.batches)
    assert FULL.server == ServerSettings() == ServerSettings(KvDtype.BF16, 0.90, 4096, "")


def test_the_kernel_scan_smoke_matches_experiment_md():
    assert SMOKE.rounds == 1
    assert SMOKE.treatments == ("MX", "NV", "NVx", "NVc", "NVt", "NVd", "NVv")
    assert SMOKE.batches == (1, 32, 128)
    assert SMOKE.m1_reps == 3
    assert SMOKE.m1_reps <= settings.M1_SETS - 1
    assert SMOKE.m2_duration_s == 10
    assert SMOKE.require_published is False
    assert set(SMOKE.batches) <= set(FULL.batches)
    assert SMOKE.server == FULL.server and SMOKE.primary_batches == FULL.primary_batches


def test_the_fusion_smoke_matches_experiment_md():
    assert (
        replace(
            SMOKE,
            name="smoke-nf",
            treatments=(T.MX, T.NV, T.NVNF),
            require_published=True,
            kernel_scan=None,
            crosscheck=None,
        )
        == SMOKE_NF
    )


def test_experiment_b_matches_section_13():
    assert EXPB.rounds == 5
    assert EXPB.treatments == ("MX", "NV", "MXp")
    assert EXPB.batches == (128, 256, 512)
    assert (EXPB.m1_reps, EXPB.m2_duration_s) == (5, 30)
    assert EXPB.require_published is True
    assert EXPB.server == ServerSettings(kv_dtype=KvDtype.FP8, gpu_memory_utilization=0.95)
    assert EXPB.primary_batches == (256, 512)


EXPC_CELLS = (
    (1, 1024),
    (1, 4096),
    (1, 16384),
    (1, 32768),
    (1, 65536),
    (1, 127360),
    (8, 15360),
    (32, 3360),
    (128, 360),
)
ROPE_OVERRIDE = '{"max_position_embeddings": 131072}'


def test_experiment_c_matches_section_16():
    assert (EXPC.rounds, EXPC.treatments, EXPC.cells) == (5, ("MX", "NV", "MXp"), EXPC_CELLS)
    assert (EXPC.m1_reps, EXPC.m2_duration_s, EXPC.require_published) == (5, 0, True)
    assert EXPC.primary_batches == (1,) and EXPC.prompts is Prompts.CELLS
    assert EXPC.server == ServerSettings(max_model_len=131072, hf_overrides=ROPE_OVERRIDE)
    assert [p for c, p in EXPC.cells if c == 1] == [1024, 4096, 16384, 32768, 65536, 127360]
    batch_arm = [(c, p) for c, p in EXPC.cells if p == 127360 or c > 1]
    assert batch_arm == [(1, 127360), (8, 15360), (32, 3360), (128, 360)]
    assert {c * (p + (settings.M1_N1 + settings.M1_N2) // 2) for c, p in batch_arm} == {128_000}


def test_the_experiment_c_smoke_matches_section_16():
    assert (
        replace(
            EXPC,
            name="smoke-c",
            treatments=(T.MX, T.NV),
            cells=(Cell(1, 1024), Cell(1, 32768), Cell(1, 127360), Cell(128, 360)),
            rounds=1,
            m1_reps=3,
            extension_rounds=None,
            smoke=True,
        )
        == SMOKE_C
    )
    assert set(SMOKE_C.cells) <= set(EXPC.cells)


def test_every_study_with_its_own_prompts_is_covered_by_the_prompt_file_prepare_c_builds():
    for study in STUDIES.values():
        if study.prompts is Prompts.CELLS:
            assert set(study.cells) <= set(C_CELLS)
            assert study.m2_duration_s == 0
    assert EXPC.cells == C_CELLS


@pytest.mark.parametrize("name", list(STUDIES))
def test_every_study_is_runnable_with_the_configured_treatments_and_prompts(name):
    study = STUDIES[name]
    assert set(study.treatments) <= set(model.TREATMENTS)
    assert len(set(study.treatments)) == len(study.treatments)
    assert len(set(study.cells)) == len(study.cells)
    if name in ("full", "expb", "expc"):
        assert set(study.primary_batches) <= set(study.batches)
    assert study.m1_reps <= settings.M1_SETS - 1
    assert max(study.batches) <= settings.M1_SET_SIZE
    args = study.server.args()
    assert max(study.batches) <= int(args[args.index("--max-num-seqs") + 1])
    max_model_len = int(args[args.index("--max-model-len") + 1])
    for cell in study.cells:
        assert cell.prompt_len + settings.M1_N2 <= max_model_len


def test_a_study_refuses_more_m1_reps_than_the_prompt_sets_after_the_warmup_set():
    assert replace(FULL, m1_reps=settings.M1_SETS - 1).m1_reps == 5
    with pytest.raises(
        ValueError,
        match=r"^study full: its 6 M1 reps and the warmup need 7 prompt sets; "
        r"there are 6 \(settings.M1_SETS\)$",
    ):
        replace(FULL, m1_reps=settings.M1_SETS)


def test_a_study_refuses_a_batch_above_the_prompt_sets_or_the_servers_max_num_seqs(monkeypatch):
    assert max(EXPB.batches) == settings.M1_SET_SIZE == base.MAX_NUM_SEQS == 512
    with pytest.raises(
        ValueError,
        match=r"^study expb: its largest batch 513 is above the 512 prompts of a prompt set "
        r"\(settings.M1_SET_SIZE\)$",
    ):
        replace(EXPB, cells=cells_at(1024, (128, 256, 513)), primary_batches=(256,))
    monkeypatch.setattr(base, "MAX_NUM_SEQS", 256)
    with pytest.raises(
        ValueError, match=r"^study expb: its largest batch 512 is above the server's 256 "
    ):
        replace(EXPB)
    assert replace(EXPB, cells=cells_at(1024, (128, 256)), primary_batches=(256,))


def test_a_study_refuses_a_longest_prompt_and_n2_wave_beyond_the_servers_max_model_len():
    needed = 127360 + settings.M1_N2
    assert replace(EXPC, server=replace(EXPC.server, max_model_len=needed))
    with pytest.raises(
        ValueError,
        match=rf"^study expc: its longest prompt \(127360 tokens\) and the {settings.M1_N2}-token "
        rf"N2 wave need a max_model_len of {needed}; its server's is {needed - 1}$",
    ):
        replace(EXPC, server=replace(EXPC.server, max_model_len=needed - 1))


def test_m2_runs_at_the_batches_of_the_studies_with_m2():
    assert FULL.m2_batches() == (1, 8, 32, 64, 128)
    assert EXPB.m2_batches() == (128, 256, 512)
    assert EXPC.m2_batches() == SMOKE_C.m2_batches() == ()
    assert replace(FULL, m2_duration_s=0).m2_batches() == ()


def test_the_kv_capacity_experiment_b_needs():
    assert EXPB.min_kv_tokens() == 512 * (settings.M1_INPUT_LEN + settings.M1_N2) == 1_114_112
    assert FULL.min_kv_tokens() == 128 * 2176


def test_min_kv_tokens_for_a_largest_batch_is_what_every_study_on_the_m1_prompts_needs():
    assert min_kv_tokens_for(512) == 1_114_112
    assert min_kv_tokens_for(1) == settings.M1_INPUT_LEN + settings.M1_N2
    for study in STUDIES.values():
        if study.prompts is Prompts.M1:
            assert study.min_kv_tokens() == min_kv_tokens_for(max(study.batches))
            assert study.largest_wave() == (max(study.batches), settings.M1_INPUT_LEN)


def test_the_kv_capacity_experiment_c_needs_is_its_largest_cells_wave():
    per_cell = {cell: cell[0] * (cell[1] + settings.M1_N2) for cell in EXPC.cells}
    assert per_cell[Cell(1, 127360)] == 128_512 and per_cell[Cell(128, 360)] == 193_536
    assert EXPC.min_kv_tokens() == max(per_cell.values()) == 193_536
    assert EXPC.largest_wave() == (128, 360)
    assert SMOKE_C.min_kv_tokens() == 193_536
    one_cell = Study(
        name="x",
        treatments=(T.MX,),
        cells=(Cell(1, 127360),),
        server=EXPC.server,
        prompts=Prompts.CELLS,
        primary_batches=(1,),
    )
    assert one_cell.min_kv_tokens() == 128_512


OLD_SERVER_ARGS = (
    "--served-model-name",
    "fp4bench",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
    "--max-model-len",
    "4096",
    "--max-num-seqs",
    "512",
    "--max-num-batched-tokens",
    "16384",
    "--gpu-memory-utilization",
    "0.90",
    "--no-enable-prefix-caching",
    "--seed",
    "0",
    "--api-server-count",
    "4",
    "--kv-cache-dtype",
    "bfloat16",
)


@pytest.mark.parametrize("study", [FULL, SMOKE, SMOKE_NF])
def test_the_main_server_settings_give_the_argv_of_before_byte_for_byte(study):
    assert study.server.args() == ServerSettings().args() == OLD_SERVER_ARGS


def test_common_server_args_use_bf16_kv_and_no_kernel_pin():
    args = FULL.server.args()
    assert args[args.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert "--linear-backend" not in args
    assert "--compilation-config" not in args
    assert args[args.index("--max-model-len") + 1] == "4096"


def test_experiment_b_serves_with_fp8_kv_and_095_in_the_same_positions():
    args = EXPB.server.args()
    assert len(args) == len(OLD_SERVER_ARGS)
    changed = {
        i: (old, new)
        for i, (old, new) in enumerate(zip(OLD_SERVER_ARGS, args, strict=True))
        if old != new
    }
    assert changed == {
        OLD_SERVER_ARGS.index("--gpu-memory-utilization") + 1: ("0.90", "0.95"),
        OLD_SERVER_ARGS.index("--kv-cache-dtype") + 1: ("bfloat16", "fp8"),
    }


def test_experiment_c_serves_with_a_131072_window_and_the_rope_override_at_the_end():
    args = EXPC.server.args()
    assert args[: len(OLD_SERVER_ARGS)] == tuple(
        "131072" if i == OLD_SERVER_ARGS.index("--max-model-len") + 1 else a
        for i, a in enumerate(OLD_SERVER_ARGS)
    )
    assert args[len(OLD_SERVER_ARGS) :] == ("--hf-overrides", ROPE_OVERRIDE)
    assert json.loads(args[-1]) == {"max_position_embeddings": 131072}
    assert SMOKE_C.server.args() == args


def test_the_context_window_is_a_server_setting_in_its_old_position():
    args = ServerSettings(max_model_len=8192).args()
    changed = {
        i: (old, new)
        for i, (old, new) in enumerate(zip(OLD_SERVER_ARGS, args, strict=True))
        if old != new
    }
    assert changed == {OLD_SERVER_ARGS.index("--max-model-len") + 1: ("4096", "8192")}
    assert len(args) == len(OLD_SERVER_ARGS)


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '"x"', "{}", '{"a": 1'])
def test_server_args_refuses_an_hf_override_that_is_not_a_json_object(bad):
    with pytest.raises(ValueError, match="hf_overrides"):
        ServerSettings(hf_overrides=bad).args()


@pytest.mark.parametrize("bad", [0, -4096, 4096.0, True, "4096"])
def test_server_args_refuses_a_context_window_that_is_not_a_positive_int(bad):
    with pytest.raises(ValueError, match="max_model_len"):
        ServerSettings(max_model_len=bad).args()


def test_server_args_refuses_a_memory_fraction_it_cannot_write_exactly():
    assert "0.85" in ServerSettings(gpu_memory_utilization=0.85).args()
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        ServerSettings(gpu_memory_utilization=0.925).args()
