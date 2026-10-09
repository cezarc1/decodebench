import pytest

from fp4bench import bytes_model as bm
from fp4bench.core.types import Format, KvDtype


def test_quantized_param_count_matches_config_arithmetic():
    assert bm.QUANT_PARAMS == 31_205_621_760
    per_layer = (5120 * 8192 + 2 * 5120 * 1024 + 8192 * 5120) + 3 * 5120 * 25600
    assert 64 * per_layer == bm.QUANT_PARAMS


def test_lm_head_is_bf16_and_read_every_step():
    assert bm.LM_HEAD_BYTES == 1_555_824_640
    assert bm.LM_HEAD_BYTES == 2 * 151_936 * 5120


def test_kv_bytes_per_token():
    assert bm.kv_bytes_per_token() == 262_144
    assert bm.kv_bytes_per_seq(1664) == 1664 * 262_144


def test_format_byte_ratio_on_weights():
    assert bm.BITS[Format.NVFP4] / bm.BITS[Format.MXFP4] == pytest.approx(1.0588, abs=1e-4)


def test_weight_sizes_match_experiment_md():
    assert bm.weight_bytes(Format.MXFP4) / 1e9 == pytest.approx(16.58, abs=0.005)
    assert bm.weight_bytes(Format.NVFP4) / 1e9 == pytest.approx(17.55, abs=0.005)


def test_every_token_reads_every_weight_so_step_bytes_are_linear_in_c():
    one, two, ten = (bm.step_bytes(c, 1664, Format.MXFP4) for c in (1, 2, 10))
    assert two - one == pytest.approx(bm.kv_bytes_per_seq(1664))
    assert ten - one == pytest.approx(9 * bm.kv_bytes_per_seq(1664))
    assert one == bm.weight_bytes(Format.MXFP4) + bm.LM_HEAD_BYTES + bm.kv_bytes_per_seq(1664)


def test_step_bytes_rejects_an_unknown_format():
    with pytest.raises(KeyError):
        bm.step_bytes(1, 1664, "fp8")  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "c,expected", [(1, 1.052), (8, 1.045), (32, 1.030), (64, 1.021), (128, 1.013)]
)
def test_r_ideal_matches_experiment_md_table(c, expected):
    assert bm.r_ideal(c, ctx=1664) == pytest.approx(expected, abs=0.0015)


def test_r_ideal_falls_towards_one_as_the_kv_cache_dilutes_the_weights():
    values = [bm.r_ideal(c, 1664) for c in (1, 8, 32, 64, 128)]
    assert values == sorted(values, reverse=True) and values[-1] > 1.0


def test_kv_bytes_per_element_by_dtype():
    assert {kv: bm.kv_bytes_per_element(kv) for kv in KvDtype} == {
        KvDtype.BF16: 2,
        KvDtype.FP8: 1,
        KvDtype.FP8_E4M3: 1,
        KvDtype.FP8_E5M2: 1,
    }
    assert bm.kv_bytes_per_token() == 262_144
    assert bm.kv_bytes_per_token(KvDtype.FP8) == 131_072
    assert bm.kv_bytes_per_seq(1664, KvDtype.FP8) == 1664 * 131_072


def test_fp8_kv_halves_the_kv_bytes_of_a_step_and_leaves_the_weights():
    bf16, fp8 = (
        bm.step_bytes(512, 1664, Format.NVFP4),
        bm.step_bytes(512, 1664, Format.NVFP4, kv_dtype=KvDtype.FP8),
    )
    assert bf16 - fp8 == pytest.approx(512 * 1664 * 131_072)
    assert bm.step_bytes(512, 1664, Format.NVFP4, KvDtype.BF16) == bf16
    assert (fp8 - bm.weight_bytes(Format.NVFP4) - bm.LM_HEAD_BYTES) / 1e9 == pytest.approx(
        111.67, abs=0.01
    )


@pytest.mark.parametrize("c,expected", [(128, 1.0212), (256, 1.0132), (512, 1.0075)])
def test_r_ideal_with_fp8_kv_matches_the_experiment_b_table(c, expected):
    assert bm.r_ideal(c, 1664, KvDtype.FP8) == pytest.approx(expected, abs=1e-4)
    assert bm.r_ideal(c, 1664, kv_dtype=KvDtype.FP8) == bm.r_ideal(c, 1664, KvDtype.FP8)
    assert bm.r_ideal(c, 1664, KvDtype.FP8) > bm.r_ideal(c, 1664)


def test_an_unknown_kv_dtype_is_refused_rather_than_guessed():
    with pytest.raises(TypeError):
        bm.r_ideal(128, 1664, "int4")  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
