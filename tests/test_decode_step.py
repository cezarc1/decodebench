import math

import pytest

from fp4bench.decode_step import decode_block, step_time


def test_step_time_differences_out_fixed_costs():
    assert step_time(1.5, 5.5, 128, 1152) == pytest.approx(4.0 / 1024)


def test_step_time_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-positive decode window"):
        step_time(2.0, 2.0, 128, 1152)
    with pytest.raises(ValueError, match="n_long must exceed n_short"):
        step_time(1.0, 2.0, 1152, 128)


def test_step_time_lenient_returns_nan_for_non_positive_window_only():
    assert math.isnan(step_time(2.0, 2.0, 128, 1152, strict=False))
    assert math.isnan(step_time(2.0, 1.0, 128, 1152, strict=False))
    with pytest.raises(ValueError, match="n_long must exceed n_short"):
        step_time(1.0, 2.0, 1152, 128, strict=False)


def test_decode_block_runs_warmup_plus_reps_and_alternates_order():
    calls = []

    def fake_wave(prompts, n):
        calls.append((len(prompts), n))
        return 0.25 + 0.004 * n

    sets = [[[i]] * 8 for i in range(4)]
    rows = decode_block(fake_wave, sets, c=4, n1=128, n2=1152, reps=3)
    assert len(rows) == 4
    assert rows[0]["warmup"] and not any(r["warmup"] for r in rows[1:])
    assert [r["set"] for r in rows] == [0, 1, 2, 3]
    assert all(r["step_s"] == pytest.approx(0.004) for r in rows)
    assert all(r["decode_tok_s"] == pytest.approx(1000.0) for r in rows)
    assert calls == [
        (4, 128),
        (4, 1152),
        (4, 1152),
        (4, 128),
        (4, 128),
        (4, 1152),
        (4, 1152),
        (4, 128),
    ]


def test_decode_block_rows_record_order_and_lengths_for_order_effect_analysis():
    sets = [[[i]] * 8 for i in range(4)]
    rows = decode_block(lambda p, n: 0.25 + 0.004 * n, sets, c=4, n1=128, n2=1152, reps=3)
    assert [r["first"] for r in rows] == ["n1", "n2", "n1", "n2"]
    assert all(r["n1"] == 128 and r["n2"] == 1152 for r in rows)


def test_decode_block_warmup_with_bad_window_records_nan_instead_of_aborting():
    calls = []

    def fake_wave(prompts, n):
        calls.append(n)
        if len(calls) <= 2:
            return {128: 2.0, 1152: 1.5}[n]
        return 0.25 + 0.004 * n

    sets = [[[i]] * 8 for i in range(3)]
    rows = decode_block(fake_wave, sets, c=4, n1=128, n2=1152, reps=2)
    assert rows[0]["warmup"]
    assert math.isnan(rows[0]["step_s"]) and math.isnan(rows[0]["decode_tok_s"])
    assert (rows[0]["t1_s"], rows[0]["t2_s"]) == (2.0, 1.5)
    assert all(r["step_s"] == pytest.approx(0.004) for r in rows[1:])


def test_decode_block_measured_pair_with_bad_window_still_raises():
    def fake_wave(prompts, n):
        return 1.0

    sets = [[[i]] * 8 for i in range(3)]
    with pytest.raises(ValueError, match="non-positive decode window"):
        decode_block(fake_wave, sets, c=4, n1=128, n2=1152, reps=2)


def test_decode_block_rejects_small_prompt_sets():
    with pytest.raises(ValueError, match="fewer than 4 prompts"):
        decode_block(lambda p, n: 1.0, [[[1]] * 2], c=4, n1=1, n2=2, reps=0)
