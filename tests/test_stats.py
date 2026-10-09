import math
import statistics

import pytest
from scipy.stats import t as student_t

from fp4bench.analysis import stats as st
from fp4bench.core.types import Verdict


def test_paired_log_ratios():
    assert st.paired_log_ratios([2.0, 3.0], [1.0, 3.0]) == pytest.approx([math.log(2), 0.0])


def test_paired_log_ratios_rejects_unpaired_or_single():
    with pytest.raises(ValueError, match="unpaired samples"):
        st.paired_log_ratios([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="need at least 2 pairs"):
        st.paired_log_ratios([1.0], [1.0])


def test_ratio_ci_known_value():
    d = [-0.01, -0.005, 0.0, 0.005, 0.01]
    est, lo, hi = st.ratio_ci(d, level=0.90)
    half = 2.131847 * 0.0079057 / math.sqrt(5)
    assert est == pytest.approx(1.0)
    assert lo == pytest.approx(math.exp(-half), rel=1e-5)
    assert hi == pytest.approx(math.exp(half), rel=1e-5)


def test_ratio_ci_nonzero_mean_orientation():
    est, lo, hi = st.ratio_ci([0.02, 0.03, 0.025, 0.035, 0.028], level=0.90)
    assert (est, lo, hi) == pytest.approx((1.02798, 1.02252, 1.03348), abs=2e-5)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (0.985, 1.015, Verdict.EQUIVALENT),
        (1.025, 1.040, Verdict.MXFP4_FASTER),
        (0.950, 0.970, Verdict.NVFP4_FASTER),
        (1.005, 1.030, Verdict.DIFFERENT_SMALL),
        (0.970, 0.995, Verdict.DIFFERENT_SMALL),
        (0.990, 1.030, Verdict.INCONCLUSIVE),
        (0.960, 1.040, Verdict.INCONCLUSIVE),
        (0.98, 1.02, Verdict.EQUIVALENT),
        (1.02, 1.05, Verdict.DIFFERENT_SMALL),
        (0.95, 0.98, Verdict.DIFFERENT_SMALL),
        (1.00, 1.03, Verdict.INCONCLUSIVE),
    ],
)
def test_classify_follows_preregistered_table(lo, hi, expected):
    assert st.classify(lo, hi, delta=0.02) == expected


def test_overall_verdict():
    primary = (8, 32, 128)
    eq = {
        8: Verdict.EQUIVALENT,
        32: Verdict.EQUIVALENT,
        128: Verdict.EQUIVALENT,
        1: Verdict.INCONCLUSIVE,
    }
    assert st.overall_verdict(eq, primary) == "validated"
    mx = {**eq, 32: Verdict.MXFP4_FASTER}
    assert st.overall_verdict(mx, primary) == "nullified_mxfp4_faster"
    nv = {**eq, 128: Verdict.NVFP4_FASTER}
    assert st.overall_verdict(nv, primary) == "nullified_nvfp4_faster"
    mixed = {**eq, 32: Verdict.MXFP4_FASTER, 128: Verdict.NVFP4_FASTER}
    assert st.overall_verdict(mixed, primary) == "nullified_mixed"
    unclear = {**eq, 8: Verdict.DIFFERENT_SMALL}
    assert st.overall_verdict(unclear, primary) == "inconclusive"
    assert type(st.overall_verdict(unclear, primary)) is st.OverallVerdict


def test_overall_verdict_rejects_empty_primary():
    with pytest.raises(ValueError, match="no primary endpoints"):
        st.overall_verdict({}, ())


def test_k_relabels_the_format_verdicts_as_pinned_and_alt_kernel_verdicts():
    assert st.relabel_for_k(Verdict.MXFP4_FASTER) == Verdict.PINNED_FASTER
    assert st.relabel_for_k(Verdict.NVFP4_FASTER) == Verdict.ALT_FASTER
    for unchanged in (Verdict.EQUIVALENT, Verdict.DIFFERENT_SMALL, Verdict.INCONCLUSIVE):
        assert st.relabel_for_k(unchanged) == unchanged


GOLDEN_CI = (0.0015, 0.0115)
DIFFS = [0.010, -0.004, 0.022, 0.007, 0.001, 0.015, -0.009, 0.012, 0.003, 0.018, 0.006, -0.002]


def test_bootstrap_is_deterministic_for_the_same_inputs():
    first = st.paired_bootstrap_ci(DIFFS)
    assert st.paired_bootstrap_ci(list(DIFFS)) == first
    assert st.paired_bootstrap_ci(DIFFS, resamples=10_000, seed=0, level=0.95) == first


def test_bootstrap_depends_on_the_seed():
    assert st.paired_bootstrap_ci(DIFFS, seed=1) != st.paired_bootstrap_ci(DIFFS, seed=0)


def test_bootstrap_matches_a_pinned_value_so_that_the_interval_is_reproducible():
    mean, lo, hi = st.paired_bootstrap_ci(DIFFS, level=0.95, resamples=10_000, seed=0)
    assert mean == pytest.approx(sum(DIFFS) / len(DIFFS), abs=1e-15)
    assert (lo, hi) == pytest.approx(GOLDEN_CI, abs=1e-12)


def test_bootstrap_of_identical_differences_is_a_point():
    assert st.paired_bootstrap_ci([0.25] * 8) == pytest.approx((0.25, 0.25, 0.25))


def test_bootstrap_interval_brackets_the_mean_and_tracks_the_normal_interval():
    mean, lo, hi = st.paired_bootstrap_ci(DIFFS)
    sd = statistics.stdev(DIFFS)
    assert lo < mean < hi
    assert (hi - lo) / 2 == pytest.approx(1.96 * sd / math.sqrt(len(DIFFS)), rel=0.15)


def test_bootstrap_level_widens_the_interval():
    _, lo95, hi95 = st.paired_bootstrap_ci(DIFFS, level=0.95)
    _, lo80, hi80 = st.paired_bootstrap_ci(DIFFS, level=0.80)
    assert lo95 < lo80 and hi80 < hi95


def test_bootstrap_rejects_too_few_or_non_finite_differences():
    with pytest.raises(ValueError, match="at least 2"):
        st.paired_bootstrap_ci([0.1])
    with pytest.raises(ValueError, match="finite"):
        st.paired_bootstrap_ci([0.1, float("nan"), 0.2])
    with pytest.raises(ValueError, match="finite"):
        st.paired_bootstrap_ci([0.1, None, 0.2])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]


def test_phi_relabels_the_format_verdicts_as_fusion_verdicts():
    assert st.relabel_for_phi(Verdict.MXFP4_FASTER) == Verdict.FUSION_SLOWER
    assert st.relabel_for_phi(Verdict.NVFP4_FASTER) == Verdict.FUSION_FASTER
    for unchanged in (Verdict.EQUIVALENT, Verdict.DIFFERENT_SMALL, Verdict.INCONCLUSIVE):
        assert st.relabel_for_phi(unchanged) == unchanged


def test_phi_labels_follow_the_decision_table_orientation():
    assert st.relabel_for_phi(st.classify(1.03, 1.05, 0.02)) == Verdict.FUSION_SLOWER
    assert st.relabel_for_phi(st.classify(0.95, 0.97, 0.02)) == Verdict.FUSION_FASTER


def test_mean_ci_is_a_t_interval_with_n_minus_1_degrees_of_freedom():
    xs = [0.40, 0.45, 0.43, 0.38, 0.44]
    m, lo, hi = st.mean_ci(xs, 0.90)
    half = student_t.ppf(0.95, 4) * statistics.stdev(xs) / math.sqrt(5)
    assert m == pytest.approx(statistics.fmean(xs))
    assert (lo, hi) == pytest.approx((m - half, m + half))
    assert st.mean_ci([0.4], 0.90) == (0.4, None, None)
    with pytest.raises(ValueError, match="no values"):
        st.mean_ci([], 0.90)
