"""The ty ratchet holds: no new diagnostics, and a baseline that matches what ty reports."""

import pytest

from tests import ty_baseline

A = "fp4bench/x.py:3:1: error[invalid-argument-type] Argument to `f` is incorrect: Expected `int`"
B = "fp4bench/x.py:9:5: error[invalid-argument-type] Argument to `g` is incorrect: Expected `str`"
C = "tests/t.py:1:1: warning[redundant-condition] Condition is always true"


@pytest.fixture
def ty_reports(tmp_path, monkeypatch):
    """Point the ratchet at a scratch baseline of `baseline` lines; ty reports `lines`."""

    def setup(baseline: list[str], lines: list[str]):
        monkeypatch.setattr(ty_baseline, "BASELINE", tmp_path / "ty_baseline.txt")
        ty_baseline.write_baseline(ty_baseline.counts(baseline))
        monkeypatch.setattr(ty_baseline, "run_ty", lambda: lines)
        return ty_baseline.BASELINE

    return setup


def test_ty_reports_exactly_what_its_baseline_allows(capsys):
    assert ty_baseline.main([]) == 0, capsys.readouterr().err


def test_diagnostics_are_counted_per_file_rule_and_message():
    assert ty_baseline.counts([A, A.replace(":3:1:", ":30:7:"), C]) == {
        (
            "fp4bench/x.py",
            "invalid-argument-type",
            "Argument to `f` is incorrect: Expected `int`",
        ): 2,
        ("tests/t.py", "redundant-condition", "Condition is always true"): 1,
    }


def test_the_baseline_round_trips(ty_reports):
    path = ty_reports([A, A, B, C], [])
    assert ty_baseline.read_baseline() == ty_baseline.counts([A, A, B, C])
    assert "# 4 diagnostics" in path.read_text()


def test_a_new_diagnostic_fails_and_update_will_not_add_it(ty_reports, capsys):
    path = ty_reports([A], [A, C])
    before = path.read_bytes()
    assert ty_baseline.main([]) == 1
    assert C in capsys.readouterr().err
    for argv in (["--update"], ["--update", "--reworded"]):
        assert ty_baseline.main(argv) == 1
        assert path.read_bytes() == before


def test_one_more_of_a_known_diagnostic_fails_too(ty_reports):
    path = ty_reports([A], [A, A])
    before = path.read_bytes()
    assert ty_baseline.main([]) == 1
    assert ty_baseline.main(["--update"]) == 1
    assert path.read_bytes() == before


def test_a_fix_and_a_new_diagnostic_in_one_file_and_rule_do_not_cancel_out(ty_reports, capsys):
    path = ty_reports([A], [B])
    before = path.read_bytes()
    assert ty_baseline.main([]) == 1
    assert B in capsys.readouterr().err
    assert ty_baseline.main(["--update"]) == 1
    assert path.read_bytes() == before
    # A retyping that only rewords a diagnostic it does not fix is accepted on request.
    assert ty_baseline.main(["--update", "--reworded"]) == 0
    assert ty_baseline.read_baseline() == ty_baseline.counts([B])


def test_a_fixed_diagnostic_asks_for_a_lower_baseline(ty_reports, capsys):
    path = ty_reports([A, C], [A])
    assert ty_baseline.main([]) == 1
    assert "1 fewer" in capsys.readouterr().err
    assert ty_baseline.main(["--update"]) == 0
    assert ty_baseline.read_baseline() == ty_baseline.counts([A])
    assert "# 1 diagnostics" in path.read_text()


def test_reworded_needs_update():
    with pytest.raises(SystemExit):
        ty_baseline.main(["--reworded"])


def test_compare_lists_what_rose_and_what_fell():
    now = ty_baseline.counts([A, C, C])
    baseline = ty_baseline.counts([A, A, B, C])
    more, fewer = ty_baseline.compare(now, baseline)
    assert more == [next(iter(ty_baseline.counts([C])))]
    assert fewer == sorted(ty_baseline.counts([A, B]))
