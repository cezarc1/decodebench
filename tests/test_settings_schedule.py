import json

from fp4bench import settings
from fp4bench.schedule import round_order
from fp4bench.studies.main import FULL
from tests import RUNS_DIR


def test_the_full_schedule_is_a_latin_square_every_treatment_holds_every_position_once():
    square = [round_order(r, FULL.treatments) for r in range(FULL.rounds)]
    assert len(square) == len(FULL.treatments) == 5
    for position in range(5):
        assert sorted(row[position] for row in square) == sorted(FULL.treatments)


def test_the_expected_versions_are_the_ones_every_committed_run_recorded():
    manifests = sorted(RUNS_DIR.glob("*/manifests.jsonl"))
    assert RUNS_DIR / "full-1" / "manifests.jsonl" in manifests
    for path in manifests:
        for line in path.read_text().splitlines():
            packages = json.loads(line)["packages"]
            recorded = {pkg: packages[pkg].split("+", 1)[0] for pkg in settings.EXPECTED_VERSIONS}
            assert recorded == settings.EXPECTED_VERSIONS, path
