from pathlib import Path

import pytest

from gridguard.ppo_experiment import (
    CONFIG,
    TRAINING_SEEDS,
    read_json,
    training_batches,
    write_json,
)


def test_training_schedule_is_balanced_and_disjoint() -> None:
    totals = [0, 0, 0]
    seeds = set()
    for run in range(len(TRAINING_SEEDS)):
        for iteration in range(CONFIG["iterations"]):
            seed, batches = training_batches(run, iteration)
            assert sum(batches) == 256
            assert seed not in seeds
            seeds.add(seed)
            for i, count in enumerate(batches):
                totals[i] += count
    assert len(set(totals)) == 1
    assert totals[0] * 3 * 24 == 18_432_000
    assert not seeds.intersection(range(112101, 112211))


def test_evidence_write_refuses_overwrite_and_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    digest = write_json(path, {"value": 1})
    assert read_json(path, digest) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_json(path, {"value": 2})
    path.write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        read_json(path, digest)


@pytest.mark.parametrize("run,iteration", [(-1, 0), (5, 0), (0, -1), (0, 600)])
def test_training_schedule_rejects_out_of_protocol_indices(run: int, iteration: int) -> None:
    with pytest.raises(ValueError):
        training_batches(run, iteration)
