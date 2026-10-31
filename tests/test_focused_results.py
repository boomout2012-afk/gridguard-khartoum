"""Current-study presentation must retain its pinned identity and adverse findings."""

import shutil
from pathlib import Path

import pytest

from gridguard.focused_results import (
    PINNED_FILES,
    FocusedStudyError,
    load_focused_results,
)

ROOT = Path(__file__).resolve().parents[1]


def test_current_study_formats_all_three_methods_and_honest_tradeoff() -> None:
    view = load_focused_results(root=ROOT)
    assert view.means.columns == ("Outcome", "Fixed Rotation", "Constrained optimizer", "PPO")
    assert view.means.rows[1] == ("Jain service fairness", "87.76%", "94.78%", "97.02%")
    assert view.means.rows[2] == ("Supply utilization", "71.23%", "97.41%", "97.02%")
    assert "did not pass" in view.optimizer_conclusion
    assert "+2.237 pp" in view.tradeoff
    assert "−0.399 pp" in view.tradeoff
    assert "97.5%" in view.interval_statement
    assert "five" in view.count_statement
    assert "validation-selected PPO" in view.live_statement
    assert "five-run" in view.live_statement
    assert "single scenario" in view.live_statement
    assert "earlier" not in view.live_statement
    assert len(view.runs.rows) == 5
    assert "+0.0499" in view.paired[1][1].rows[3][1]


@pytest.mark.parametrize("relative", tuple(PINNED_FILES))
def test_missing_current_artifacts_fail_closed(tmp_path, relative) -> None:
    for source in PINNED_FILES:
        if source != relative:
            destination = tmp_path / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, destination)
    with pytest.raises(FocusedStudyError, match="verify"):
        load_focused_results(root=tmp_path)


@pytest.mark.parametrize("relative", tuple(PINNED_FILES))
def test_changed_current_artifacts_fail_closed(tmp_path, relative) -> None:
    for source in PINNED_FILES:
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    with (tmp_path / relative).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(FocusedStudyError, match="verify"):
        load_focused_results(root=tmp_path)
