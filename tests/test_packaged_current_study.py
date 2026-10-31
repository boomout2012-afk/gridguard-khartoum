"""Frozen smoke must reject missing or mismatched current-study displays."""

from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

from gridguard.focused_results import load_focused_results


@pytest.mark.parametrize("corrupt_cell", [False, True])
def test_packaged_current_table_agrees_with_pinned_study(corrupt_cell: bool) -> None:
    view = load_focused_results()
    rows = [list(row) for row in view.means.rows]
    if corrupt_cell:
        rows[2][3] = "100.00%"
    table = SimpleNamespace(
        rowCount=lambda: len(rows),
        columnCount=lambda: len(view.means.columns),
        item=lambda row, column: SimpleNamespace(text=lambda: rows[row][column]),
    )
    page = SimpleNamespace(study_view=view, current_results_table=table)
    launcher = run_path(str(Path(__file__).parents[1] / "packaging/gridguard_launcher.py"))
    check = launcher["_focused_study_smoke_facts"]
    if corrupt_cell:
        with pytest.raises(RuntimeError, match="differs"):
            check(page)
    else:
        facts = check(page)
        assert facts["display_matches_pinned_study"] is True
        assert facts["displayed_outcomes"] == 5
        assert facts["evidence"] == dict(view.evidence)


def test_packaged_current_study_cannot_silently_fall_back_to_history() -> None:
    launcher = run_path(str(Path(__file__).parents[1] / "packaging/gridguard_launcher.py"))
    with pytest.raises(RuntimeError, match="current study"):
        launcher["_focused_study_smoke_facts"](
            SimpleNamespace(study_view=None, current_results_table=None)
        )
