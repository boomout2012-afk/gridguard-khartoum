from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

from gridguard.committee_results import load_committee_results


@pytest.mark.parametrize("tamper", [None, "means", "paired", "stress", "missing"])
def test_packaged_followup_requires_every_pinned_display_cell(tamper) -> None:
    view = load_committee_results()

    def table(data, altered=False):
        rows = [list(row) for row in data.rows]
        if altered:
            rows[0][1] = "unverified"
        return SimpleNamespace(
            rowCount=lambda: len(rows),
            columnCount=lambda: len(data.columns),
            item=lambda row, column: SimpleNamespace(text=lambda: rows[row][column]),
        )

    followup = SimpleNamespace(
        view=view,
        means_table=table(view.means, tamper == "means"),
        paired_tables=[table(data, tamper == "paired") for _, data in view.paired],
        stress_table=table(view.stress, tamper == "stress"),
    )
    if tamper == "missing":
        followup.means_table = None
    launcher = run_path(str(Path(__file__).parents[1] / "packaging/gridguard_launcher.py"))
    check = launcher["_followup_smoke_facts"]
    page = SimpleNamespace(followup_page=followup)
    if tamper:
        with pytest.raises(RuntimeError, match="follow-up"):
            check(page)
    else:
        facts = check(page)
        assert facts["status"] == "PASS"
        assert facts["display_matches_pinned_audit"] is True
        assert facts["controller_rows"] == 5
        assert facts["stress_rows"] == 10
        assert facts["evidence"] == dict(view.evidence)
