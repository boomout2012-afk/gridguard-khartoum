from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gridguard.committee_results import PINNED_FILES, CommitteeResultsError, load_committee_results

ROOT = Path(__file__).resolve().parents[1]


def test_followup_view_keeps_post_hoc_scope_and_adverse_energy_tradeoff() -> None:
    view = load_committee_results(root=ROOT)
    assert "Post-hoc" in view.scope_statement
    assert "960" in view.count_statement and "five" in view.count_statement
    assert len(view.means.rows) == 5
    assert view.means.rows[-1][0] == "PPO"
    assert view.means.rows[-1][1] == "6528.02"
    assert view.means.rows[1][0] == "Critical-aware Fixed Rotation"
    assert view.paired[0][1].rows[0][1] == "+241.63"
    assert view.paired[1][1].rows[0][1] == "−16.68"
    assert "tradeoff" in view.tradeoff_statement
    assert len(view.stress.rows) == 10
    assert view.stress.rows[4][2:5] == ("24", "168", "0")
    assert "not uninterrupted" in view.stress_statement


@pytest.mark.parametrize("relative", tuple(PINNED_FILES))
@pytest.mark.parametrize("tamper", ["missing", "changed"])
def test_followup_artifact_and_receipt_fail_closed(tmp_path, relative, tamper) -> None:
    for source in PINNED_FILES:
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    target = tmp_path / relative
    if tamper == "missing":
        target.unlink()
    else:
        with target.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(CommitteeResultsError, match="verify"):
        load_committee_results(root=tmp_path)
