"""Public focused widgets used to compose the live-scenario page."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gridguard.ui.components import ClaimBoundaryNote
from gridguard.ui.live_controls import (
    LiveComparisonSummary,
    LiveComparisonTable,
    LiveScenarioControls,
)
from gridguard.ui.live_visualization import LiveLocalityPanel, LiveMapPanel
from gridguard.ui.power_trace import PowerTrace

__all__ = [
    "LiveComparisonTable",
    "LiveLocalityPanel",
    "LiveMapPanel",
    "LiveScenarioControls",
]


def compose_live_page(page, asset) -> None:
    """Compose approved presentation surfaces without scientific calculations."""

    page.setObjectName("approvedLivePage")
    page.setStyleSheet(_LIVE_STYLE)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(16, 12, 16, 10)
    layout.setSpacing(8)
    page.controls = LiveScenarioControls()
    for name in ("shortage_combo", "seed_input", "run_button", "state_label"):
        setattr(page, name, getattr(page.controls, name))
    header = QWidget()
    header.setFixedHeight(64)
    row = QHBoxLayout(header)
    row.setContentsMargins(0, 0, 0, 0)
    text = QVBoxLayout()
    text.setContentsMargins(0, 0, 0, 0)
    text.setSpacing(1)
    title = QLabel("Khartoum, hour by hour.")
    title.setObjectName("livePageTitle")
    title.setAccessibleName("Page title: Live Scenario")
    page.description = QLabel("Explore how power reaches each locality.")
    page.description.setObjectName("liveDescription")
    text.addWidget(title)
    text.addWidget(page.description)
    text.addWidget(page.state_label)
    page.description.hide()
    page.state_label.show()
    row.addLayout(text, 1)
    page.scope_note = ClaimBoundaryNote()
    page.scope_note.setText("Synthetic scenario · Decision support only")
    page.scope_note.setObjectName("liveScopeNote")
    page.scope_note.setWordWrap(False)
    row.addWidget(page.scope_note, 0, Qt.AlignmentFlag.AlignTop)
    layout.addWidget(header)
    layout.addWidget(page.controls)
    page.map_panel = LiveMapPanel(asset)
    for name in ("hour_label", "controller_combo", "timeline", "map_view"):
        setattr(page, name, getattr(page.map_panel, name))
    page.controls.add_controller_selector(page.map_panel.controller_selector)
    page.controls.setFixedHeight(52)
    page.body = QWidget()
    body = QHBoxLayout(page.body)
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(22)
    body.addWidget(page.map_panel, 1)
    page.locality_panel = LiveLocalityPanel(asset, page.map_view)
    for name in (
        "locality_list",
        "locality_card",
        "locality_name",
        "locality_context",
        "locality_status",
        "locality_demand",
        "locality_served",
        "locality_critical",
        "locality_projection",
        "sector_service_labels",
    ):
        setattr(page, name, getattr(page.locality_panel, name))
    body.addWidget(page.locality_panel)
    layout.addWidget(page.body, 1)
    page.map_panel.timeline_bar.setFixedHeight(72)
    layout.addWidget(page.map_panel.timeline_bar)
    page.comparison_summary = LiveComparisonSummary()
    page.comparison_summary.setFixedHeight(92)
    layout.addWidget(page.comparison_summary)
    page.comparison_caption = QLabel(
        "Live demonstration · matched aggregate metrics from this run", page
    )
    page.comparison_caption.hide()
    page.comparison_table = LiveComparisonTable(page)
    page.comparison_table.hide()
    page.power_trace = PowerTrace(page)
    page.power_trace.hide()
    page.controls.state_changed.connect(lambda state: page.description.setVisible(state == "ready"))
    page.controls.state_changed.connect(lambda state: page.state_label.setVisible(state != "ready"))


_LIVE_STYLE = """
QWidget#approvedLivePage { background: #F4F7F5;
    }
QLabel#livePageTitle { font-family: "Atkinson Hyperlegible";
    font-size: 36px;
    font-weight: 700;
    color: #183D40;
    }
QLabel#liveDescription, QLabel#liveRunState, QLabel#liveScopeNote { font-size: 14px;
    color: #536A67;
    background: transparent;
    border: none;
    padding: 0;
    }
QFrame#liveScenarioControls { background: transparent;
    border: none;
    }
QWidget#liveControllerSelector { background: #E4EEEA;
    border-radius: 15px;
    }
QPushButton#liveControllerButton { background: transparent;
    border: 2px solid transparent;
    border-radius: 11px;
    padding: 3px 12px;
    color: #536A67;
    font-size: 15px;
    }
QPushButton#liveControllerButton:checked { background: white;
    color: #126F68;
    font-weight: 700;
    }
QPushButton#liveControllerButton:disabled { color: #647571;
    }
QLabel#liveMapCredit, QLabel#liveSmallCaption { color: #536A67;
    font-size: 12px;
    }
QPushButton#liveMapButton { background: white;
    border: 2px solid transparent;
    border-radius: 12px;
    color: #183D40;
    padding: 0;
    font-size: 22px;
    }
QFrame#liveLocalityCard, QFrame#liveTimeline { background: white;
    border: none;
    border-radius: 22px;
    }
QComboBox#liveLocalitySelector { font-family: "Atkinson Hyperlegible";
    font-size: 27px;
    font-weight: 700;
    border: 2px solid transparent;
    background: white;
    padding: 0 24px 0 0;
    color: #183D40;
    min-height: 40px;
    max-height: 40px;
    }
QLabel#liveServiceTier { color: #126F68;
    font-size: 14px;
    padding: 0;
    }
QLabel#livePowerValue { font-family: "Atkinson Hyperlegible";
    font-size: 25px;
    font-weight: 400;
    color: #183D40;
    }
QLabel#liveSectorLabel { color: #183D40;
    font-size: 14px;
    }
QProgressBar#liveSectorBar { background: #E5EEEA;
    border: none;
    border-radius: 2px;
    min-height: 0;
    }
QProgressBar#liveSectorBar::chunk { background: #559D89;
    border-radius: 2px;
    }
QWidget#liveSectorSurface { background: transparent;
    }
QLabel#liveLocalityArrow { font-size: 18px;
    color: #183D40;
    }
QFrame#liveTimeline QSlider { background: transparent;
    }
QLabel#liveProjection { color: #536A67;
    font-size: 12px;
    }
QPushButton#livePlayButton { background: #E9F2ED;
    color: #126F68;
    border: 2px solid transparent;
    border-radius: 22px;
    padding: 0;
    font-size: 21px;
    }
QPushButton#liveControllerButton:focus, QPushButton#liveMapButton:focus,
QPushButton#livePlayButton:focus, QComboBox#liveLocalitySelector:focus {
    border-color: #005FCC;
    }
QLabel#liveHourValue { font-family: "Atkinson Hyperlegible";
    font-size: 27px;
    color: #183D40;
    }
QLabel#liveTimelinePower { font-size: 13px;
    color: #183D40;
    }
QWidget#liveComparisonSummary { background: transparent;
    }
QLabel#liveComparisonTitle { font-family: "Atkinson Hyperlegible";
    font-size: 19px;
    font-weight: 700;
    color: #183D40;
    }
QLabel#liveMetricCaption { font-size: 13px;
    color: #536A67;
    }
QLabel#liveMetricValue { font-family: "Atkinson Hyperlegible";
    font-size: 27px;
    color: #126F68;
    }
QLabel#liveMetricBaseline { font-size: 14px;
    color: #536A67;
    }
"""
