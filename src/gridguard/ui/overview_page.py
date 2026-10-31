"""Approved map-first Overview for the offline native presentation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gridguard.domain import HORIZON_HOURS, SECTORS
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_view import LocalityMapView
from gridguard.ui.presentation_model import PresentationSnapshot
from gridguard.ui.theme import BODY_FONT_FAMILY, PALETTE


class OverviewPage(QWidget):
    """Geographic introduction and scope, without prepopulated scientific results."""

    def __init__(self, snapshot: PresentationSnapshot, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("approvedOverview")
        self.setAccessibleName("Overview page")
        self.setAccessibleDescription(
            "GridGuard Khartoum: seven localities, a 24-hour scenario and five service sectors. "
            "Explore a live scenario or open the verified comparison in Results."
        )
        self.setStyleSheet(_OVERVIEW_STYLE)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(16, 32, 16, 24)
        self._layout.setHorizontalSpacing(32)
        self._layout.setVerticalSpacing(24)
        self._layout.setRowStretch(0, 1)
        self._layout.setColumnStretch(1, 1)

        self._hero = QWidget()
        self._hero.setFixedWidth(460)
        self._hero_layout = QVBoxLayout(self._hero)
        self._hero_layout.setContentsMargins(0, 22, 0, 0)
        self._hero_layout.setSpacing(0)
        self.title_label = QLabel(f'Khartoum<span style="color:{PALETTE.selection}">.</span>')
        self.title_label.setTextFormat(Qt.TextFormat.RichText)
        self.title_label.setAccessibleName("Khartoum")
        self.title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._hero_layout.addWidget(self.title_label)
        self._hero_layout.addSpacing(25)
        self.lead_label = QLabel("Power allocation.\nSeven localities in view.")
        self.lead_label.setObjectName("overviewLead")
        self._hero_layout.addWidget(self.lead_label)
        self._hero_layout.addSpacing(22)
        description = QLabel(
            "Explore the current trained PPO, Fixed Rotation and constrained optimizer "
            "on one live scenario, or compare their frozen study results."
        )
        description.setObjectName("overviewDescription")
        description.setWordWrap(True)
        description.setMaximumWidth(365)
        self._hero_layout.addWidget(description)
        self._hero_layout.addSpacing(28)
        actions = QHBoxLayout()
        actions.setSpacing(15)
        self.explore_button = QPushButton("Explore scenario")
        self.explore_button.setProperty("primary", True)
        self.explore_button.setAccessibleName("Explore scenario in Live Scenario")
        self.explore_button.setMinimumHeight(54)
        self.results_button = QPushButton("Compare results")
        self.results_button.setProperty("quiet", True)
        self.results_button.setAccessibleName("Compare frozen controller results")
        self.results_button.setMinimumHeight(54)
        actions.addWidget(self.explore_button)
        actions.addWidget(self.results_button)
        actions.addStretch(1)
        self._hero_layout.addLayout(actions)
        self._hero_layout.addSpacing(23)
        self.scenario_caption = QLabel("Choose a shortage level and seed to begin.")
        self.scenario_caption.setObjectName("overviewScenarioCaption")
        self.scenario_caption.setAccessibleName(
            "No scenario has been run. Choose a shortage level and seed in Live Scenario."
        )
        self._hero_layout.addWidget(self.scenario_caption)
        self._hero_layout.addStretch(1)
        self._layout.addWidget(self._hero, 0, 0)

        self._map_panel = _OverviewMapPanel(asset)
        self.map_view = self._map_panel.map_view
        self.selected_locality_label = self._map_panel.selected_locality_label
        self._layout.addWidget(self._map_panel, 0, 1)

        self._facts_panel = QFrame()
        self._facts_panel.setObjectName("overviewFacts")
        self._facts_panel.setAccessibleName(
            "Scenario scope: seven localities, 24 hours, five sectors"
        )
        facts = QHBoxLayout(self._facts_panel)
        facts.setContentsMargins(27, 18, 27, 20)
        facts.setSpacing(22)
        self.scope_fact_labels: dict[str, QLabel] = {}
        for label, value in (
            ("Localities", len(asset.localities)),
            ("Hours", HORIZON_HOURS),
            ("Service sectors", len(SECTORS)),
        ):
            column = QVBoxLayout()
            column.setSpacing(3)
            number = QLabel(str(value))
            number.setObjectName("overviewScopeCount")
            number.setAccessibleName(f"{value} {label.lower()}")
            caption = QLabel(label)
            caption.setProperty("muted", True)
            self.scope_fact_labels[label] = number
            column.addWidget(number)
            column.addWidget(caption)
            facts.addLayout(column, 1)
        self._layout.addWidget(self._facts_panel, 1, 0)

        self._controllers_panel = _controller_introduction()
        self._layout.addWidget(self._controllers_panel, 1, 1)
        self.scope_note = QLabel("Synthetic scenarios · Decision support only")
        self.scope_note.setProperty("muted", True)
        self.scope_note.setAccessibleName(
            "Synthetic scenarios. Decision support only. This is not an operational dispatch tool."
        )
        self._layout.addWidget(self.scope_note, 2, 0)
        self.source_attribution = _source_attribution(asset)
        self._layout.addWidget(self.source_attribution, 2, 1)
        self._set_compact(False)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._set_compact(self.height() < 710)
        super().resizeEvent(event)

    def _set_compact(self, compact: bool) -> None:
        title_size = 88 if compact else 100
        font = QFont(BODY_FONT_FAMILY)
        font.setPixelSize(title_size)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -4.8 if compact else -5.5)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet(
            f'font-family: "{BODY_FONT_FAMILY}"; font-size: {title_size}px; font-weight: 700;'
        )
        self._hero.setFixedWidth(424 if compact else 460)
        self._hero_layout.setContentsMargins(0, 12 if compact else 22, 0, 0)
        for index, height in (
            (1, 16 if compact else 25),
            (3, 16 if compact else 22),
            (5, 22 if compact else 28),
            (7, 16 if compact else 23),
        ):
            spacer = self._hero_layout.itemAt(index).spacerItem()
            if spacer is not None:
                spacer.changeSize(0, height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.lead_label.setStyleSheet(f"font-size: {24 if compact else 27}px;")
        self._layout.setContentsMargins(16, 20 if compact else 32, 16, 20 if compact else 24)
        self._layout.setHorizontalSpacing(28 if compact else 32)
        self._layout.setVerticalSpacing(16 if compact else 24)
        self._facts_panel.setFixedHeight(110 if compact else 128)
        self._controllers_panel.setFixedHeight(110 if compact else 128)
        self._facts_panel.layout().setContentsMargins(25, 16, 25, 16)
        for number in self.scope_fact_labels.values():
            number.setStyleSheet(f"font-size: {42 if compact else 47}px; font-weight: 700;")
        self._hero_layout.invalidate()


class _OverviewMapPanel(QWidget):
    """Open map with a floating selection card, retaining native navigation."""

    def __init__(self, asset: MapAsset) -> None:
        super().__init__()
        self.setAccessibleName("Khartoum locality geography")
        self.setMinimumWidth(650)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.map_view = LocalityMapView(asset)
        self.map_view.setMinimumSize(0, 0)
        layout.addWidget(self.map_view)
        self._selection = QFrame(self)
        self._selection.setObjectName("overviewSelectedLocality")
        selection_layout = QVBoxLayout(self._selection)
        selection_layout.setContentsMargins(20, 14, 20, 16)
        selection_layout.setSpacing(5)
        heading_row = QHBoxLayout()
        heading_row.setSpacing(7)
        swatch = QFrame()
        swatch.setFixedSize(8, 8)
        swatch.setStyleSheet(f"background: {PALETTE.selection}; border-radius: 4px;")
        swatch.setAccessibleName("Coral denotes the selected locality")
        heading_row.addWidget(swatch)
        heading = QLabel("Selected locality")
        heading.setObjectName("overviewSelectionCaption")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        selection_layout.addLayout(heading_row)
        self.selected_locality_label = QLabel("East Nile")
        self.selected_locality_label.setObjectName("overviewSelectedName")
        selection_layout.addWidget(self.selected_locality_label)
        self._selection.setFixedSize(190, 92)
        self.map_view.locality_selected.connect(self._select_locality)
        self.map_view.select_locality("east_nile")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._selection.move(max(0, self.width() - self._selection.width() - 8), 8)
        self._selection.raise_()

    def _select_locality(self, locality_id: str) -> None:
        name = self.map_view.asset.locality(locality_id).name
        self.selected_locality_label.setText(name)
        self._selection.setAccessibleName(f"Selected locality: {name}. Coral marks selection.")


def _controller_introduction() -> QFrame:
    panel = QFrame()
    panel.setObjectName("overviewControllers")
    panel.setAccessibleName("Controllers compared on identical matched scenarios")
    layout = QHBoxLayout(panel)
    layout.setContentsMargins(28, 20, 28, 20)
    layout.setSpacing(25)
    for icon, name, description in (
        ("↗", "PPO", "Learned controller"),
        ("↻", "Fixed Rotation", "Baseline controller"),
    ):
        column = QVBoxLayout()
        column.setSpacing(7)
        title = QLabel(f'<span style="color:{PALETTE.accent}">{icon}</span>  {name}')
        title.setObjectName("overviewControllerName")
        title.setAccessibleName(name)
        caption = QLabel(description)
        caption.setProperty("muted", True)
        column.addWidget(title)
        column.addWidget(caption)
        layout.addLayout(column, 1)
    matched = QLabel("One scenario.\nA matched comparison.")
    matched.setProperty("muted", True)
    matched.setObjectName("overviewMatchedNote")
    layout.addWidget(matched)
    return panel


def _source_attribution(asset: MapAsset) -> QLabel:
    label = QLabel(
        f"Boundaries: OCHA COD-AB {asset.source.boundary_year}; Rivers: Natural Earth 5.0.\n"
        "Geographic context, not an electrical network."
    )
    label.setObjectName("overviewGeographyCredit")
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse
        | Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    label.setToolTip(
        f"Boundary source: {asset.source.dataset}; {asset.source.provider}; "
        f"revision {asset.source.source_revision}; {asset.source.license}; "
        f"{asset.source.source_url}\n"
        "River source: Natural Earth 1:10m rivers and lake centerlines; "
        "revision 5.0.0; public domain"
    )
    label.setAccessibleName(
        f"Map sources: OCHA COD-AB {asset.source.boundary_year}, "
        "CC BY 3.0 IGO; Natural Earth 5.0 rivers, public domain"
    )
    return label


_OVERVIEW_STYLE = f"""
QWidget#approvedOverview {{ background: {PALETTE.canvas}; }}
QFrame#overviewFacts, QFrame#overviewControllers, QFrame#overviewSelectedLocality {{
    background: {PALETTE.surface}; border: 0; border-radius: 22px;
}}
QLabel#overviewDescription {{ color: {PALETTE.muted}; font-size: 16px; }}
QLabel#overviewScenarioCaption {{ color: {PALETTE.muted}; font-size: 14px; }}
QLabel#overviewScopeCount {{ font-size: 47px; font-weight: 700; }}
QLabel#overviewControllerName {{ font-size: 21px; font-weight: 700; }}
QLabel#overviewMatchedNote {{ border-left: 1px solid {PALETTE.border}; padding-left: 20px; }}
QLabel#overviewSelectionCaption {{ font-size: 12px; color: {PALETTE.muted}; }}
QLabel#overviewSelectedName {{ font-size: 23px; font-weight: 700; }}
QLabel#overviewGeographyCredit {{ font-size: 12px; color: {PALETTE.muted}; }}
QWidget#approvedOverview QPushButton {{
    font-size: 16px; font-weight: 700; border-radius: 16px; min-height: 42px;
}}
"""
