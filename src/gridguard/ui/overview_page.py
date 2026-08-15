"""Overview page for the presentation shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gridguard.ui.components import CardFrame, ControllerCard, page_intro
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_view import LocalityMapView
from gridguard.ui.presentation_model import PresentationSnapshot


class OverviewPage(QWidget):
    """At-a-glance system scope and verified comparison evidence."""

    def __init__(self, snapshot: PresentationSnapshot, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Overview page")
        self.setAccessibleDescription(
            "Verified benchmark summary beside the seven-locality boundary map."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 18, 26, 18)
        layout.setSpacing(10)

        kicker = QLabel("OVERVIEW  /  KHARTOUM STATE")
        kicker.setProperty("kicker", True)
        kicker.setAccessibleName("Overview, Khartoum State")
        layout.addWidget(kicker)
        title, description = page_intro(
            "A service-allocation atlas",
            "Explore the seven-locality boundary context and keep the frozen PPO comparison "
            "in view. "
            "All traces are synthetic; this interface supports decisions, not grid operations.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        body = QHBoxLayout()
        body.setSpacing(12)
        map_card = CardFrame(accessible_name="Khartoum locality map overview")
        map_header = QHBoxLayout()
        map_title = QLabel(asset.title)
        map_title.setProperty("heading", "section")
        map_title.setAccessibleName(f"Map title: {asset.title}")
        boundary_badge = QLabel("7 LOCALITIES · BOUNDARY CONTEXT")
        boundary_badge.setProperty("badge", "neutral")
        boundary_badge.setProperty("compact", True)
        boundary_badge.setAccessibleName("Seven locality administrative boundary context")
        reset_button = QPushButton("Reset view")
        reset_button.setProperty("quiet", True)
        reset_button.setAccessibleName("Reset overview map zoom and pan")
        map_header.addWidget(map_title)
        map_header.addStretch(1)
        map_header.addWidget(boundary_badge)
        map_header.addWidget(reset_button)
        map_card.content.addLayout(map_header)
        self.map_view = LocalityMapView(asset)
        self.map_view.setAccessibleDescription(
            "Interactive offline boundary context for seven Khartoum State localities. "
            "Use arrow keys to change locality and plus or minus to zoom."
        )
        reset_button.clicked.connect(self.map_view.reset_view)
        map_card.content.addWidget(self.map_view, 1)
        disclaimer = QLabel(asset.disclaimer)
        disclaimer.setProperty("muted", True)
        disclaimer.setProperty("compact", True)
        disclaimer.setWordWrap(True)
        disclaimer.setAccessibleName(f"Map limitation: {asset.disclaimer}")
        map_card.content.addWidget(disclaimer)
        self.source_attribution = QLabel(
            f"Sources: OCHA COD-AB {asset.source.boundary_year} "
            f"({asset.source.license}) · Natural Earth 5.0 rivers (public domain)"
        )
        self.source_attribution.setProperty("muted", True)
        self.source_attribution.setProperty("compact", True)
        self.source_attribution.setWordWrap(True)
        self.source_attribution.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.source_attribution.setToolTip(
            f"Boundary source: {asset.source.dataset}; {asset.source.provider}; "
            f"revision {asset.source.source_revision}; {asset.source.source_url}\n"
            "River source: Natural Earth 1:10m rivers and lake centerlines; "
            "revision 5.0.0; public domain"
        )
        self.source_attribution.setAccessibleName(
            f"Map sources: OCHA COD-AB {asset.source.boundary_year}, "
            "CC BY 3.0 IGO; Natural Earth 5.0 rivers, public domain"
        )
        map_card.content.addWidget(self.source_attribution)
        body.addWidget(map_card, 7)

        summaries = QVBoxLayout()
        summaries.setSpacing(10)
        evidence = CardFrame(
            accessible_name="Evidence protocol summary",
            variant="accent",
        )
        evidence.setAccessibleDescription(
            f"Hash-verified evidence {snapshot.benchmark_id}. {snapshot.count_statement}."
        )
        evidence_header = QHBoxLayout()
        evidence_title = QLabel("Frozen comparison")
        evidence_title.setProperty("heading", "section")
        self.evidence_status = QLabel("HASH VERIFIED")
        self.evidence_status.setProperty("badge", "verified")
        self.evidence_status.setProperty("compact", True)
        self.evidence_status.setAccessibleName(
            f"Evidence hash verified for {snapshot.benchmark_id}"
        )
        evidence_header.addWidget(evidence_title)
        evidence_header.addStretch(1)
        evidence_header.addWidget(self.evidence_status)
        self.protocol_summary = QLabel(snapshot.count_statement)
        self.protocol_summary.setWordWrap(True)
        self.protocol_summary.setProperty("muted", True)
        self.protocol_summary.setAccessibleName(
            f"Verified evaluation protocol: {snapshot.count_statement}"
        )
        evidence.content.addLayout(evidence_header)
        evidence.content.addWidget(self.protocol_summary)
        evidence_id = QLabel(snapshot.benchmark_id)
        evidence_id.setProperty("compact", True)
        evidence_id.setProperty("muted", True)
        evidence_id.setProperty("mono", True)
        evidence_id.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
        evidence_id.setAccessibleName(f"Evidence identifier: {snapshot.benchmark_id}")
        evidence.content.addWidget(evidence_id)
        summaries.addWidget(evidence)
        summaries.addWidget(ControllerCard(snapshot, "PPO"))
        summaries.addWidget(ControllerCard(snapshot, "Fixed Rotation"))
        summaries.addStretch(1)

        summary_widget = QWidget()
        summary_widget.setAccessibleName("Controller comparison summary")
        summary_widget.setLayout(summaries)
        body.addWidget(summary_widget, 5)
        layout.addLayout(body, 1)
