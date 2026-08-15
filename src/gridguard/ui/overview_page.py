"""Overview page for the presentation shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gridguard.ui.components import CardFrame, ControllerCard, page_intro
from gridguard.ui.map_asset import MapAsset
from gridguard.ui.map_view import SchematicMapView
from gridguard.ui.presentation_model import PresentationSnapshot


class OverviewPage(QWidget):
    """At-a-glance system scope and verified comparison evidence."""

    def __init__(self, snapshot: PresentationSnapshot, asset: MapAsset, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Overview page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        title, description = page_intro(
            "Power allocation at a glance",
            "A proposal-aligned, offline decision-support simulation for seven "
            "Khartoum State localities. The schematic is contextual, not geographic.",
        )
        layout.addWidget(title)
        layout.addWidget(description)

        body = QHBoxLayout()
        body.setSpacing(12)
        map_card = CardFrame(accessible_name="Khartoum locality map overview")
        map_header = QHBoxLayout()
        map_title = QLabel(asset.title)
        map_title.setProperty("heading", "section")
        reset_button = QPushButton("Reset map")
        reset_button.setAccessibleName("Reset overview map zoom and pan")
        map_header.addWidget(map_title)
        map_header.addStretch(1)
        map_header.addWidget(reset_button)
        map_card.content.addLayout(map_header)
        self.map_view = SchematicMapView(asset)
        reset_button.clicked.connect(self.map_view.reset_view)
        map_card.content.addWidget(self.map_view, 1)
        disclaimer = QLabel(asset.disclaimer)
        disclaimer.setProperty("muted", True)
        disclaimer.setWordWrap(True)
        map_card.content.addWidget(disclaimer)
        body.addWidget(map_card, 7)

        summaries = QVBoxLayout()
        evidence = CardFrame(accessible_name="Evidence protocol summary")
        verified = QLabel("● PAIRED EVIDENCE LOADED")
        verified.setProperty("badge", "verified")
        verified.setAlignment(Qt.AlignmentFlag.AlignCenter)
        protocol = QLabel(snapshot.count_statement)
        protocol.setWordWrap(True)
        protocol.setProperty("muted", True)
        evidence.content.addWidget(verified)
        evidence.content.addWidget(protocol)
        summaries.addWidget(evidence)
        summaries.addWidget(ControllerCard(snapshot, "PPO"))
        summaries.addWidget(ControllerCard(snapshot, "Fixed Rotation"))
        summaries.addStretch(1)
        body.addLayout(summaries, 5)
        layout.addLayout(body, 1)
