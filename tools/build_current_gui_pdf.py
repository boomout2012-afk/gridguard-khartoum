"""Publish the captured app pages with the shared academic cover."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/current-app-screenshots"
OUTPUT = ROOT / "output/pdf/gridguard-khartoum-gui-pages.pdf"
SIZE = (1366, 768)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cover(pdf: canvas.Canvas, metadata: dict, count: int) -> None:
    """Use the original institutional logos and the shared cover wording."""
    ink = HexColor("#183D40")
    teal = HexColor("#117E78")
    muted = HexColor("#536A67")
    fonts = ROOT / "src/gridguard/assets/fonts"
    pdfmetrics.registerFont(TTFont("GuiBody", str(fonts / "AtkinsonHyperlegible-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("GuiBold", str(fonts / "AtkinsonHyperlegible-Bold.ttf")))
    logo = ROOT / metadata["logos"]
    with Image.open(logo) as image:
        width, height = image.size
    logo_width = 570
    logo_height = logo_width * height / width
    pdf.drawImage(str(logo), (SIZE[0] - logo_width) / 2, 610, logo_width, logo_height)

    def line(text: str, y: float, size: float, font: str = "GuiBody", color=ink) -> None:
        pdf.setFillColor(color)
        pdf.setFont(font, size)
        pdf.drawCentredString(SIZE[0] / 2, y, text)

    line(metadata["university"], 590, 24, "GuiBold")
    line(metadata["faculty"], 560, 20)
    line(metadata["project_name"], 512, 17, "GuiBold", teal)
    title = Paragraph(
        metadata["project_title"],
        ParagraphStyle(
            "GuiCoverTitle",
            fontName="GuiBold",
            fontSize=38,
            leading=44,
            alignment=TA_CENTER,
            textColor=ink,
        ),
    )
    _, height = title.wrap(1130, 145)
    title.drawOn(pdf, (SIZE[0] - 1130) / 2, 480 - height)
    line("Graduation project - " + metadata["degree"], 335, 20)
    for index, author in enumerate(metadata["authors"]):
        line(f"{author['name']} ({author['index']})", 285 - index * 29, 19)
    line("Supervisor: " + metadata["supervisor"], 170, 19)
    line(metadata["submission_date"], 130, 18, color=muted)
    line(f"Application interface - {count} captured screens", 68, 18, "GuiBold", teal)


def main() -> None:
    manifest_path = ASSETS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = ROOT / "docs/academic-cover.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pages = manifest["pages"]
    for item in pages:
        path = ASSETS / item["filename"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Capture changed: {path}")
        with Image.open(path) as image:
            if image.size != SIZE:
                raise ValueError(f"Unexpected screenshot dimensions: {path}")
    build = ROOT / "tmp/gui-document-refresh"
    build.mkdir(parents=True, exist_ok=True)
    staged = build / "gui-pages.pdf"
    pdf = canvas.Canvas(str(staged), pagesize=SIZE, pageCompression=1)
    pdf.setTitle(metadata["project_name"] + " - Application Interface")
    pdf.setAuthor("; ".join(author["name"] for author in metadata["authors"]))
    pdf.setSubject("All application pages and detail tabs with the deployed trained PPO")
    pdf.bookmarkPage("cover")
    pdf.addOutlineEntry("Cover", "cover")
    cover(pdf, metadata, len(pages))
    pdf.showPage()
    for index, item in enumerate(pages, 1):
        bookmark = f"screen-{index}"
        pdf.bookmarkPage(bookmark)
        pdf.addOutlineEntry(item["title"], bookmark)
        pdf.drawImage(str(ASSETS / item["filename"]), 0, 0, width=SIZE[0], height=SIZE[1])
        pdf.showPage()
    pdf.save()
    document = PdfReader(staged)
    assert len(document.pages) == len(pages) + 1
    assert len(document.outline) == len(pages) + 1
    assert all(len(page.images) == 1 for page in document.pages)
    previous = OUTPUT.parent / ".previous"
    previous.mkdir(exist_ok=True)
    if OUTPUT.exists():
        backup = previous / f"gridguard-khartoum-gui-pages-{digest(OUTPUT)[:12]}.pdf"
        if not backup.exists():
            shutil.copy2(OUTPUT, backup)
    staged.replace(OUTPUT)
    receipt = {
        "schema": "gridguard.current_gui_pdf.1",
        "pdf": {
            "path": OUTPUT.relative_to(ROOT).as_posix(),
            "sha256": digest(OUTPUT),
            "pages": len(document.pages),
            "captured_screens": len(pages),
        },
        "cover_metadata_sha256": digest(metadata_path),
        "logos_sha256": digest(ROOT / metadata["logos"]),
        "capture_manifest_sha256": digest(manifest_path),
        "screenshots_sha256": {item["filename"]: item["sha256"] for item in pages},
        "builder_sha256": digest(Path(__file__)),
    }
    OUTPUT.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["pdf"]))


if __name__ == "__main__":
    main()
