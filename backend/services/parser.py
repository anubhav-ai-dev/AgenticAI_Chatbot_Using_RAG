"""
Multi-format document parser.
Supports: PDF, DOCX, CSV, XLSX, TXT/MD/RST and web URLs.
Returns a list of {page_number, text, metadata} dicts ("pages").
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Optional

from backend.client.web_client import WebPageClient
from backend.core.exceptions import DocumentParseError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "docx", "csv", "xlsx", "txt", "md", "rst"}
MAX_PDF_PAGES = 1000
MIN_PAGE_CHARS = 30

Page = dict[str, Any]


class DocumentParser:
    def __init__(self, web_client: Optional[WebPageClient] = None) -> None:
        self._web = web_client or WebPageClient()

    # ── public API ───────────────────────────────────────────────────────────

    def parse_file(self, file_path: str, filename: str) -> list[Page]:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in SUPPORTED_EXTENSIONS:
            raise DocumentParseError(
                f"Unsupported file type '.{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        path = Path(file_path)
        handlers = {
            "pdf": self._parse_pdf, "docx": self._parse_docx, "csv": self._parse_csv,
            "xlsx": self._parse_excel, "txt": self._parse_text, "md": self._parse_text,
            "rst": self._parse_text,
        }
        return handlers[ext](path)

    def parse_url(self, url: str) -> list[Page]:
        html = self._web.fetch_html(url)
        text, parser_name = self._extract_article(html)
        if not text or len(text.strip()) < 100:
            raise DocumentParseError("No readable article text found at that URL.")
        return [
            {"page_number": i + 1, "text": chunk,
             "metadata": {"parser": parser_name, "url": url}}
            for i, chunk in enumerate(self._split_into_chunks(text, 1200))
        ]

    @staticmethod
    def _extract_article(html: str) -> tuple[str, str]:
        try:
            import trafilatura

            text = trafilatura.extract(html, include_tables=True, include_links=False)
            if text and len(text.strip()) > 100:
                return text, "trafilatura"
        except Exception as exc:  # noqa: BLE001 — fall through to BeautifulSoup
            logger.debug("trafilatura failed: %s", exc)
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
            return text, "beautifulsoup"
        except Exception as exc:  # noqa: BLE001
            logger.warning("BeautifulSoup failed: %s", exc)
        return "", "none"

    # ── file parsers ─────────────────────────────────────────────────────────

    def _parse_pdf(self, path: Path) -> list[Page]:
        pages: list[Page] = []
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                for num, page in enumerate(pdf.pages[:MAX_PDF_PAGES], start=1):
                    text = (page.extract_text() or "").strip()   # includes table text
                    if len(text) >= MIN_PAGE_CHARS:
                        pages.append({"page_number": num, "text": text,
                                      "metadata": {"parser": "pdfplumber"}})
            if pages:
                return pages
        except Exception as exc:  # noqa: BLE001 — try the fallback parser
            logger.warning("pdfplumber failed: %s", exc)
            pages = []

        try:
            import PyPDF2

            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for num, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
                    text = (page.extract_text() or "").strip()
                    if len(text) >= MIN_PAGE_CHARS:
                        pages.append({"page_number": num, "text": text,
                                      "metadata": {"parser": "PyPDF2"}})
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyPDF2 failed: %s", exc)
            if not pages:
                raise DocumentParseError("The PDF could not be read (corrupt or encrypted).") from exc
        return pages

    def _parse_docx(self, path: Path) -> list[Page]:
        try:
            from docx import Document
        except ImportError as exc:
            raise DocumentParseError("DOCX support is not installed.") from exc
        try:
            doc = Document(str(path))
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError("The DOCX file could not be read.") from exc

        pages: list[Page] = []
        current: list[str] = []
        section = 1
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style is not None and para.style.name.startswith("Heading") and current:
                pages.append({"page_number": section, "text": "\n".join(current),
                              "metadata": {"parser": "python-docx", "type": "section"}})
                section += 1
                current = [text]
            else:
                current.append(text)
        for table in doc.tables:
            rows = ["\t".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            current.append("\n".join(rows))
        if current:
            pages.append({"page_number": section, "text": "\n".join(current),
                          "metadata": {"parser": "python-docx", "type": "section"}})
        return pages

    def _parse_csv(self, path: Path) -> list[Page]:
        try:
            with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
                rows = list(csv.reader(fh))
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError("The CSV file could not be read.") from exc
        if not rows:
            return []

        headers = rows[0]
        pages: list[Page] = [{
            "page_number": 1,
            "text": f"CSV File Summary\nColumns ({len(headers)}): {', '.join(headers)}\n"
                    f"Total rows: {len(rows) - 1}",
            "metadata": {"parser": "csv", "type": "summary", "columns": headers},
        }]
        step = 50
        for i, start in enumerate(range(1, len(rows), step), start=2):
            block = rows[start:start + step]
            lines = ["\t".join(headers)] + ["\t".join(str(c) for c in row) for row in block]
            pages.append({
                "page_number": i, "text": "\n".join(lines),
                "metadata": {"parser": "csv", "type": "data",
                             "row_start": start, "row_end": min(start + step - 1, len(rows) - 1)},
            })
        return pages

    def _parse_excel(self, path: Path) -> list[Page]:
        try:
            import pandas as pd

            xl = pd.ExcelFile(str(path))
            pages: list[Page] = []
            num = 1
            step = 50
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                pages.append({
                    "page_number": num,
                    "text": f"Sheet: {sheet}\nColumns ({len(df.columns)}): "
                            f"{', '.join(str(c) for c in df.columns)}\nRows: {len(df)}",
                    "metadata": {"parser": "pandas", "sheet": sheet, "type": "summary"},
                })
                num += 1
                for start in range(0, len(df), step):
                    block = df.iloc[start:start + step]
                    pages.append({
                        "page_number": num, "text": block.to_string(index=False),
                        "metadata": {"parser": "pandas", "sheet": sheet, "type": "data",
                                     "row_start": start, "row_end": min(start + step - 1, len(df) - 1)},
                    })
                    num += 1
            return pages
        except ImportError as exc:
            raise DocumentParseError("Excel support is not installed.") from exc
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError("The Excel file could not be read.") from exc

    def _parse_text(self, path: Path) -> list[Page]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise DocumentParseError("The text file could not be read.") from exc
        sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]

        pages: list[Page] = []
        current: list[str] = []
        num = 1
        for section in sections:
            current.append(section)
            if sum(len(s) for s in current) >= 800:
                pages.append({"page_number": num, "text": "\n\n".join(current),
                              "metadata": {"parser": "text"}})
                num += 1
                current = []
        if current:
            pages.append({"page_number": num, "text": "\n\n".join(current),
                          "metadata": {"parser": "text"}})
        return pages

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int = 1200) -> list[str]:
        """Split text into roughly equal chunks at paragraph boundaries."""
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for para in re.split(r"\n{2,}", text):
            if length + len(para) > chunk_size and current:
                chunks.append("\n\n".join(current))
                current, length = [para], len(para)
            else:
                current.append(para)
                length += len(para)
        if current:
            chunks.append("\n\n".join(current))
        return chunks or [text[:chunk_size]]
