"""
Multi-format document parser.
Supports: PDF, DOCX, CSV, XLSX, TXT/MD, and web URLs.
Returns a list of {page_number, text, metadata} dicts.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any


def parse_file(file_path: str, filename: str) -> list[dict[str, Any]]:
    """
    Dispatch to the correct parser based on file extension.
    Returns list of page dicts: {page_number, text, metadata}.
    """
    ext = Path(filename).suffix.lower()
    path = Path(file_path)

    if ext == ".pdf":
        return _parse_pdf(path)
    elif ext == ".docx":
        return _parse_docx(path)
    elif ext in (".csv",):
        return _parse_csv(path)
    elif ext in (".xlsx", ".xls"):
        return _parse_excel(path)
    elif ext in (".txt", ".md", ".rst"):
        return _parse_text(path)
    else:
        # Fallback: try as plain text
        return _parse_text(path)


def parse_url(url: str) -> list[dict[str, Any]]:
    """Extract article text from a web URL."""
    return _parse_url(url)


# ── Internal parsers ──────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    # Primary: pdfplumber (handles tables and complex layouts)
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                # Also extract tables as CSV-like text
                for table in (page.extract_tables() or []):
                    rows = ["\t".join(str(c or "") for c in row) for row in table if row]
                    text += "\n" + "\n".join(rows)
                text = text.strip()
                if len(text) >= 30:
                    pages.append({
                        "page_number": num,
                        "text": text,
                        "metadata": {"parser": "pdfplumber"},
                    })
        if pages:
            return pages
    except ImportError:
        pass
    except Exception as exc:
        print(f"[parser] pdfplumber failed: {exc}")

    # Fallback: PyPDF2
    try:
        import PyPDF2
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if len(text) >= 30:
                    pages.append({
                        "page_number": num,
                        "text": text,
                        "metadata": {"parser": "PyPDF2"},
                    })
    except Exception as exc:
        print(f"[parser] PyPDF2 failed: {exc}")

    return pages


def _parse_docx(path: Path) -> list[dict[str, Any]]:
    try:
        from docx import Document
        doc = Document(str(path))
        pages: list[dict[str, Any]] = []
        current_section: list[str] = []
        section_num = 1

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Treat Heading styles as section breaks
            if para.style.name.startswith("Heading") and current_section:
                pages.append({
                    "page_number": section_num,
                    "text": "\n".join(current_section),
                    "metadata": {"parser": "python-docx", "type": "section"},
                })
                section_num += 1
                current_section = [text]
            else:
                current_section.append(text)

        # Also extract tables
        for table in doc.tables:
            rows = [
                "\t".join(cell.text.strip() for cell in row.cells)
                for row in table.rows
            ]
            current_section.append("\n".join(rows))

        if current_section:
            pages.append({
                "page_number": section_num,
                "text": "\n".join(current_section),
                "metadata": {"parser": "python-docx", "type": "section"},
            })

        return pages
    except ImportError:
        print("[parser] python-docx not installed")
        return []
    except Exception as exc:
        print(f"[parser] DOCX error: {exc}")
        return []


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            rows = list(reader)

        if not rows:
            return []

        headers = rows[0]
        total   = len(rows) - 1  # exclude header

        # Schema summary as page 1
        summary = f"CSV File Summary\nColumns ({len(headers)}): {', '.join(headers)}\nTotal rows: {total}"
        pages: list[dict[str, Any]] = [{
            "page_number": 1,
            "text": summary,
            "metadata": {"parser": "csv", "type": "summary", "columns": headers},
        }]

        # Chunk data rows into groups of 50
        chunk_size = 50
        for i, start in enumerate(range(1, len(rows), chunk_size), start=2):
            chunk = rows[start: start + chunk_size]
            lines = ["\t".join(headers)]
            lines += ["\t".join(str(c) for c in row) for row in chunk]
            pages.append({
                "page_number": i,
                "text": "\n".join(lines),
                "metadata": {
                    "parser": "csv",
                    "type": "data",
                    "row_start": start,
                    "row_end": min(start + chunk_size - 1, len(rows) - 1),
                },
            })

        return pages
    except Exception as exc:
        print(f"[parser] CSV error: {exc}")
        return []


def _parse_excel(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
        xl = pd.ExcelFile(str(path))
        pages: list[dict[str, Any]] = []
        page_num = 1

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            # Schema summary
            summary = (
                f"Sheet: {sheet_name}\n"
                f"Columns ({len(df.columns)}): {', '.join(str(c) for c in df.columns)}\n"
                f"Rows: {len(df)}"
            )
            pages.append({
                "page_number": page_num,
                "text": summary,
                "metadata": {"parser": "pandas", "sheet": sheet_name, "type": "summary"},
            })
            page_num += 1

            # Data chunks
            chunk_size = 50
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start: start + chunk_size]
                pages.append({
                    "page_number": page_num,
                    "text": chunk.to_string(index=False),
                    "metadata": {
                        "parser": "pandas", "sheet": sheet_name, "type": "data",
                        "row_start": start, "row_end": min(start + chunk_size - 1, len(df) - 1),
                    },
                })
                page_num += 1

        return pages
    except ImportError:
        print("[parser] pandas/openpyxl not installed")
        return []
    except Exception as exc:
        print(f"[parser] Excel error: {exc}")
        return []


def _parse_text(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Split on double newlines (paragraphs / sections)
        sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]

        pages: list[dict[str, Any]] = []
        current: list[str] = []
        page_num = 1

        for section in sections:
            current.append(section)
            # Group ~800 chars per "page"
            if sum(len(s) for s in current) >= 800:
                pages.append({
                    "page_number": page_num,
                    "text": "\n\n".join(current),
                    "metadata": {"parser": "text"},
                })
                page_num += 1
                current = []

        if current:
            pages.append({
                "page_number": page_num,
                "text": "\n\n".join(current),
                "metadata": {"parser": "text"},
            })

        return pages
    except Exception as exc:
        print(f"[parser] Text error: {exc}")
        return []


def _parse_url(url: str) -> list[dict[str, Any]]:
    # Primary: trafilatura (best article extraction)
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_tables=True, include_links=False)
            if text and len(text.strip()) > 100:
                # Split into page-sized chunks
                chunks = _split_into_chunks(text, chunk_size=1200)
                return [
                    {
                        "page_number": i + 1,
                        "text": chunk,
                        "metadata": {"parser": "trafilatura", "url": url},
                    }
                    for i, chunk in enumerate(chunks)
                ]
    except ImportError:
        pass
    except Exception as exc:
        print(f"[parser] trafilatura failed: {exc}")

    # Fallback: requests + BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        chunks = _split_into_chunks(text, chunk_size=1200)
        return [
            {
                "page_number": i + 1,
                "text": chunk,
                "metadata": {"parser": "beautifulsoup", "url": url},
            }
            for i, chunk in enumerate(chunks)
        ]
    except Exception as exc:
        print(f"[parser] BeautifulSoup failed: {exc}")

    return []


def _split_into_chunks(text: str, chunk_size: int = 1200) -> list[str]:
    """Split text into roughly equal chunks at paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text[:chunk_size]]
