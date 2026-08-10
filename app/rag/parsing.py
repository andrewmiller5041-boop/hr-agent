"""Parsers for the two supported corpus source formats: Markdown and HTML.

Each parser returns a dict: {"doc_id": str, "title": str, "sections": [(heading, text), ...]}
so the chunker can work identically regardless of source format.
"""
import re
from pathlib import Path

from bs4 import BeautifulSoup

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str):
    """Very small YAML-subset parser for our two-key frontmatter block."""
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_block, body = m.group(1), m.group(2)
    meta = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body


def parse_markdown(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    doc_id = meta.get("doc_id", path.stem)
    title = meta.get("title", path.stem)

    sections = []
    current_heading = title
    current_lines = []
    for line in body.splitlines():
        heading_match = re.match(r"^(#{1,3})\s+(.*)$", line.strip())
        if heading_match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = heading_match.group(2).strip()
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    sections = [(h, t) for h, t in sections if t.strip()]
    return {"doc_id": doc_id, "title": title, "sections": sections, "source_format": "markdown"}


def parse_html(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    meta_tag = soup.find("meta", attrs={"name": "doc_id"})
    doc_id = meta_tag["content"] if meta_tag else path.stem
    title_tag = soup.find("title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else path.stem

    sections = []
    current_heading = title
    current_parts = []
    body = soup.find("body") or soup
    for el in body.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ("h1", "h2", "h3"):
            if current_parts:
                sections.append((current_heading, "\n".join(current_parts).strip()))
                current_parts = []
            current_heading = el.get_text(strip=True)
        else:
            text = el.get_text(strip=True)
            if text:
                current_parts.append(text)
    if current_parts:
        sections.append((current_heading, "\n".join(current_parts).strip()))

    sections = [(h, t) for h, t in sections if t.strip()]
    return {"doc_id": doc_id, "title": title, "sections": sections, "source_format": "html"}


def parse_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return parse_markdown(path)
    if suffix in (".html", ".htm"):
        return parse_html(path)
    raise ValueError(f"Unsupported corpus file type: {path}")
