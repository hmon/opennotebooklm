"""Structure-aware parsing and chunking. Pure functions: no DB, no models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from app.config import CHUNK_OVERLAP, CHUNK_TOKENS


@dataclass
class Page:
    number: int | None
    text: str


@dataclass
class Chunk:
    page: int | None
    section: str | None
    ord: int
    start_char: int
    end_char: int
    text: str


# --- parsing -------------------------------------------------------------

class _HTMLToText(HTMLParser):
    """Flatten HTML to text, turning <h1>-<h6> into markdown headings so the
    chunker's section detection works on HTML the same way it does on Markdown."""

    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self._heading = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = int(tag[1])
            self.parts.append("\n\n" + "#" * self._heading + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = 0
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def parse(data: bytes, source_type: str) -> list[Page]:
    """Return pages of plain text. Non-paginated formats yield a single page."""
    if source_type == "pdf":
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return [Page(i + 1, page.get_text()) for i, page in enumerate(doc)]
    text = data.decode("utf-8", errors="replace")
    if source_type == "html":
        parser = _HTMLToText()
        parser.feed(text)
        text = parser.text()
    return [Page(None, text)]


def guess_title(pages: list[Page], file_name: str) -> str:
    """Prefer the document's own title; fall back to the file name.

    The title travels with every citation and is shown to the verifier, so a
    real title ("The Alpine Trial") grounds a claim that names the study in a
    way that "alpine-trial.md" cannot.
    """
    if not pages:
        return file_name
    lines = [line.strip() for line in pages[0].text.splitlines()[:40]]
    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            return heading.group(2).strip()
    # No markdown heading (a PDF, say): a short opening line that is not a
    # sentence is almost always the title.
    for line in lines:
        if line and len(line) <= 120 and not line.endswith((".", ":", ";", ",")):
            return line
    return file_name


def guess_source_type(file_name: str) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    return {
        "pdf": "pdf",
        "md": "markdown",
        "markdown": "markdown",
        "txt": "text",
        "text": "text",
        "html": "html",
        "htm": "html",
    }.get(ext, "text")


# --- chunking ------------------------------------------------------------

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
# A short standalone line like "3.2 Results" also reads as a section header.
_NUMBERED_HEADING = re.compile(r"^\s*((?:\d+\.)*\d+\.?)\s+(\S.{0,70})$")


def _blocks(text: str) -> list[tuple[int, str]]:
    """Split into (start_char, block) on blank lines, keeping exact offsets."""
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"[^\n]+(?:\n(?![ \t]*\n)[^\n]+)*", text):
        block = m.group(0)
        lead = len(block) - len(block.lstrip())
        stripped = block.strip()
        if stripped:
            out.append((m.start() + lead, stripped))
    return out


def _lines(block: str, offset: int) -> list[tuple[int, str]]:
    """Split one block into (start_char, line) pairs, offsets absolute."""
    out = []
    for m in re.finditer(r"[^\n]+", block):
        stripped = m.group(0).strip()
        if stripped:
            lead = len(m.group(0)) - len(m.group(0).lstrip())
            out.append((offset + m.start() + lead, stripped))
    return out


def _split_oversized(
    blocks: list[tuple[int, str]], max_tokens: int
) -> list[tuple[int, str]]:
    """Break blocks that are too large to chunk on their own.

    Extracted PDF text separates lines with single newlines and almost never
    with blank ones, so a whole page arrives as one block. Without this, every
    page would become a single oversized chunk and headings inside it would
    never be seen.
    """
    out: list[tuple[int, str]] = []
    for start, block in blocks:
        if _ntokens(block) <= max_tokens or "\n" not in block:
            out.append((start, block))
        else:
            out.extend(_lines(block, start))
    return out


def _is_heading(block: str) -> str | None:
    if "\n" in block:
        return None
    m = _HEADING.match(block)
    if m:
        return m.group(2).strip()
    m = _NUMBERED_HEADING.match(block)
    if m and len(block) <= 80:
        return block.strip()
    return None


def _ntokens(text: str) -> int:
    return len(text.split())


def chunk_page(
    page: Page,
    start_ord: int = 0,
    max_tokens: int = CHUNK_TOKENS,
    overlap: float = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Pack paragraphs into chunks, never crossing a page or a heading boundary.

    Offsets are into `page.text`, so a citation can be highlighted in the original.
    """
    chunks: list[Chunk] = []
    section: str | None = None
    buf: list[tuple[int, str]] = []  # (start_char, block)
    ordinal = start_ord

    def flush() -> None:
        nonlocal buf, ordinal
        if not buf:
            return
        start = buf[0][0]
        end = buf[-1][0] + len(buf[-1][1])
        chunks.append(
            Chunk(
                page=page.number,
                section=section,
                ord=ordinal,
                start_char=start,
                end_char=end,
                text=page.text[start:end].strip(),
            )
        )
        ordinal += 1
        # carry trailing blocks as overlap for the next chunk
        if overlap > 0 and len(buf) > 1:
            budget = max_tokens * overlap
            carried: list[tuple[int, str]] = []
            total = 0
            for item in reversed(buf[1:]):
                total += _ntokens(item[1])
                if total > budget and carried:
                    break
                carried.insert(0, item)
                if total > budget:
                    break
            buf = carried
        else:
            buf = []

    for start, block in _split_oversized(_blocks(page.text), max_tokens):
        heading = _is_heading(block)
        if heading is not None:
            flush()
            buf = []
            section = heading
            continue
        if buf and _ntokens(" ".join(b for _, b in buf)) + _ntokens(block) > max_tokens:
            flush()
        buf.append((start, block))
    flush()
    return [c for c in chunks if c.text]


def chunk_document(pages: list[Page], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        out.extend(chunk_page(page, start_ord=0, **kw))
    return out
