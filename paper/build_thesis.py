"""Compile the thesis chapters (docs/thesis/*.md) into one Word draft.

    python paper/build_thesis.py      -> docs/thesis/khmer_asr_thesis_draft.docx

The chapters are written in a small Markdown subset: headings, paragraphs, bullet and numbered
lists, pipe tables, italic captions ("*Table 3.1. ...*"), images ("![caption](path)"), block quotes,
**bold**, *italic*, `code` and [links](url). The reference list of Chapter 2 becomes the thesis's
reference list; working notes ("To verify before submission") are left out. Layout is a neutral A4
layout until the university template is available. Open the file in Word and update the table of
contents (or let render_thesis.ps1 do it).
"""
import re
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent))
from build_paper import AFFILIATION, set_cell_shading, set_fonts

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "docs" / "thesis"
OUT = THESIS / "khmer_asr_thesis_draft.docx"
TITLE = "An Experimental Study of Fine-Tuning Strategies for Pretrained Models in Khmer ASR"
CHAPTERS = ["ch1_introduction.md", "ch2_literature_review.md", "ch3_data_and_methodology.md",
            "ch4_implementation.md", "ch5_results.md", "ch6_conclusion.md"]
SKIP_SECTIONS = {"To verify before submission"}
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_inline(par, text: str, size: float | None = None, italic: bool = False) -> None:
    """Add text with **bold**, *italic*, `code` and [link](url) (link text only) to a paragraph."""
    for part in INLINE.split(text):
        if not part:
            continue
        bold = ital = code = False
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, code = part[1:-1], True
        elif part.startswith("[") and "](" in part:
            part = part[1:].split("](", 1)[0]  # keep the link text; the URL is in the source
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, ital = part[1:-1], True
        run = par.add_run(part)
        run.bold = bold or None
        run.italic = (ital or italic) or None
        if code:
            run.font.name = "Consolas"
        if size:
            run.font.size = Pt(size)


def add_field(par, instr: str) -> None:
    """A Word field (PAGE, TOC, ...) that Word fills in when fields are updated."""
    run = par.add_run()
    for tag, text in (("begin", None), ("instr", instr), ("separate", None), ("end", None)):
        if tag == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        run._r.append(el)
        if tag == "separate":
            run._r.append(OxmlElement("w:t"))
            run._r[-1].text = "Right-click and choose Update Field." if "TOC" in instr else ""


def add_table(doc, lines: list[str]) -> None:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    header, body = rows[0], [r for r in rows[2:]]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = table.rows[0].cells[i]
        add_inline(cell.paragraphs[0], f"**{text}**" if text else "", size=9.5)
        set_cell_shading(cell, "E8EEF7")
    for row in body:
        cells = table.add_row().cells
        for i in range(len(header)):
            add_inline(cells[i].paragraphs[0], row[i] if i < len(row) else "", size=9.5)
    last = len(table.rows) - 1
    for n, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for cell in row.cells:
            for p in cell.paragraphs:
                fmt = p.paragraph_format
                fmt.space_after = Pt(0)
                fmt.line_spacing = 1.0
                fmt.keep_with_next = n < last
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_caption(doc, text: str) -> None:
    p = doc.add_paragraph(style="Caption")
    add_inline(p, text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True


def add_markdown(doc, md: str, chapter_heading: bool = True, drop_preamble: bool = True) -> list[str]:
    """Render one chapter; returns the reference lines found under '## References'."""
    lines = md.splitlines()
    # Drop the draft preamble: everything between the title and the first '---' line.
    if drop_preamble and lines and lines[0].startswith("# ") and "---" in lines:
        lines = [lines[0], *lines[lines.index("---") + 1:]]
    refs: list[str] = []
    mode, skip_level = None, None
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        m = re.match(r"^(#{1,4}) (.*)", ln)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if skip_level and level > skip_level:
                i += 1
                continue
            skip_level = None
            mode = None
            if text == "References":
                mode = "refs"
            elif text in SKIP_SECTIONS:
                skip_level = level
            elif level == 1:
                if chapter_heading:
                    doc.add_heading(text.replace(" — ", ": "), level=1)
            else:
                doc.add_heading(text, level=level)
            i += 1
            continue
        if skip_level:
            i += 1
            continue
        if mode == "refs":
            r = re.match(r"^(\d+)\. (.*)", stripped)
            if r:
                refs.append(f"[{r.group(1)}] {r.group(2)}")
            elif stripped and refs:
                refs[-1] += " " + stripped
            i += 1
            continue
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        img = re.match(r"^!\[(.*)\]\((.*)\)$", stripped)
        if img:
            path = (THESIS / img.group(2)).resolve()
            doc.add_picture(str(path), width=Cm(15))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.paragraphs[-1].paragraph_format.keep_with_next = True
            add_caption(doc, img.group(1))
            i += 1
            continue
        if re.match(r"^\*(Table|Figure) [\d.]+\.", stripped):
            text = stripped
            i += 1
            while not text.endswith("*") and i < len(lines) and lines[i].strip():
                text += " " + lines[i].strip()
                i += 1
            add_caption(doc, text.strip("*"))
            continue
        item = re.match(r"^(\s*)(- \[ \] |- |\d+\. )(.*)", ln)
        if item:
            indent = len(item.group(1))
            text = item.group(3)
            i += 1
            while i < len(lines) and lines[i].startswith(" " * (indent + 2)) and lines[i].strip() \
                    and not re.match(r"^\s*(- |\d+\. )", lines[i]):
                text += " " + lines[i].strip()
                i += 1
            style = "List Number" if item.group(2)[0].isdigit() else "List Bullet"
            if indent >= 2:
                style += " 2"
            p = doc.add_paragraph(style=style)
            add_inline(p, ("☐ " if "[ ]" in item.group(2) else "") + text)
            continue
        quote = stripped.startswith(">")
        text = stripped.lstrip("> ")
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4} |\||!\[|\s*- |\s*\d+\. |---)", lines[i]):
            text += " " + lines[i].strip().lstrip("> ")
            i += 1
        p = doc.add_paragraph()
        add_inline(p, text, italic=quote)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if quote:
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xB0, 0x3A, 0x2E)
    return refs


def setup(doc) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    set_fonts(normal)
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12), ("Title", 20),
                       ("Caption", 10)):
        st = doc.styles[name]
        set_fonts(st)
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
    doc.styles["Heading 1"].paragraph_format.page_break_before = True
    for name in ("List Bullet", "List Number", "List Bullet 2", "List Number 2"):
        doc.styles[name].paragraph_format.line_spacing = 1.5


def page_numbers(section, fmt: str) -> None:
    """Centred page number in the footer, restarting at 1 in format `fmt` (decimal/lowerRoman)."""
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(p, "PAGE")
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), "1")
    cols = section._sectPr.find(qn("w:cols"))  # schema order: pgNumType comes before cols
    if cols is not None:
        cols.addprevious(pg)
    else:
        section._sectPr.append(pg)


def build() -> Path:
    doc = Document()
    setup(doc)

    # Title page
    for text, size, bold, space in (
        ("DRAFT FOR ADVISOR REVIEW — " + f"{date.today():%d %B %Y}", 9, False, 60),
        (TITLE, 20, True, 36), ("Master's Thesis", 14, False, 48), ("Sem Chitra", 14, True, 6),
        ("Advisor: Dr. Mao Makara", 12, False, 48), (AFFILIATION, 12, False, 6),
        (f"{date.today():%B %Y}", 12, False, 0),
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(space)
        r = p.add_run(text)
        r.font.size, r.bold = Pt(size), bold
        if text.startswith("DRAFT"):
            r.font.color.rgb = RGBColor(0xB0, 0x3A, 0x2E)
    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = note.add_run("Results are provisional until the test set and normalization protocol are "
                     "approved and locked. Layout is a placeholder until the university template "
                     "is available.")
    r.italic, r.font.size = True, Pt(9)

    # Front matter: abstracts (English and Khmer draft) and contents, roman page numbers.
    doc.add_section(WD_SECTION.NEW_PAGE)
    add_markdown(doc, (THESIS / "abstract.md").read_text(encoding="utf-8"), drop_preamble=False)
    doc.add_heading("Contents", level=1)
    add_field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u')

    # Chapters, arabic page numbers. Page numbering is set once all sections exist: python-docx
    # returns the document's last section from add_section, which later becomes the chapters.
    doc.add_section(WD_SECTION.NEW_PAGE)
    refs: list[str] = []
    first = True
    for name in CHAPTERS:
        refs += add_markdown(doc, (THESIS / name).read_text(encoding="utf-8"))
        if first:
            # The section break already starts a page; avoid an empty page before Chapter 1.
            heading = next(p for p in doc.paragraphs if p.text.startswith("Chapter 1"))
            heading.paragraph_format.page_break_before = False
            first = False

    doc.add_heading("References", level=1)
    for ref in refs:
        p = doc.add_paragraph()
        add_inline(p, ref, size=10.5)
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
        p.paragraph_format.line_spacing = 1.15
    doc.paragraphs[-1].add_run().add_break(WD_BREAK.LINE)
    _title_page, front, body = doc.sections  # title page keeps no page number
    page_numbers(front, "lowerRoman")
    page_numbers(body, "decimal")
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
