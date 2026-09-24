"""
The PDF text extractors compared in ROUTES.md (plan: ROUTES_PLAN.md).

    python -m audit.extractors            # extract every PDF with every extractor
    python -m audit.extractors ocr        # one extractor

Each extractor returns one string per page. Outputs are cached in
cache/extractions/<name>.json ({file name: [page texts]}) so that every later
step reads exactly the same text. docling needs its own environment and is
run by audit/extract_docling.py, which writes to the same place.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / 'data_pdf'
CACHE = ROOT / 'cache' / 'extractions'


def plumber(x_tolerance):
    def run(path):
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return [pg.extract_text(x_tolerance=x_tolerance) or '' for pg in pdf.pages]
    return run


def pdfminer(path):
    from pdfminer.high_level import extract_text
    from pdfminer.pdfpage import PDFPage
    with open(path, 'rb') as fh:
        n = sum(1 for _ in PDFPage.get_pages(fh))
    return [extract_text(str(path), page_numbers=[i]) for i in range(n)]


def pymupdf(sort):
    def run(path):
        import pymupdf as fitz
        with fitz.open(path) as doc:
            return [page.get_text('text', sort=sort) for page in doc]
    return run


def pypdf(layout):
    def run(path):
        from pypdf import PdfReader
        return [p.extract_text(extraction_mode='layout' if layout else 'plain') or '' for p in PdfReader(path).pages]
    return run


def pdfium(path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(path))
    try:
        return [doc[i].get_textpage().get_text_range() for i in range(len(doc))]
    finally:
        doc.close()


def pdftotext(layout):
    def run(path):
        args = ['pdftotext', '-enc', 'UTF-8'] + (['-layout'] if layout else []) + [str(path), '-']
        text = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        pages = text.split('\f')
        return pages[:-1] if pages and not pages[-1].strip() else pages
    return run


def ocr(path):
    """Tesseract (eng+fra) on the pages rendered at 300 dpi."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['pdftoppm', '-r', '300', '-gray', '-png', str(path), f'{tmp}/p'], check=True)
        out = []
        for png in sorted(Path(tmp).glob('p-*.png'), key=lambda p: int(p.stem.split('-')[-1])):
            out.append(subprocess.run(['tesseract', str(png), 'stdout', '-l', 'eng+fra'],
                                      capture_output=True, text=True, check=True).stdout)
        return out


EXTRACTORS = {
    'pdfplumber-1.5': plumber(1.5),         # current
    'pdfplumber-1': plumber(1),
    'pdfplumber-2': plumber(2),
    'pdfplumber-3': plumber(3),             # library default
    'pdfminer': pdfminer,
    'pymupdf': pymupdf(False),
    'pymupdf-sort': pymupdf(True),
    'pypdf': pypdf(False),
    'pypdf-layout': pypdf(True),
    'pdfium': pdfium,
    'pdftotext': pdftotext(False),
    'pdftotext-layout': pdftotext(True),
    'ocr': ocr,
}


def load(name):
    """{file name: [page texts]} for one extractor (from the cache)."""
    return json.loads((CACHE / f'{name}.json').read_text())


def main(names):
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        fn = EXTRACTORS[name]
        out = {p.name: fn(p) for p in sorted(PDF_DIR.glob('*.pdf'))}
        (CACHE / f'{name}.json').write_text(json.dumps(out, ensure_ascii=False, indent=0))
        print(f'{name}: {sum(len(v) for v in out.values())} pages, {sum(len(t) for v in out.values() for t in v)} chars')


if __name__ == '__main__':
    main(sys.argv[1:] or list(EXTRACTORS))
