"""
docling (ML layout model) as a PDF text extractor, for ROUTES.md.

Run in its own environment, so its dependencies do not touch the app's:
    python -m venv /tmp/docling-env && /tmp/docling-env/bin/pip install docling
    /tmp/docling-env/bin/python audit/extract_docling.py

Writes cache/extractions/docling.json ({file name: [page texts]}): the plain
text of each page, in docling's reading order.
"""
import json
from pathlib import Path

from docling.document_converter import DocumentConverter

ROOT = Path(__file__).resolve().parent.parent
converter = DocumentConverter()
out = {}
for path in sorted((ROOT / 'data_pdf').glob('*.pdf')):
    doc = converter.convert(str(path)).document
    out[path.name] = [doc.export_to_text(page_no=p) for p in sorted(doc.pages)]
    print(path.name, len(out[path.name]), 'pages')
(ROOT / 'cache' / 'extractions').mkdir(parents=True, exist_ok=True)
(ROOT / 'cache' / 'extractions' / 'docling.json').write_text(json.dumps(out, ensure_ascii=False, indent=0))
