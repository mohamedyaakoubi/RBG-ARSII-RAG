# Which PDF extraction and which chunking are optimal?

The plan was written and frozen before any of this ran: [ROUTES_PLAN.md](ROUTES_PLAN.md) (commit `4e21bf7`, with TEST-5).

## What can be proven

No experiment proves a route optimal among all possible routes. This study makes three narrower claims:
1. **Best in a stated family:** 14 extractors and 10 chunking routes, compared on the 297 known questions.
2. **Confirmed or not on fresh questions:** the winner against the current system, once, on TEST-5.
3. **An upper bound for chunking:** the span oracle tries every possible verbatim fragment of the right document for every miss.

## 1. Extraction: fidelity to the page

**Ground truth is physical, not a vote between extractors.** It comes from the position of every visible glyph on the page ([`extraction_fidelity.py`](extraction_fidelity.py)).
- Space characters are ignored, because they draw nothing.
- Two letters are separated when the gap between them is at least 0.1 em. Measured gaps are either under 0.03 em or over 0.21 em, so the threshold doesn't matter.
- Two spots were checked by eye on the rendered page, and the rendering agrees with this truth.

| extractor | text found in reading order | spaces inside a word | words run together | errors per 1,000 words |
|---|---:|---:|---:|---:|
| **pdftotext** (poppler) | 99.3% | 4 | 1 | **0.5** |
| pdftotext `-layout` | 100% | 4 | 3 | 0.8 |
| PyMuPDF, sorted | 100% | 23 | 0 | 2.5 |
| pdfminer.six | 98.7% | 68 | 0 | 7.4 |
| pypdf, layout mode | 100% | 67 | 34 | 10.9 |
| **pdfplumber, `x_tolerance` 1, 1.5 (current) or 2** (identical output) | 100% | 169 | 0 | **18.3** |
| pdfplumber, `x_tolerance` 3 (default) | 100% | 169 | 119 | 31.2 |
| docling (ML layout model) | 80.6% | 41 | 49 | 9.8 |
| Tesseract OCR, 300 dpi | 69.5% | 28 | 33 | 6.6 |
| PyMuPDF, pdfium, pypdf (drawing order) | 66.5% | – | – | – |

- **pdfplumber's errors come from the sheets themselves.** They contain stray space characters from another font, sitting over letters that touch. pdfplumber folds them into the line and splits the word: "gly cerides", "netwo rks", "10-10 0 pp m", "Science and Technolog y" on every letterhead. pdftotext ignores them. On the page, these read "glycerides", "networks", "10-100 ppm".
- **"esterbonds" and "breadapplications" are glued on the page itself.** Every extractor, and OCR, agrees.
- **Extractors that follow the PDF's drawing order are unusable here.** PyMuPDF, pypdf and pdfium in their default modes interleave labels and values ("2-12 / Dosage / ppm"), so a third of the text is out of reading order.
- **OCR and docling lose text.** OCR recovers 69.5% of it, docling 80.6%.

## 2. Extraction: right answers

*In progress:* [results/routes_extractors.md](results/routes_extractors.md) holds the retrieval results for every extractor on the 297 known questions. Chunking routes, the span oracle and the TEST-5 confirmation follow.
