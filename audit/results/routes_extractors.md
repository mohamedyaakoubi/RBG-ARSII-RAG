# Routes: extractors (dev pool)

Current chunking (corpus v2), every extractor. Dev pool = DEV + TEST + TEST-2 + TEST-3 + TEST-4, all already seen: this selects, it does not prove. "Fully answered" = right answer in the top 3, or every named product covered. p = exact McNemar against the current extractor.

## Retrieval with each extractor

| route | fragments | fully answered, transparent (S+F) | vs current: gained / lost / p | fully answered, strict (S) | vs current: gained / lost / p | right answer 1st (S+F, single) |
|---|---:|---:|---|---:|---|---:|
| pdfplumber-1.5 (current) | 609 | 251/281 | +0 / -0 / 1.00 | 184/281 | +0 / -0 / 1.00 | 201/245 |
| pdfplumber-1 | 609 | 251/281 | +0 / -0 / 1.00 | 184/281 | +0 / -0 / 1.00 | 201/245 |
| pdfplumber-2 | 609 | 251/281 | +0 / -0 / 1.00 | 184/281 | +0 / -0 / 1.00 | 201/245 |
| pdfplumber-3 | 609 | 252/281 | +1 / -0 / 1.00 | 184/281 | +2 / -2 / 1.00 | 202/245 |
| pdfminer | 576 | 250/281 | +3 / -4 / 1.00 | 182/281 | +3 / -5 / 0.73 | 197/245 |
| pymupdf | 605 | 194/281 | +4 / -61 / 0.00 | 147/281 | +9 / -46 / 0.00 | 149/245 |
| pymupdf-sort | 609 | 252/281 | +1 / -0 / 1.00 | 184/281 | +0 / -0 / 1.00 | 201/245 |
| pypdf | 601 | 196/281 | +4 / -59 / 0.00 | 148/281 | +9 / -45 / 0.00 | 150/245 |
| pypdf-layout | 609 | 253/281 | +2 / -0 / 0.50 | 185/281 | +1 / -0 / 1.00 | 200/245 |
| pdfium | 595 | 197/281 | +4 / -58 / 0.00 | 146/281 | +8 / -46 / 0.00 | 150/245 |
| pdftotext | 613 | 250/281 | +1 / -2 / 1.00 | 182/281 | +0 / -2 / 0.50 | 199/245 |
| pdftotext-layout | 609 | 252/281 | +1 / -0 / 1.00 | 184/281 | +0 / -0 / 1.00 | 201/245 |
| ocr | 430 | 201/281 | +8 / -58 / 0.00 | 144/281 | +6 / -46 / 0.00 | 160/245 |
| docling | 518 | 245/281 | +6 / -12 / 0.24 | 185/281 | +12 / -11 / 1.00 | 191/245 |
