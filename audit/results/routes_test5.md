# TEST-5: one-shot confirmation

TEST-5 was frozen with ROUTES_PLAN.md (commit 4e21bf7) before any route ran. The selected configuration was chosen on the dev pool by the pre-registered rule. Primary test: fully answered questions in transparent mode (S+F), selected against current, exact McNemar.

- **S+F**: current 88/100, selected 87/100; selected gains 0, loses 1; p = 1.000
- **S**: current 61/100, selected 61/100; selected gains 1, loses 1; p = 1.000

## Every candidate on TEST-5 (exploratory except the first two rows)

| route | fragments | fully answered, transparent (S+F) | vs current: gained / lost / p | fully answered, strict (S) | vs current: gained / lost / p | right answer 1st (S+F, single) |
|---|---:|---:|---|---:|---|---:|
| current (pdfplumber-1.5, current chunking) | 609 | 88/100 | +0 / -0 / 1.00 | 61/100 | +0 / -0 / 1.00 | 70/88 |
| selected (pypdf-layout, unmerge) | 677 | 87/100 | +0 / -1 / 1.00 | 61/100 | +1 / -1 / 1.00 | 69/88 |
| pdfplumber-1, current chunking | 609 | 88/100 | +0 / -0 / 1.00 | 61/100 | +0 / -0 / 1.00 | 70/88 |
| pdfplumber-2, current chunking | 609 | 88/100 | +0 / -0 / 1.00 | 61/100 | +0 / -0 / 1.00 | 70/88 |
| pdfplumber-3, current chunking | 609 | 88/100 | +0 / -0 / 1.00 | 60/100 | +0 / -1 / 1.00 | 70/88 |
| pdfminer, current chunking | 576 | 87/100 | +0 / -1 / 1.00 | 61/100 | +4 / -4 / 1.00 | 67/88 |
| pymupdf, current chunking | 605 | 77/100 | +2 / -13 / 0.01 | 51/100 | +4 / -14 / 0.03 | 56/88 |
| pymupdf-sort, current chunking | 609 | 88/100 | +0 / -0 / 1.00 | 62/100 | +1 / -0 / 1.00 | 70/88 |
| pypdf, current chunking | 601 | 76/100 | +1 / -13 / 0.00 | 50/100 | +3 / -14 / 0.01 | 57/88 |
| pdfium, current chunking | 595 | 76/100 | +1 / -13 / 0.00 | 52/100 | +4 / -13 / 0.05 | 57/88 |
| pdftotext, current chunking | 613 | 87/100 | +0 / -1 / 1.00 | 57/100 | +1 / -5 / 0.22 | 70/88 |
| pdftotext-layout, current chunking | 609 | 88/100 | +0 / -0 / 1.00 | 62/100 | +1 / -0 / 1.00 | 70/88 |
| ocr, current chunking | 430 | 72/100 | +2 / -18 / 0.00 | 48/100 | +2 / -15 / 0.00 | 55/88 |
| docling, current chunking | 518 | 83/100 | +0 / -5 / 0.06 | 60/100 | +2 / -3 / 1.00 | 66/88 |
| pypdf-layout, items-spec | 983 | 88/100 | +0 / -0 / 1.00 | 63/100 | +2 / -0 / 0.50 | 70/88 |
| pypdf-layout, items-all | 983 | 88/100 | +0 / -0 / 1.00 | 63/100 | +2 / -0 / 0.50 | 70/88 |
| pypdf-layout, sentences | 826 | 87/100 | +3 / -4 / 1.00 | 61/100 | +4 / -4 / 1.00 | 68/88 |
| pypdf-layout, windows-3 | 2068 | 77/100 | +7 / -18 / 0.04 | 51/100 | +7 / -17 / 0.06 | 63/88 |
| pypdf-layout, windows-6 | 663 | 77/100 | +6 / -17 / 0.03 | 55/100 | +7 / -13 / 0.26 | 60/88 |
| pypdf-layout, fixed-128 | 313 | 81/100 | +7 / -14 / 0.19 | 61/100 | +12 / -12 / 1.00 | 63/88 |
| pypdf-layout, semantic | 569 | 82/100 | +6 / -12 / 0.24 | 46/100 | +4 / -19 / 0.00 | 54/88 |
| pypdf-layout, fr-350 | 613 | 88/100 | +0 / -0 / 1.00 | 62/100 | +1 / -0 / 1.00 | 71/88 |
| pypdf-layout, fr-1000 | 609 | 88/100 | +0 / -0 / 1.00 | 62/100 | +1 / -0 / 1.00 | 70/88 |
