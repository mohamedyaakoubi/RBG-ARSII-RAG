# Plan: which PDF extraction and which chunking are optimal?

Written and committed with TEST-5, before any route below is run. Results go in `ROUTES.md`.

## What "optimal" can mean here

No experiment proves a route optimal among all possible routes. This study makes three narrower claims, each checkable:
1. **Best in a stated family, on a stated measure:** the extractors and chunking routes listed below, compared on the 297 questions of DEV, TEST, TEST-2, TEST-3 and TEST-4.
2. **Confirmed on fresh questions:** the winner is compared with the current system once, on TEST-5, frozen with this plan.
3. **An upper bound for any chunking of the PDF text:** the span oracle below. If even the best possible fragment cannot fix an error, no chunking can.

The imposed parts stay fixed: `all-MiniLM-L6-v2`, cosine similarity, top 3.

## Candidates

**Extractors** (page text; the rest of the pipeline unchanged, French tables always read by pdfplumber):
pdfplumber with `x_tolerance` 1, 1.5 (current), 2 and 3 (library default); pdfminer.six; PyMuPDF, plain and sorted; pypdf, plain and layout mode; pypdfium2; pdftotext (poppler), plain and `-layout`; Tesseract OCR of the rendered pages; docling (ML layout model), run in its own environment.

**Chunking routes** (with the chosen extractor):
- current (v2);
- one fragment per item of the specification sections (Microbiology, Heavy metals, Organoleptic, GMO status, Allergens), added to the sections;
- Physicochemical and Ionization status as their own sections instead of merged;
- one fragment per "Label: value" item of every section, added;
- one fragment per sentence, added;
- sliding windows of 3 and of 6 lines, replacing the sections;
- fixed chunks of 128 tokens overlapping by 64, replacing the sections;
- semantic chunking (a new chunk where consecutive lines' similarity drops), replacing the sections;
- the French document's sections split at 350 or 1000 characters instead of 700;
- the combination of the routes above that each improve the measure.

Every fragment keeps the product header, and every route is content-preserving.

## Measure and selection

- **Primary measure:** fully answered questions in the app's default mode (transparent, with the product filter). A single-answer question counts if a right answer is in the top 3. A question naming several things counts if every one of them is covered.
- **Secondary measures:** the same in strict mode, right answer ranked 1st, and extraction fidelity (below).
- **Selection:**
  - Extractor first, with the current chunking.
  - Then chunking routes, on the chosen extractor.
  - A challenger replaces the incumbent only if it gains at least 1 question on the primary measure, and ties keep the incumbent.
- **Confirmation:**
  - On TEST-5, the selected configuration against the current system: exact McNemar test on fully answered questions, transparent mode.
  - The app changes only if the selected configuration is better with p < 0.05.
  - All other candidates are also run on TEST-5, as exploratory results only.

## Extraction fidelity

- **Completeness:** every extractor should output the same characters. Letters and digits are compared with whitespace removed, against the majority.
- **Spacing:** split words ("gly cerides") and glued words ("esterbonds") are counted with a dictionary (wordfreq, English and French).
- **Numbers:** broken numbers ("10-10 0") are counted.
- **Manual check:** a random sample of disagreements is checked by hand against the rendered page.

## Span oracle (upper bound for chunking)

For every question the current system gets wrong:
- The target document is cut into every possible verbatim fragment: any run of consecutive lines, with the product header and optionally its section heading.
- The failure counts as fixable by chunking only if some such fragment containing the answer would enter the top 3, holding the rest of the index fixed.
- The count of fixable failures bounds what any chunking of the verbatim text can gain.
- The oracle is computed per extractor. The best extractor per question bounds the extraction choice.
- It is computed on the 297 known questions and on TEST-5.
