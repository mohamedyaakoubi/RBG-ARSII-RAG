# Which PDF extraction and which chunking are optimal?

The plan was written and frozen before any of this ran: [ROUTES_PLAN.md](ROUTES_PLAN.md) (commit `4e21bf7`, with TEST-5).

## Short answer

- **Extraction:** for right answers, the current pdfplumber setup is tied for best. No extractor answers more fresh questions. It is not the most faithful, though: pdftotext `-layout` makes 23 times fewer word-break errors ("10-100 ppm" rather than "10-10 0 pp m") and answers exactly as many.
- **Chunking:** the current section-based chunking is tied for best on fresh questions.
  - Routes that ignore the sheets' structure lose up to 62 of 281 questions.
  - Putting every possible fragment in the index loses 35.
- **What cannot be proven:** that no chunking at all could do better.
  - For most misses, the span oracle finds a fragment that would win if chosen knowing the answer.
  - The same fragments, put together, make results worse.
  - So the limit is established across everything tested, not mathematically.

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

Each extractor feeds the unchanged pipeline (current chunking) and answers the 281 answerable known questions (DEV, TEST, TEST-2, TEST-3, TEST-4). A question is **fully answered** if a right answer is in the top 3, or, if it names several things, all of them are covered. Full table: [results/routes_extractors.md](results/routes_extractors.md).

| extractor | fully answered, transparent | against current: gained / lost / p | fully answered, strict |
|---|---:|---|---:|
| pypdf, layout mode | **253** | +2 / −0 / 0.50 | 185 |
| pdftotext `-layout`, PyMuPDF sorted, pdfplumber default (3) | 252 | +1 / −0 / 1.00 | 184 |
| **pdfplumber 1.5 (current)**, and 1 or 2 (identical text) | **251** | – | 184 |
| pdftotext | 250 | +1 / −2 / 1.00 | 182 |
| pdfminer.six | 250 | +3 / −4 / 1.00 | 182 |
| docling | 245 | +6 / −12 / 0.24 | 185 |
| Tesseract OCR | 201 | +8 / −58 / < 0.001 | 144 |
| PyMuPDF, pypdf, pdfium (drawing order) | 194–197 | about +4 / −60 / < 0.001 | 146–148 |

- **Fidelity doesn't buy answers here.** pdftotext is 35 times cleaner than pdfplumber, yet answers one question fewer. Every extractor that keeps reading order lands within 2 questions of the current one, which is noise on 281 questions.
- **Why:** a word split here and there changes a fragment's vector little, and relevance is judged with spaces removed. Every missed answer is present in a fragment, with any extractor.
- **Reading order is what matters.** The drawing-order extractors separate each label from its value, which breaks the sections; they lose about 60 questions. OCR loses text and about 50 questions.
- **Selected by the pre-registered rule: pypdf in layout mode.** It gains 2 and loses none (p = 0.50), a difference within noise that TEST-5 has to confirm or not.

## 3. Chunking routes

Every route on pypdf's layout-mode text, same questions ([results/routes_chunking.md](results/routes_chunking.md)).

| route | fragments | fully answered, transparent | against current chunking: gained / lost / p | fully answered, strict |
|---|---:|---:|---|---:|
| current chunking (v2) | 609 | 253 | – | 185 |
| + one fragment per specification item ("Lead: < 5 mg/kg") | 983 | 255 | +2 / −0 / 0.50 | 185 |
| + the same for every section | 983 | 255 | +2 / −0 / 0.50 | 185 |
| Physicochemical and Ionization status as their own sections | 677 | **255** | +2 / −0 / 0.50 | **186** |
| both of the above | 949 | 255 | +2 / −0 / 0.50 | 186 |
| + one fragment per sentence | 826 | 246 | +2 / −9 / 0.07 | 177 |
| French sections split at 350 or 1000 characters | 609–613 | 253 | +0 / −0 / 1.00 | 185 |
| fixed chunks of 128 tokens, overlapping by 64 | 313 | 226 | +16 / −43 / < 0.001 | 172 |
| semantic chunking | 569 | 206 | +11 / −58 / < 0.001 | 134 |
| sliding windows of 3 lines | 2068 | 207 | +9 / −55 / < 0.001 | 136 |
| sliding windows of 6 lines | 663 | 191 | +11 / −73 / < 0.001 | 140 |

- **The document's own structure is worth 27 to 62 questions.** That is the net loss of every route that ignores it (windows, fixed-size, semantic). This is the clearest result of the study.
- **Finer fragments inside the structure gain at most 2 questions.** Four variants tie at 255 (items, items for every section, unmerged sections, both). Items for every section is identical to items for the specification sections, because the other sections have no labelled items.
- **Sentences hurt.** A sentence is too short to carry its context.
- **Selected, with the plan's secondary measures breaking the tie: unmerged sections.** It is highest in strict mode (186) and ranks the right answer first most often (202). Its gain over current chunking, +2 / −0, is again within noise.

The configuration taken to TEST-5 is therefore **pypdf in layout mode, with Physicochemical and Ionization status as their own sections**. On the dev pool it has 255 fully answered, against 251 for the current system.

## 4. Confirmation on TEST-5

TEST-5 is 100 fresh questions plus 4 unanswerable ones. It was frozen with the plan before any route ran, and run once ([results/routes_test5.md](results/routes_test5.md)).

| configuration | fully answered, transparent | fully answered, strict |
|---|---:|---:|
| **current**: pdfplumber 1.5, current chunking | **88/100** | 61/100 |
| selected: pypdf layout mode, unmerged sections | 87/100 (+0 / −1, p = 1.0) | 61/100 (+1 / −1) |

- **The selected configuration is not better.** Its +4 on the known questions was noise, so by the plan the app does not change.
- **No other candidate beats the current system in transparent mode.** The best ties at 88. These runs are exploratory, and the pattern of the known questions repeats:
  - extractors that keep reading order: 87–88;
  - docling: 83; drawing-order extractors: 76–77; OCR: 72;
  - routes that ignore the structure: 77–82;
  - item fragments and French section sizes: 88; sentences: 87.
- **In strict mode,** item fragments reach 63 (+2 / −0) and a few extractors 62 (+1). All within noise.

## 5. The span oracle: could any chunking do better?

**The test.** For every single-answer question the current system misses, the oracle tries every verbatim fragment of the right document:
- every run of consecutive lines, with the product header, also after its section heading;
- about 66,000 candidates per extractor.

It asks whether the best of them would enter the top 3, with the rest of the index unchanged ([results/routes_oracle.md](results/routes_oracle.md), [results/routes_oracle_test5.md](results/routes_oracle_test5.md)).

| current extractor | misses, transparent | some fragment would win | no fragment can | misses, strict | some would win | none can |
|---|---:|---:|---:|---:|---:|---:|
| known questions | 20 | 18 | 2 | 66 | 51 | 15 |
| TEST-5 | 10 | 10 | 0 | 27 | 23 | 4 |

The other extractors give nearly the same numbers.

- **The bound does not close.** For most misses there is a fragment that would win, but it is picked for one question, knowing where the answer is.
- **The winners are of three kinds:**
  - one-line fragments: "BVZyme L MAX X (lipase) - Lead: < 5 mg/kg", "Ascorbic Acid (E300) - Density: ~1.65 g/cm3";
  - the letterhead under a product name, which version 1 of the corpus used and which hijacked unrelated questions;
  - page-long runs that win by a hair.
- **The misses that no fragment can fix are proven to be beyond chunking.** In strict mode they are mostly French questions: "teneur maximale en plomb" scores at most 0.17 against any cut of the English sheets.

## 6. Post-hoc: every possible fragment at once

This test is not in the plan. I added it after the oracle to check whether its fragments still win when they compete. All 65,772 candidates go into one index, and a fragment is skipped when its lines are already shown ([results/routes_all_spans.md](results/routes_all_spans.md)).

| | fully answered, transparent | against current | fully answered, strict |
|---|---:|---|---:|
| known questions: current chunking (609 fragments) | **251/281** | – | **184/281** |
| known questions: every possible fragment | 216/281 | +12 / −47, p < 0.001 | 147/281 |
| TEST-5: current chunking | **88/100** | – | **61/100** |
| TEST-5: every possible fragment | 75/100 | +5 / −18, p = 0.01 | 50/100 |

More fragments means more near-copies competing for three slots, and the right one no longer stands out. Chunking pushed to the maximum is clearly worse. The oracle's wins exist only when the answer is known in advance.

## 7. Conclusion

**Proven within everything tested, and confirmed on fresh questions:** the current system is tied for best.
- Extraction needs reading order. Beyond that, fidelity does not change the answers.
- Chunking needs the sheets' own structure. Finer, coarser or structure-free cuts do not help.

**Not proven:** that no other chunking could do better. The oracle's bound is too loose. The one direction it points to that was not tested is one fragment per "Label: value" line of the French specification section (density, pH, solubility). TEST-5 misses suggested it, so only a new frozen test set could measure it, and it concerns a few questions at most.

**An optional change that costs no answers:** extracting with pdftotext `-layout`.
- It answers the same number of questions: 252 vs 251 known, 88 vs 88 on TEST-5.
- It shows cleaner text to the user: "10-100 ppm" rather than "10-10 0 pp m", and "glycerides".
- The pre-registered rule counts answers only, so this is a product choice, not a measured gain in right answers.

**A lesson for the stop checklist ([ERROR_ANALYSIS.md](ERROR_ANALYSIS.md), step 6):** the selected configuration looked 4 questions better on the 281 known questions. On fresh questions it was 1 worse. Gains of a few questions are noise until a fresh set confirms them.
