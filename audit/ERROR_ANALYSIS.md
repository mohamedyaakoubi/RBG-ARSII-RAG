# Are the answers really correct, and are the errors really limits?

This checks the results of [IMPROVEMENT.md](IMPROVEMENT.md) by hand and asks, for every remaining error, whether anything inside the challenge's rules could fix it. The last section turns this into a checklist for knowing when to stop.

## 1. Hand check of the automatic scoring

The scores so far came from an automatic check: a result counted as right if it came from the right PDF and contained the expected answer, e.g. that product's real dosage range. That can be wrong both ways, so I read every result of the final system on TEST-3, the last held-out set, myself. That was all 59 answerable questions in transparent mode, plus every question where strict mode differed.

**The automatic judgment matched my reading on every question.** I found no "right" result that doesn't really answer, and no "wrong" one that does. But the numbers hide things a reader would notice:

| issue | questions | what the user sees |
|---|---|---|
| right but incomplete | Z07, Z17 | A family question ("dosage range of the xylanases", "bacterial or fungal?") has 9 products, and only 3 fit. Z17 shows only fungal xylanases, although the bacterial HCB ones exist. |
| right answer at rank 2, another product's value at rank 1 | Z03 | GO MAX 63's dose above TG MAX64's. The product name is shown, but a hasty reader could take the wrong value. |
| PDF extraction glitch in the text shown | Z08, Z53 | "10-10 0 ppm" for 10-100 ppm (the PDF itself spaces the digits) |
| translation quirk | Z52 | "Malte" (malt) became "Malta"; the French original is shown next to it |
| wasted slot (bug) | Z57 | In a multi-product answer, the same AMG880 application text appears twice. Duplicates are only removed within each sub-search, not across them. |
| no "I don't know" | Y01–Y03 | Unanswerable questions still get three results with ordinary-looking scores (0.41–0.65 here) |

## 2. Why the remaining errors happen

For each error, [`error_analysis.py`](error_analysis.py) measures three things ([results/error_analysis.md](results/error_analysis.md)):
- **Rank:** where the correct fragment sits in the full ranking.
- **Oracle test:** whether the same question, asked in the documents' own words ("glucose oxidase dosage"), finds it.
- **Product filter:** whether answering only from the sheet of a product named in the question would help.

**Transparent mode: 16 errors out of 122 held-out questions** (TEST-2 + TEST-3):

| cause | errors | evidence | fixable inside the rules? |
|---|---:|---|---|
| product code misread (e.g. "L MAX64" read as "TG MAX64") | 4 | V18, V32, Z01, Z14. Even ideal wording fails for Z14 (rank 6) | **Yes.** A product-code filter answers from the named product's sheet, with cosine ranking inside it. It fixes all 4 and breaks none of the 22 questions naming a product. **Not implemented yet.** |
| lost at the top-3 cut among near-identical sheets | 5 | V05, V34, V54, V63, V70: correct fragment at rank 4–5, within 0.008 of the third | No. `Top K = 3` is imposed, and 34 near-identical sheets make the order of the top results nearly random. |
| user wording ≠ document wording | 5 | V11 "per tonne of flour", V37 "conditionnement" (translated as "conditioning"), V39 "ASR (anaérobies…)", V62 "densité", Z29 "packed". The ideal wording finds 4 at rank 1 and density at rank 3. | Only with hand-written synonym rules or text added to the corpus. That generalizes badly, and it is what the previous version did. |
| the PDF only implies the answer | 2 | V44, Z34 "who manufactures BVZyme?". The letterhead names VTR&beyond but never says "manufacturer". | Only by writing a "manufacturer" label the PDF doesn't contain. |

**Strict mode: 37 errors.** 31 are French questions against English sheets; translating the question fixes 25 of them. That is the English-only model, measured, and strict mode forbids the workaround.

So, of the 16 errors in transparent mode:
- **12 are limits** of the challenge's setup: the imposed model, the imposed top-3, and what the PDFs say.
- **4 are unfinished work.** One general rule I hadn't found.

## 3. How to know when to stop

1. **Check the metric by hand before trusting it.** Read a sample of "right" *and* "wrong" results. Here it held, but it hid incomplete answers and display glitches.
2. **Find where each correct answer ranks.** Rank 4–5 within a hair of the cut means the cut decides. Rank 50+ means the system fundamentally doesn't connect the question to the answer.
3. **Run the oracle test.** Ask again in the documents' own words.
   - Still fails: the model or the fragment is the limit. Only changing a constraint helps.
   - Works: the gap is wording. Go to step 4.
4. **Ask whether the fix is general or specific.** Would you have designed it without seeing this failing question? Does it apply to a whole class of questions ("any product code") or to one word ("conditionnement → packaging")?
   - General: do it, then measure it on questions that played no part in designing it.
   - Specific: you are tuning to the test. Stop.
5. **Blame the constraints explicitly.** Which errors disappear if you lift one rule? Here: 5 slots instead of 3, a multilingual model, translating the question. Those errors are the challenge's limits, not yours, so say so instead of hacking around them.
6. **Know your noise floor.** With N test questions, a difference smaller than about 2×√(p(1−p)/N) can't be told from luck. That is ±0.08–0.12 here. Fixes worth 1–2 questions can't be proven without a bigger test set.
7. **Watch the honesty line.** Once the remaining fixes require writing text the documents don't contain, or rules tuned to known questions, you have reached the point where the previous version went wrong.

**Stop when every remaining error is explained and falls under 5 (a constraint) or 7 (only fixable dishonestly), and the general fixes from step 4 are done or too small to measure.** Here, one general fix (the product-code filter) is still open. After it, the remaining errors are limits.
