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
- **Oracle test:** whether the question, rephrased in the documents' own words ("glucose oxidase dosage"), finds it.
- **Product filter:** whether answering only from the sheet of a product named in the question would help.

The oracle test is a diagnostic, not a fix. I wrote those phrasings knowing where each answer is, and no user can be expected to guess them. The test separates "the answer can't be found" from "the answer can't be found from the user's words". Either way, the user's question still gets a wrong answer, and it stays counted as one.

**Transparent mode: 16 errors out of 122 held-out questions** (TEST-2 + TEST-3):

| cause | errors | evidence | fixable inside the rules? |
|---|---:|---|---|
| product code misread (e.g. "L MAX64" read as "TG MAX64") | 4 | V18, V32, Z01, Z14. Z14 fails even when rephrased in the documents' words (rank 6): the code itself is misread. | **Yes.** A product-code filter answers from the named product's sheet, with cosine ranking inside it. It fixes all 4 and breaks none of the 22 questions naming a product. **Not implemented yet.** |
| lost at the top-3 cut among near-identical sheets | 5 | V05, V34, V54, V63, V70: correct fragment at rank 4–5, within 0.008 of the third | No. `Top K = 3` is imposed, and 34 near-identical sheets make the order of the top results nearly random. |
| user wording ≠ document wording | 5 | V11 "per tonne of flour", V37 "conditionnement" (translated as "conditioning"), V39 "ASR (anaérobies…)", V62 "densité", Z29 "packed". Rephrased in the documents' words, 4 rank 1st and density 3rd. So the answers are in the index, but the model doesn't connect the user's words to them. | **Not that I found.** A user can't be asked to guess the documents' vocabulary, so a fix has to be automatic. With the imposed model, the only fixes I found are hand-written synonym rules or text added to the corpus. Those only cover words already seen failing, and it is what the previous version did. V37 is also a translation error; another translation model might avoid it (untested, one question). |
| the PDF only implies the answer | 2 | V44, Z34 "who manufactures BVZyme?". The letterhead names VTR&beyond but never says "manufacturer". | Only by writing a "manufacturer" label the PDF doesn't contain. |

**Strict mode: 37 errors.** 31 are French questions against English sheets; translating the question fixes 25 of them. That is the English-only model, measured, and strict mode forbids the workaround.

So, of the 16 errors in transparent mode:
- **12 are limits** of the challenge's setup: the imposed model, the imposed top-3, and what the PDFs say. "Limit" doesn't mean acceptable: each is still a wrong answer for the user. It only means I found no honest, general change inside the rules that fixes it.
- **4 are unfinished work.** One general rule I hadn't found.

## 3. How to know when to stop

1. **Check the metric by hand before trusting it.** Read a sample of "right" *and* "wrong" results. Here it held, but it hid incomplete answers and display glitches.
2. **Find where each correct answer ranks.** Rank 4–5 within a hair of the cut means the cut decides. Rank 50+ means the system fundamentally doesn't connect the question to the answer.
3. **Use the oracle test as a diagnostic only.** Rephrase the failing question in the documents' own words. You can do that only because you know the answer, so it tells you where the failure is, never that the question was fine. Users ask their own way, and the system has to cope with that.
   - Still fails: wording isn't the cause. Check the fragment, or look for a general rule (step 4). Here: product codes.
   - Works: the answer is in the index, but not reachable from the user's words. Only an automatic change applied to every question counts (step 4). Asking users to rephrase doesn't.
4. **Ask whether the fix is general or specific.** Would you have designed it without seeing this failing question? Does it apply to a whole class of questions ("any product code") or to one word ("conditionnement → packaging")?
   - General: do it, then measure it on questions that played no part in designing it.
   - Specific: you are tuning to the test. Stop.
5. **Blame the constraints explicitly.** Which errors disappear if you lift one rule? Here: 5 slots instead of 3, a multilingual model, translating the question. Those errors are the challenge's limits, not yours, so say so instead of hacking around them.
6. **Know your noise floor.** With N test questions, a difference smaller than about 2×√(p(1−p)/N) can't be told from luck. That is ±0.08–0.12 here. Fixes worth 1–2 questions can't be proven without a bigger test set.
7. **Watch the honesty line.** Once the remaining fixes require writing text the documents don't contain, or rules tuned to known questions, you have reached the point where the previous version went wrong.

**Stop when every remaining error is explained and falls under 5 (a constraint) or 7 (only fixable dishonestly), and the general fixes from step 4 are done or too small to measure.** Here, one general fix (the product-code filter) is still open. After it, the remaining errors are limits. They are still wrong answers users will get, so the write-up has to say so.
