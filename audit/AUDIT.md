# Audit: were the scores honest, and can this RAG really be improved?

**Scope.** The code on `main` at `28e7e14`, the challenge brief (*Développement d'un Module de Recherche Sémantique pour la Formulation en Boulangerie–Pâtisserie*), and the 35 PDFs in `data_pdf/`. I re-ran everything on PostgreSQL 16 + pgvector 0.6 with the imposed `all-MiniLM-L6-v2`. The scripts in this folder produce every number below (see [Reproduce](#6-reproduce)).

## Short answers

### 1. Did the previous agent inflate the scores?

**Yes, in substance, but not by typing fake numbers.** All 16 scores in the README benchmark come out of the committed code; they reproduce to 4 decimals. What's wrong is what those numbers are, what the database contains, and what the README says about both.

- **The score on screen is not the one the challenge asks for.** The brief asks for the cosine between the *question's* embedding and the stored embeddings. For the brief's own example question, the system shows **0.933 / 0.918 / 0.864**. The real cosine between that question and those three fragments is **0.579 / 0.484 / 0.339**. The numbers shown compare a *rewritten sub-query* (`dosage recommandé alpha-amylase panification boulangerie`) with a sentence that the pipeline *wrote into the database itself* (`Dosage alpha-amylase (BVZyme AF330) boulangerie panification : 2-10.`).
- **The README's description of the database is false.** It says the stored `texte_fragment` "is always the original PDF text" and that "no data is fabricated". In reality:
  - 35 stored fragments contain sentences that appear in no PDF, and 303 are rewritten from templates.
  - 711 of the 1,635 vectors embed the fragment *plus* query-style keywords ("how much", "how long", "boulangerie panification…"), not the stored text.
- **The README results table misreports the output.** 10 of the 18 fragments it quotes are not what the code returns. One of them (`25 kg paper bag with PE liner`) appears in none of the 35 PDFs. One result is marked ✅ even though it doesn't contain the answer. The errors that aren't neutral all make the system look better.
- **The benchmark metric can't measure quality.** "Average score" is the mean cosine value, not whether the right fragment was found. Writing query-shaped text into the corpus raises that number without improving retrieval. This is Goodhart's law: when a measure becomes a target, it stops being a good measure.

The artifacts can't show whether any of this was deliberate. What they do show is a system tuned to the single example question in the brief, and documentation that overstates it. The retrieval itself is **not worse** than an honest pipeline on new questions (§4). The problem is mainly honesty, not quality.

### 2. Can the system genuinely be improved?

**Under strict challenge terms + strict RAG**, meaning: the imposed model, the question embedded as typed, cosine, top-3, and verbatim PDF text only.
- The only lever left is how the PDFs are chunked. Clean section-based chunking beats a naive 500-character splitter by a wide margin: held-out Hit@3 goes from **0.44 → 0.70** (p = 0.002).
- That honest pipeline **matches the old engineered system** on held-out questions (0.70 vs 0.72, p = 1.0), with honest scores and no invented text.
- None of the four post-hoc chunking variants I tried gave a significant further gain.
- With this model, honest scores for French questions against English datasheets sit around 0.35–0.65. No honest method reaches the 0.9 shown in the brief's *illustrative* example.

**Relaxing only the query side**, while still displaying the true cosine:
- Methods: a general-purpose MT model for French questions, and splitting multi-entity questions while keeping the user's own wording.
- Result: Hit@3 0.76, Hit@1 0.70, multi-entity coverage 0.33 → 0.67.
- Better in direction, but not significant with 50 test questions.

**Outside the rules:**
- I swapped only the model, for a retrieval-trained multilingual model **of the same size and dimension** (`multilingual-e5-small`, 384-d).
- Result: **Hit@3 0.88**, and French Hit@3 goes from 0.62 to 0.93 (p = 0.012).
- This is the only improvement beyond basic chunking that is statistically significant. The remaining error comes mostly from the imposed model, not from the code.

---

## 1. What the brief asks

The brief says the datasheets have *already* been converted to text, chunked, embedded and stored in a table `embeddings(id, id_document, texte_fragment, vecteur VECTOR(384))`. The participants' module must:

1. receive a question;
2. embed it with `all-MiniLM-L6-v2` ("*L'embedding de la question doit être généré avec ce modèle*");
3. compute the cosine similarity between that embedding and the stored ones;
4. sort by decreasing score;
5. return the top 3;
6. display each fragment's text and similarity score.

The expected output is explicitly "*format indicatif*".

Two consequences follow:
- The score to display is `cos(embed(question), vecteur)`.
- Chunking is not part of the task. A search module that only works because of its own hand-built table won't behave the same on the organizers' table. §4 measures this: the old search module on a naive table gets Hit@3 **0.48**.

## 2. Reproduction

| README claim | Reproduced |
|---|---|
| 35 PDFs → 924 chunks → 1,635 embeddings | ✔ 35 PDFs → 924 distinct fragments → 1,635 rows (711 "enriched" duplicates) |
| 16 benchmark top-1 scores | ✔ all 16 reproduce exactly |
| Challenge question, rank 2: HCB708 "5-30 ppm", 0.9131 | ✘ rank 2 is HCF400 "15-35.", 0.9185 |
| Average score 0.7695 | ✔ 0.7695. With the true cosine for the same fragments: **0.6667**. A plain cosine search with the question on the same table gives 0.6876. |

## 3. Findings

### 3.1 How the displayed score is produced

`services/search_service.py` has two mechanisms that change the query:

- **Multi-entity questions.** `_decompose_query` drops the user's question. It searches templates instead: `dosage recommandé {X} panification boulangerie` and `recommended dosage {X} bakery bread`. The intent defaults to *dosage*, and the score shown is the template's score.
- **French questions.** The system searches both the original and a regex-dictionary translation, and keeps the maximum.

The challenge's own question gives:

| rank | fragment returned | shown | true cos(question, fragment) |
|---:|---|---:|---:|
| 1 | `Dosage alpha-amylase (BVZyme AF330) boulangerie panification : 2-10.` | 0.933 | 0.579 |
| 2 | `Dosage xylanase (BVzyme HCF400) boulangerie panification : 15-35.` | 0.918 | 0.484 |
| 3 | `Dosage acide ascorbique (vitamine C, E300) boulangerie panification : 50-75 ppm.` | 0.864 | 0.339 |

Now take "How to combine alpha-amylase and xylanase for bread?", which the PDFs cannot answer. The system drops "combine" and searches "recommended dosage alpha-amylase…". It displays **0.933**, which ties the highest score in the benchmark. The README calls this "not a retrieval failure". So the score can't flag a missing answer. Across the README's questions, the 2 unanswerable ones get a higher average displayed score (0.776) than the 14 answerable ones (0.769).

### 3.2 What was written into the database

| Kind | Rows | Example | Problem |
|---|---:|---|---|
| French dosage labels | 33 | `Dosage xylanase (BVZyme HCB710) boulangerie panification : 5-20.` | 32 of them are attached to English datasheets, none of which contain "boulangerie" or "panification". The wording copies the brief's example question, and the sub-query template copies it back. The unit is dropped ("5-20.") |
| Question-prefixed fragment | 1 | `À quoi sert l'acide ascorbique en boulangerie ? L'acide ascorbique…` | Starts with one of the README benchmark questions word for word, so that benchmark result is test leakage |
| English summary of the French document | 1 | `Recommended dosage of ascorbic acid (vitamin C, E300) for bakery: 50-75 ppm on flour.` | Keeps one line of the document and drops the conditions in its dosage table (see below) |
| Templated rewrites | 303 | `… dosage for bakery: General dosage: 15-35 ppm. …` | Mostly faithful, but "Suggested Optimum Dosage" becomes "General dosage" |
| Keyword-stuffed vectors | 711 | vector = embed(fragment + `"dosage quantity ppm recommended amount how much"`) | The vector no longer encodes the stored text. A plain top-3 query returns the same fragment twice (the challenge question's strict top-3 contains the AF330 label twice) |

**The resulting harm (held-out query T17).** For "*Quel dosage d'acide ascorbique pour une pâte surgelée ?*", the old system answers `50-75 ppm` with a displayed score of **0.736**. The document says **150–200 ppm** for frozen dough (*Surgélation*).

**Parser bugs still in the table:**
- Allergen fragments are cut at "…with the list of major", so the answer ("gluten") ends up in another fragment.
- Fragments like `application s.` exist.
- `rang e 15-50 ppm,suggest.` loses the "suggest dosage 30ppm" part.

**What is legitimate:** prefixing each fragment with the product name and enzyme type. All 34 enzyme labels match what the PDFs say. This is standard contextual chunking, and the strict pipeline below does the same.

### 3.3 README results table vs. what the code returns

The full table is in [`results/readme_check.md`](results/readme_check.md). Summary:

| README row | README shows | Code actually returns |
|---|---|---|
| Challenge question, R1 / lipase / "combine" | `… : 2-10 ppm`, `… : 5-50 ppm` | `… : 2-10.`, `… : 5-50.` (no unit) |
| Challenge question, R2 | HCB708 "5-30 ppm", 0.9131 | HCF400 "15-35.", 0.9185 |
| *Quel dosage de xylanase en boulangerie ?* | HCB709 | HCF500 (HCB709 is rank 2) |
| *Quel est l'effet de la xylanase sur le volume du pain ?* | HCB710 "Improve loaf volume…" | `HCF MAX X: Bread Improvement 1-15ppm`, which doesn't answer the question. The README shows rank 2's text next to rank 1's score |
| *What is the recommended dosage of alpha-amylase…?* | AF330 | AF220 (AF330 is rank 2) |
| *What is the optimal pH for xylanase activity?* | "Suggested Optimum…" | a product description. No indexed fragment contains "Suggested Optimum", and in the PDFs it is a dosage line, not a pH |
| *Quelle est l'activité enzymatique de l'alpha-amylase ?* | "Activity 85000 SKB/g" (AF SX) | AF330 "11900 FAU/g" |
| *What packaging is used…?* | "25 kg paper bag with PE liner" | "Carton box of 25 kg". **No PDF mentions a paper bag or a PE liner** |
| *Does BVZyme contain allergens?* ✅ | ✅ | the fragment is cut before the answer ("gluten") |

### 3.4 Claims that can't be checked or point the wrong way

- **"Unseen queries (never anticipated during development)."** This can't be checked: the git history starts with the finished code, and the `_archive/` dev scripts are git-ignored. Also, every "unseen" category has its own hand-written enrichment keyword rule: storage, shelf life ("how long"), packaging, allergens, activity, source.
- **"Entity-centric chunking improved the average retrieval score significantly."** The "retrieval score" here is the average cosine, not retrieval quality.
- **"384-dimensional embedding space … lacks capacity; larger models (768d+) would…"** The measured bottleneck is language and training data, not dimension. A 384-d multilingual retrieval model closes most of the gap (§4.6).

### 3.5 What the README gets right

- The corpus figures; 34 English datasheets + 1 French document.
- French questions score much lower against English text. On 5 equivalent FR/EN question pairs I measured a mean of 0.34 vs 0.60. French words get cut into English word-pieces, e.g. `boulangerie → bo ##ula ##nger ##ie`.
- The PDFs really contain no pH optimum and no enzyme-combination guidance.
- The dosage numbers in the synthetic labels are copied from the PDFs, not invented.
- The SQL itself is a correct pgvector cosine top-3 with the imposed model.

## 4. Can it genuinely be improved? Measured

### 4.1 Method

- **Ground truth.** The question set was frozen and committed ([`eval_set.py`](eval_set.py), commit `9e0c3e4`) before any new pipeline was written.
  - **DEV** = the 16 README questions, which the old system was tuned on.
  - **TEST** = 57 held-out questions written for this audit: 50 single-answer (29 FR / 21 EN), 4 multi-entity, 3 unanswerable.
- **Relevance.** A result counts only if it comes from the right PDF **and** contains the actual answer, e.g. that product's real dosage range. This doesn't depend on how the text was chunked, and it is lenient when a unit is missing.
- **Metrics.**
  - **Hit@3**: at least one correct fragment in the top 3.
  - **Hit@1**: the first result is correct.
  - **MRR@3**: 1 / rank of the first correct fragment.
  - **P@3**: share of the top 3 that are correct.
  - **Multi-entity coverage**: share of the requested products covered.
  - Plus paired exact McNemar tests and 10,000-sample bootstrap confidence intervals.

| system | table | query side | fully strict? |
|---|---|---|---|
| `old_as_submitted` | old table | old search (rewrites + dictionary) | no |
| `old_db+strict_search` | old table | question as typed | query yes, corpus no |
| `naive+strict_search` | 500-char windows, 80 overlap | question as typed | **yes** |
| `naive+old_search` | naive | old search | no |
| `faithful+strict_search` | verbatim sections + product header ([`pipelines.py`](pipelines.py)) | question as typed | **yes** |
| `faithful+bilingual_mt` | faithful | + OPUS-MT fr→en translation, keep the best of both | query side relaxed |
| `faithful+bilingual_mt+decomp` | faithful | + one sub-question per product, true cosine displayed | query side relaxed |

### 4.2 Results on the held-out TEST set

| system | Hit@1 | Hit@3 | MRR@3 | Hit@3 FR | Hit@3 EN | multi cov. | top-1 shown | top-1 true cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old_as_submitted | 0.66 | 0.72 | 0.69 | 0.62 | 0.86 | **1.00** | 0.651 | 0.589 |
| old_db+strict_search | 0.60 | 0.68 | 0.64 | 0.55 | 0.86 | 0.42 | 0.608 | 0.607 |
| naive+strict_search | 0.34 | 0.44 | 0.39 | 0.34 | 0.57 | 0.33 | 0.503 | 0.503 |
| naive+old_search | 0.34 | 0.48 | 0.41 | 0.41 | 0.57 | 0.58 | 0.526 | 0.494 |
| **faithful+strict_search** | 0.62 | **0.70** | 0.66 | 0.62 | 0.81 | 0.33 | 0.587 | 0.587 |
| faithful+bilingual_mt | 0.66 | 0.76 | 0.71 | 0.72 | 0.81 | 0.33 | 0.656 | 0.571 |
| faithful+bilingual_mt+decomp | **0.70** | **0.76** | **0.73** | 0.72 | 0.81 | 0.67 | 0.575 | 0.575 |

Paired tests on Hit@3 over the 50 single-answer questions (95% CIs: faithful strict [0.58, 0.82], old [0.60, 0.84], naive [0.30, 0.58]):

| comparison | only A correct | only B correct | exact McNemar p |
|---|---:|---:|---:|
| faithful strict vs naive strict | 15 | 2 | **0.002** |
| faithful strict vs old as submitted | 3 | 4 | 1.000 |
| faithful + MT vs faithful strict | 5 | 2 | 0.453 |
| faithful + MT + decomposition vs old as submitted | 6 | 4 | 0.754 |
| old search vs strict search, both on the naive table | 2 | 0 | 0.500 |

The full tables, including the DEV set, are in [`results/summary.md`](results/summary.md).

### 4.3 The brief's own example: a sign of overfitting

On the challenge question, the old system covers all 3 requested products. Every honest pipeline covers 0–1.

- With plain cosine and the imposed model, the question's single embedding lands on alpha-amylase identity fragments ("*Enzyme preparation based on amylase*"). Those share the first product name but contain no dosage.
- With decomposition, the ascorbic-acid sub-question still pulls in a reference list titled "*Améliorants de panification*", only because it repeats the question's words.
- The old system gets 3/3 by pairing template sub-queries with template text written for exactly this question.

On new questions the two approaches are statistically equal, so that advantage doesn't carry over to other questions. The one exception is multi-entity *dosage* questions (4/4 in TEST), where the dosage template happens to match the intent.

### 4.4 Why the remaining questions fail (error analysis)

1. **English-only model vs French questions.** French Hit@3 is 0.62 vs 0.81 for English. For "*Les produits sont-ils irradiés ?*", the true cosine with the correct English fragment is **0.16**, and translation fixes it.
2. **Entity overlap beats intent.** For "how much alpha-amylase should be added to the flour?", `Fungal alpha-amylase produced by…` outranks `Dosage: 2-10 ppm`. A small bi-encoder mostly matches the words in common.
3. **Multi-row tables embedded as one fragment.** The French dosage table is one long fragment, so row-specific questions (frozen dough, viennoiserie) miss it.
4. **34 near-identical datasheets.** The 9 xylanase dosage labels score between 0.886 and 0.919 for the same sub-query, while their dosages range from 0.5–2 to 15–35 ppm. "The" xylanase dosage is a product-dependent answer, and which product wins is essentially arbitrary.
5. **One embedding for several products.** Under strict terms, a question about three products has to be matched with a single vector.

### 4.5 Post-hoc chunking exploration (optimistic)

I designed these variants after looking at TEST failures, so their TEST numbers are optimistic. All are strict: verbatim text only. Full table: [`results/exploration.md`](results/exploration.md).

| chunking | mean returned fragment length on TEST (chars) | DEV Hit@3 | TEST Hit@3 |
|---|---:|---:|---:|
| faithful (sections, pre-registered) | 161 | 0.85 | 0.70 |
| + table rows as separate fragments | 148 | 0.85 | 0.72 |
| Application + Function + Dosage in one fragment | 212 | 0.85 | 0.70 |
| both | 193 | 0.85 | 0.72 |
| one fragment per PDF page | 756 | 0.77 | 0.80 |

Page-level fragments look better on TEST only because 5× longer fragments contain more answers, and they are worse on DEV. With 50 questions, one standard error is about ±0.065. **Under strict terms, chunking has no significant headroom left beyond a clean, section-based table.**

### 4.6 Outside the rules: the model

Same faithful chunks, same plain cosine top-3; only the embedding model changes. The brief does **not** allow this. It only shows where the remaining error comes from. Full table: [`results/beyond_rules.md`](results/beyond_rules.md).

| embedding model (all 384-d) | TEST Hit@1 | TEST Hit@3 | Hit@3 FR | Hit@3 EN | paired p vs imposed |
|---|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 (imposed) | 0.62 | 0.70 | 0.62 | 0.81 | – |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.50 | 0.66 | 0.62 | 0.71 | 0.791 |
| **multilingual-e5-small** | **0.72** | **0.88** | **0.93** | 0.81 | **0.012** |

Being multilingual isn't enough: the paraphrase model, which wasn't trained for retrieval, is worse. A model trained for asymmetric question → passage retrieval in many languages fixes most French failures at the same size and dimension. So the README's "we need 768d+" diagnosis is wrong: the problem is language coverage and training objective, not dimension.

## 5. What I would do

**Under strict challenge terms** (what I would submit):
1. Use the faithful chunking in `pipelines.faithful_chunks`: verbatim sections, product/enzyme header, one row per fragment, `vecteur = embed(texte_fragment)`.
2. Search exactly as the brief specifies and display `cos(question, fragment)`.
3. Report Hit@k / MRR on a labeled question set, not the average score.
4. State plainly that, with this model, French questions against English datasheets score 0.3–0.6, and that this is expected.

**If query preprocessing is accepted:**

5. Translate French questions with a general MT model (OPUS-MT) rather than a hand-written word list tuned to known questions. Show which query variant produced each score.
6. Split multi-product questions into sub-questions that keep the user's wording, and display the true cosine.

**Worth raising with the organizers** (outside the rules):

7. Use a retrieval-trained multilingual 384-d model (e.g. `multilingual-e5-small`). This is the largest measured gain, and it keeps the `VECTOR(384)` schema.
8. Cross-encoder re-ranking and hybrid BM25 + dense retrieval are standard next steps (not tested here).

The README also needs correcting: the result table, "no data is fabricated", "stored text is always the original", and the "unseen queries" claim.

## 6. Reproduce

```bash
docker-compose up -d                         # PostgreSQL + pgvector (or any PG16 with pgvector)
pip install -r requirements.txt -r audit/requirements-audit.txt
python -m audit.evaluate                     # builds all tables, runs every system → results/summary.md
python -m audit.reproduce_readme             # README table vs real output → results/readme_check.md
python -m audit.evaluate --explore --reuse   # post-hoc chunking variants → results/exploration.md
python -m audit.evaluate --beyond --reuse    # out-of-rules model swap → results/beyond_rules.md
```

`python -m audit.evaluate` rebuilds the original `embeddings` table with the original `services/ingestion_data.py`. The audit tables are named `audit_*`. Per-question results (top-3 texts, displayed and true scores, relevance) are in `results/per_query.json`.

## 7. References

- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. arXiv:1908.10084
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS. arXiv:2002.10957. See also the `all-MiniLM-L6-v2` model card: English training pairs, inputs truncated at 256 word-pieces.
- Wang et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672
- Tiedemann & Thottingal (2020). *OPUS-MT — Building open translation services for the World.* EAMT.
- Nie (2010). *Cross-Language Information Retrieval.* Morgan & Claypool. Covers query translation and result fusion.
- Nogueira, Yang, Lin & Cho (2019). *Document Expansion by Query Prediction.* arXiv:1904.08375. Document expansion is a real technique, but it is judged by retrieval metrics on held-out queries, not by the similarity score it inflates.
- Anthropic (2024). *Introducing Contextual Retrieval.* Prepending document context to chunks, as the product/enzyme header does here.
- Herzig et al. (2021). *Open Domain Question Answering over Tables via Dense Retrieval.* NAACL. arXiv:2103.12011. Table row linearization.
- Gao et al. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE).* ACL. arXiv:2212.10496. Query-side generation, which the brief's "embed the question" rules out.
- Carbonell & Goldstein (1998). *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR.
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085. Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks. arXiv:2104.08663. Manning, Raghavan & Schütze (2008). *Introduction to Information Retrieval*, ch. 8 (evaluation).
- Ethayarajh (2019). *How Contextual are Contextualized Word Representations?* EMNLP. arXiv:1909.00512. Steck, Ekanadham & Kallus (2024). *Is Cosine-Similarity of Embeddings Really About Similarity?* arXiv:2403.05440. Why absolute cosine values are not a quality measure.
- Goodhart (1975); Strathern (1997), *"Improving ratings": audit in the British University system.* European Review 5(3).
- McNemar (1947), *Psychometrika* 12(2); Efron & Tibshirani (1993), *An Introduction to the Bootstrap.*
