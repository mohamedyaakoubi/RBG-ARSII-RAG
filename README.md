# RAG — Module de Recherche Sémantique pour la Formulation en Boulangerie

A semantic search module that answers technical questions in **French or English** about bakery ingredient data sheets (BVZyme enzymes and ascorbic acid). It follows the ARSII RAG Challenge rules: `all-MiniLM-L6-v2` embeddings, cosine similarity, top 3.

---

## Context

This project was developed for the **ARSII RAG Challenge** organized by **STE AGRO MELANGE TECHNOLOGIE — Rose Blanche Group**.

**Objective:** build the semantic search module of a RAG system over technical data sheets of bakery ingredients (enzymes, dough improvers, oxidizing agents). Given a question in natural language, the module must:

1. embed the question;
2. compare it with the stored fragment embeddings using **cosine similarity**;
3. rank the fragments by decreasing score;
4. return the **3 most relevant fragments** with their text and similarity score.

**Constraints imposed by the challenge:**

| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector dimension | 384 |
| Similarity metric | Cosine similarity |
| Results returned | Top K = 3 |
| Language | Python |
| Database | PostgreSQL + pgvector, table `embeddings(id, id_document, texte_fragment, vecteur)` |

**Dataset:** 35 PDFs: 34 English technical data sheets (BVZyme alpha-amylase, maltogenic amylase, amyloglucosidase, glucose oxidase, xylanase, lipase, transglutaminase) and 1 French document on ascorbic acid (E300).

**Example question from the challenge:**
> *Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?*

---

## How it works

```
PDFs ──► fragments (text from the PDFs) ──► all-MiniLM-L6-v2 ──► PostgreSQL + pgvector
                                                                       │
question ──► all-MiniLM-L6-v2 ──► cosine similarity, ranked ◄──────────┘ ──► top 3 (text + score)
```

**Fragments** (`services/pdf_processor.py`) are text taken from the PDFs. Nothing is added that the PDFs do not say.
- One fragment per section of each data sheet (Dosage, Function, Storage, Allergens…), headed by the product and enzyme names read from the same PDF.
- Neighboring sections are also grouped (Application + Dosage, Function + Dosage, page-1 product card), because a dosage line alone is too short to be found.
- Tables are kept whole and also split into one fragment per row, with the column headings.
- Words split apart by the PDF extraction are re-joined ("Amyloglu cosidase" → "Amyloglucosidase").

**French document also indexed in English** (`services/translation.py`). The embedding model works best in English, so French fragments are also indexed as English translations (OPUS-MT).
- Translation is sentence by sentence, with product names protected and section labels taken from a glossary.
- A translation is rejected if a number changes.
- Results show the original French text next to the translation.

**Search** (`services/search_service.py`) embeds the question with `all-MiniLM-L6-v2`, ranks the stored fragments by pgvector cosine similarity, and returns the top 3. A fragment whose content is already shown (its translation, or a section of a card already shown) is skipped. Two modes, chosen with `SEARCH_MODE` or in the UI:

| mode | the question is embedded… |
|---|---|
| `transparent` (default) | as typed and, if French, also in English (the better match is kept); a question naming several products is split into one sub-question per product, keeping the user's wording; a question naming a product by its code (`L MAX64`, `lmax64`, `L-MAX 64`) is answered from that product's sheet |
| `strict` | exactly as typed |

Every result shows its score, the source PDF, and **the formulation of the question that produced the score**. The score is the true cosine similarity between that formulation and the fragment shown. It is not a probability of being correct. When the search was restricted to the sheets a question names, a notice above the results says so.

---

## Results

Measured on held-out questions that were frozen in git before the configuration was chosen. "Right answer" means a result from the right PDF that contains the actual answer (e.g. that product's real dosage range).

| system | right answer in the top 3 | ranked 1st | French questions | multi-product questions (share of products covered) |
|---|---:|---:|---:|---:|
| previous version of this project | 33/52 = 63% | 52% | 61% | 43% |
| **strict mode** | **42/52 = 81%** | 67% | 79% | 50% |
| **transparent mode** (before the product filter) | **48/52 = 92%** | 85% | 97% | 86% |

The challenge's example question in transparent mode returns one dosage fragment per product: xylanase (BVZyme HCF MAX X), ascorbic acid (Pain de mie CBP: 75 ppm), and alpha-amylase (BVZyme AF220: 2-10 ppm).

**Product filter.** Transparent mode later gained the product filter. It was measured on 77 fresh questions that all name a product (TEST-4), frozen before the filter was written:
- **Single-product questions:** 53/60 → 56/60 right. It fixed 3 and broke none, but that is too few to rule out luck (p = 0.25).
- **Questions naming two products, or a product and another family:** everything named was covered in 12 of 14 instead of 5 (8 gained, 1 lost; p = 0.039).

Full method, per-component contributions and remaining limits are in [audit/IMPROVEMENT.md](audit/IMPROVEMENT.md). [audit/ROUTES.md](audit/ROUTES.md) compares 14 PDF extractors and 10 chunking methods, confirmed on fresh questions: the current pdfplumber extraction and section-based chunking are tied for best. [audit/AUDIT.md](audit/AUDIT.md) is an audit of the previous version. Its benchmark scores were real outputs, but the shown scores came from rewritten queries matched against text written into the database, and its results table did not match the code's output.

**Limits.** Scores cannot tell when the corpus has no answer: unanswerable questions score in the same range as answerable ones. For a question about a product family (e.g. "xylanase"), dosages differ between products, so read the product name. The product filter needs the full code: a partial code ("MAX64" could be L, TG or HCF MAX64) or a typo is answered as before. A retrieval-trained multilingual model would help most with French questions, but the challenge imposes `all-MiniLM-L6-v2`.

---

## Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose

### 1. Clone and install dependencies

```bash
git clone https://github.com/mohamedyaakoubi/RBG-ARSII-RAG.git
cd RBG-ARSII-RAG
pip install -r requirements.txt
```

The first run downloads `all-MiniLM-L6-v2` and, in transparent mode, the OPUS-MT French→English model. Translations of the provided PDFs are already cached in `cache/translations.json`.

### 2. Start the PostgreSQL + pgvector database

```bash
docker-compose up -d
```

This builds a PostgreSQL 16 image with pgvector and starts it on `localhost:5432`.

### 3. Configure environment

Create a `.env` file (or use the provided one):

```env
DB_USER=rag
DB_PASSWORD=ragpassword
DB_NAME=ragdb
DB_HOST=localhost
PDF_FOLDER=data_pdf
# optional: SEARCH_MODE=strict
```

### 4. Run

**Streamlit UI:**

```bash
streamlit run app.py
```

**CLI:**

```bash
python main.py
```

From the menu: **1** ingests the PDFs (rebuilds the tables), **2** searches, **4** switches between transparent and strict mode.

---

## Project Structure

```
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entry point
├── config/settings.py       # DB, top-k, search mode, translation cache
├── database/
│   ├── connection_pg.py     # PostgreSQL connection
│   └── models.py            # tables, batch insert, cosine similarity search
├── services/
│   ├── pdf_processor.py     # PDF → fragments
│   ├── translation.py       # guarded French → English translation
│   ├── embedding_service.py # all-MiniLM-L6-v2
│   ├── ingestion_data.py    # PDFs → fragments → embeddings → database
│   └── search_service.py    # strict / transparent search
├── audit/                   # audit of the previous version, evaluation sets and scripts
├── cache/translations.json  # translation cache
├── data_pdf/                # the 35 PDFs
├── docker-compose.yml, Dockerfile, init.sql
└── requirements.txt
```
