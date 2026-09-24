"""
Chunking pipelines compared in the audit.

naive     What a typical participant (or the organisers' pre-built table)
          would have: raw pdfplumber text cut into 500-character windows with
          80 characters of overlap. No cleaning, no headers.

faithful  Strict-RAG chunking. Every stored fragment is verbatim PDF text
          (after Unicode/whitespace normalisation and removal of the
          letterhead boilerplate), split on the documents' own section
          headings and prefixed with the product name and the enzyme named in
          the same document. One row per fragment, and
          vecteur = embedding(texte_fragment), as the challenge table implies.
          Nothing is added that the PDF does not say: no synonyms, no
          translations, no question-like text.
"""
import re
import unicodedata
from pathlib import Path

import pdfplumber

from audit.eval_set import doc_key

# ── Extraction & normalisation ─────────────────────────────────────────────

def extract_pages(path, x_tolerance=3):
    with pdfplumber.open(path) as pdf:
        return [pg.extract_text(x_tolerance=x_tolerance) or '' for pg in pdf.pages]


def normalize(text):
    """Unicode NFKC (full-width colon, subscripts), Greek alpha spelled out,
    PDF bullet glyphs to '-', collapsed spaces. No words are added."""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('α', 'alpha').replace('®', '')
    text = re.sub(r'[-•]', '-', text)   # private-use bullet glyphs
    text = re.sub(r'[ \t]+', ' ', text)
    return text


_BOILERPLATE = re.compile(
    r'^(VTR\s*&\s*beyond|No\.\s*8,|Zone,\s*Nanping|Stresemann|Tel\s*:|Mail\s*:|Website\s*:|'
    r'TECHNICAL DATA SHEET|FOOD SAFTY DATA|Bakery\s*Enzyme|Last updating)', re.I)


def squash(s):
    return re.sub(r'\s+', '', s.lower())


# ── Product name / enzyme (both read from the document itself) ─────────────

def product_name(path):
    code = re.sub(r'\.pdf$', '', Path(path).name, flags=re.I)
    for junk in (r'BVZyme', r'TDS', r'pdf', r'\(1\)'):
        code = re.sub(junk, ' ', code, flags=re.I)
    code = re.sub(r'\s+', ' ', code).strip()
    return f'BVZyme {code}'


_ENZYMES = [  # (pattern on squashed PDF text, name as written in the PDF)
    (r'maltogenicamylase', 'maltogenic amylase'),
    (r'alpha-amylase', 'alpha-amylase'),
    (r'amyloglu?cosidase', 'amyloglucosidase'),
    (r'glucoseoxidase', 'glucose oxidase'),
    (r'xylanase', 'xylanase'),
    (r'transglutaminase', 'transglutaminase'),
    (r'lipase|lipolytic', 'lipase'),
]


def enzyme_name(identity_text):
    s = squash(identity_text)
    for pat, name in _ENZYMES:
        if re.search(pat, s):
            return name
    return 'enzyme'


# ── English TDS: section parsing on the documents' own headings ────────────

_TDS_HEADERS = [
    'Product Description', 'Effective material', 'Activity', 'Application',
    'Function', 'Dosage', 'Organoleptic', 'Physicochemical', 'Microbiology',
    'Heavy metals', 'Allergens', 'GMO status', 'Ionization status',
    'Packaging', 'Package', 'Storage',
]
# sections merged into one fragment (they describe one thing together)
_MERGE = {'Physicochemical': 'Organoleptic', 'Ionization status': 'GMO status'}


def _match_header(line):
    s = squash(line)
    for h in _TDS_HEADERS:
        if s.startswith(squash(h)):
            # keep the rest of the line verbatim (split words stay as printed)
            rest = re.sub(r'^\s*' + r'\s*'.join(map(re.escape, h.replace(' ', ''))) + r'\s*:?\s*', '', line, flags=re.I)
            return ('Packaging' if h == 'Package' else h), rest
    return None, None


def tds_sections(pages):
    lines = []
    for page in pages:
        for line in normalize(page).split('\n'):
            line = line.strip()
            if line and not _BOILERPLATE.match(line):
                lines.append(line)
    sections, current = {}, None
    for line in lines:
        header, rest = _match_header(line)
        if header and header != current:
            current = _MERGE.get(header, header)
            sections.setdefault(current, [])
            if header in _MERGE:           # keep the merged sub-heading visible
                rest = f'{header}: {rest}'.strip()
            if rest:
                sections[current].append(rest)
        elif current:
            sections[current].append(line)  # same heading repeated -> content
    return {h: ' '.join(v).strip() for h, v in sections.items() if ' '.join(v).strip()}


def faithful_tds_chunks(path):
    pages = extract_pages(path, x_tolerance=1.5)
    sec = tds_sections(pages)
    enzyme = enzyme_name(sec.get('Product Description', '') + ' ' + sec.get('Effective material', ''))
    prefix = f'{product_name(path)} ({enzyme})'
    labels = {'Organoleptic': 'Organoleptic', 'GMO status': 'GMO status'}
    return [f'{prefix} - {labels.get(h, h)}: {body}' for h, body in sec.items()]


# ── French ascorbic-acid document ──────────────────────────────────────────

_FR_HEADERS = [
    'Résumé Général', 'Propriétés Principales', 'Points Importants',
    'Dosages Recommandés', 'Table de Conversion Rapide', 'Spécifications Techniques',
    'Caractéristiques du Produit', 'Conditionnement Recommandé',
    "Mode d'Emploi en Production", 'Points de Contrôle', 'Avantages et Limitations',
    'Avantages', 'Limitations', 'Alternatives et Complémentarité', 'Réglementation',
    'Statut Légal', 'Dosage Maximum Autorisé', 'Recommandations pour ta Production',
    'Test et Validation', 'Stockage et Sécurité', 'Références',
]
_FR_TITLE = re.compile(r'^(Acide Ascorbique \(E300\)|Améliorant de Panification)$')


def faithful_fr_chunks(path, max_chars=700):
    text = normalize('\n'.join(extract_pages(path, x_tolerance=1.5)))
    prefix = 'Acide Ascorbique (E300)'
    head_keys = [squash(h) for h in _FR_HEADERS]
    sections = []                                   # [label, [lines]]
    for raw in text.split('\n'):
        line = raw.strip()
        if not line or _FR_TITLE.match(line):
            continue
        s = squash(line)
        # a heading is a short line starting with a known heading (the length
        # guard keeps table rows such as "Dosage maximum autorisé 300 ..." as content)
        if any(s.startswith(k) and len(s) <= len(k) + 12 for k in head_keys):
            if sections and not sections[-1][1]:
                sections[-1][0] += ' > ' + line     # empty heading = parent of this one
            else:
                sections.append([line, []])
        elif sections:
            sections[-1][1].append(line)
    chunks = []
    for label, lines in sections:
        part = []
        for line in lines:                          # split long sections on line breaks
            if part and len(' '.join(part + [line])) > max_chars:
                chunks.append(f'{prefix} - {label}: ' + ' '.join(part))
                part = []
            part.append(line)
        if part:
            chunks.append(f'{prefix} - {label}: ' + ' '.join(part))
    return chunks


def faithful_chunks(path):
    if doc_key(Path(path).name) == 'aa':
        return faithful_fr_chunks(path)
    return faithful_tds_chunks(path)


# ── Naive fixed-size baseline ──────────────────────────────────────────────

def naive_chunks(path, size=500, overlap=80):
    text = re.sub(r'\s+', ' ', '\n'.join(extract_pages(path))).strip()
    chunks, start = [], 0
    while True:
        end = min(len(text), start + size)
        if end < len(text):                         # do not cut a word in half
            cut = text.rfind(' ', start + size // 2, end)
            end = cut if cut > 0 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            return [c for c in chunks if c]
        nxt = text.find(' ', end - overlap, end)    # overlap starts on a word boundary
        start = nxt + 1 if nxt > start else end - overlap


PIPELINES = {'naive': naive_chunks, 'faithful': faithful_chunks}


# ── Post-hoc exploration ───────────────────────────────────────────────────
# Designed AFTER looking at TEST failures of the pre-registered pipelines, so
# their TEST numbers are optimistic. Still strict: verbatim text only.

def faithful_page_chunks(path):
    """One fragment per PDF page (page 2 of a TDS has no product name, so the
    same product header is used)."""
    pages = extract_pages(path, x_tolerance=1.5)
    if doc_key(Path(path).name) == 'aa':
        prefix = 'Acide Ascorbique (E300)'
    else:
        sec = tds_sections(pages)
        prefix = f'{product_name(path)} ({enzyme_name(sec.get("Product Description", "") + " " + sec.get("Effective material", ""))})'
    out = []
    for page in pages:
        lines = [l.strip() for l in normalize(page).split('\n')
                 if l.strip() and not _BOILERPLATE.match(l.strip())]
        if lines:
            out.append(f'{prefix} - ' + ' '.join(lines))
    return out


def _fr_table_rows(path, prefix):
    """French tables linearised row by row with their column headings."""
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables():
                rows = [[normalize(c or '').replace('\n', ' ').strip() for c in r] for r in tbl]
                header = rows[0]
                label = {'Type de Production': 'Dosages Recommandés (ppm = g/tonne de farine)',
                         'Poids Farine': 'Table de Conversion Rapide (en grammes)',
                         'Alternative': 'Alternatives et Complémentarité'}.get(header[0], header[0])
                merged = []
                for r in rows[1:]:
                    if merged and not r[0]:             # continuation of a wrapped cell
                        merged[-1] = [f'{a} {b}'.strip() for a, b in zip(merged[-1], r)]
                    else:
                        merged.append(r)
                for r in merged:
                    cells = [f'{h}: {v}' if h else v for h, v in zip(header, r) if v]
                    out.append(f'{prefix} - {label}: ' + ' | '.join(cells))
    return out


_FR_TABLE_SECTIONS = ('Dosages Recommandés', 'Table de Conversion', 'Alternatives et Complémentarité')


def faithful_rows_chunks(path):
    """Section fragments, but tables of the French document become one
    fragment per row."""
    if doc_key(Path(path).name) != 'aa':
        return faithful_tds_chunks(path)
    prefix = 'Acide Ascorbique (E300)'
    keep = [c for c in faithful_fr_chunks(path)
            if not c[len(prefix) + 3:].startswith(_FR_TABLE_SECTIONS)]
    return keep + [f'{prefix} - Dosages Recommandés (ppm*): *ppm = parties par million = g/tonne de farine'] \
        + _fr_table_rows(path, prefix)


def faithful_usage_chunks(path, rows=False):
    """Section fragments, but Application + Function + Dosage of a TDS form a
    single 'usage' fragment, so the dosage line keeps its context."""
    if doc_key(Path(path).name) == 'aa':
        return faithful_rows_chunks(path) if rows else faithful_fr_chunks(path)
    pages = extract_pages(path, x_tolerance=1.5)
    sec = tds_sections(pages)
    prefix = f'{product_name(path)} ({enzyme_name(sec.get("Product Description", "") + " " + sec.get("Effective material", ""))})'
    usage = [h for h in ('Application', 'Function', 'Dosage') if h in sec]
    out = [f'{prefix} - ' + ' '.join(f'{h}: {sec[h]}' for h in usage)] if usage else []
    return out + [f'{prefix} - {h}: {b}' for h, b in sec.items() if h not in usage]


EXPLORATION = {
    'faithful_page': faithful_page_chunks,
    'faithful_rows': faithful_rows_chunks,
    'faithful_usage': faithful_usage_chunks,
    'faithful_usage_rows': lambda p: faithful_usage_chunks(p, rows=True),
}


# ── Storage (same schema as the challenge's `embeddings` table) ────────────

def build_table(conn, table, chunker, model, pdf_folder='data_pdf'):
    """Create `table`, one row per fragment, vecteur = embedding(texte_fragment).
    Returns {id_document: doc_key}."""
    pdfs = sorted(Path(pdf_folder).glob('*.pdf'))
    rows, id2doc = [], {}
    for i, p in enumerate(pdfs, 1):
        id2doc[i] = doc_key(p.name)
        rows += [(i, c) for c in chunker(p)]
    vecs = model.encode([c for _, c in rows], batch_size=64, normalize_embeddings=True)
    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS {table}')
    cur.execute(f'CREATE TABLE {table} (id SERIAL PRIMARY KEY, id_document INT, '
                f'texte_fragment TEXT, vecteur vector(384))')
    cur.executemany(f'INSERT INTO {table} (id_document, texte_fragment, vecteur) VALUES (%s, %s, %s)',
                    [(d, c, v.tolist()) for (d, c), v in zip(rows, vecs)])
    conn.commit()
    return id2doc
