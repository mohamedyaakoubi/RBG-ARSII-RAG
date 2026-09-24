"""
PDF → fragments.

Every fragment is text taken from the PDF (whitespace/Unicode normalised,
words split by the PDF extraction re-joined), prefixed with the product or
document name found in the same PDF. Nothing is added that the PDF does not
say. Fragments of French documents are also indexed in English (the embedding
model's language) by services/translation.py.

A fragment is a dict:
  id_document, fichier, produit (header), parts [(label, content)],
  texte (what is embedded and shown), langue, cle (content identity: an
  original and its translation share it), couvre (section ids it contains,
  used to avoid showing the same content twice).
"""
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION & NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

def extract_pages(path):
    """Page texts in reading order, read by pdftotext -layout (poppler).

    Of 14 extractors compared in audit/ROUTES.md, it answers as many questions
    as the best and reads the pages most faithfully: 23 times fewer broken or
    merged words than pdfplumber, which splits words where the sheets contain
    stray space characters ("10-10 0 ppm" for 10-100 ppm). Without poppler,
    pdfplumber is used instead (the configuration measured before)."""
    try:
        text = subprocess.run(['pdftotext', '-enc', 'UTF-8', '-layout', str(path), '-'],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning(f'pdftotext indisponible ({e}) : extraction avec pdfplumber')
        with pdfplumber.open(path) as pdf:
            return [pg.extract_text(x_tolerance=1.5) or '' for pg in pdf.pages]
    pages = text.split('\f')
    return pages[:-1] if pages and not pages[-1].strip() else pages


def normalize(text):
    """Unicode NFKC, Greek alpha spelled out, PDF bullet glyphs to '-'."""
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('α', 'alpha').replace('®', '')
    text = re.sub(r'[-•]', '-', text)
    return re.sub(r'[ \t]+', ' ', text)


def squash(s):
    return re.sub(r'\s+', '', s.lower())


class Desplitter:
    """Re-join a word that PDF extraction cut in two ('Amyloglu cosidase',
    'applicatio ns', 'rang e', 'pp m') when the joined form is a known word and
    one piece is not a word on its own. Known words: the embedding model's
    English vocabulary plus words occurring unsplit at least twice in the corpus."""

    def __init__(self, corpus_texts, english_vocab):
        counts = Counter(w for t in corpus_texts for w in re.findall(r'[A-Za-zÀ-ÿ]+', t.lower()))
        self.english = {w for w in english_vocab if w.isalpha() and len(w) > 1}
        self.known = self.english | {w for w, n in counts.items() if n >= 2 and len(w) >= 3}

    def __call__(self, text):
        toks = text.split(' ')
        words = [re.sub(r'\W', '', t).lower() for t in toks]
        piece = lambda w: w not in self.english and w not in ('a', 'i')

        def joinable(i):
            if i + 1 >= len(toks):
                return 0
            a, b = words[i], words[i + 1]
            ok = a.isalpha() and b.isalpha() and (a + b) in self.known and (piece(a) or piece(b))
            return len(a + b) if ok else 0

        out, i = [], 0
        while i < len(toks):
            here = joinable(i)
            if here and joinable(i + 1) <= here:      # prefer the longer word when joins compete
                out.append(toks[i] + toks[i + 1])
                i += 2
            else:
                out.append(toks[i])
                i += 1
        return ' '.join(out)


def fragment(doc_id, fichier, produit, parts, langue, cle, couvre, original=None):
    texte = f'{produit} - ' + ' '.join(f'{label}: {content}' if label else content for label, content in parts)
    return {'id_document': doc_id, 'fichier': fichier, 'produit': produit, 'parts': parts,
            'texte': texte, 'texte_original': original or texte, 'langue': langue,
            'cle': cle, 'couvre': sorted(couvre)}


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL DATA SHEETS (BVZyme layout)
# ═══════════════════════════════════════════════════════════════════════════

_LETTERHEAD = re.compile(
    r'^(VTR\s*&\s*beyond|No\.\s*8,|Zone,\s*Nanping|Stresemann|Tel\s*:|Mail\s*:|Website\s*:)', re.I)
_SKIP = re.compile(r'^(TECHNICAL DATA SHEET|FOOD SAFTY DATA|Bakery\s*Enzyme|Last updating)', re.I)

_TDS_HEADERS = [
    'Product Description', 'Effective material', 'Activity', 'Application',
    'Function', 'Dosage', 'Organoleptic', 'Physicochemical', 'Microbiology',
    'Heavy metals', 'Allergens', 'GMO status', 'Ionization status',
    'Packaging', 'Package', 'Storage',
]
_MERGED = {'Physicochemical': 'Organoleptic', 'Ionization status': 'GMO status'}
PAGE1 = ('Product Description', 'Effective material', 'Activity', 'Application',
         'Function', 'Dosage', 'Organoleptic')

_ENZYMES = [  # (pattern on squashed text of the sheet, name as written in the sheets)
    (r'maltogenicamylase', 'maltogenic amylase'),
    (r'alpha-amylase', 'alpha-amylase'),
    (r'amyloglu?cosidase', 'amyloglucosidase'),
    (r'glucoseoxidase', 'glucose oxidase'),
    (r'xylanase', 'xylanase'),
    (r'transglutaminase', 'transglutaminase'),
    (r'lipase|lipolytic', 'lipase'),
]


def product_name(path):
    code = re.sub(r'\.pdf$', '', Path(path).name, flags=re.I)
    for junk in (r'BVZyme', r'TDS', r'pdf', r'\(1\)'):
        code = re.sub(junk, ' ', code, flags=re.I)
    return 'BVZyme ' + re.sub(r'\s+', ' ', code).strip()


def enzyme_name(text):
    s = squash(text)
    return next((name for pat, name in _ENZYMES if re.search(pat, s)), 'enzyme')


# header of a data sheet's fragments: f'{product_name(path)} ({enzyme_name(...)})'
_SHEET_HEADER = re.compile(r'(BVZyme .+) \((?:%s|enzyme)\)' % '|'.join(re.escape(n) for _, n in _ENZYMES))


def sheet_product(produit):
    """'BVZyme L MAX64 (lipase)' -> 'BVZyme L MAX64' when produit is the header
    of a technical data sheet's fragments, else None."""
    m = _SHEET_HEADER.fullmatch(produit or '')
    return m.group(1) if m else None


def _header_of(line):
    s = squash(line)
    for h in _TDS_HEADERS:
        if s.startswith(squash(h)):
            rest = re.sub(r'^\s*' + r'\s*'.join(map(re.escape, h.replace(' ', ''))) + r'\s*:?\s*', '', line, flags=re.I)
            return ('Packaging' if h == 'Package' else h), rest
    return None, None


def tds_sections(pages):
    """{section heading: text} following the sheet's own headings."""
    sections, current = {}, None
    for page in pages:
        for line in normalize(page).split('\n'):
            line = line.strip()
            if not line or _LETTERHEAD.match(line) or _SKIP.match(line):
                continue
            header, rest = _header_of(line)
            if header and header != current:
                current = _MERGED.get(header, header)
                sections.setdefault(current, [])
                if header in _MERGED:                  # keep the merged sub-heading visible
                    rest = f'{header}: {rest}'.strip()
                if rest:
                    sections[current].append(rest)
            elif current:
                sections[current].append(line)
    return {h: ' '.join(v).strip() for h, v in sections.items() if ' '.join(v).strip()}


def tds_fragments(doc_id, path, pages, desplit):
    """Section fragments plus a few groupings of neighbouring sections, all
    verbatim: the page-1 product card, Application+Dosage and Function+Dosage
    (a dosage line alone is too short to be matched), and Product Description
    merged into Effective material (alone it only repeats the header)."""
    fichier = Path(path).name
    sec = {h: desplit(b) for h, b in tds_sections(pages).items()}
    text = normalize('\n'.join(pages))
    last_update = re.search(r'Last updat\w*\s*:?\s*[\d/]+', text)
    if last_update and 'Storage' in sec:                # the date closes the Storage block
        sec['Storage'] = f"{sec['Storage']} {last_update.group(0)}"
    letterhead = ' '.join(l.strip() for l in text.split('\n')[:12] if _LETTERHEAD.match(l.strip()))

    enzyme = enzyme_name(sec.get('Product Description', '') + ' ' + sec.get('Effective material', ''))
    produit = f'{product_name(path)} ({enzyme})'
    key = lambda *p: ':'.join([str(doc_id), *p])
    out = []
    merge_identity = 'Product Description' in sec and 'Effective material' in sec
    for h, body in sec.items():
        if merge_identity and h == 'Product Description':
            continue
        if merge_identity and h == 'Effective material':
            parts = [('Product Description', sec['Product Description']), (h, body)]
            out.append(fragment(doc_id, fichier, produit, parts, 'en', key('identity'), {'Product Description', h}))
        else:
            out.append(fragment(doc_id, fichier, produit, [(h, body)], 'en', key(h), {h}))
    card = [(h, sec[h]) for h in PAGE1 if h in sec]
    if card:
        out.append(fragment(doc_id, fichier, produit, card, 'en', key('card'), {h for h, _ in card}))
    if 'Dosage' in sec:
        for group in (('Application', 'Dosage'), ('Function', 'Dosage')):
            parts = [(h, sec[h]) for h in group if h in sec]
            if len(parts) == 2:
                out.append(fragment(doc_id, fichier, produit, parts, 'en', key('+'.join(group)), set(group)))
    if letterhead:                                      # identical on every sheet: one content key
        out.append(fragment(doc_id, fichier, 'TECHNICAL DATA SHEET', [(None, letterhead)], 'en',
                            f'letterhead:{squash(letterhead)}', {'letterhead'}))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENTS WITH THEIR OWN HEADINGS (e.g. the French ascorbic-acid sheet)
# ═══════════════════════════════════════════════════════════════════════════

_FR_HEADINGS = [
    'Résumé Général', 'Propriétés Principales', 'Points Importants',
    'Dosages Recommandés', 'Table de Conversion Rapide', 'Spécifications Techniques',
    'Caractéristiques du Produit', 'Conditionnement Recommandé',
    "Mode d'Emploi en Production", 'Points de Contrôle', 'Avantages et Limitations',
    'Avantages', 'Limitations', 'Alternatives et Complémentarité', 'Réglementation',
    'Statut Légal', 'Dosage Maximum Autorisé', 'Recommandations pour ta Production',
    'Test et Validation', 'Stockage et Sécurité', 'Références',
]
_TABLE_LABELS = {'Type de Production': 'Dosages Recommandés (ppm = g/tonne de farine)',
                 'Poids Farine': 'Table de Conversion Rapide (en grammes)',
                 'Alternative': 'Alternatives et Complémentarité'}


def heading_sections(pages, title_lines=2):
    """[(heading, [lines])] using known headings; a heading with no text of its
    own becomes the parent of the next one ('Réglementation > Statut Légal')."""
    lines = [l.strip() for l in normalize('\n'.join(pages)).split('\n') if l.strip()]
    keys = [squash(h) for h in _FR_HEADINGS]
    sections = []
    for line in lines[title_lines:]:
        s = squash(line)
        if any(s.startswith(k) and len(s) <= len(k) + 12 for k in keys):
            if sections and not sections[-1][1]:
                sections[-1][0] += ' > ' + line
            else:
                sections.append([line, []])
        elif sections:
            sections[-1][1].append(line)
    return [(h, ls) for h, ls in sections if ls]


def table_rows(path):
    """Tables linearised row by row with their column headings."""
    rows_out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables():
                rows = [[normalize(c or '').replace('\n', ' ').strip() for c in r] for r in tbl]
                header = rows[0]
                label = _TABLE_LABELS.get(header[0], header[0])
                merged = []
                for r in rows[1:]:
                    if merged and not r[0]:             # continuation of a wrapped cell
                        merged[-1] = [f'{a} {b}'.strip() for a, b in zip(merged[-1], r)]
                    else:
                        merged.append(r)
                for r in merged:
                    cells = [f'{h}: {v}' if h else v for h, v in zip(header, r) if v]
                    rows_out.append((label, ' | '.join(cells)))
    return rows_out


def document_fragments(doc_id, path, pages, max_chars=700):
    """Section fragments (long sections split on line breaks); tables kept whole
    and also one fragment per row (a shown table hides its rows)."""
    fichier = Path(path).name
    lines = [l.strip() for l in normalize('\n'.join(pages)).split('\n') if l.strip()]
    produit = lines[0] if lines and len(lines[0]) < 80 else Path(path).stem
    key = lambda *p: ':'.join([str(doc_id), *p])
    rows = table_rows(path)
    row_ids = {}
    for i, (label, _) in enumerate(rows):
        row_ids.setdefault(label.split(' (')[0], []).append(f'row{i}')
    out = []
    for heading, body in heading_sections(pages):
        covers = {heading} | set(row_ids.get(heading.split(' (')[0], []))
        part = []
        for line in body + [None]:
            if line is None or (part and len(' '.join(part + [line])) > max_chars):
                if part:
                    out.append(fragment(doc_id, fichier, produit, [(heading, ' '.join(part))], 'fr',
                                        key(heading, str(len(out))), covers))
                part = []
            if line is not None:
                part.append(line)
    for i, (label, content) in enumerate(rows):
        out.append(fragment(doc_id, fichier, produit, [(label, content)], 'fr', key(f'row{i}'), {f'row{i}'}))
    if not out:                                         # unknown layout: plain paragraphs
        text = re.sub(r'\s+', ' ', ' '.join(lines[1:]))
        for i, chunk in enumerate(re.findall(r'.{1,%d}(?:[.!?](?=\s)|$)' % max_chars, text)):
            if chunk.strip():
                out.append(fragment(doc_id, fichier, produit, [(None, chunk.strip())], 'fr', key(f'p{i}'), {f'p{i}'}))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def build_fragments(pdf_folder, english_vocab):
    """All fragments of all PDFs in pdf_folder (without translations)."""
    pdfs = sorted(Path(pdf_folder).glob('*.pdf'))
    pages = {p: extract_pages(p) for p in pdfs}
    desplit = Desplitter([normalize(t) for ps in pages.values() for t in ps], english_vocab)
    documents, frags = [], []
    for doc_id, p in enumerate(pdfs, 1):
        documents.append((doc_id, p.name))
        if len(tds_sections(pages[p])) >= 3:
            out = tds_fragments(doc_id, p, pages[p], desplit)
        else:
            out = document_fragments(doc_id, p, pages[p])
        logger.info(f'{p.name}: {len(out)} fragments')
        frags += out
    return documents, frags
