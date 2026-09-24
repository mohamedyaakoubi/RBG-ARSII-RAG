"""
In-rules optimisation: what can be changed inside the challenge's rules to
get more right answers, without inventing data?

Fixed by the brief: all-MiniLM-L6-v2 embeds the question, cosine similarity
against stored vectors, results ranked by score, top-3 fragments shown with
text and score.

Free (and used here), always content-preserving:
  - how the PDFs are turned into fragments (extraction, cleaning, granularity);
  - bilingual fragments: a machine translation of a fragment is stored as its
    own fragment and displayed together with the original. Translation is done
    sentence by sentence with product names masked and section labels taken
    from a fixed glossary; it is rejected if any sentence changes a number, a
    product name, or its length implausibly;
  - collapsing duplicates (the same content in two languages or granularities).

Question-side preprocessing (translation to English, one sub-question per
product) is a separate, clearly labelled level ("S+"): the question is still
embedded by all-MiniLM-L6-v2, but a transformed version of it.

Everything is scored in memory with NumPy (same cosine as pgvector), so many
configurations can be compared on the development pool (DEV + TEST).
"""
import json
import logging
import re
from collections import Counter
from pathlib import Path

logging.disable(logging.CRITICAL)

import numpy as np
from sentence_transformers import SentenceTransformer

from audit.eval_set import doc_key, relevant
from audit.pipelines import (_BOILERPLATE, _FR_TABLE_SECTIONS, _fr_table_rows, enzyme_name,
                             extract_pages, faithful_fr_chunks, normalize, product_name,
                             tds_sections)

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / 'data_pdf'
CACHE = ROOT / 'audit' / 'results' / 'translations.json'
TOP_K = 3

MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# ── Fragments ──────────────────────────────────────────────────────────────
# A fragment = header + labelled parts. Its text (what is embedded AND shown)
# is "header - label: content label: content ...". `key` identifies the
# content (an original and its translation share it); `covers` lists the
# section ids it contains, to collapse duplicates at search time.

def frag(doc, header, parts, lang, key, covers, orig=None):
    text = f'{header} - ' + ' '.join(f'{label}: {content}' for label, content in parts)
    return {'doc': doc, 'header': header, 'parts': parts, 'text': text, 'lang': lang,
            'key': key, 'covers': frozenset(covers), 'orig': orig or text}


PAGE1 = ('Product Description', 'Effective material', 'Activity', 'Application',
         'Function', 'Dosage', 'Organoleptic')


_WORDS = None


def _known_words():
    """Whole words of the embedding model's (English) vocabulary, plus words
    that occur unsplit somewhere in the corpus."""
    corpus = Counter()
    for p in PDF_DIR.glob('*.pdf'):
        for page in extract_pages(p, x_tolerance=1.5):
            corpus.update(re.findall(r'[A-Za-zÀ-ÿ]+', normalize(page).lower()))
    english = {w for w in MODEL.tokenizer.vocab if w.isalpha() and len(w) > 1}
    return english, english | {w for w, n in corpus.items() if n >= 2 and len(w) >= 3}


def desplit(text):
    """Re-join a word the PDF extraction cut in two ('Amyloglu cosidase',
    'applicatio ns', 'rang e', 'pp m') when the joined form is a known word and
    one piece is not a word on its own. Nothing else is changed."""
    global _WORDS
    if _WORDS is None:
        _WORDS = _known_words()
    english, known = _WORDS
    piece = lambda w: w not in english and w not in ('a', 'i')
    toks = text.split(' ')
    words = [re.sub(r'\W', '', t).lower() for t in toks]

    def joinable(i):
        if i + 1 >= len(toks):
            return 0
        a, b = words[i], words[i + 1]
        ok = a.isalpha() and b.isalpha() and (a + b) in known and (piece(a) or piece(b))
        return len(a + b) if ok else 0

    out, i = [], 0
    while i < len(toks):
        here = joinable(i)
        if here and joinable(i + 1) <= here:          # prefer the longer word when joins compete
            out.append(toks[i] + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return ' '.join(out)


def tds_fragments(path, card=False, docinfo=False, fix_splits=False, usage=False):
    """Section fragments, plus optionally:
    card    - page 1 of the sheet as one fragment (identity, use, dosage...);
    usage   - Application + Dosage together (how the product is used, how much);
    docinfo - the letterhead (issuer, addresses) and the update date, headed by
              the product code only so it does not match enzyme questions."""
    pages = extract_pages(path, x_tolerance=1.5)
    sec = tds_sections(pages)
    if fix_splits:
        sec = {h: desplit(b) for h, b in sec.items()}
    doc = doc_key(path.name)
    enzyme = enzyme_name(sec.get('Product Description', '') + ' ' + sec.get('Effective material', ''))
    header = f'{product_name(path)} ({enzyme})'
    out = [frag(doc, header, [(h, b)], 'en', (doc, h), {h}) for h, b in sec.items()]
    if card:
        parts = [(h, sec[h]) for h in PAGE1 if h in sec]
        out.append(frag(doc, header, parts, 'en', (doc, 'card'), {h for h, _ in parts}))
    if usage and 'Dosage' in sec:
        parts = [(h, sec[h]) for h in ('Application', 'Dosage') if h in sec]
        out.append(frag(doc, header, parts, 'en', (doc, 'usage'), {h for h, _ in parts}))
    if docinfo:
        text = normalize('\n'.join(pages))
        lines = [l.strip() for l in text.split('\n')[:12]
                 if _BOILERPLATE.match(l.strip()) and not re.match(r'(TECHNICAL|FOOD|Bakery)', l.strip(), re.I)]
        last = re.search(r'Last updat\w*\s*:?\s*[\d/]+', text)
        info = ' '.join(lines) + (f' {last.group(0)}' if last else '')
        out.append(frag(doc, product_name(path), [('Document', info)], 'en', (doc, 'Document'), {'Document'}))
    return out


def aa_fragments(path, rows=True):
    """rows: False = tables stay whole; True = one fragment per table row;
    'both' = whole tables and their rows (a shown table hides its rows)."""
    prefix = 'Acide Ascorbique (E300)'
    table_rows = {}
    if rows:
        for i, r in enumerate(_fr_table_rows(path, prefix)):
            label, content = r[len(prefix) + 3:].split(': ', 1)
            table_rows.setdefault(label.split(' (')[0], []).append((f'row{i}', label, content))
    out = []
    for c in faithful_fr_chunks(path):
        label, content = c[len(prefix) + 3:].split(': ', 1)
        is_table = label.startswith(_FR_TABLE_SECTIONS)
        if is_table and rows is True:
            continue
        covers = {label} | ({rid for rid, _, _ in table_rows.get(label.split(' (')[0], [])} if is_table else set())
        out.append(frag('aa', prefix, [(label, content)], 'fr', ('aa', label), covers))
    if rows:                        # the ppm footnote is kept in each row's label
        for group in table_rows.values():
            for rid, label, content in group:
                out.append(frag('aa', prefix, [(label, content)], 'fr', ('aa', rid), {rid}))
    return out


# ── Translation: content-preserving and guarded ────────────────────────────

_MT = {}
_TRANS = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _mt(direction):
    if direction not in _MT:
        from transformers import MarianMTModel, MarianTokenizer
        name = f'Helsinki-NLP/opus-mt-{direction}'
        _MT[direction] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name))
    return _MT[direction]


def translate(texts, direction):
    todo = [t for t in dict.fromkeys(texts) if f'{direction}|{t}' not in _TRANS]
    if todo:
        tok, mt = _mt(direction)
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
            out = mt.generate(**enc, num_beams=4, max_new_tokens=256)
            for src, o in zip(batch, out):
                _TRANS[f'{direction}|{src}'] = tok.decode(o, skip_special_tokens=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(_TRANS, ensure_ascii=False, indent=0))
    return [_TRANS[f'{direction}|{t}'] for t in texts]


def numbers(s):
    """Multiset of numbers, ignoring separators ('0,5' == '0.5', '1 000' == '1,000')."""
    return sorted(re.sub(r'\D', '', n) for n in re.findall(r'\d+(?:[ ., ]\d+)*', s))


EN_FR_LABELS = {
    'Product Description': 'Description du produit', 'Effective material': 'Matière active',
    'Activity': 'Activité', 'Application': 'Application', 'Function': 'Fonction', 'Dosage': 'Dosage',
    'Organoleptic': 'Caractéristiques organoleptiques', 'Microbiology': 'Microbiologie',
    'Heavy metals': 'Métaux lourds', 'Allergens': 'Allergènes', 'GMO status': 'Statut OGM',
    'Packaging': 'Conditionnement', 'Storage': 'Stockage', 'Document': 'Document',
}
FR_EN_LABELS = {
    'Résumé Général': 'General summary', 'Propriétés Principales': 'Main properties',
    'Points Importants': 'Key points', 'Dosages Recommandés (ppm*)': 'Recommended dosages (ppm*)',
    'Dosages Recommandés (ppm = g/tonne de farine)': 'Recommended dosages (ppm = g/tonne of flour)',
    'Table de Conversion Rapide (en grammes)': 'Quick conversion table (in grams)',
    'Spécifications Techniques > Caractéristiques du Produit': 'Technical specifications > Product characteristics',
    'Conditionnement Recommandé': 'Recommended packaging', "Mode d'Emploi en Production": 'Directions for use in production',
    'Points de Contrôle': 'Control points', 'Avantages et Limitations > Avantages': 'Advantages and limitations > Advantages',
    'Limitations': 'Limitations', 'Alternatives et Complémentarité': 'Alternatives and complementarity',
    'Réglementation > Statut Légal': 'Regulation > Legal status', 'Dosage Maximum Autorisé': 'Maximum authorised dosage',
    'Recommandations pour ta Production > Test et Validation': 'Recommendations for your production > Testing and validation',
    'Stockage et Sécurité': 'Storage and safety', 'Références': 'References',
}
EN_FR_ENZYMES = {'maltogenic amylase': 'amylase maltogénique', 'glucose oxidase': 'glucose oxydase'}
MASK = 'PRODX'


_SEP = r'((?<=[.;])\s+(?=[A-Z0-9*(\-])|\s+\|\s+|\s+(?=- ))'


def _segments(content):
    """Sentences, table cells and bullets are translated one by one, so MT
    cannot silently drop a clause of a long input. Separators are kept."""
    pieces = re.split(_SEP, content)
    return [p for p in pieces[0::2] if p.strip()], [s if '|' in s else ' ' for s in pieces[1::2]]


def _join(segs, seps):
    out = segs[0]
    for sep, seg in zip(seps, segs[1:]):
        out += sep + seg
    return out


def _mask(text, product):
    if not product:
        return text, 0
    code = product.replace('BVZyme', '').replace(' ', '')
    pat = r'BVZ\s*y\s*m\s*e\s*' + r'\s*'.join(map(re.escape, code))
    return re.subn(pat, MASK, text, flags=re.I)


def _plan(f):
    """(segments to translate, how to rebuild the translated fragment)."""
    if f['lang'] == 'en':
        m = re.match(r'(.*) \((.*)\)$', f['header'])
        name, enzyme = m.groups() if m else (f['header'], None)
        header = f'{name} ({EN_FR_ENZYMES.get(enzyme, enzyme)})' if enzyme else name
        labels, direction, product = EN_FR_LABELS, 'en-fr', name
    else:
        header, labels, direction, product = 'Ascorbic Acid (E300)', FR_EN_LABELS, 'fr-en', None
    parts = []
    for label, content in f['parts']:
        masked, n_masks = _mask(content, product)
        segs, seps = _segments(masked)
        parts.append((labels.get(label, label), segs, seps, n_masks))
    return direction, header, product, parts


def translated(frags, rejected=None):
    plans = [(f, _plan(f)) for f in frags]
    for direction in ('en-fr', 'fr-en'):
        translate([s for _, (d, _, _, parts) in plans if d == direction for _, segs, _, _ in parts for s in segs], direction)
    out = []
    for f, (direction, header, product, parts) in plans:
        new_parts, ok = [], True
        for label, segs, seps, n_masks in parts:
            tsegs = translate(segs, direction)
            for s, t in zip(segs, tsegs):
                if numbers(s) != numbers(t) or (len(s) > 15 and not 0.5 <= len(t) / len(s) <= 2.5):
                    ok = False
            text = _join(tsegs, seps)
            if text.count(MASK) != n_masks:
                ok = False
            new_parts.append((label, text.replace(MASK, product or '')))
        if not ok:
            if rejected is not None:
                rejected.append(f['text'])
            continue
        out.append(frag(f['doc'], header, new_parts, 'fr' if f['lang'] == 'en' else 'en',
                        f['key'], f['covers'], orig=f['text']))
    return out


def build(card=False, docinfo=False, fix_splits=False, rows=True, usage=False, bilingual=False, rejected=None):
    """bilingual: False; True = translate every fragment into the other
    language; 'fr-en' = only the French document is also indexed in English
    (the embedding model's language)."""
    frags = []
    for p in sorted(PDF_DIR.glob('*.pdf')):
        if doc_key(p.name) == 'aa':
            frags += aa_fragments(p, rows=rows)
        else:
            frags += tds_fragments(p, card=card, docinfo=docinfo, fix_splits=fix_splits, usage=usage)
    if bilingual:
        source = [f for f in frags if bilingual is True or f['lang'] == 'fr']
        frags += translated(source, rejected)
    return frags


# ── Search ─────────────────────────────────────────────────────────────────

_EMB = {}


def embed_all(texts):
    todo = [t for t in dict.fromkeys(texts) if t not in _EMB]
    if todo:
        for t, v in zip(todo, MODEL.encode(todo, batch_size=64, normalize_embeddings=True)):
            _EMB[t] = v
    return np.stack([_EMB[t] for t in texts])


class Index:
    def __init__(self, frags):
        self.frags = frags
        self.mat = embed_all([f['text'] for f in frags])

    def search(self, queries, k=TOP_K, dedupe=True):
        """Rank fragments by cosine with the question (max over the given
        formulations of it); skip a fragment whose content is already shown."""
        qv = embed_all(list(queries))
        scores = (self.mat @ qv.T).max(axis=1)
        picked, shown = [], {}
        for i in np.argsort(-scores):
            f = self.frags[i]
            if dedupe:
                seen = shown.get(f['doc'], set())
                if f['covers'] <= seen:
                    continue
                shown[f['doc']] = seen | f['covers']
            picked.append((f, float(scores[i])))
            if len(picked) == k:
                break
        return picked


# ── Question-side preprocessing (level S+) ─────────────────────────────────

_LANGID = None


def is_french(q):
    global _LANGID
    if _LANGID is None:
        import langid
        langid.set_languages(['fr', 'en'])
        _LANGID = langid
    return _LANGID.classify(q)[0] == 'fr'


def question_variants(q, translate_q):
    """translate_q: False; True = keep the best of the question and its English
    translation; 'only' = use the English translation alone."""
    if translate_q and is_french(q):
        en = translate([q], 'fr-en')[0]
        return [en] if translate_q == 'only' else [q, en]
    return [q]


ENTITIES = [
    ('alpha-amylase', r"alpha[\s-]*amylases?|α[\s-]*amylases?|amylases?\s+fongiques?|fungal\s+(?:alpha[\s-]*)?amylases?"),
    ('maltogenic amylase', r"amylases?\s+maltog[ée]niques?|maltogenic\s+amylases?"),
    ('amyloglucosidase', r"amyloglucosidases?|glucoamylases?"),
    ('xylanase', r"xylanases?"),
    ('lipase', r"(?:phospho)?lipases?"),
    ('glucose oxidase', r"glucose[\s-]*ox[yi]dases?"),
    ('transglutaminase', r"transglutaminases?"),
    ('ascorbic acid', r"acides?\s+ascorbiques?|ascorbic\s+acid|vitamine?\s*C\b|E300"),
]
_GLUE = r"(?:\s|,|;|/|&|\+|\bet\b|\band\b|\bou\b|\bor\b|\bde\b|\bd['’]|\bdu\b|\bdes\b|\bla\b|\ble\b|\bl['’]|\bles\b|\bof\b|\bthe\b)*"


def split_by_entity(q):
    """One sub-question per product family named in the question, keeping the
    user's own wording (a coordinated list is reduced to one of its members)."""
    spans = sorted([(n, m) for n, p in ENTITIES for m in [re.search(p, q, re.I)] if m], key=lambda x: x[1].start())
    if len(spans) < 2:
        return [q]
    subs = []
    for keep, kept in spans:
        if all(re.fullmatch(_GLUE, q[a.end():b.start()], re.I) for (_, a), (_, b) in zip(spans, spans[1:])):
            s = q[:spans[0][1].start()] + kept.group(0) + q[spans[-1][1].end():]
        else:
            s = q
            for name, m in reversed(spans):
                if name != keep:
                    s = s[:m.start()] + s[m.end():]
        subs.append(re.sub(r'\s+', ' ', s).strip())
    return subs


def answer(index, q, translate_q=False, decompose=False, k=TOP_K):
    """Top-k fragments for question q: [(fragment, score, formulations used)]."""
    subs = split_by_entity(q) if decompose else [q]
    if len(subs) == 1:
        variants = question_variants(q, translate_q)
        return [(f, s, variants) for f, s in index.search(variants, k)]
    picked, keys = [], set()
    patterns = dict(ENTITIES)
    for (name, _), sq in zip(sorted([(n, m) for n, p in ENTITIES for m in [re.search(p, q, re.I)] if m],
                                    key=lambda x: x[1].start()), subs):
        # best fragment for this sub-question that is about the product family it names
        variants = question_variants(sq, translate_q)
        hits = index.search(variants, 10)
        about = [(f, s) for f, s in hits if re.search(patterns[name], f['header'], re.I)]
        for f, s in about or hits:
            if f['key'] not in keys:
                picked.append((f, s, variants))
                keys.add(f['key'])
                break
    variants = question_variants(q, translate_q)      # fill with the whole question
    for f, s in index.search(variants, 2 * k):
        if len(picked) >= k:
            break
        if f['key'] not in keys:
            picked.append((f, s, variants))
            keys.add(f['key'])
    return sorted(picked[:k], key=lambda x: -x[1])


# ── Configuration chosen on the development pool (frozen before TEST-2) ────

FINAL_CORPUS = dict(card=True, usage=True, rows='both', docinfo=True, fix_splits=True, bilingual='fr-en')
MODES = {
    'S': {},                                              # question embedded as typed
    'S+': dict(translate_q=True, decompose=True),         # + EN translation, one sub-question per product
}


# ── Evaluation ─────────────────────────────────────────────────────────────

def is_relevant(target, f):
    return relevant(target, f['doc'], f['orig']) or relevant(target, f['doc'], f['text'])


def evaluate(index, queries, **opts):
    single, multi, fr, en = [], [], [], []
    for q in queries:
        if not q['answerable']:
            continue
        res = answer(index, q['q'], **opts)
        if len(q['targets']) == 1:
            flags = [is_relevant(q['targets'][0], f) for f, _, _ in res]
            first = next((i for i, x in enumerate(flags, 1) if x), None)
            row = (flags[:1] == [True], first is not None, 1 / first if first else 0.0)
            single.append(row)
            (fr if q['lang'] == 'fr' else en).append(row[1])
        else:
            covered = sum(any(is_relevant(t, f) for f, _, _ in res) for t in q['targets'])
            multi.append(covered / len(q['targets']))
    m = lambda xs: round(float(np.mean(xs)), 3) if xs else None
    return {'hit@1': m([r[0] for r in single]), 'hit@3': m([r[1] for r in single]),
            'mrr@3': m([r[2] for r in single]), 'hit@3_fr': m(fr), 'hit@3_en': m(en),
            'multi_cov': m(multi), 'n_single': len(single), 'hits3': [r[1] for r in single]}
