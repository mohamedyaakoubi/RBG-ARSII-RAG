"""
Extraction and chunking routes, compared as pre-registered in ROUTES_PLAN.md.

    python -m audit.routes extractors            # every extractor, current chunking
    python -m audit.routes chunking <extractor>  # every chunking route
    python -m audit.routes test5 <extractor> <route>   # one-shot confirmation

The pipeline is the experiment code of the current system (audit/inrules.py,
corpus v2, strict mode S and transparent mode with product filter S+F). A
route only changes where the page text comes from, or which fragments are
built from it; everything else is identical. Dev pool: DEV, TEST, TEST-2,
TEST-3 and TEST-4 (all already seen).
"""
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

from audit import inrules, pipelines
from audit.eval_set import DEV, TEST
from audit.eval_set_v2 import TEST2
from audit.eval_set_v3 import TEST3
from audit.eval_set_v4 import TEST4
from audit.eval_set_v5 import TEST5
from audit.extractors import load

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'audit' / 'results'
DEV_POOL = DEV + TEST + TEST2 + TEST3 + TEST4
inrules.CACHE = ROOT / 'cache' / 'extractions' / 'translations.json'    # keep the committed cache as it is
_ORIGINAL_EXTRACT = pipelines.extract_pages


def use_extractor(name):
    """Make the whole pipeline read page text from `name` (None = pdfplumber, as committed).
    French tables are always read by pdfplumber (plan)."""
    if name is None:
        fn = _ORIGINAL_EXTRACT
    else:
        pages = load(name)
        fn = lambda path, x_tolerance=None: pages[Path(path).name]
    pipelines.extract_pages = fn
    inrules.extract_pages = fn
    inrules._WORDS = None                 # split-word repair: vocabulary from this extractor's text


def outcomes(index, questions, mode):
    """Per answerable question: (fully answered, right answer ranked 1st or None for multi)."""
    out = {}
    for q in questions:
        if not q['answerable']:
            continue
        res = inrules.answer(index, q['q'], **inrules.MODES[mode])
        flags = [[inrules.is_relevant(t, f) for f, _, _ in res] for t in q['targets']]
        full = all(any(fl) for fl in flags)
        first = flags[0][:1] == [True] if len(q['targets']) == 1 else None
        out[q['id']] = (full, first)
    return out


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (only a, only b, p)."""
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p


def run(frags, questions):
    index = inrules.Index(frags)
    return {m: outcomes(index, questions, m) for m in ('S', 'S+F')}, len(frags)


def table(results, base, questions, title):
    ids = [q['id'] for q in questions if q['answerable']]
    lines = [f'## {title}', '',
             '| route | fragments | fully answered, transparent (S+F) | vs current: gained / lost / p | '
             'fully answered, strict (S) | vs current: gained / lost / p | right answer 1st (S+F, single) |',
             '|---|---:|---:|---|---:|---|---:|']
    for name, (res, n_frags) in results.items():
        cells = []
        for m in ('S+F', 'S'):
            full = [res[m][i][0] for i in ids]
            ref = [base[m][i][0] for i in ids]
            g, l, p = mcnemar(full, ref)
            cells.append((sum(full), f'+{g} / -{l} / {p:.2f}'))
        first = [res['S+F'][i][1] for i in ids if res['S+F'][i][1] is not None]
        lines.append(f'| {name} | {n_frags} | {cells[0][0]}/{len(ids)} | {cells[0][1]} | '
                     f'{cells[1][0]}/{len(ids)} | {cells[1][1]} | {sum(first)}/{len(first)} |')
    return lines


def _load(stem):
    """Stored results {name: (outcomes, fragments)} of an earlier run."""
    path = RESULTS / f'{stem}.json'
    if not path.exists():
        return {}
    return {k: ({m: {i: tuple(x) for i, x in r.items()} for m, r in v['outcomes'].items()}, v['fragments'])
            for k, v in json.loads(path.read_text()).items()}


def _save(stem, results):
    (RESULTS / f'{stem}.json').write_text(json.dumps({k: {'outcomes': v[0], 'fragments': v[1]}
                                                      for k, v in results.items()}))


def extractors_study(names, merge=False):
    results = _load('routes_extractors') if merge else {}
    for name in names:
        use_extractor(None if name == 'pdfplumber-1.5 (current)' else name)
        results[name] = run(inrules.build(**inrules.FINAL_CORPUS_V2), DEV_POOL)
        print(name, {m: sum(v[0] for v in r.values()) for m, r in results[name][0].items()}, flush=True)
    use_extractor(None)
    base = results['pdfplumber-1.5 (current)'][0]
    out = ['# Routes: extractors (dev pool)', '',
           'Current chunking (corpus v2), every extractor. Dev pool = DEV + TEST + TEST-2 + TEST-3 + TEST-4, '
           'all already seen: this selects, it does not prove. "Fully answered" = right answer in the top 3, or '
           'every named product covered. p = exact McNemar against the current extractor.', '']
    out += table(results, base, DEV_POOL, 'Retrieval with each extractor')
    (RESULTS / 'routes_extractors.md').write_text('\n'.join(out) + '\n')
    _save('routes_extractors', results)
    print('\n'.join(out))


# ── Chunking routes ────────────────────────────────────────────────────────
# Additive routes keep the current fragments and add smaller ones; a shown
# fragment hides the smaller fragments it contains (same rule as a table and
# its rows). Replacing routes cut each document's lines without its structure.

SPEC = ('Microbiology', 'Heavy metals', 'Organoleptic', 'GMO status', 'Allergens')


def _sheets():
    return [p for p in sorted(inrules.PDF_DIR.glob('*.pdf')) if inrules.doc_key(p.name) != 'aa']


def _doc_lines(path):
    """Non-empty lines of the document as the current extractor gives them."""
    text = pipelines.normalize('\n'.join(inrules.extract_pages(path, x_tolerance=1.5)))
    return [l.strip() for l in text.split('\n') if l.strip()]


def section_lines(path):
    """{section: [lines]} with the same parsing as tds_sections, lines kept."""
    lines = [l for l in _doc_lines(path) if not pipelines._BOILERPLATE.match(l)]
    sections, current = {}, None
    for line in lines:
        header, rest = pipelines._match_header(line)
        if header and header != current:
            current = pipelines._MERGE.get(header, header)
            sections.setdefault(current, [])
            if header in pipelines._MERGE:
                rest = f'{header}: {rest}'.strip()
            if rest:
                sections[current].append(rest)
        elif current:
            sections[current].append(line)
    return {h: [inrules.desplit(l) for l in v] for h, v in sections.items() if v}


def sheet_header(path):
    sec = inrules.tds_sections(inrules.extract_pages(path, x_tolerance=1.5))
    return f"{inrules.product_name(path)} ({inrules.enzyme_name(sec.get('Product Description', '') + ' ' + sec.get('Effective material', ''))})"


def items_of(lines):
    """A new item starts at a capitalised 'Label:' line; other lines continue
    it. A label with no value of its own is kept with the next item."""
    items = []
    for line in lines:
        new = re.match(r'[A-Z][^:]{0,40}:', line) is not None
        if items and (not new or re.fullmatch(r'[^:]+:', items[-1])):
            items[-1] = f'{items[-1]} {line}'
        else:
            items.append(line)
    return items


def sentences_of(text):
    """Sentences, bullets and table cells, as segmented for translation; a
    piece shorter than 3 words is kept with the next."""
    pieces = [p.strip() for p in re.split(r'(?<=[.;])\s+(?=[A-Z0-9*(\-])|\s+\|\s+|\s+(?=- )', text) if p.strip()]
    out = []
    for p in pieces:
        if out and len(re.findall(r'\w+', out[-1])) < 3:
            out[-1] = f'{out[-1]} {p}'
        else:
            out.append(p)
    return [re.sub(r'^(?:-\s+)+', '- ', p) for p in out]      # a run of bullet glyphs is one bullet


def with_parts(frags, extra, tag=''):
    """Add fragments `extra` = {(doc, section, header, language): [parts]}:
    each part becomes a fragment, and every fragment covering `section` now
    also covers its parts."""
    new, ids = [], {}
    for (doc, section, header, lang), parts in extra.items():
        if len(parts) < 2:
            continue
        pid = [f'{section}#{tag}{i}' for i in range(len(parts))]
        ids[(doc, section)] = set(pid)
        for i, part in zip(pid, parts):
            new.append(inrules.frag(doc, header, [(section, part)], lang, (doc, i), {i}))
    out = []
    for f in frags:
        add = set().union(*[ids.get((f['doc'], s), set()) for s in f['covers']])
        out.append({**f, 'covers': f['covers'] | add} if add else f)
    return out, new


def route_items(frags, sections=SPEC):
    extra = {}
    for path in _sheets():
        doc, header = inrules.doc_key(path.name), sheet_header(path)
        for s, lines in section_lines(path).items():
            if sections is None or s in sections:
                extra[(doc, s, header, 'en')] = items_of(lines)
    frags, new = with_parts(frags, extra)
    return frags + new


def route_sentences(frags):
    extra = {}
    for path in _sheets():
        doc, header = inrules.doc_key(path.name), sheet_header(path)
        for s, lines in section_lines(path).items():
            extra[(doc, s, header, 'en')] = sentences_of(' '.join(lines))
    for f in frags:                                       # the French document, by its own sections
        if f['doc'] == 'aa' and f['lang'] == 'fr' and len(f['parts']) == 1 and not f['key'][1].startswith('row'):
            label, content = f['parts'][0]
            extra[('aa', label, f['header'], 'fr')] = sentences_of(content)
    frags, new = with_parts(frags, extra, tag='s')
    return frags + new + inrules.translated([f for f in new if f['lang'] == 'fr'])


ADDITIVE = ('items-spec', 'items-all', 'unmerge', 'sentences', 'fr-350', 'fr-1000')


def build_combo(parts):
    """Several additive routes at once ('combo:unmerge+items-spec')."""
    base = dict(inrules.FINAL_CORPUS_V2)
    merge, original = pipelines._MERGE, pipelines.faithful_fr_chunks
    try:
        if 'unmerge' in parts:
            pipelines._MERGE = {}
        size = next((int(p.split('-')[1]) for p in parts if p.startswith('fr-')), None)
        if size:
            pipelines.faithful_fr_chunks = lambda path, max_chars=size: original(path, max_chars=size)
            inrules.faithful_fr_chunks = pipelines.faithful_fr_chunks
        frags = inrules.build(**base)
        if 'items-spec' in parts or 'items-all' in parts:
            frags = route_items(frags, sections=None if 'items-all' in parts else SPEC)
        if 'sentences' in parts:
            frags = route_sentences(frags)
        return frags
    finally:
        pipelines._MERGE = merge
        pipelines.faithful_fr_chunks = inrules.faithful_fr_chunks = original


def build_route(route):
    """Fragments of the corpus under a chunking route (current extractor)."""
    base = dict(inrules.FINAL_CORPUS_V2)
    if route.startswith('combo:'):
        return build_combo(route[len('combo:'):].split('+'))
    if route == 'current':
        return inrules.build(**base)
    if route == 'items-spec':
        return route_items(inrules.build(**base))
    if route == 'items-all':
        return route_items(inrules.build(**base), sections=None)
    if route == 'unmerge':
        merge, pipelines._MERGE = pipelines._MERGE, {}
        try:
            return inrules.build(**base)
        finally:
            pipelines._MERGE = merge
    if route == 'sentences':
        return route_sentences(inrules.build(**base))
    if route in ('fr-350', 'fr-1000'):
        size = int(route.split('-')[1])
        original = pipelines.faithful_fr_chunks
        pipelines.faithful_fr_chunks = lambda path, max_chars=size: original(path, max_chars=size)
        inrules.faithful_fr_chunks = pipelines.faithful_fr_chunks
        try:
            return inrules.build(**base)
        finally:
            pipelines.faithful_fr_chunks = inrules.faithful_fr_chunks = original
    return structureless(route)


def _units(path):
    """(header, language, lines) of one document for the structure-agnostic routes."""
    if inrules.doc_key(path.name) == 'aa':
        return 'Acide Ascorbique (E300)', 'fr', _doc_lines(path)
    return sheet_header(path), 'en', [inrules.desplit(l) for l in _doc_lines(path)]


def structureless(route):
    """windows-N (N lines, half overlapping), fixed-128 (128 word pieces
    overlapping by 64), semantic (a cut where consecutive lines' similarity
    drops below the document's 25th percentile, at most 12 lines)."""
    frags = []
    tok = inrules.MODEL.tokenizer
    for path in sorted(inrules.PDF_DIR.glob('*.pdf')):
        doc = inrules.doc_key(path.name)
        header, lang, lines = _units(path)
        if route.startswith('windows-'):
            n = int(route.split('-')[1])
            spans = [(i, min(i + n, len(lines))) for i in range(0, max(len(lines) - n, 0) + 1, max(n // 2, 1))]
        elif route == 'fixed-128':
            ids = [(li, t) for li, l in enumerate(lines) for t in tok.tokenize(l)]
            spans = []
            for start in range(0, max(len(ids) - 64, 1), 64):
                chunk = ids[start:start + 128]
                spans.append((chunk[0][0], chunk[-1][0] + 1))
            spans = sorted(set(spans))
        elif route == 'semantic':
            v = inrules.embed_all(lines)
            sims = (v[:-1] * v[1:]).sum(axis=1)
            cut = np.percentile(sims, 25) if len(sims) else 0
            spans, start = [], 0
            for i, s in enumerate(sims, 1):
                if s < cut or i - start >= 12:
                    spans.append((start, i))
                    start = i
            spans.append((start, len(lines)))
        else:
            raise ValueError(route)
        for a, b in spans:
            frags.append(inrules.frag(doc, header, [(None, ' '.join(lines[a:b]))], lang, (doc, route, a, b),
                                      {f'L{i}' for i in range(a, b)}))
    return frags + inrules.translated([f for f in frags if f['lang'] == 'fr'])


# ── Span oracle: an upper bound for any chunking of the verbatim text ──────

def _fr_heading(line):
    s = pipelines.squash(line)
    return any(s.startswith(k) and len(s) <= len(k) + 12 for k in map(pipelines.squash, pipelines._FR_HEADERS))


def doc_spans(path, max_lines=60):
    """Every run of consecutive lines of the document, as a fragment with the
    product header, also preceded by its section heading when it starts
    inside a section; French runs also in English (line by line translation).
    [(doc, text, original text)]"""
    doc = inrules.doc_key(path.name)
    header, lang, lines = _units(path)
    heading = _fr_heading if lang == 'fr' else (lambda l: pipelines._match_header(l)[0] is not None)
    versions = [(header, lines, lines)]
    if lang == 'fr':
        versions.append(('Ascorbic Acid (E300)', inrules.translate(lines, 'fr-en'), lines))
    out = []
    for head, shown, orig in versions:
        last = None
        heads = []
        for i, l in enumerate(orig):
            if heading(l):
                last = i
            heads.append(last)
        for i in range(len(shown)):
            for j in range(i + 1, min(len(shown), i + max_lines) + 1):
                body, body_o = ' '.join(shown[i:j]), ' '.join(orig[i:j])
                out.append((doc, f'{head} - {body}', f'{header} - {body_o}'))
                h = heads[i]
                if h is not None and h < i:
                    out.append((doc, f'{head} - {shown[h]}: {body}', f'{header} - {orig[h]}: {body_o}'))
    return out


def span_oracle(frags, questions, modes=('S', 'S+F')):
    """For each single-answer question the current index misses: the best
    score any verbatim span containing the answer could get, against the score
    it would need (the 3rd result shown). Returns {mode: [(id, need, best, span)]}."""
    index = inrules.Index(frags)
    spans = [s for p in sorted(inrules.PDF_DIR.glob('*.pdf')) for s in doc_spans(p)]
    mat = inrules.embed_all([t for _, t, _ in spans])
    by_doc = {}
    for k, (doc, _, _) in enumerate(spans):
        by_doc.setdefault(doc, []).append(k)
    out = {}
    for mode in modes:
        rows = []
        for q in questions:
            if not q['answerable'] or len(q['targets']) != 1:
                continue
            t = q['targets'][0]
            res = inrules.answer(index, q['q'], **inrules.MODES[mode])
            if any(inrules.is_relevant(t, f) for f, _, _ in res):
                continue
            need = res[2][1] if len(res) >= 3 else -1.0
            variants = inrules.question_variants(q['q'], inrules.MODES[mode].get('translate_q', False))
            qv = inrules.embed_all(variants)
            cand = [k for d in t['docs'] for k in by_doc.get(d, [])
                    if inrules.relevant(t, d, spans[k][1]) or inrules.relevant(t, d, spans[k][2])]
            if not cand:
                rows.append((q['id'], need, None, None))
                continue
            scores = (mat[cand] @ qv.T).max(axis=1)
            best = int(np.argmax(scores))
            rows.append((q['id'], need, float(scores[best]), spans[cand[best]][1]))
        out[mode] = rows
    return out


def oracle_study(extractors, questions=DEV_POOL, name='dev pool', stem='routes_oracle'):
    lines = [f'# Span oracle ({name})', '',
             'For every single-answer question the current chunking misses: could ANY run of consecutive lines of '
             'the right document, with the product header (and optionally its section heading), enter the top 3, '
             'the rest of the index unchanged? "Fixable" = the best such fragment scores above the 3rd result.', '',
             '| extractor | mode | misses | fixable by some fragment | not fixable by any fragment |',
             '|---|---|---:|---:|---:|']
    detail, fixable_ids = [], {}
    for ex in extractors:
        use_extractor(None if ex == 'pdfplumber-1.5' else ex)
        res = span_oracle(inrules.build(**inrules.FINAL_CORPUS_V2), questions)
        for mode, rows in res.items():
            fix = [r for r in rows if r[2] is not None and r[2] > r[1]]
            fixable_ids.setdefault(mode, {})[ex] = {r[0] for r in fix}
            lines.append(f'| {ex} | {mode} | {len(rows)} | {len(fix)} | {len(rows) - len(fix)} |')
            detail += [f'- {ex} {mode} **{r[0]}**: needs {r[1]:.3f}, best span '
                       + (f'{r[2]:.3f} ({"fixable" if r[2] > r[1] else "not fixable"}): `{r[3][:160]}`' if r[2] is not None
                          else 'none contains the answer') for r in rows]
        print(ex, {m: (len(r), sum(1 for x in r if x[2] is not None and x[2] > x[1])) for m, r in res.items()}, flush=True)
    use_extractor(None)
    lines += ['', 'Across extractors (the best extractor for each question): '
              + ', '.join(f'{m}: {len(set().union(*v.values()))} fixable' for m, v in fixable_ids.items()), '',
              '## Every miss', ''] + detail
    (RESULTS / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:12 + 2 * len(extractors)]))


ROUTES = ['current', 'items-spec', 'items-all', 'unmerge', 'sentences', 'windows-3', 'windows-6',
          'fixed-128', 'semantic', 'fr-350', 'fr-1000']


def chunking_study(extractor, routes=ROUTES, merge=False):
    use_extractor(None if extractor == 'pdfplumber-1.5' else extractor)
    results = _load('routes_chunking') if merge else {}
    for route in routes:
        results[route] = run(build_route(route), DEV_POOL)
        print(route, {m: sum(v[0] for v in r.values()) for m, r in results[route][0].items()}, flush=True)
    base = results['current'][0]
    out = ['# Routes: chunking (dev pool)', '',
           f'Extractor: {extractor}. Every chunking route of ROUTES_PLAN.md; p = exact McNemar against the '
           'current chunking with the same extractor. Dev pool, all already seen: this selects, it does not prove.', '']
    out += table(results, base, DEV_POOL, 'Retrieval with each chunking route')
    (RESULTS / 'routes_chunking.md').write_text('\n'.join(out) + '\n')
    _save('routes_chunking', results)
    print('\n'.join(out))


def test5(extractor, route, explore_extractors, explore_routes):
    """One-shot confirmation on TEST-5: the selected (extractor, route) against
    the current system, then every other candidate as exploration, then the
    span oracle for both configurations."""
    ids = [q['id'] for q in TEST5 if q['answerable']]
    configs = {'current (pdfplumber-1.5, current chunking)': ('pdfplumber-1.5', 'current'),
               f'selected ({extractor}, {route})': (extractor, route)}
    configs.update({f'{e}, current chunking': (e, 'current') for e in explore_extractors})
    configs.update({f'{extractor}, {r}': (extractor, r) for r in explore_routes})
    results, done = {}, {}
    for name, (ex, r) in configs.items():
        if (ex, r) not in done:
            use_extractor(None if ex == 'pdfplumber-1.5' else ex)
            done[(ex, r)] = run(build_route(r), TEST5)
            print(name, {m: sum(v[0] for v in res.values()) for m, res in done[(ex, r)][0].items()}, flush=True)
        results[name] = done[(ex, r)]
    use_extractor(None)
    cur, sel = list(configs)[:2]
    out = ['# TEST-5: one-shot confirmation', '',
           'TEST-5 was frozen with ROUTES_PLAN.md (commit 4e21bf7) before any route ran. The selected configuration '
           'was chosen on the dev pool by the pre-registered rule. Primary test: fully answered questions in '
           'transparent mode (S+F), selected against current, exact McNemar.', '']
    for m in ('S+F', 'S'):
        a = [results[sel][0][m][i][0] for i in ids]
        b = [results[cur][0][m][i][0] for i in ids]
        g, l, p = mcnemar(a, b)
        out.append(f'- **{m}**: current {sum(b)}/{len(ids)}, selected {sum(a)}/{len(ids)}; '
                   f'selected gains {g}, loses {l}; p = {p:.3f}')
    out += [''] + table(results, results[cur][0], TEST5, 'Every candidate on TEST-5 (exploratory except the first two rows)')
    (RESULTS / 'routes_test5.md').write_text('\n'.join(out) + '\n')
    _save('routes_test5', results)
    print('\n'.join(out))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'test5':
        ex, rt = sys.argv[2], sys.argv[3]
        from audit.extractors import EXTRACTORS
        others = [e for e in list(EXTRACTORS) + ['docling'] if e not in ('pdfplumber-1.5', ex)
                  and (ROOT / 'cache' / 'extractions' / f'{e}.json').exists()]
        test5(ex, rt, others, [r for r in ROUTES if r not in ('current', rt)])
    if cmd == 'chunking':
        chunking_study(sys.argv[2], sys.argv[3:] or ROUTES, merge=bool(sys.argv[3:]))
    if cmd == 'oracle':
        oracle_study(sys.argv[2:])
    if cmd == 'extractors':
        from audit.extractors import EXTRACTORS
        names = sys.argv[2:] or ['pdfplumber-1.5 (current)'] + [
            n for n in list(EXTRACTORS) + ['docling']
            if n != 'pdfplumber-1.5' and (ROOT / 'cache' / 'extractions' / f'{n}.json').exists()]
        extractors_study(names, merge=bool(sys.argv[2:]))
