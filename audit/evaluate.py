"""
Evaluate the original RAG system and strict / relaxed alternatives on the
frozen ground-truth set in audit/eval_set.py.

    python -m audit.evaluate                    # build tables, run everything
    python -m audit.evaluate --reuse            # reuse tables built by a previous run
    python -m audit.evaluate --explore --reuse  # post-hoc chunking variants (strict search)
    python -m audit.evaluate --beyond --reuse   # out-of-rules embedding model swap

Requires the PostgreSQL + pgvector database from docker-compose / .env.
Writes audit/results/{summary.md, metrics.json, per_query.json,
exploration.md, beyond_rules.md}.

"displayed" score = the score the system shows to the user.
"honest" score    = cosine(embedding of the user's question as typed,
                             embedding of the returned fragment's text),
                    i.e. what the challenge asks to display.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psycopg2

from config.settings import config
from services.embedding_service import model            # all-MiniLM-L6-v2, as imposed
import services.search_service as old_search_module
from services.ingestion_data import ingest_pdfs
from audit.eval_set import DEV, TEST, doc_key, relevant
from audit.pipelines import EXPLORATION, PIPELINES, build_table

TOP_K = config.TOP_K
RESULTS = ROOT / 'audit' / 'results'


def connect():
    return psycopg2.connect(host=config.DB_HOST, user=config.DB_USER,
                            password=config.DB_PASSWORD, dbname=config.DB_NAME)


CONN = connect()


def embed(text):
    return model.encode(text, normalize_embeddings=True)


_frag_cache = {}


def honest_score(question, text):
    if text not in _frag_cache:
        _frag_cache[text] = embed(text)
    return float(np.dot(embed(question), _frag_cache[text]))


def cosine_topk(table, qvec, k=TOP_K):
    cur = CONN.cursor()
    cur.execute(f'SELECT id_document, texte_fragment, 1 - (vecteur <=> %s::vector) AS score '
                f'FROM {table} ORDER BY vecteur <=> %s::vector LIMIT %s',
                (qvec.tolist(), qvec.tolist(), k))
    return [(d, t, float(s)) for d, t, s in cur.fetchall()]


# ── Query translation (relaxed variants only) ──────────────────────────────

_mt = {}


def mt_fr_en(text):
    """General-purpose MT (Helsinki-NLP opus-mt-fr-en), gated by language ID."""
    if 'model' not in _mt:
        import langid
        from transformers import MarianMTModel, MarianTokenizer
        langid.set_languages(['fr', 'en'])
        _mt['langid'] = langid
        _mt['tok'] = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        _mt['model'] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        _mt['cache'] = {}
    if _mt['langid'].classify(text)[0] != 'fr':
        return None
    if text not in _mt['cache']:
        batch = _mt['tok']([text], return_tensors='pt')
        out = _mt['model'].generate(**batch, num_beams=4, max_new_tokens=128)
        _mt['cache'][text] = _mt['tok'].decode(out[0], skip_special_tokens=True)
    return _mt['cache'][text]


def old_dict_fr_en(text):
    """The original system's hand-written regex dictionary."""
    return old_search_module._translate_fr_to_en(text) if old_search_module._is_french(text) else None


def bilingual_search(table, question, translate, k=TOP_K):
    """CombMAX fusion of the question as typed and its English translation."""
    best = {}
    for variant in filter(None, [question, translate(question)]):
        for d, t, s in cosine_topk(table, embed(variant), k):
            if t not in best or s > best[t][2]:
                best[t] = (d, t, s)
    return sorted(best.values(), key=lambda r: r[2], reverse=True)[:k]


# ── Entity decomposition (relaxed, intent-preserving) ──────────────────────
# Lexicon = the additive families present in the corpus (not query-derived).
_ENTITIES = [
    ('alpha-amylase', r"alpha[\s-]*amylases?|α[\s-]*amylases?|amylases?\s+fongiques?|fungal\s+(?:alpha[\s-]*)?amylases?"),
    ('maltogenic amylase', r"amylases?\s+maltog[ée]niques?|maltogenic\s+amylases?"),
    ('amyloglucosidase', r"amyloglucosidases?|glucoamylases?"),
    ('xylanase', r"xylanases?"),
    ('lipase', r"(?:phospho)?lipases?"),
    ('glucose oxidase', r"glucose[\s-]*ox[yi]dases?"),
    ('transglutaminase', r"transglutaminases?"),
    ('ascorbic acid', r"acides?\s+ascorbiques?|ascorbic\s+acid|vitamine?\s*C\b|E300"),
]


def entity_spans(question):
    found = []
    for name, pat in _ENTITIES:
        m = re.search(pat, question, re.I)
        if m:
            found.append((name, m))
    return found


_GLUE = r"(?:\s|,|;|/|&|\+|\bet\b|\band\b|\bou\b|\bor\b|\bde\b|\bd['’]|\bdu\b|\bdes\b|\bla\b|\ble\b|\bl['’]|\bles\b|\bof\b|\bthe\b)*"


def sub_question(question, keep, spans):
    """The user's own question, restricted to one entity: a coordinated list
    ("alpha-amylase, xylanase et d'acide ascorbique") is replaced by the kept
    entity; otherwise the other mentions are simply removed."""
    spans = sorted(spans, key=lambda x: x[1].start())
    kept = next(m for name, m in spans if name == keep)
    if all(re.fullmatch(_GLUE, question[a.end():b.start()], re.I)
           for (_, a), (_, b) in zip(spans, spans[1:])):
        q = question[:spans[0][1].start()] + kept.group(0) + question[spans[-1][1].end():]
    else:
        q = question
        for name, m in reversed(spans):
            if name != keep:
                q = q[:m.start()] + q[m.end():]
    return re.sub(r'\s+', ' ', q).strip()


def decomposed_search(table, question, translate, k=TOP_K):
    spans = entity_spans(question)
    full = bilingual_search(table, question, translate, k)
    if len(spans) < 2:
        picked = full
    else:
        picked, seen = [], set()
        for name, _ in spans:
            for d, t, s in bilingual_search(table, sub_question(question, name, spans), translate, k):
                if t not in seen:
                    picked.append((d, t, s)); seen.add(t)
                    break
        for d, t, s in full:
            if len(picked) >= k:
                break
            if t not in seen:
                picked.append((d, t, s)); seen.add(t)
        picked = picked[:k]
    # Display the honest score (question as typed vs fragment), ranked by it.
    shown = [(d, t, honest_score(question, t)) for d, t, _ in picked]
    return sorted(shown, key=lambda r: r[2], reverse=True)


# ── The original search module, pointed at any table ───────────────────────

def old_search(table, question):
    original = old_search_module.search_cosine_similarity
    old_search_module.search_cosine_similarity = lambda vec, top_k=TOP_K: cosine_topk(table, vec, top_k)
    try:
        res = old_search_module.search(question)
    finally:
        old_search_module.search_cosine_similarity = original
    return [(r['id_document'], r['texte'], r['score']) for r in res]


# ── Systems ────────────────────────────────────────────────────────────────

SYSTEMS = {
    'old_as_submitted':        ('embeddings', 'relaxed', lambda q: old_search('embeddings', q)),
    'old_db+strict_search':    ('embeddings', 'strict',  lambda q: cosine_topk('embeddings', embed(q))),
    'naive+strict_search':     ('audit_naive', 'strict', lambda q: cosine_topk('audit_naive', embed(q))),
    'naive+old_search':        ('audit_naive', 'relaxed', lambda q: old_search('audit_naive', q)),
    'faithful+strict_search':  ('audit_faithful', 'strict', lambda q: cosine_topk('audit_faithful', embed(q))),
    'faithful+old_search':     ('audit_faithful', 'relaxed', lambda q: old_search('audit_faithful', q)),
    'faithful+bilingual_dict': ('audit_faithful', 'relaxed', lambda q: bilingual_search('audit_faithful', q, old_dict_fr_en)),
    'faithful+bilingual_mt':   ('audit_faithful', 'relaxed', lambda q: bilingual_search('audit_faithful', q, mt_fr_en)),
    'faithful+bilingual_mt+decomp': ('audit_faithful', 'relaxed', lambda q: decomposed_search('audit_faithful', q, mt_fr_en)),
}


# ── Metrics ────────────────────────────────────────────────────────────────

def auc(pos, neg):
    """P(score of a relevant result > score of an irrelevant one)."""
    if not pos or not neg:
        return None
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def evaluate_query(q, results, id2doc):
    rows = []
    for rank, (d, text, shown) in enumerate(results, 1):
        doc = id2doc[d]
        rel = [t['label'] for t in q['targets'] if relevant(t, doc, text)]
        rows.append({'rank': rank, 'doc': doc, 'relevant_to': rel, 'displayed': round(shown, 4),
                     'honest': round(honest_score(q['q'], text), 4), 'text': text})
    covered = {lab for r in rows for lab in r['relevant_to']}
    rel_flags = [bool(r['relevant_to']) for r in rows]
    first = next((i for i, f in enumerate(rel_flags, 1) if f), None)
    return {
        'id': q['id'], 'lang': q['lang'], 'q': q['q'], 'answerable': q['answerable'],
        'n_targets': len(q['targets']),
        'hit1': bool(rel_flags[:1] and rel_flags[0]), 'hit3': first is not None,
        'rr': 1.0 / first if first else 0.0, 'p3': sum(rel_flags) / TOP_K,
        'coverage': len(covered) / len(q['targets']) if q['targets'] else None,
        'results': rows,
    }


def aggregate(per_query):
    single = [r for r in per_query if r['answerable'] and r['n_targets'] == 1]
    multi = [r for r in per_query if r['n_targets'] > 1]
    unans = [r for r in per_query if not r['answerable']]
    mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None
    rows = [x for r in per_query for x in r['results']]
    pos_d = [x['displayed'] for x in rows if x['relevant_to']]
    neg_d = [x['displayed'] for x in rows if not x['relevant_to']]
    pos_h = [x['honest'] for x in rows if x['relevant_to']]
    neg_h = [x['honest'] for x in rows if not x['relevant_to']]
    top1 = lambda rs, key: mean([r['results'][0][key] for r in rs if r['results']])
    return {
        'n_single': len(single), 'n_multi': len(multi), 'n_unanswerable': len(unans),
        'hit@1': mean([r['hit1'] for r in single]), 'hit@3': mean([r['hit3'] for r in single]),
        'mrr@3': mean([r['rr'] for r in single]), 'p@3': mean([r['p3'] for r in single]),
        'hit@3_fr': mean([r['hit3'] for r in single if r['lang'] == 'fr']),
        'hit@3_en': mean([r['hit3'] for r in single if r['lang'] == 'en']),
        'multi_coverage': mean([r['coverage'] for r in multi]),
        'multi_full': mean([r['coverage'] == 1 for r in multi]),
        'top1_displayed': top1(per_query, 'displayed'), 'top1_honest': top1(per_query, 'honest'),
        'top1_displayed_unanswerable': top1(unans, 'displayed'),
        'auc_displayed': auc(pos_d, neg_d), 'auc_honest': auc(pos_h, neg_h),
    }


def explore(reuse):
    """Post-hoc chunking variants, strict search only. Designed after seeing
    TEST failures, so TEST numbers here are optimistic."""
    pdfs = sorted(Path(config.PDF_FOLDER).glob('*.pdf'))
    ids = {i: doc_key(p.name) for i, p in enumerate(pdfs, 1)}
    variants = {'faithful': 'audit_faithful', 'naive': 'audit_naive'}
    for name, chunker in EXPLORATION.items():
        variants[name] = f'audit_{name}'
        if not reuse:
            print(f'building `audit_{name}` ...')
            build_table(CONN, f'audit_{name}', chunker, model, config.PDF_FOLDER)
    cols = [('hit@1', 'Hit@1'), ('hit@3', 'Hit@3'), ('mrr@3', 'MRR@3'), ('p@3', 'P@3'),
            ('hit@3_fr', 'Hit@3 FR'), ('hit@3_en', 'Hit@3 EN'), ('multi_coverage', 'Multi cov.')]
    out = ['# Post-hoc chunking exploration (strict search only)', '',
           'These variants were designed after looking at TEST failures of the pre-registered',
           'pipelines, so their TEST numbers are optimistic. All fragments are verbatim PDF text.', '']
    cur = CONN.cursor()
    for set_name, qs in (('dev', DEV), ('test', TEST)):
        out += [f'## {set_name.upper()}', '',
                '| chunking | fragments | mean returned length (chars) | ' + ' | '.join(c[1] for c in cols) + ' |',
                '|---|---:|---:|' + '---:|' * len(cols)]
        for name, table in variants.items():
            cur.execute(f'SELECT count(*) FROM {table}')
            n = cur.fetchone()[0]
            pq = [evaluate_query(q, cosine_topk(table, embed(q['q'])), ids) for q in qs]
            agg = aggregate(pq)
            length = np.mean([len(x['text']) for r in pq for x in r['results']])
            out.append(f'| {name} | {n} | {length:.0f} | ' + ' | '.join(fmt(agg[c[0]]) for c in cols) + ' |')
        out.append('')
    (RESULTS / 'exploration.md').write_text('\n'.join(out))
    print('\n'.join(out))


class _Prefixed:
    """Adds the query/passage prefixes that E5 models expect."""
    def __init__(self, st_model, prefix):
        self.m, self.prefix = st_model, prefix

    def encode(self, texts, **kw):
        if isinstance(texts, str):
            return self.m.encode(self.prefix + texts, **kw)
        return self.m.encode([self.prefix + t for t in texts], **kw)


def beyond_rules(reuse):
    """OUT OF THE CHALLENGE RULES: same faithful chunks and plain cosine top-3,
    but a multilingual 384-d embedding model instead of the imposed one.
    Quantifies how much of the remaining gap is due to the imposed model."""
    from sentence_transformers import SentenceTransformer
    from audit.pipelines import faithful_chunks
    pdfs = sorted(Path(config.PDF_FOLDER).glob('*.pdf'))
    ids = {i: doc_key(p.name) for i, p in enumerate(pdfs, 1)}
    specs = {
        'paraphrase-multilingual-MiniLM-L12-v2': ('audit_faithful_pmml12', '', ''),
        'multilingual-e5-small': ('audit_faithful_me5s', 'query: ', 'passage: '),
    }
    hub = {'paraphrase-multilingual-MiniLM-L12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
           'multilingual-e5-small': 'intfloat/multilingual-e5-small'}
    cols = [('hit@1', 'Hit@1'), ('hit@3', 'Hit@3'), ('mrr@3', 'MRR@3'), ('p@3', 'P@3'),
            ('hit@3_fr', 'Hit@3 FR'), ('hit@3_en', 'Hit@3 EN'), ('multi_coverage', 'Multi cov.')]
    rows = {'all-MiniLM-L6-v2 (imposed)': (lambda q: cosine_topk('audit_faithful', embed(q)))}
    for name, (table, qp, pp) in specs.items():
        st = SentenceTransformer(hub[name])
        if not reuse:
            print(f'building `{table}` with {name} ...')
            build_table(CONN, table, faithful_chunks, _Prefixed(st, pp), config.PDF_FOLDER)
        rows[name] = (lambda q, st=st, table=table, qp=qp:
                      cosine_topk(table, st.encode(qp + q, normalize_embeddings=True)))
    out = ['# Beyond the challenge rules: embedding model swap', '',
           'Same faithful chunks, same plain cosine top-3; only the embedding model changes.',
           'Swapping the model is NOT allowed by the challenge (all-MiniLM-L6-v2 is imposed);',
           'this only measures how much of the remaining error is due to that model.', '']
    hits = {}
    for set_name, qs in (('dev', DEV), ('test', TEST)):
        out += [f'## {set_name.upper()}', '', '| embedding model | ' + ' | '.join(c[1] for c in cols) + ' |',
                '|---|' + '---:|' * len(cols)]
        for name, fn in rows.items():
            pq = [evaluate_query(q, fn(q['q']), ids) for q in qs]
            hits[set_name, name] = [r['hit3'] for r in pq if r['answerable'] and r['n_targets'] == 1]
            agg = aggregate(pq)
            out.append(f'| {name} | ' + ' | '.join(fmt(agg[c[0]]) for c in cols) + ' |')
        out.append('')
    base = hits['test', 'all-MiniLM-L6-v2 (imposed)']
    out += ['## Paired test vs the imposed model (TEST, single-answer, Hit@3)', '',
            '| model | model-only hits | imposed-only hits | exact McNemar p |', '|---|---:|---:|---:|']
    for name in specs:
        n01, n10, p = mcnemar_exact(hits['test', name], base)
        out.append(f'| {name} | {n01} | {n10} | {p:.3f} |')
    (RESULTS / 'beyond_rules.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', action='store_true', help='reuse existing tables')
    ap.add_argument('--explore', action='store_true', help='post-hoc chunking exploration')
    ap.add_argument('--beyond', action='store_true', help='out-of-rules embedding model swap')
    args = ap.parse_args()
    if args.explore or args.beyond:
        RESULTS.mkdir(parents=True, exist_ok=True)
        return explore(args.reuse) if args.explore else beyond_rules(args.reuse)

    pdfs = list(Path(config.PDF_FOLDER).glob('*.pdf'))   # same order as ingest_pdfs()
    old_ids = {i: doc_key(p.name) for i, p in enumerate(pdfs, 1)}
    id2doc = {'embeddings': old_ids}
    if not args.reuse:
        print('building original table `embeddings` with services/ingestion_data.py ...')
        ingest_pdfs(config.PDF_FOLDER)
    for name, chunker in PIPELINES.items():
        table = f'audit_{name}'
        sorted_ids = {i: doc_key(p.name) for i, p in enumerate(sorted(pdfs), 1)}
        if not args.reuse:
            print(f'building `{table}` ...')
            build_table(CONN, table, chunker, model, config.PDF_FOLDER)
        id2doc[table] = sorted_ids

    cur = CONN.cursor()
    table_stats = {}
    for t in id2doc:
        cur.execute(f'SELECT count(*), count(DISTINCT texte_fragment) FROM {t}')
        table_stats[t] = dict(zip(('rows', 'distinct_fragments'), cur.fetchone()))

    metrics, per_query_all = {}, {}
    for sname, (table, mode, fn) in SYSTEMS.items():
        print(f'running {sname} ...')
        for set_name, qs in (('dev', DEV), ('test', TEST)):
            pq = [evaluate_query(q, fn(q['q']), id2doc[table]) for q in qs]
            metrics.setdefault(sname, {'table': table, 'mode': mode})[set_name] = aggregate(pq)
            per_query_all.setdefault(sname, {})[set_name] = pq

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'metrics.json').write_text(json.dumps({'tables': table_stats, 'systems': metrics}, indent=1, ensure_ascii=False))
    (RESULTS / 'per_query.json').write_text(json.dumps(per_query_all, ensure_ascii=False, separators=(',', ':')))
    write_summary(table_stats, metrics, significance(per_query_all))
    print((RESULTS / 'summary.md').read_text())


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired binary outcomes."""
    from math import comb
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = 2 * sum(comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def bootstrap_ci(values, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), (reps, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


PAIRS = [('faithful+strict_search', 'naive+strict_search'),
         ('faithful+strict_search', 'old_as_submitted'),
         ('faithful+strict_search', 'old_db+strict_search'),
         ('faithful+bilingual_mt', 'faithful+strict_search'),
         ('faithful+bilingual_mt+decomp', 'old_as_submitted'),
         ('naive+old_search', 'naive+strict_search')]


def significance(per_query_all, set_name='test'):
    single = lambda s: [r for r in per_query_all[s][set_name] if r['answerable'] and r['n_targets'] == 1]
    out = ['', f'## Uncertainty ({set_name.upper()}, single-answer queries)', '',
           '| system | Hit@3 | 95% bootstrap CI |', '|---|---:|---|']
    for s in per_query_all:
        hits = [r['hit3'] for r in single(s)]
        lo, hi = bootstrap_ci(hits)
        out.append(f'| {s} | {np.mean(hits):.3f} | [{lo:.2f}, {hi:.2f}] |')
    out += ['', '| A vs B | A-only hits | B-only hits | exact McNemar p (Hit@3) |', '|---|---:|---:|---:|']
    for a, b in PAIRS:
        n01, n10, p = mcnemar_exact([r['hit3'] for r in single(a)], [r['hit3'] for r in single(b)])
        out.append(f'| {a} vs {b} | {n01} | {n10} | {p:.3f} |')
    return out


def fmt(x):
    return '–' if x is None else f'{x:.3f}'


def write_summary(table_stats, metrics, extra=()):
    out = ['# Evaluation summary', '', '## Tables', '',
           '| table | rows | distinct fragments |', '|---|---:|---:|']
    out += [f'| `{t}` | {s["rows"]} | {s["distinct_fragments"]} |' for t, s in table_stats.items()]
    cols = [('hit@1', 'Hit@1'), ('hit@3', 'Hit@3'), ('mrr@3', 'MRR@3'), ('p@3', 'P@3'),
            ('hit@3_fr', 'Hit@3 FR'), ('hit@3_en', 'Hit@3 EN'), ('multi_coverage', 'Multi cov.'),
            ('top1_displayed', 'Top-1 shown'), ('top1_honest', 'Top-1 honest'),
            ('top1_displayed_unanswerable', 'Top-1 shown (unanswerable)'), ('auc_displayed', 'AUC shown'),
            ('auc_honest', 'AUC honest')]
    for set_name in ('dev', 'test'):
        any_sys = next(iter(metrics.values()))[set_name]
        out += ['', f'## {set_name.upper()} set  ({any_sys["n_single"]} single-answer, '
                    f'{any_sys["n_multi"]} multi-entity, {any_sys["n_unanswerable"]} unanswerable queries)', '',
                '| system | mode | ' + ' | '.join(c[1] for c in cols) + ' |',
                '|---|---|' + '---:|' * len(cols)]
        for sname, m in metrics.items():
            s = m[set_name]
            out.append(f'| {sname} | {m["mode"]} | ' + ' | '.join(fmt(s[c[0]]) for c in cols) + ' |')
    (RESULTS / 'summary.md').write_text('\n'.join(out + list(extra)) + '\n')


if __name__ == '__main__':
    main()
