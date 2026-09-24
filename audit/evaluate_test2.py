"""
One-shot evaluation on TEST-2 (audit/eval_set_v2.py), frozen before the
in-rules configuration was chosen on the development pool (DEV + TEST).

    python -m audit.evaluate_test2

Systems:
  old_as_submitted     original table + original search module
  faithful+strict      pre-registered strict baseline from the audit
  in-rules S           final corpus, question embedded as typed (strict)
  in-rules S+          final corpus, + English translation of French questions
                       and one sub-question per product (transparent)
Writes audit/results/test2.md and test2_per_query.json.
"""
import json
import logging

logging.disable(logging.CRITICAL)

import numpy as np

from audit.evaluate import (RESULTS, bootstrap_ci, cosine_topk, embed, honest_score, mcnemar_exact,
                            old_search, CONN)
from audit.eval_set import DEV, TEST, doc_key, relevant
from audit.eval_set_v2 import TEST2
from audit import inrules
from config.settings import config
from pathlib import Path


def pg_system(table, fn):
    pdfs = list(Path(config.PDF_FOLDER).glob('*.pdf'))
    ids = ({i: doc_key(p.name) for i, p in enumerate(pdfs, 1)} if table == 'embeddings'
           else {i: doc_key(p.name) for i, p in enumerate(sorted(pdfs), 1)})

    def run(q):
        return [{'doc': ids[d], 'text': t, 'orig': t, 'shown': s} for d, t, s in fn(q)]
    return run


def inrules_system(index, mode):
    def run(q):
        return [{'doc': f['doc'], 'text': f['text'], 'orig': f['orig'], 'shown': s,
                 'embedded_as': v} for f, s, v in inrules.answer(index, q, **inrules.MODES[mode])]
    return run


def score(q, results):
    rel = lambda t, r: relevant(t, r['doc'], r['orig']) or relevant(t, r['doc'], r['text'])
    rows = [{**r, 'relevant_to': [t['label'] for t in q['targets'] if rel(t, r)],
             'honest': round(honest_score(q['q'], r['text']), 4)} for r in results]
    flags = [bool(r['relevant_to']) for r in rows]
    first = next((i for i, f in enumerate(flags, 1) if f), None)
    covered = {lab for r in rows for lab in r['relevant_to']}
    return {'id': q['id'], 'lang': q['lang'], 'q': q['q'], 'answerable': q['answerable'],
            'n_targets': len(q['targets']), 'hit1': bool(flags[:1] and flags[0]), 'hit3': first is not None,
            'rr': 1 / first if first else 0.0,
            'coverage': len(covered) / len(q['targets']) if q['targets'] else None, 'results': rows}


def summarize(pq):
    single = [r for r in pq if r['answerable'] and r['n_targets'] == 1]
    multi = [r for r in pq if r['n_targets'] > 1]
    unans = [r for r in pq if not r['answerable']]
    m = lambda xs: float(np.mean(xs)) if xs else float('nan')
    hits = [r['hit3'] for r in single]
    return {'n': len(single), 'hit@1': m([r['hit1'] for r in single]), 'hit@3': m(hits),
            'ci': bootstrap_ci(hits), 'mrr': m([r['rr'] for r in single]),
            'fr': m([r['hit3'] for r in single if r['lang'] == 'fr']),
            'en': m([r['hit3'] for r in single if r['lang'] == 'en']),
            'multi': m([r['coverage'] for r in multi]), 'multi_full': m([r['coverage'] == 1 for r in multi]),
            'unans_top1': m([r['results'][0]['shown'] for r in unans if r['results']]),
            'ans_top1': m([r['results'][0]['shown'] for r in pq if r['answerable'] and r['results']])}


def main():
    index = inrules.Index(inrules.build(**inrules.FINAL_CORPUS))
    systems = {
        'old_as_submitted': pg_system('embeddings', lambda q: old_search('embeddings', q)),
        'faithful+strict': pg_system('audit_faithful', lambda q: cosine_topk('audit_faithful', embed(q))),
        'in-rules S (strict)': inrules_system(index, 'S'),
        'in-rules S+ (transparent preprocessing)': inrules_system(index, 'S+'),
    }
    out = ['# TEST-2: one-shot evaluation of the in-rules system', '',
           f'Final corpus: {len(index.frags)} fragments ({inrules.FINAL_CORPUS}).',
           'TEST-2 was frozen (commit 9cb134a) before the configuration was chosen on DEV + TEST.', '']
    per_query, sums = {}, {}
    for set_name, qs in (('dev pool (DEV+TEST)', DEV + TEST), ('TEST-2 (held out)', TEST2)):
        out += [f'## {set_name}', '',
                '| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
        for name, run in systems.items():
            pq = [score(q, run(q['q'])) for q in qs]
            per_query[f'{set_name} | {name}'] = pq
            s = sums[set_name, name] = summarize(pq)
            n_hit = int(round(s['hit@3'] * s['n']))
            out.append(f"| {name} | {n_hit}/{s['n']} = **{s['hit@3']:.3f}** | [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | "
                       f"{s['hit@1']:.3f} | {s['mrr']:.3f} | {s['fr']:.3f} | {s['en']:.3f} | {s['multi']:.3f} | "
                       f"{s['multi_full']:.2f} | {s['ans_top1']:.3f} / {s['unans_top1']:.3f} |")
        out.append('')
    t2 = lambda name: [r['hit3'] for r in per_query[f'TEST-2 (held out) | {name}'] if r['answerable'] and r['n_targets'] == 1]
    out += ['## Paired tests on TEST-2 (single-answer questions, Hit@3)', '',
            '| A vs B | only A right | only B right | exact McNemar p |', '|---|---:|---:|---:|']
    for a, b in [('in-rules S (strict)', 'faithful+strict'), ('in-rules S (strict)', 'old_as_submitted'),
                 ('in-rules S+ (transparent preprocessing)', 'in-rules S (strict)'),
                 ('in-rules S+ (transparent preprocessing)', 'old_as_submitted')]:
        n01, n10, p = mcnemar_exact(t2(a), t2(b))
        out.append(f'| {a} vs {b} | {n01} | {n10} | {p:.4f} |')
    (RESULTS / 'test2.md').write_text('\n'.join(out) + '\n')
    (RESULTS / 'test2_per_query.json').write_text(json.dumps(per_query, ensure_ascii=False, separators=(',', ':')))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
