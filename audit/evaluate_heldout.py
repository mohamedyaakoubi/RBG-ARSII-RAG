"""
One-shot evaluations on the held-out sets.

    python -m audit.evaluate_heldout --set test2   # frozen before v1 was chosen
    python -m audit.evaluate_heldout --set test3   # frozen before v2 (post-hoc) was chosen
    python -m audit.evaluate_heldout --set test4   # frozen before the product filter was built

Systems:
  old_as_submitted     original table + original search module
  faithful+strict      pre-registered strict baseline from the audit
  in-rules v1 S / S+   configuration chosen on DEV + TEST (run once on TEST-2)
  in-rules v2 S / S+   + fixes suggested by TEST-2 failures (run once on TEST-3)
  in-rules v2 S+F      + product-code filter, suggested by TEST-2/TEST-3
                       failures (run once on TEST-4; the app's transparent mode)
  S   = question embedded as typed (strict);
  S+  = + English translation of French questions and one sub-question per
        product family (transparent preprocessing);
  S+F = S+ + a question naming a product by its code is answered from its sheet.
Writes audit/results/<set>.md and <set>_per_query.json.
"""
import argparse
import json
import logging

logging.disable(logging.CRITICAL)

import numpy as np

from audit.evaluate import (RESULTS, bootstrap_ci, cosine_topk, embed, honest_score, mcnemar_exact,
                            old_search)
from audit.eval_set import doc_key, relevant
from audit.eval_set_v2 import TEST2
from audit.eval_set_v3 import TEST3
from audit.eval_set_v4 import TEST4
from audit import inrules
from config.settings import config
from pathlib import Path


def pg_system(table, fn):
    pdfs = list(Path(config.PDF_FOLDER).glob('*.pdf'))
    ids = ({i: doc_key(p.name) for i, p in enumerate(pdfs, 1)} if table == 'legacy_embeddings'
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


def test4_details(qs, per_query, index):
    """TEST-4 breakdowns: detection of the named products, results by
    attribute group, by how the code was written, and by kind of question."""
    from audit.sample_test4_slots import slots
    answerable = [q for q in qs if q['answerable']]
    slot = {q['id']: s for q, s in zip(answerable, slots)}
    detected = sum(sorted(d for d, _ in inrules.named_products(index, q['q']))
                   == sorted(t['docs'][0] for t in q['targets'] if len(t['docs']) == 1 and t['docs'][0] in index.sheets)
                   for q in answerable)
    names = ['old_as_submitted', 'in-rules v2 S (strict)', 'in-rules v2 S+ (transparent preprocessing)',
             'in-rules v2 S+F (transparent + product filter)']
    rows = {n: {r['id']: r for r in per_query[n]} for n in names}
    single = [q for q in answerable if len(q['targets']) == 1]
    out = ['', '## TEST-4 details', '',
           f'Named products found by the filter\'s code matching: {detected}/{len(answerable)} answerable questions '
           '(exactly the products the question is about).', '',
           '### Single-product questions by attribute group (right answer in top 3)', '',
           '| group | n | ' + ' | '.join(names) + ' |', '|---|---:|' + '---:|' * len(names)]
    for group in ('specific', 'header', 'shared'):
        g = [q for q in single if q['targets'][0]['group'] == group]
        out.append(f'| {group} | {len(g)} | ' + ' | '.join(
            f"{sum(rows[n][q['id']]['hit3'] for q in g)}/{len(g)}" for n in names) + ' |')
    out += ['', '### Single-product questions by how the code is written (right answer in top 3)', '',
            '| code written | n | ' + ' | '.join(names[2:]) + ' |', '|---|---:|---:|---:|']
    for form in ('printed+brand', 'printed', 'lower', 'respaced', 'hyphenated'):
        g = [q for q in single if slot[q['id']][4] == form]
        out.append(f'| {form} | {len(g)} | ' + ' | '.join(
            f"{sum(rows[n][q['id']]['hit3'] for q in g)}/{len(g)}" for n in names[2:]) + ' |')
    g = [q for q in single if slot[q['id']][5]]
    out.append(f'| (also names its own family) | {len(g)} | ' + ' | '.join(
        f"{sum(rows[n][q['id']]['hit3'] for q in g)}/{len(g)}" for n in names[2:]) + ' |')
    out += ['', '### Questions naming several things (share of named products / families covered in the top 3)', '',
            '| kind | n | ' + ' | '.join(names) + ' |', '|---|---:|' + '---:|' * len(names)]
    for kind in ('two products', 'product + family'):
        g = [q for q in answerable if slot[q['id']][0] == kind]
        out.append(f'| {kind} | {len(g)} | ' + ' | '.join(
            f"{np.mean([rows[n][q['id']]['coverage'] for q in g]):.2f}" for n in names) + ' |')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['test2', 'test3', 'test4'], default='test4')
    args = ap.parse_args()
    held_out = {'test2': ('TEST-2', TEST2, '9cb134a'), 'test3': ('TEST-3', TEST3, 'b6e31ab'),
                'test4': ('TEST-4', TEST4, '9ef1b1d')}[args.set]
    v1 = inrules.Index(inrules.build(**inrules.FINAL_CORPUS))
    v2 = inrules.Index(inrules.build(**inrules.FINAL_CORPUS_V2))
    systems = {
        'old_as_submitted': pg_system('legacy_embeddings', lambda q: old_search('legacy_embeddings', q)),
        'faithful+strict': pg_system('audit_faithful', lambda q: cosine_topk('audit_faithful', embed(q))),
        'in-rules v1 S (strict)': inrules_system(v1, 'S'),
        'in-rules v1 S+ (transparent preprocessing)': inrules_system(v1, 'S+'),
        'in-rules v2 S (strict)': inrules_system(v2, 'S'),
        'in-rules v2 S+ (transparent preprocessing)': inrules_system(v2, 'S+'),
        'in-rules v2 S+F (transparent + product filter)': inrules_system(v2, 'S+F'),
    }
    name, qs, frozen = held_out
    out = [f'# {name}: one-shot evaluation', '',
           f'v1 corpus: {len(v1.frags)} fragments {inrules.FINAL_CORPUS} (chosen on DEV + TEST).',
           f'v2 corpus: {len(v2.frags)} fragments {inrules.FINAL_CORPUS_V2} (post-hoc, chosen on DEV + TEST + TEST-2).',
           'S+F = S+ with the product-code filter (suggested by TEST-2 and TEST-3 failures, so only TEST-4 measures it).',
           f'{name} was frozen at commit {frozen}, before the configuration it tests was chosen.', '',
           '| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    per_query = {}
    for sname, run in systems.items():
        pq = per_query[sname] = [score(q, run(q['q'])) for q in qs]
        s = summarize(pq)
        n_hit = int(round(s['hit@3'] * s['n']))
        out.append(f"| {sname} | {n_hit}/{s['n']} = **{s['hit@3']:.3f}** | [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | "
                   f"{s['hit@1']:.3f} | {s['mrr']:.3f} | {s['fr']:.3f} | {s['en']:.3f} | {s['multi']:.3f} | "
                   f"{s['multi_full']:.2f} | {s['ans_top1']:.3f} / {s['unans_top1']:.3f} |")
    hits = lambda sname: [r['hit3'] for r in per_query[sname] if r['answerable'] and r['n_targets'] == 1]
    out += ['', f'## Paired tests ({name}, single-answer questions, Hit@3)', '',
            '| A vs B | only A right | only B right | exact McNemar p |', '|---|---:|---:|---:|']
    for a, b in [('in-rules v1 S (strict)', 'faithful+strict'), ('in-rules v1 S (strict)', 'old_as_submitted'),
                 ('in-rules v1 S+ (transparent preprocessing)', 'old_as_submitted'),
                 ('in-rules v2 S (strict)', 'faithful+strict'), ('in-rules v2 S (strict)', 'old_as_submitted'),
                 ('in-rules v2 S+ (transparent preprocessing)', 'in-rules v2 S (strict)'),
                 ('in-rules v2 S+ (transparent preprocessing)', 'old_as_submitted'),
                 ('in-rules v2 S+F (transparent + product filter)', 'in-rules v2 S+ (transparent preprocessing)'),
                 ('in-rules v2 S+F (transparent + product filter)', 'old_as_submitted')]:
        n01, n10, p = mcnemar_exact(hits(a), hits(b))
        out.append(f'| {a} vs {b} | {n01} | {n10} | {p:.4f} |')
    if args.set == 'test4':
        out += test4_details(qs, per_query, v2)
    (RESULTS / f'{args.set}.md').write_text('\n'.join(out) + '\n')
    (RESULTS / f'{args.set}_per_query.json').write_text(json.dumps(per_query, ensure_ascii=False, separators=(',', ':')))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
