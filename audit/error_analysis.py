"""
Why do the remaining answers fail, and could anything inside the rules fix them?

    python -m audit.error_analysis

For every held-out question the final system (v2) misses, on TEST-2 and TEST-3:
  1. rank of the first correct fragment in the full cosine ranking, and its
     score gap to the 3rd result (near miss vs far miss);
  2. oracle test: the same question asked in the documents' own words. If
     even that fails, no pipeline under these rules can fix it; if it works,
     the failure is a gap between user wording and document wording;
  3. whether a product-code filter (answer from the named product's sheet)
     would fix it, checked on every question that names a product.
Writes audit/results/error_analysis.md.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit import inrules
from audit.eval_set_v2 import TEST2
from audit.eval_set_v3 import TEST3
from audit.pipelines import product_name

# The failing question asked in the vocabulary of the documents.
ORACLE = {
    'V02': 'transglutaminase dosage', 'V05': 'glucose oxidase dosage', 'V07': 'fungal alpha-amylase dosage',
    'V08': 'transglutaminase dosage', 'V11': 'glucose oxidase dosage', 'V13': 'xylanase function',
    'V14': 'glucose oxidase function', 'V16': 'glucose oxidase function', 'V18': 'BVZyme HCF400 function',
    'V27': 'BVZyme AF110 activity', 'V31': 'alpha-amylase storage', 'V32': 'BVZyme A FRESH101 storage',
    'V34': 'allergens gluten', 'V37': 'BVZyme packaging', 'V39': 'microbiology ASR', 'V40': 'microbiology',
    'V41': 'heavy metals lead', 'V42': 'moisture', 'V43': 'aspect color powder',
    'V44': 'manufacturer company address', 'V45': 'last updating date', 'V47': 'ascorbic acid dosage freezing',
    'V52': 'maximum authorised dosage ascorbic acid', 'V54': 'conversion table 50 kg 75 ppm grams',
    'V57': 'ascorbic acid optimal temperature', 'V60': 'ascorbic acid chemical formula',
    'V61': 'ascorbic acid chemical formula', 'V62': 'ascorbic acid density', 'V63': 'ascorbic acid packaging formats',
    'V67': 'ascorbic acid volume improvement', 'V70': 'alternatives to ascorbic acid',
    'Z01': 'BVZyme A SOFT305 dosage', 'Z08': 'amyloglucosidase dosage', 'Z14': 'BVZyme L MAX64 function',
    'Z27': 'GMO status labeling', 'Z29': 'BVZyme packaging', 'Z31': 'heavy metals lead',
    'Z34': 'manufacturer company address', 'Z43': 'ascorbic acid dilute in water', 'Z45': 'ascorbic acid pH',
    'Z48': 'ascorbic acid shelf life',
}


def product_codes():
    codes = {}
    for p in (ROOT / 'data_pdf').glob('*.pdf'):
        name = product_name(p)
        if 'ascorbique' not in name.lower():
            code = name.replace('BVZyme ', '')
            codes[code] = r'\b' + r'\s*'.join(map(re.escape, code.replace(' ', ''))) + r'\b'
    return codes


def main():
    idx = inrules.Index(inrules.build(**inrules.FINAL_CORPUS_V2))
    codes = product_codes()
    first_ok = lambda target, ranking: next(
        ((i, s) for i, (f, s) in enumerate(ranking, 1) if inrules.is_relevant(target, f)), (None, None))
    rows = []
    for set_name, qs in (('TEST-2', TEST2), ('TEST-3', TEST3)):
        for q in qs:
            if not q['answerable'] or len(q['targets']) != 1:
                continue
            t = q['targets'][0]
            for mode in ('S', 'S+'):
                res = inrules.answer(idx, q['q'], **inrules.MODES[mode])
                if any(inrules.is_relevant(t, f) for f, _, _ in res):
                    continue
                variants = inrules.question_variants(q['q'], mode == 'S+')
                ranking = idx.search(variants, k=len(idx.frags))
                rank, score = first_ok(t, ranking)
                gap = score - ranking[2][1] if rank else None
                o_rank, _ = first_ok(t, idx.search([ORACLE[q['id']]], k=len(idx.frags)))
                named = [c for c, pat in codes.items() if re.search(pat, q['q'], re.I)]
                fixed_by_filter = None
                if named:
                    pat = codes[max(named, key=len)]
                    kept = [(f, s) for f, s in ranking if re.search(pat, f['header'], re.I)][:3]
                    fixed_by_filter = any(inrules.is_relevant(t, f) for f, _ in kept)
                rows.append(dict(set=set_name, id=q['id'], mode=mode, q=q['q'], rank=rank, gap=gap,
                                 oracle=ORACLE[q['id']], oracle_rank=o_rank, product_filter=fixed_by_filter))
    out = ['# Error analysis of the final system (v2)', '',
           'Every single-answer question of TEST-2 and TEST-3 that the final system misses.',
           '"rank" = rank of the first correct fragment in the full cosine ranking; "gap" = its score minus the',
           '3rd result\'s. "oracle" = the same question asked in the documents\' own words.', '',
           '| set | id | mode | question | rank | gap | oracle phrasing | oracle rank | product filter fixes it |',
           '|---|---|---|---|---:|---:|---|---:|---|']
    for r in rows:
        out.append(f"| {r['set']} | {r['id']} | {r['mode']} | {r['q']} | {r['rank']} | {r['gap']:+.3f} | "
                   f"{r['oracle']} | {r['oracle_rank']} | "
                   f"{'' if r['product_filter'] is None else ('yes' if r['product_filter'] else 'no')} |")
    for mode in ('S', 'S+'):
        rs = [r for r in rows if r['mode'] == mode]
        near = sum(1 for r in rs if r['rank'] and r['rank'] <= 5)
        oracle_ok = sum(1 for r in rs if r['oracle_rank'] and r['oracle_rank'] <= 3)
        filt = sum(1 for r in rs if r['product_filter'])
        out += ['', f'**{mode}**: {len(rs)} failures; correct fragment at rank 4-5 (lost to the top-3 cut): {near}; '
                    f'found in the top 3 by the oracle phrasing: {oracle_ok}; fixed by a product-code filter: {filt}.']
    (ROOT / 'audit' / 'results' / 'error_analysis.md').write_text('\n'.join(out) + '\n')
    (ROOT / 'audit' / 'results' / 'error_analysis.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
