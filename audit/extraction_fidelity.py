"""
How faithful is each extractor to what the page shows? (ROUTES.md, part 1)

    python -m audit.extraction_fidelity

Ground truth is physical, not a vote between extractors: the position of every
visible glyph on the page. Space characters are ignored because they draw
nothing (the sheets contain stray ones that overlap letters). Two letters on
the same line are separated by a space when the gap between their glyphs is at
least 0.1 em. Measured gaps are either under 0.03 em or over 0.21 em, so any
threshold in between gives the same truth.

For each extractor, compared with that truth after aligning the letters and
digits:
  - coverage: share of the page's letters and digits found in reading order;
  - false splits: a space inside a word ("gly cerides", "10-10 0");
  - false joins: two words run together.
Writes audit/results/extraction_fidelity.md.
"""
import difflib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber
from pdfplumber.utils import cluster_objects

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'cache' / 'extractions'
GAP_EM = 0.1


def truth(path):
    """[page: [(char, space_before)]] from the visible glyphs."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            visible = [c for c in page.chars if not c['text'].isspace()]
            seq = []
            for line in cluster_objects(visible, 'top', 3):
                line = sorted(line, key=lambda c: c['x0'])
                gap_seen = True                           # a new line is a break
                prev = None
                for c in line:
                    if prev is not None and (c['x0'] - prev['x1']) / max(prev['size'], 1e-6) >= GAP_EM:
                        gap_seen = True
                    for ch in unicodedata.normalize('NFKC', c['text']):
                        if ch.isalnum():
                            seq.append((ch, gap_seen))
                            gap_seen = False
                    prev = c
            pages.append(seq)
    return pages


def from_text(text):
    """[(char, space_before)] of an extractor's page text."""
    seq, ws = [], True
    for ch in unicodedata.normalize('NFKC', text):
        if ch.isspace():
            ws = True
        elif ch.isalnum():
            seq.append((ch, ws))
            ws = False
    return seq


def compare(ref, ext):
    """(matched chars, comparable pairs, false splits, false joins, examples)."""
    a = ''.join(c for c, _ in ref)
    b = ''.join(c for c, _ in ext)
    matched = pairs = splits = joins = 0
    examples = []
    for i, j, n in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        matched += n
        for k in range(1, n):                             # pairs inside a block are consecutive in both
            t, e = ref[i + k][1], ext[j + k][1]
            pairs += 1
            if e and not t:
                splits += 1
                examples.append(('split', a[max(i + k - 8, 0):i + k] + '|' + a[i + k:i + k + 8]))
            elif t and not e:
                joins += 1
                examples.append(('join', a[max(i + k - 8, 0):i + k] + '+' + a[i + k:i + k + 8]))
    return matched, pairs, splits, joins, examples


def main():
    pdfs = sorted((ROOT / 'data_pdf').glob('*.pdf'))
    truths = {p.name: truth(p) for p in pdfs}
    n_truth = sum(len(s) for pages in truths.values() for s in pages)
    words = sum(ws for pages in truths.values() for s in pages for _, ws in s)
    rows = []
    for path in sorted(CACHE.glob('*.json')):
        if path.stem == 'translations':                  # the routes' translation cache, not an extractor
            continue
        data = json.loads(path.read_text())
        tot = Counter()
        kinds = Counter()
        for name, pages in truths.items():
            ext_pages = data.get(name, [])
            for p, ref in enumerate(pages):
                ext = from_text(ext_pages[p]) if p < len(ext_pages) else []
                m, pairs, s, j, ex = compare(ref, ext)
                tot.update(matched=m, pairs=pairs, splits=s, joins=j, ext=len(ext))
                kinds.update(f'{k}: {x}' for k, x in ex)
        rows.append((path.stem, tot, kinds))
    rows.sort(key=lambda r: (-(r[1]['matched'] / n_truth), r[1]['splits'] + r[1]['joins']))
    out = ['# Extraction fidelity against the page itself', '',
           f'Truth: {n_truth} letters and digits in {words} words, from the positions of the visible glyphs '
           f'(space characters ignored; a gap of at least {GAP_EM} em between two glyphs is a space).', '',
           '| extractor | letters and digits found in reading order | false splits (space inside a word) | '
           'false joins (words run together) | errors per 1,000 words |',
           '|---|---:|---:|---:|---:|']
    for name, t, _ in rows:
        err = t['splits'] + t['joins']
        out.append(f"| {name} | {t['matched'] / n_truth:.1%} | {t['splits']} | {t['joins']} | "
                   f"{1000 * err / max(words, 1):.1f} |")
    out += ['', '## Most frequent errors per extractor', '']
    for name, t, kinds in rows:
        if kinds:
            out.append(f"- **{name}**: " + '; '.join(f'`{k}` ×{n}' for k, n in kinds.most_common(6)))
    (ROOT / 'audit' / 'results' / 'extraction_fidelity.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
