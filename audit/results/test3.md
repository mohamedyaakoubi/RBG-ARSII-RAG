# TEST-3: one-shot evaluation

v1 corpus: 608 fragments {'card': True, 'usage': True, 'rows': 'both', 'docinfo': True, 'fix_splits': True, 'bilingual': 'fr-en'} (chosen on DEV + TEST).
v2 corpus: 609 fragments {'card': True, 'usage': 'AD+FD', 'rows': 'both', 'docinfo': 'plain', 'fix_splits': True, 'bilingual': 'fr-en', 'identity': True} (post-hoc, chosen on DEV + TEST + TEST-2).
TEST-3 was frozen at commit b6e31ab, before the configuration it tests was chosen.

| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| old_as_submitted | 33/52 = **0.635** | [0.50, 0.77] | 0.519 | 0.577 | 0.606 | 0.684 | 0.429 | 0.29 | 0.643 / 0.528 |
| faithful+strict | 33/52 = **0.635** | [0.50, 0.77] | 0.519 | 0.567 | 0.606 | 0.684 | 0.429 | 0.14 | 0.595 / 0.531 |
| in-rules v1 S (strict) | 40/52 = **0.769** | [0.65, 0.88] | 0.615 | 0.679 | 0.758 | 0.789 | 0.429 | 0.14 | 0.606 / 0.531 |
| in-rules v1 S+ (transparent preprocessing) | 46/52 = **0.885** | [0.79, 0.96] | 0.769 | 0.827 | 0.939 | 0.789 | 0.786 | 0.57 | 0.667 / 0.578 |
| in-rules v2 S (strict) | 42/52 = **0.808** | [0.69, 0.90] | 0.673 | 0.731 | 0.788 | 0.842 | 0.500 | 0.29 | 0.604 / 0.531 |
| in-rules v2 S+ (transparent preprocessing) | 48/52 = **0.923** | [0.85, 0.98] | 0.846 | 0.885 | 0.970 | 0.842 | 0.857 | 0.71 | 0.667 / 0.575 |

## Paired tests (TEST-3, single-answer questions, Hit@3)

| A vs B | only A right | only B right | exact McNemar p |
|---|---:|---:|---:|
| in-rules v1 S (strict) vs faithful+strict | 7 | 0 | 0.0156 |
| in-rules v1 S (strict) vs old_as_submitted | 10 | 3 | 0.0923 |
| in-rules v1 S+ (transparent preprocessing) vs old_as_submitted | 14 | 1 | 0.0010 |
| in-rules v2 S (strict) vs faithful+strict | 9 | 0 | 0.0039 |
| in-rules v2 S (strict) vs old_as_submitted | 12 | 3 | 0.0352 |
| in-rules v2 S+ (transparent preprocessing) vs in-rules v2 S (strict) | 6 | 0 | 0.0312 |
| in-rules v2 S+ (transparent preprocessing) vs old_as_submitted | 16 | 1 | 0.0003 |
