# TEST-2: one-shot evaluation

v1 corpus: 608 fragments {'card': True, 'usage': True, 'rows': 'both', 'docinfo': True, 'fix_splits': True, 'bilingual': 'fr-en'} (chosen on DEV + TEST).
v2 corpus: 609 fragments {'card': True, 'usage': 'AD+FD', 'rows': 'both', 'docinfo': 'plain', 'fix_splits': True, 'bilingual': 'fr-en', 'identity': True} (post-hoc, chosen on DEV + TEST + TEST-2).
S+F = S+ with the product-code filter (suggested by TEST-2 and TEST-3 failures, so only TEST-4 measures it).
TEST-2 was frozen at commit 9cb134a, before the configuration it tests was chosen.

| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| old_as_submitted | 43/70 = **0.614** | [0.50, 0.73] | 0.457 | 0.524 | 0.556 | 0.812 | 0.517 | 0.30 | 0.651 / 0.612 |
| faithful+strict | 36/70 = **0.514** | [0.40, 0.63] | 0.386 | 0.448 | 0.444 | 0.750 | 0.333 | 0.00 | 0.581 / 0.564 |
| in-rules v1 S (strict) | 40/70 = **0.571** | [0.46, 0.69] | 0.443 | 0.498 | 0.481 | 0.875 | 0.433 | 0.10 | 0.595 / 0.572 |
| in-rules v1 S+ (transparent preprocessing) | 54/70 = **0.771** | [0.67, 0.87] | 0.657 | 0.702 | 0.741 | 0.875 | 0.800 | 0.70 | 0.662 / 0.623 |
| in-rules v2 S (strict) | 43/70 = **0.614** | [0.50, 0.73] | 0.471 | 0.531 | 0.556 | 0.812 | 0.483 | 0.10 | 0.589 / 0.571 |
| in-rules v2 S+ (transparent preprocessing) | 58/70 = **0.829** | [0.74, 0.91] | 0.700 | 0.750 | 0.833 | 0.812 | 0.800 | 0.70 | 0.658 / 0.627 |
| in-rules v2 S+F (transparent + product filter) | 60/70 = **0.857** | [0.77, 0.93] | 0.714 | 0.774 | 0.852 | 0.875 | 0.800 | 0.70 | 0.658 / 0.625 |

## Paired tests (TEST-2, single-answer questions, Hit@3)

| A vs B | only A right | only B right | exact McNemar p |
|---|---:|---:|---:|
| in-rules v1 S (strict) vs faithful+strict | 7 | 3 | 0.3438 |
| in-rules v1 S (strict) vs old_as_submitted | 6 | 9 | 0.6072 |
| in-rules v1 S+ (transparent preprocessing) vs old_as_submitted | 16 | 5 | 0.0266 |
| in-rules v2 S (strict) vs faithful+strict | 9 | 2 | 0.0654 |
| in-rules v2 S (strict) vs old_as_submitted | 8 | 8 | 1.0000 |
| in-rules v2 S+ (transparent preprocessing) vs in-rules v2 S (strict) | 19 | 4 | 0.0026 |
| in-rules v2 S+ (transparent preprocessing) vs old_as_submitted | 19 | 4 | 0.0026 |
| in-rules v2 S+F (transparent + product filter) vs in-rules v2 S+ (transparent preprocessing) | 2 | 0 | 0.5000 |
| in-rules v2 S+F (transparent + product filter) vs old_as_submitted | 20 | 3 | 0.0005 |
