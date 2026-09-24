# TEST-4: one-shot evaluation

v1 corpus: 608 fragments {'card': True, 'usage': True, 'rows': 'both', 'docinfo': True, 'fix_splits': True, 'bilingual': 'fr-en'} (chosen on DEV + TEST).
v2 corpus: 609 fragments {'card': True, 'usage': 'AD+FD', 'rows': 'both', 'docinfo': 'plain', 'fix_splits': True, 'bilingual': 'fr-en', 'identity': True} (post-hoc, chosen on DEV + TEST + TEST-2).
S+F = S+ with the product-code filter (suggested by TEST-2 and TEST-3 failures, so only TEST-4 measures it).
TEST-4 was frozen at commit 9ef1b1d, before the configuration it tests was chosen.

| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| old_as_submitted | 43/60 = **0.717** | [0.60, 0.83] | 0.633 | 0.675 | 0.559 | 0.923 | 0.429 | 0.14 | 0.652 / 0.469 |
| faithful+strict | 35/60 = **0.583** | [0.45, 0.70] | 0.483 | 0.528 | 0.412 | 0.808 | 0.393 | 0.07 | 0.608 / 0.460 |
| in-rules v1 S (strict) | 38/60 = **0.633** | [0.52, 0.75] | 0.517 | 0.572 | 0.471 | 0.846 | 0.393 | 0.07 | 0.614 / 0.480 |
| in-rules v1 S+ (transparent preprocessing) | 52/60 = **0.867** | [0.78, 0.95] | 0.717 | 0.792 | 0.882 | 0.846 | 0.607 | 0.29 | 0.691 / 0.511 |
| in-rules v2 S (strict) | 42/60 = **0.700** | [0.58, 0.82] | 0.500 | 0.589 | 0.559 | 0.885 | 0.464 | 0.14 | 0.608 / 0.461 |
| in-rules v2 S+ (transparent preprocessing) | 53/60 = **0.883** | [0.80, 0.97] | 0.800 | 0.839 | 0.882 | 0.885 | 0.643 | 0.36 | 0.687 / 0.498 |
| in-rules v2 S+F (transparent + product filter) | 56/60 = **0.933** | [0.87, 0.98] | 0.850 | 0.892 | 0.912 | 0.962 | 0.929 | 0.86 | 0.695 / 0.498 |

## Paired tests (TEST-4, single-answer questions, Hit@3)

| A vs B | only A right | only B right | exact McNemar p |
|---|---:|---:|---:|
| in-rules v1 S (strict) vs faithful+strict | 4 | 1 | 0.3750 |
| in-rules v1 S (strict) vs old_as_submitted | 3 | 8 | 0.2266 |
| in-rules v1 S+ (transparent preprocessing) vs old_as_submitted | 12 | 3 | 0.0352 |
| in-rules v2 S (strict) vs faithful+strict | 7 | 0 | 0.0156 |
| in-rules v2 S (strict) vs old_as_submitted | 4 | 5 | 1.0000 |
| in-rules v2 S+ (transparent preprocessing) vs in-rules v2 S (strict) | 11 | 0 | 0.0010 |
| in-rules v2 S+ (transparent preprocessing) vs old_as_submitted | 11 | 1 | 0.0063 |
| in-rules v2 S+F (transparent + product filter) vs in-rules v2 S+ (transparent preprocessing) | 3 | 0 | 0.2500 |
| in-rules v2 S+F (transparent + product filter) vs old_as_submitted | 14 | 1 | 0.0010 |

## TEST-4 details

Named products found by the filter's code matching: 74/74 answerable questions (exactly the products the question is about).

### Single-product questions by attribute group (right answer in top 3)

| group | n | old_as_submitted | in-rules v2 S (strict) | in-rules v2 S+ (transparent preprocessing) | in-rules v2 S+F (transparent + product filter) |
|---|---:|---:|---:|---:|---:|
| specific | 34 | 30/34 | 31/34 | 33/34 | 34/34 |
| header | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| shared | 24 | 11/24 | 9/24 | 18/24 | 20/24 |

### Single-product questions by how the code is written (right answer in top 3)

| code written | n | in-rules v2 S+ (transparent preprocessing) | in-rules v2 S+F (transparent + product filter) |
|---|---:|---:|---:|
| printed+brand | 28 | 27/28 | 27/28 |
| printed | 12 | 10/12 | 11/12 |
| lower | 6 | 6/6 | 6/6 |
| respaced | 7 | 4/7 | 6/7 |
| hyphenated | 7 | 6/7 | 6/7 |
| (also names its own family) | 18 | 15/18 | 16/18 |

### Questions naming several things (share of named products / families covered in the top 3)

| kind | n | old_as_submitted | in-rules v2 S (strict) | in-rules v2 S+ (transparent preprocessing) | in-rules v2 S+F (transparent + product filter) |
|---|---:|---:|---:|---:|---:|
| two products | 8 | 0.44 | 0.50 | 0.62 | 0.94 |
| product + family | 6 | 0.42 | 0.42 | 0.67 | 0.92 |
