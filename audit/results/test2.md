# TEST-2: one-shot evaluation of the in-rules system

Final corpus: 608 fragments ({'card': True, 'usage': True, 'rows': 'both', 'docinfo': True, 'fix_splits': True, 'bilingual': 'fr-en'}).
TEST-2 was frozen (commit 9cb134a) before the configuration was chosen on DEV + TEST.

## dev pool (DEV+TEST)

| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| old_as_submitted | 48/63 = **0.762** | [0.65, 0.86] | 0.698 | 0.728 | 0.686 | 0.857 | 1.000 | 1.00 | 0.680 / 0.627 |
| faithful+strict | 46/63 = **0.730** | [0.62, 0.84] | 0.667 | 0.698 | 0.629 | 0.857 | 0.267 | 0.00 | 0.606 / 0.524 |
| in-rules S (strict) | 50/63 = **0.794** | [0.68, 0.89] | 0.683 | 0.735 | 0.686 | 0.929 | 0.333 | 0.00 | 0.614 / 0.526 |
| in-rules S+ (transparent preprocessing) | 58/63 = **0.921** | [0.84, 0.98] | 0.810 | 0.860 | 0.914 | 0.929 | 0.767 | 0.40 | 0.689 / 0.595 |

## TEST-2 (held out)

| system | right answer in top 3 (Hit@3) | 95% CI | Hit@1 | MRR@3 | Hit@3 FR | Hit@3 EN | multi-product coverage | all products covered | mean top-1 score: answerable / unanswerable |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| old_as_submitted | 43/70 = **0.614** | [0.50, 0.73] | 0.457 | 0.524 | 0.556 | 0.812 | 0.517 | 0.30 | 0.651 / 0.612 |
| faithful+strict | 36/70 = **0.514** | [0.40, 0.63] | 0.386 | 0.448 | 0.444 | 0.750 | 0.333 | 0.00 | 0.581 / 0.564 |
| in-rules S (strict) | 40/70 = **0.571** | [0.46, 0.69] | 0.443 | 0.498 | 0.481 | 0.875 | 0.433 | 0.10 | 0.595 / 0.572 |
| in-rules S+ (transparent preprocessing) | 54/70 = **0.771** | [0.67, 0.87] | 0.657 | 0.702 | 0.741 | 0.875 | 0.800 | 0.70 | 0.662 / 0.623 |

## Paired tests on TEST-2 (single-answer questions, Hit@3)

| A vs B | only A right | only B right | exact McNemar p |
|---|---:|---:|---:|
| in-rules S (strict) vs faithful+strict | 7 | 3 | 0.3438 |
| in-rules S (strict) vs old_as_submitted | 6 | 9 | 0.6072 |
| in-rules S+ (transparent preprocessing) vs in-rules S (strict) | 18 | 4 | 0.0043 |
| in-rules S+ (transparent preprocessing) vs old_as_submitted | 16 | 5 | 0.0266 |
