# Evaluation summary

## Tables

| table | rows | distinct fragments |
|---|---:|---:|
| `legacy_embeddings` | 1635 | 924 |
| `audit_naive` | 180 | 145 |
| `audit_faithful` | 456 | 456 |

## DEV set  (13 single-answer, 1 multi-entity, 2 unanswerable queries)

| system | mode | Hit@1 | Hit@3 | MRR@3 | P@3 | Hit@3 FR | Hit@3 EN | Multi cov. | Top-1 shown | Top-1 honest | Top-1 shown (unanswerable) | AUC shown | AUC honest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_as_submitted | relaxed | 0.846 | 0.923 | 0.885 | 0.795 | 1.000 | 0.857 | 1.000 | 0.769 | 0.667 | 0.776 | 0.663 | 0.717 |
| old_db+strict_search | strict | 0.846 | 0.846 | 0.846 | 0.821 | 0.833 | 0.857 | 0.333 | 0.688 | 0.687 | 0.631 | 0.734 | 0.734 |
| naive+strict_search | strict | 0.308 | 0.308 | 0.308 | 0.205 | 0.500 | 0.143 | 0.000 | 0.510 | 0.510 | 0.494 | 0.859 | 0.859 |
| naive+old_search | relaxed | 0.462 | 0.538 | 0.500 | 0.256 | 1.000 | 0.143 | 0.333 | 0.539 | 0.489 | 0.484 | 0.696 | 0.418 |
| faithful+strict_search | strict | 0.846 | 0.846 | 0.846 | 0.769 | 0.667 | 1.000 | 0.000 | 0.647 | 0.647 | 0.614 | 0.846 | 0.846 |
| faithful+old_search | relaxed | 0.923 | 0.923 | 0.923 | 0.795 | 0.833 | 1.000 | 0.667 | 0.714 | 0.625 | 0.688 | 0.730 | 0.707 |
| faithful+bilingual_dict | relaxed | 0.923 | 0.923 | 0.923 | 0.795 | 0.833 | 1.000 | 0.333 | 0.698 | 0.629 | 0.614 | 0.859 | 0.639 |
| faithful+bilingual_mt | relaxed | 0.923 | 0.923 | 0.923 | 0.821 | 0.833 | 1.000 | 0.333 | 0.696 | 0.628 | 0.614 | 0.895 | 0.657 |
| faithful+bilingual_mt+decomp | relaxed | 0.846 | 0.923 | 0.885 | 0.821 | 0.833 | 1.000 | 0.333 | 0.642 | 0.642 | 0.614 | 0.725 | 0.725 |

## TEST set  (50 single-answer, 4 multi-entity, 3 unanswerable queries)

| system | mode | Hit@1 | Hit@3 | MRR@3 | P@3 | Hit@3 FR | Hit@3 EN | Multi cov. | Top-1 shown | Top-1 honest | Top-1 shown (unanswerable) | AUC shown | AUC honest |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_as_submitted | relaxed | 0.660 | 0.720 | 0.687 | 0.533 | 0.621 | 0.857 | 1.000 | 0.651 | 0.589 | 0.528 | 0.749 | 0.625 |
| old_db+strict_search | strict | 0.600 | 0.680 | 0.637 | 0.553 | 0.552 | 0.857 | 0.417 | 0.608 | 0.607 | 0.477 | 0.721 | 0.708 |
| naive+strict_search | strict | 0.340 | 0.440 | 0.387 | 0.267 | 0.345 | 0.571 | 0.333 | 0.503 | 0.503 | 0.399 | 0.662 | 0.662 |
| naive+old_search | relaxed | 0.340 | 0.480 | 0.407 | 0.260 | 0.414 | 0.571 | 0.583 | 0.526 | 0.494 | 0.407 | 0.604 | 0.591 |
| faithful+strict_search | strict | 0.620 | 0.700 | 0.660 | 0.493 | 0.621 | 0.810 | 0.333 | 0.587 | 0.587 | 0.464 | 0.662 | 0.662 |
| faithful+old_search | relaxed | 0.620 | 0.740 | 0.680 | 0.547 | 0.690 | 0.810 | 0.917 | 0.617 | 0.579 | 0.510 | 0.661 | 0.582 |
| faithful+bilingual_dict | relaxed | 0.620 | 0.740 | 0.680 | 0.547 | 0.690 | 0.810 | 0.333 | 0.609 | 0.580 | 0.510 | 0.652 | 0.602 |
| faithful+bilingual_mt | relaxed | 0.660 | 0.760 | 0.710 | 0.573 | 0.724 | 0.810 | 0.333 | 0.656 | 0.571 | 0.565 | 0.562 | 0.522 |
| faithful+bilingual_mt+decomp | relaxed | 0.700 | 0.760 | 0.730 | 0.573 | 0.724 | 0.810 | 0.667 | 0.575 | 0.575 | 0.440 | 0.518 | 0.518 |

## Uncertainty (TEST, single-answer queries)

| system | Hit@3 | 95% bootstrap CI |
|---|---:|---|
| old_as_submitted | 0.720 | [0.60, 0.84] |
| old_db+strict_search | 0.680 | [0.54, 0.80] |
| naive+strict_search | 0.440 | [0.30, 0.58] |
| naive+old_search | 0.480 | [0.34, 0.62] |
| faithful+strict_search | 0.700 | [0.58, 0.82] |
| faithful+old_search | 0.740 | [0.62, 0.86] |
| faithful+bilingual_dict | 0.740 | [0.62, 0.86] |
| faithful+bilingual_mt | 0.760 | [0.64, 0.88] |
| faithful+bilingual_mt+decomp | 0.760 | [0.64, 0.88] |

| A vs B | A-only hits | B-only hits | exact McNemar p (Hit@3) |
|---|---:|---:|---:|
| faithful+strict_search vs naive+strict_search | 15 | 2 | 0.002 |
| faithful+strict_search vs old_as_submitted | 3 | 4 | 1.000 |
| faithful+strict_search vs old_db+strict_search | 3 | 2 | 1.000 |
| faithful+bilingual_mt vs faithful+strict_search | 5 | 2 | 0.453 |
| faithful+bilingual_mt+decomp vs old_as_submitted | 6 | 4 | 0.754 |
| naive+old_search vs naive+strict_search | 2 | 0 | 0.500 |
