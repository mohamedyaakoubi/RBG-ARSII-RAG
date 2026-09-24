# Post-hoc chunking exploration (strict search only)

These variants were designed after looking at TEST failures of the pre-registered
pipelines, so their TEST numbers are optimistic. All fragments are verbatim PDF text.

## DEV

| chunking | fragments | mean returned length (chars) | Hit@1 | Hit@3 | MRR@3 | P@3 | Hit@3 FR | Hit@3 EN | Multi cov. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faithful | 456 | 110 | 0.846 | 0.846 | 0.846 | 0.769 | 0.667 | 1.000 | 0.000 |
| naive | 180 | 471 | 0.308 | 0.308 | 0.308 | 0.205 | 0.500 | 0.143 | 0.000 |
| faithful_page | 73 | 687 | 0.615 | 0.769 | 0.692 | 0.692 | 1.000 | 0.571 | 0.333 |
| faithful_rows | 471 | 110 | 0.846 | 0.846 | 0.846 | 0.769 | 0.667 | 1.000 | 0.000 |
| faithful_usage | 388 | 184 | 0.692 | 0.846 | 0.756 | 0.667 | 0.667 | 1.000 | 0.000 |
| faithful_usage_rows | 403 | 184 | 0.692 | 0.846 | 0.756 | 0.667 | 0.667 | 1.000 | 0.000 |

## TEST

| chunking | fragments | mean returned length (chars) | Hit@1 | Hit@3 | MRR@3 | P@3 | Hit@3 FR | Hit@3 EN | Multi cov. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faithful | 456 | 161 | 0.620 | 0.700 | 0.660 | 0.493 | 0.621 | 0.810 | 0.333 |
| naive | 180 | 484 | 0.340 | 0.440 | 0.387 | 0.267 | 0.345 | 0.571 | 0.333 |
| faithful_page | 73 | 756 | 0.640 | 0.800 | 0.707 | 0.553 | 0.690 | 0.952 | 0.583 |
| faithful_rows | 471 | 148 | 0.600 | 0.720 | 0.657 | 0.507 | 0.655 | 0.810 | 0.333 |
| faithful_usage | 388 | 212 | 0.640 | 0.700 | 0.670 | 0.467 | 0.586 | 0.857 | 0.333 |
| faithful_usage_rows | 403 | 193 | 0.620 | 0.720 | 0.667 | 0.480 | 0.621 | 0.857 | 0.333 |
