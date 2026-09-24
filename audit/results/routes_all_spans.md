# Post-hoc: every possible fragment in one index

Not in ROUTES_PLAN.md: added after the span oracle, to test whether its fragments still win when they compete with each other. 65772 fragments (every run of consecutive lines of every document, with the product header, also after its section heading), current extractor.

## known questions

| route | fragments | fully answered, transparent (S+F) | vs current: gained / lost / p | fully answered, strict (S) | vs current: gained / lost / p | right answer 1st (S+F, single) |
|---|---:|---:|---|---:|---|---:|
| current chunking | 609 | 251/281 | +0 / -0 / 1.00 | 184/281 | +0 / -0 / 1.00 | 201/245 |
| every possible fragment | 65772 | 216/281 | +12 / -47 / 0.00 | 147/281 | +15 / -52 / 0.00 | 167/245 |

## TEST-5

| route | fragments | fully answered, transparent (S+F) | vs current: gained / lost / p | fully answered, strict (S) | vs current: gained / lost / p | right answer 1st (S+F, single) |
|---|---:|---:|---|---:|---|---:|
| current chunking | 609 | 88/100 | +0 / -0 / 1.00 | 61/100 | +0 / -0 / 1.00 | 70/88 |
| every possible fragment | 65772 | 75/100 | +5 / -18 / 0.01 | 50/100 | +7 / -18 / 0.04 | 57/88 |

