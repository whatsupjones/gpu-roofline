# Print-Ready Study Documents

Open any `.html` file in a browser and print to PDF (Ctrl+P → Save as PDF).

## Files

- `study-simulation-manuscript.html` — Full 8-section publication manuscript with table of contents
- `study-results-summary.html` — Main results tables (Tables 1–4, bootstrap CIs)
- `study-results-supplement.html` — Supplementary tables (S1–S5, sensitivity, confusion matrices)

## Regenerate

Requires [pandoc](https://pandoc.org/) 3.x:

```bash
pandoc docs/study-simulation-manuscript.md \
  --standalone --embed-resources \
  --css=docs/print/study-print.css \
  --toc --toc-depth=3 --number-sections \
  -o docs/print/study-simulation-manuscript.html

pandoc docs/study-results/summary.md \
  --standalone --embed-resources \
  --css=docs/print/study-print.css \
  -o docs/print/study-results-summary.html

pandoc docs/study-results/supplement.md \
  --standalone --embed-resources \
  --css=docs/print/study-print.css \
  -o docs/print/study-results-supplement.html
```
