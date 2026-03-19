# Print-Ready Study Documents

Open in a browser → Ctrl+P → Save as PDF.

## Master Document (use this one)

- **`gpu-waste-study-complete.html`** — Complete paper: manuscript (sections 1–8, references) + Appendix A (results tables 1–4) + Appendix B (supplementary tables S1–S5). This is the journal-submission format.

## Individual Sections (for reference)

- `study-simulation-manuscript.html` — Manuscript body only
- `study-results-summary.html` — Main results tables only
- `study-results-supplement.html` — Supplementary tables only

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
