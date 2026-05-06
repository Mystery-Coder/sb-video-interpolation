# IEEE LaTeX Paper

This folder contains an IEEE journal-style LaTeX manuscript based on the repository implementation.

## Build

From this folder:

- `pdflatex main.tex`
- `bibtex main`
- `pdflatex main.tex`
- `pdflatex main.tex`

Or (if you have latexmk):

- `latexmk -pdf main.tex`

## Notes

- Edit the title/authors in `main.tex`.
- `refs.bib` contains BibTeX entries from the project literature survey. Replace placeholder authors/venues/DOIs with verified metadata as needed.
