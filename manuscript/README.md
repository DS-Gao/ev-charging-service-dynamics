# Manuscript typesetting package

Extract this package into a clean directory and run:

```powershell
python compile_manuscript.py
```

The command writes `main_en.pdf`, `supplement_en.pdf`, and `reader_zh.pdf` in
the extracted directory. All TeX, bibliography, vector-figure, and
exact-parameter inputs use relative paths. The exact-parameter companion
preserves unrounded knots, tests, covariance factors, and related values
behind rounded display text.

## Requirements and pass order

Typesetting requires Python's standard library, XeLaTeX, BibTeX, and the
fonts Times New Roman, Arial, and SimSun. The compiler first runs the English
main article and Supplementary Information, including each bibliography
pass, then performs the additional XeLaTeX passes that resolve cross-document
references. It then performs the final reader passes. No old auxiliary files
or absolute workspace paths are required.

The package is for document production, not statistical estimation. It does
not include every raw charging or policy input or every upstream analytical
script, and it cannot by itself perform a standalone statistical rerun. The
research-materials guide and the analytical index identify the available
entrypoints, exact result locations, and their preconditions. The saved
statistical environment is Python 3.12.0 with NumPy 2.4.2, pandas 3.0.0,
SciPy 1.15.3, statsmodels 0.14.6, and matplotlib 3.10.8; these packages are
not needed by this typesetting compiler.

Formal definitions and result interpretation are in Supplementary Information
S1--S8. The compiler creates the documents from the supplied relative source
files and does not change samples, estimates, or raw data.
