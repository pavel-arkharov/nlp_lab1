# NLP Lab 1

This project answers the tasks in `Lab1_2026.docx` with a reproducible Python
analysis and an optional static presentation for GitHub Pages.

## Quick start

From the `Lab1` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python lab1_analysis.py
```

The first analysis run downloads the required NLTK corpora (`nps_chat`, `brown`,
and `stopwords`) if they are not already installed. There is no interactive input.

The terminal prints numbered progress for Tasks 1-7. Generated files are written
to:

- `outputs/lab1_report.md`: numbered written answers
- `outputs/figures/`: eleven labeled PNG plots
- `outputs/tables/`: complete numerical CSV data
- `docs/data/lab_results.json`: data used by the website

## Run the tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## Preview the presentation

The page must be served over HTTP because it loads generated JSON:

```bash
source .venv/bin/activate
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. Stop the server with `Ctrl+C`.

The displayed student name is configured in `docs/config.js`.

## Analysis choices

- Words are lowercased; punctuation and numeric tokens are excluded.
- Alphabetic NPS Chat event terms such as `JOIN` and `PART` are retained.
- The thirty least-frequent words are selected by ascending frequency, with
  alphabetical tie-breaking.
- The middle-frequency sample consists of the thirty words closest to the
  geometric midpoint of the observed frequency range. This accounts for the
  strongly skewed distribution of natural-language frequencies.
- Each NPS Chat post is treated as a sentence-like unit for modal analysis.
- Character length is measured after joining corpus tokens with one space.
- Brown sentence plots use density because many sentences overlap at identical
  length and stopword-count values.

## Publish with GitHub Pages

1. Create an empty GitHub repository for this project.
2. Initialize and push this folder to the repository's `main` branch.
3. In the repository, open **Settings > Pages**.
4. Choose **Deploy from a branch**, select `main`, select `/docs`, and save.

GitHub will display the public URL after the first deployment completes. Flask is
not used: GitHub Pages serves the static files in `docs`, while the Python script
is run locally whenever the analysis data needs to be regenerated.
