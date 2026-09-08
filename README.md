# NLP Lab 1

**Author:** Pavel Arkharov

**Subject:** Natural Language Processing

This repository presents the completed NLP laboratory work using Python and
NLTK. The primary result is a compact, reproducible corpus-analysis script
supported by numbered written answers, data tables, and static figures.

An [interactive web presentation](https://pavel-arkharov.github.io/nlp_lab1/)
accompanies the analysis. It gives the teacher a structured view of every lab
answer and demonstrates to other students how Python output can be represented
in a more accessible, explorable format.

## Lab work

### 1. NPS Chat corpus preparation

The NPS Chat corpus is imported and normalized for analysis. The program reports
the number of posts, raw tokens, normalized word tokens, distinct word types, and
lexical diversity.

### 2. Twenty most frequent words

The twenty most frequent normalized words are calculated and displayed in a
labeled frequency bar chart. The corresponding counts are also stored in a CSV
table.

### 3. Alternative graphical illustrations

Two additional views describe the frequency distribution:

- A log-log rank-frequency plot illustrates its Zipf-like shape.
- A cumulative-coverage curve shows how much of the corpus is represented as
  increasingly rare words are included.

### 4. Least- and middle-frequency words

The program selects and plots thirty least-frequent words and thirty words from
the middle of the observed frequency range. Selection rules are deterministic
and documented so that the results can be reproduced and interpreted.

### 5. Word length and frequency

Word occurrences are grouped by character length. The resulting chart compares
word length with total token frequency and also records the number of distinct
word types at each length.

### 6. Modal-word analysis

The frequencies of `will`, `must`, `might`, `may`, `could`, and `can` are
calculated. For every NPS Chat post containing one of these modal words, the
program records its length in words and characters. Summary statistics and
distribution plots are provided for each modal.

### 7. Brown corpus stopword analysis

For every sentence in the Brown corpus, the program calculates the number of
English stopwords and the sentence length in both words and characters. Density
plots show how stopword count changes with sentence length, while the full
per-sentence measurements remain available as CSV data.

## Methodology

- Words are converted to lowercase; punctuation and numeric tokens are excluded.
- Alphabetic NPS Chat event terms such as `JOIN` and `PART` are retained.
- Frequency ties in the least-frequent sample are resolved alphabetically.
- The middle-frequency sample uses the geometric midpoint of the observed
  frequency range because natural-language frequencies are strongly skewed.
- Each NPS Chat post is treated as a sentence-like unit for modal analysis.
- Character length is measured after joining corpus tokens with one space.
- Density plots are used for Brown corpus results because many sentences share
  identical length and stopword-count values.

## Project results

- [`lab1.py`](lab1.py) is the concise, self-contained laboratory submission.
- [`tools/generate_site_outputs.py`](tools/generate_site_outputs.py) creates the
  extended report files and website dataset; it is not part of the submission.
- [`outputs/lab1_report.md`](outputs/lab1_report.md) contains the written answers.
- [`outputs/figures/`](outputs/figures/) contains eleven labeled PNG figures.
- [`outputs/tables/`](outputs/tables/) contains the complete numerical CSV data.
- [`docs/data/lab_results.json`](docs/data/lab_results.json) connects the Python
  output to the interactive presentation.

## Interactive presentation

The presentation is a separate display layer rather than a replacement for the
Python work. It reads the JSON produced by the analysis and presents the same
results through numbered expandable sections, animated charts, responsive
tables, tooltips, and a compact corpus summary. No analytical values are manually
duplicated in the JavaScript interface.
