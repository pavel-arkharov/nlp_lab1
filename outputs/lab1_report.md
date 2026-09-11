# Lab 1 - Natural Language Processing

Generated: 2026-09-11 06:36 UTC

## 1. Import and prepare the NPS Chat corpus

The corpus contains **10,567 posts**, **45,010 raw tokens**, and **34,461 normalized word tokens** representing **4,689 distinct words**.

Method: Lowercase alphabetic words (including internal apostrophes); punctuation and numeric tokens are excluded. Alphabetic chat-event terms such as JOIN and PART remain part of the corpus.

## 2. Twenty most frequent words

| Rank | Word | Frequency |
| --- | --- | --- |
| 1 | i | 1224 |
| 2 | part | 1022 |
| 3 | join | 1021 |
| 4 | lol | 822 |
| 5 | you | 686 |
| 6 | to | 665 |
| 7 | the | 660 |
| 8 | hi | 656 |
| 9 | a | 580 |
| 10 | me | 428 |
| 11 | is | 380 |
| 12 | in | 364 |
| 13 | and | 357 |
| 14 | it | 355 |
| 15 | action | 347 |
| 16 | hey | 292 |
| 17 | that | 284 |
| 18 | my | 259 |
| 19 | of | 207 |
| 20 | u | 204 |

Figure: `figures/02_top_20_words.png`

## 3. Alternative graphical illustrations

The log-log rank-frequency graph tests the Zipf-like shape of the vocabulary. The cumulative-coverage graph shows how quickly common words account for the corpus tokens.

Figures: `figures/03a_zipf_rank_frequency.png` and `figures/03b_cumulative_coverage.png`

## 4. Thirty least-frequent and thirty middle-frequency words

Least-frequency method: Sort by ascending frequency, then alphabetically to resolve ties; take the first 30 words.

| Least-frequency word | Frequency |
| --- | --- |
| aaaaaaaaaaaaaaaaa | 1 |
| aaahhhh | 1 |
| abide | 1 |
| able | 1 |
| abortions | 1 |
| abou | 1 |
| abourted | 1 |
| above | 1 |
| abs | 1 |
| accent | 1 |
| access | 1 |
| accident | 1 |
| aching | 1 |
| acid | 1 |
| ack | 1 |
| acros | 1 |
| across | 1 |
| act | 1 |
| acting | 1 |
| actualy | 1 |
| adams | 1 |
| addict | 1 |
| addicted | 1 |
| addy | 1 |
| adjusts | 1 |
| admit | 1 |
| adopted | 1 |
| adoted | 1 |
| adreniline | 1 |
| adults | 1 |

Middle-frequency method: Choose the 30 frequencies nearest the geometric midpoint between the minimum and maximum observed frequencies. The target frequency is **34.9857**.

| Middle-frequency word | Frequency |
| --- | --- |
| him | 41 |
| then | 41 |
| much | 40 |
| over | 40 |
| man | 39 |
| were | 39 |
| does | 38 |
| his | 38 |
| hot | 38 |
| take | 38 |
| work | 38 |
| even | 37 |
| gonna | 37 |
| come | 36 |
| damn | 36 |
| omg | 36 |
| only | 36 |
| say | 36 |
| seen | 36 |
| more | 35 |
| s | 35 |
| still | 34 |
| brb | 33 |
| day | 33 |
| nick | 33 |
| gay | 31 |
| something | 31 |
| sorry | 31 |
| long | 30 |
| r | 30 |

Figures: `figures/04a_least_30_words.png` and `figures/04b_middle_30_words.png`

## 5. Word length versus frequency

Word frequency is summed for every distinct character length.

| Length | Token frequency | Distinct types |
| --- | --- | --- |
| 1 | 2349 | 23 |
| 2 | 6342 | 150 |
| 3 | 7533 | 425 |
| 4 | 9380 | 856 |
| 5 | 3594 | 873 |
| 6 | 2360 | 786 |
| 7 | 1392 | 633 |
| 8 | 738 | 380 |
| 9 | 374 | 232 |
| 10 | 188 | 150 |
| 11 | 85 | 68 |
| 12 | 51 | 41 |
| 13 | 14 | 14 |
| 14 | 6 | 6 |
| 15 | 9 | 7 |
| 16 | 7 | 6 |
| 17 | 4 | 4 |
| 18 | 4 | 4 |
| 19 | 2 | 2 |
| 20 | 2 | 2 |
| 21 | 2 | 2 |
| 22 | 2 | 2 |
| 23 | 2 | 2 |
| 24 | 1 | 1 |
| 25 | 3 | 3 |
| 26 | 1 | 1 |
| 27 | 1 | 1 |
| 28 | 1 | 1 |
| 29 | 1 | 1 |
| 30 | 1 | 1 |
| 31 | 1 | 1 |
| 32 | 2 | 2 |
| 34 | 1 | 1 |
| 35 | 1 | 1 |
| 36 | 2 | 2 |
| 37 | 1 | 1 |
| 38 | 1 | 1 |
| 50 | 1 | 1 |
| 51 | 1 | 1 |
| 67 | 1 | 1 |

Figure: `figures/05_word_length_frequency.png`

## 6. Modal words and sentence/post lengths

Each NPS Chat post is treated as a sentence-like unit. Modal occurrences count every use; length statistics include each matching post once for that modal.

| Modal | Occurrences | Posts | Mean words | Median words | Mean characters | Median characters |
| --- | --- | --- | --- | --- | --- | --- |
| will | 42 | 42 | 11.26 | 7.0 | 58.76 | 38.0 |
| must | 19 | 16 | 9.12 | 8.5 | 51.56 | 47.0 |
| might | 6 | 6 | 9.5 | 7.5 | 48.5 | 41.5 |
| may | 14 | 14 | 6.79 | 5.0 | 32.14 | 26.5 |
| could | 21 | 21 | 10.9 | 8.0 | 59.29 | 43.0 |
| can | 107 | 104 | 10.61 | 9.0 | 54.27 | 41.0 |

Figures: `figures/06a_modal_frequencies.png`, `figures/06b_modal_post_length_words.png`, and `figures/06c_modal_post_length_characters.png`

## 7. Brown corpus stopwords versus sentence length

The Brown corpus contains **57,340 sentences**. Using NLTK's **198 English stopwords**, the analysis finds **476,394 stopword occurrences** (8.31 per sentence on average).

Pearson correlation with sentence length in words: **0.9555**.

Pearson correlation with sentence length in characters: **0.8998**.

Density plots are used because many sentences share the same values; color records how frequently each region occurs.

Figures: `figures/07a_stopwords_vs_sentence_words.png` and `figures/07b_stopwords_vs_sentence_characters.png`

## Reproducibility notes

- Words are lowercased and punctuation-only tokens are excluded.
- Character lengths use the NLTK tokens joined by one space.
- The complete numerical results are stored in `tables/` and `../docs/data/lab_results.json`.
