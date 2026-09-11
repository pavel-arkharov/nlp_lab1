"""NLP Laboratory 1 - Pavel Arkharov.

Web presentation: https://pavel-arkharov.github.io/nlp_lab1/
"""

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown, nps_chat, stopwords
from nltk.probability import FreqDist


MODALS = ("will", "must", "might", "may", "could", "can")
WORD_PATTERN = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
FIGURE_DIR = Path(__file__).with_name("lab1_figures")


def clean_words(tokens):
    """Lowercase words and remove punctuation and numeric tokens."""
    return [token.lower() for token in tokens if WORD_PATTERN.fullmatch(token)]


def character_length(tokens):
    """Count characters after joining corpus tokens with one space."""
    return len(" ".join(tokens))


def ensure_nltk_data(allow_download=True):
    """Use installed corpora, downloading only resources that are missing."""
    resources = {
        "nps_chat": "corpora/nps_chat",
        "brown": "corpora/brown",
        "stopwords": "corpora/stopwords",
    }
    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not allow_download:
                raise RuntimeError(f"Missing NLTK resource: {package}") from None
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Could not download NLTK resource: {package}")


def task_1_prepare_nps_chat():
    """Load NPS Chat and create the frequency distribution used by Tasks 2-6."""
    posts = [list(post) for post in nps_chat.posts()]
    raw_tokens = [token for post in posts for token in post]
    words = clean_words(raw_tokens)
    frequencies = FreqDist(words)
    return posts, raw_tokens, words, frequencies


def task_2_most_frequent(frequencies):
    """Return the twenty words with the largest occurrence counts."""
    return frequencies.most_common(20)


def task_3_frequency_views(frequencies):
    """Calculate word rank and cumulative corpus coverage."""
    total_words = frequencies.N()
    cumulative = 0
    rows = []
    for rank, (word, frequency) in enumerate(frequencies.most_common(), start=1):
        cumulative += frequency
        coverage = cumulative / total_words * 100
        rows.append((rank, word, frequency, coverage))
    return rows


def task_4_frequency_samples(frequencies, sample_size=30):
    """Select the least words and words near the middle of the frequency scale."""
    least = sorted(frequencies.items(), key=lambda item: (item[1], item[0]))[
        :sample_size
    ]

    # A geometric midpoint suits the strongly skewed word-frequency scale.
    target = math.sqrt(min(frequencies.values()) * max(frequencies.values()))
    middle = sorted(
        frequencies.items(),
        key=lambda item: (
            abs(math.log(item[1]) - math.log(target)),
            item[0],
        ),
    )[:sample_size]
    middle.sort(key=lambda item: (-item[1], item[0]))
    return least, middle, target


def task_5_word_lengths(frequencies):
    """Group token frequency and distinct word types by character length."""
    token_totals = Counter()
    type_totals = Counter()
    for word, frequency in frequencies.items():
        token_totals[len(word)] += frequency
        type_totals[len(word)] += 1
    return [
        (length, token_totals[length], type_totals[length])
        for length in sorted(token_totals)
    ]


def task_6_modal_words(posts):
    """Count modal words and measure every post containing each modal."""
    occurrences = Counter({modal: 0 for modal in MODALS})
    word_lengths = {modal: [] for modal in MODALS}
    character_lengths = {modal: [] for modal in MODALS}
    matching_posts = []

    for post_number, post in enumerate(posts, start=1):
        post_words = clean_words(post)
        counts = Counter(post_words)
        for modal in MODALS:
            occurrences[modal] += counts[modal]
            if counts[modal]:
                # A matching post contributes once to that modal's length data.
                word_lengths[modal].append(len(post_words))
                character_lengths[modal].append(character_length(post))
                matching_posts.append(
                    (
                        modal,
                        post_number,
                        counts[modal],
                        len(post_words),
                        character_length(post),
                    )
                )
    return occurrences, word_lengths, character_lengths, matching_posts


def task_7_brown_stopwords():
    """Measure stopword count and length for every Brown corpus sentence."""
    english_stopwords = set(stopwords.words("english"))
    metrics = []
    for sentence_number, sentence in enumerate(brown.sents(), start=1):
        sentence_words = clean_words(sentence)
        stopword_count = sum(
            word in english_stopwords for word in sentence_words
        )
        metrics.append(
            (
                sentence_number,
                stopword_count,
                len(sentence_words),
                character_length(sentence),
            )
        )
    return english_stopwords, metrics


def save_figure(figure, filename):
    """Save one labeled figure in the lab1_figures directory."""
    path = FIGURE_DIR / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"    Figure saved: {path.name}")


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    ensure_nltk_data()

    posts, raw_tokens, words, frequencies = task_1_prepare_nps_chat()
    print("\n1. NPS CHAT CORPUS")
    print(f"   Posts: {len(posts):,}")
    print(f"   Raw tokens: {len(raw_tokens):,}")
    print(f"   Normalized word tokens: {len(words):,}")
    print(f"   Distinct words: {len(frequencies):,}")

    top_20 = task_2_most_frequent(frequencies)
    print("\n2. TWENTY MOST FREQUENT WORDS")
    for rank, (word, count) in enumerate(top_20, start=1):
        print(f"   {rank:2}. {word:<12} {count}")

    labels = [word for word, count in reversed(top_20)]
    counts = [count for word, count in reversed(top_20)]
    figure, axis = plt.subplots(figsize=(10, 7))
    bars = axis.barh(labels, counts, color="#2d6cdf")
    axis.bar_label(bars, padding=3)
    axis.set(title="Twenty most frequent words in NPS Chat",
             xlabel="Frequency", ylabel="Word")
    save_figure(figure, "02_top_20_words.png")

    rank_rows = task_3_frequency_views(frequencies)
    print("\n3. ALTERNATIVE ILLUSTRATIONS")
    print("   Creating a rank-frequency plot and cumulative-coverage plot.")
    ranks = [row[0] for row in rank_rows]
    ranked_frequencies = [row[2] for row in rank_rows]
    coverage = [row[3] for row in rank_rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(ranks, ranked_frequencies, color="#2d6cdf")
    axes[0].set(title="Rank-frequency distribution",
                xlabel="Word rank (log scale)", ylabel="Frequency (log scale)")
    axes[1].semilogx(ranks, coverage, color="#e05a47")
    axes[1].set(title="Cumulative token coverage",
                xlabel="Word types included (log scale)", ylabel="Coverage (%)")
    figure.tight_layout()
    save_figure(figure, "03_alternative_views.png")

    least_30, middle_30, middle_target = task_4_frequency_samples(frequencies)
    print("\n4. FREQUENCY-RANGE WORDS")
    print("   Least frequent:", least_30)
    print(f"   Geometric middle target: {middle_target:.2f}")
    print("   Middle frequency:", middle_30)

    figure, axes = plt.subplots(1, 2, figsize=(14, 9))
    for axis, records, title, color in (
        (axes[0], least_30, "Thirty least-frequent words", "#e05a47"),
        (axes[1], middle_30, "Thirty middle-frequency words", "#d99a2b"),
    ):
        labels = [word for word, count in reversed(records)]
        counts = [count for word, count in reversed(records)]
        axis.barh(labels, counts, color=color)
        axis.set(title=title, xlabel="Frequency", ylabel="Word")
    figure.tight_layout()
    save_figure(figure, "04_frequency_ranges.png")

    length_rows = task_5_word_lengths(frequencies)
    print("\n5. WORD LENGTH VERSUS FREQUENCY")
    for length, token_count, type_count in length_rows:
        print(f"   {length:2} characters: {token_count:,}")
    figure, axis = plt.subplots(figsize=(10, 5))
    lengths = [row[0] for row in length_rows]
    length_frequencies = [row[1] for row in length_rows]
    axis.bar(lengths, length_frequencies, color="#21867a")
    axis.plot(lengths, length_frequencies, color="#1f2925", marker="o")
    axis.set(title="NPS Chat word length versus frequency",
             xlabel="Word length in characters", ylabel="Total frequency")
    save_figure(figure, "05_word_length_frequency.png")

    modal_counts, modal_words, modal_characters, _matching_posts = (
        task_6_modal_words(posts)
    )
    print("\n6. MODAL WORDS")
    for modal in MODALS:
        print(f"   {modal}: {modal_counts[modal]} occurrences")
        print(f"      Post lengths in words: {modal_words[modal]}")
        print(f"      Post lengths in characters: {modal_characters[modal]}")

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(MODALS, [modal_counts[modal] for modal in MODALS], color="#2d6cdf")
    axes[0].set(title="Modal-word frequencies", xlabel="Modal word", ylabel="Frequency")
    axes[1].boxplot([modal_words[modal] for modal in MODALS],
                    tick_labels=MODALS, showfliers=False)
    axes[1].set(title="Matching post lengths", xlabel="Modal word", ylabel="Words")
    axes[2].boxplot([modal_characters[modal] for modal in MODALS],
                    tick_labels=MODALS, showfliers=False)
    axes[2].set(title="Matching post character lengths",
                xlabel="Modal word", ylabel="Characters")
    figure.tight_layout()
    save_figure(figure, "06_modal_words.png")

    _english_stopwords, brown_metrics = task_7_brown_stopwords()
    stopword_counts = [row[1] for row in brown_metrics]
    word_lengths = [row[2] for row in brown_metrics]
    character_lengths = [row[3] for row in brown_metrics]
    print("\n7. BROWN CORPUS STOPWORDS")
    print(f"   Sentences: {len(brown_metrics):,}")
    print(f"   Total stopwords: {sum(stopword_counts):,}")
    print(f"   Mean stopwords per sentence: "
          f"{sum(stopword_counts) / len(stopword_counts):.2f}")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, lengths, title, label in (
        (axes[0], word_lengths, "Stopwords versus sentence length", "Words"),
        (axes[1], character_lengths, "Stopwords versus character length", "Characters"),
    ):
        density = axis.hexbin(lengths, stopword_counts, gridsize=45,
                              mincnt=1, bins="log", cmap="viridis")
        axis.set(title=title, xlabel=f"Sentence length in {label.lower()}",
                 ylabel="Number of stopwords")
        figure.colorbar(density, ax=axis, label="Sentence density")
    figure.tight_layout()
    save_figure(figure, "07_brown_stopwords.png")

    print(f"\nDone. Figures are in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
