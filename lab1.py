"""NLP Laboratory 1 - Pavel Arkharov."""

import math
import re
from collections import Counter
from pathlib import Path

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


def save_figure(figure, filename):
    """Save one labeled figure in the lab1_figures directory."""
    path = FIGURE_DIR / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"    Figure saved: {path.name}")


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    for package in ("nps_chat", "brown", "stopwords"):
        nltk.download(package, quiet=True)

    # Task 1: import and prepare the NPS Chat corpus.
    posts = list(nps_chat.posts())
    raw_tokens = [token for post in posts for token in post]
    words = clean_words(raw_tokens)
    word_frequency = FreqDist(words)

    print("\n1. NPS CHAT CORPUS")
    print(f"   Posts: {len(posts):,}")
    print(f"   Raw tokens: {len(raw_tokens):,}")
    print(f"   Normalized word tokens: {len(words):,}")
    print(f"   Distinct words: {len(word_frequency):,}")

    # Task 2: plot the twenty most frequent words.
    top_20 = word_frequency.most_common(20)
    print("\n2. TWENTY MOST FREQUENT WORDS")
    for rank, (word, count) in enumerate(top_20, start=1):
        print(f"   {rank:2}. {word:<12} {count}")

    labels = [word for word, count in reversed(top_20)]
    counts = [count for word, count in reversed(top_20)]
    figure, axis = plt.subplots(figsize=(10, 7))
    bars = axis.barh(labels, counts, color="#2d6cdf")
    axis.bar_label(bars, padding=3)
    axis.set_title("Twenty most frequent words in NPS Chat")
    axis.set_xlabel("Frequency")
    axis.set_ylabel("Word")
    save_figure(figure, "02_top_20_words.png")

    # Task 3: alternative rank-frequency and cumulative-coverage plots.
    frequencies = sorted(word_frequency.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)
    running_total = 0
    coverage = []
    for frequency in frequencies:
        running_total += frequency
        coverage.append(running_total / len(words) * 100)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(ranks, frequencies, color="#2d6cdf")
    axes[0].set_title("Rank-frequency distribution")
    axes[0].set_xlabel("Word rank (log scale)")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[1].semilogx(ranks, coverage, color="#e05a47")
    axes[1].set_title("Cumulative token coverage")
    axes[1].set_xlabel("Word types included (log scale)")
    axes[1].set_ylabel("Coverage (%)")
    figure.suptitle("3. Alternative frequency illustrations")
    figure.tight_layout()
    save_figure(figure, "03_alternative_views.png")
    print("\n3. ALTERNATIVE ILLUSTRATIONS")
    print("   Created a Zipf-style rank plot and cumulative-coverage plot.")

    # Task 4: select thirty least- and middle-frequency words.
    least_30 = sorted(
        word_frequency.items(), key=lambda item: (item[1], item[0])
    )[:30]
    middle_target = math.sqrt(
        min(word_frequency.values()) * max(word_frequency.values())
    )
    middle_30 = sorted(
        word_frequency.items(),
        key=lambda item: (
            abs(math.log(item[1]) - math.log(middle_target)), item[0]
        ),
    )[:30]
    middle_30.sort(key=lambda item: (-item[1], item[0]))

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
        axis.set_title(title)
        axis.set_xlabel("Frequency")
        axis.set_ylabel("Word")
    figure.tight_layout()
    save_figure(figure, "04_frequency_ranges.png")

    # Task 5: sum word frequency for every character length.
    frequency_by_length = Counter()
    for word, frequency in word_frequency.items():
        frequency_by_length[len(word)] += frequency

    lengths = sorted(frequency_by_length)
    length_frequencies = [frequency_by_length[length] for length in lengths]
    print("\n5. WORD LENGTH VERSUS FREQUENCY")
    for length in lengths:
        print(f"   {length:2} characters: {frequency_by_length[length]:,}")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(lengths, length_frequencies, color="#21867a")
    axis.plot(lengths, length_frequencies, color="#1f2925", marker="o")
    axis.set_title("NPS Chat word length versus frequency")
    axis.set_xlabel("Word length in characters")
    axis.set_ylabel("Total frequency")
    save_figure(figure, "05_word_length_frequency.png")

    # Task 6: modal frequencies and lengths of posts containing each modal.
    modal_word_lengths = {modal: [] for modal in MODALS}
    modal_character_lengths = {modal: [] for modal in MODALS}
    for post in posts:
        post_words = clean_words(post)
        for modal in MODALS:
            if modal in post_words:
                modal_word_lengths[modal].append(len(post_words))
                modal_character_lengths[modal].append(len(" ".join(post)))

    print("\n6. MODAL WORDS")
    for modal in MODALS:
        print(f"   {modal}: {word_frequency[modal]} occurrences")
        print(f"      Post lengths in words: {modal_word_lengths[modal]}")
        print(f"      Post lengths in characters: {modal_character_lengths[modal]}")

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(MODALS, [word_frequency[m] for m in MODALS], color="#2d6cdf")
    axes[0].set_title("Modal-word frequencies")
    axes[0].set_xlabel("Modal word")
    axes[0].set_ylabel("Frequency")
    axes[1].boxplot(
        [modal_word_lengths[m] for m in MODALS], tick_labels=MODALS,
        showfliers=False
    )
    axes[1].set_title("Matching post lengths")
    axes[1].set_xlabel("Modal word")
    axes[1].set_ylabel("Words")
    axes[2].boxplot(
        [modal_character_lengths[m] for m in MODALS], tick_labels=MODALS,
        showfliers=False
    )
    axes[2].set_title("Matching post character lengths")
    axes[2].set_xlabel("Modal word")
    axes[2].set_ylabel("Characters")
    figure.tight_layout()
    save_figure(figure, "06_modal_words.png")

    # Task 7: Brown stopwords versus sentence length.
    english_stopwords = set(stopwords.words("english"))
    brown_word_lengths = []
    brown_character_lengths = []
    brown_stopword_counts = []

    for sentence in brown.sents():
        sentence_words = clean_words(sentence)
        brown_word_lengths.append(len(sentence_words))
        brown_character_lengths.append(len(" ".join(sentence)))
        brown_stopword_counts.append(
            sum(word in english_stopwords for word in sentence_words)
        )

    print("\n7. BROWN CORPUS STOPWORDS")
    print(f"   Sentences: {len(brown_word_lengths):,}")
    print(f"   Total stopwords: {sum(brown_stopword_counts):,}")
    print(
        "   Mean stopwords per sentence: "
        f"{sum(brown_stopword_counts) / len(brown_stopword_counts):.2f}"
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    word_plot = axes[0].hexbin(
        brown_word_lengths, brown_stopword_counts,
        gridsize=45, mincnt=1, bins="log", cmap="viridis"
    )
    axes[0].set_title("Stopwords versus sentence length")
    axes[0].set_xlabel("Sentence length in words")
    axes[0].set_ylabel("Number of stopwords")
    figure.colorbar(word_plot, ax=axes[0], label="Sentence density")

    character_plot = axes[1].hexbin(
        brown_character_lengths, brown_stopword_counts,
        gridsize=45, mincnt=1, bins="log", cmap="viridis"
    )
    axes[1].set_title("Stopwords versus character length")
    axes[1].set_xlabel("Sentence length in characters")
    axes[1].set_ylabel("Number of stopwords")
    figure.colorbar(character_plot, ax=axes[1], label="Sentence density")
    figure.tight_layout()
    save_figure(figure, "07_brown_stopwords.png")

    print(f"\nDone. Figures are in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
