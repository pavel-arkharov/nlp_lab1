#!/usr/bin/env python3
"""Generate all numbered analyses and artifacts for NLP Lab 1."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lab1 as submission


DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_SITE_DATA = ROOT / "docs" / "data" / "lab_results.json"
DEFAULT_SITE_REPORT = ROOT / "docs" / "downloads" / "lab1_report.md"
DEFAULT_SITE_FIGURES = ROOT / "docs" / "figures"

MODALS = submission.MODALS
COLORS = {
    "blue": "#2D6CDF",
    "coral": "#E05A47",
    "teal": "#21867A",
    "gold": "#D99A2B",
    "ink": "#1F2925",
    "paper": "#F6F8F7",
}

CODE_GUIDES = {
    "task_1": {
        "functions": (submission.clean_words, submission.task_1_prepare_nps_chat),
        "summary": "The corpus is flattened into tokens, cleaned, and converted into one reusable frequency table.",
        "steps": (
            "Read every NPS Chat post and keep each post as a list of tokens.",
            "Flatten the posts, lowercase real words, and discard punctuation and numbers.",
            "Build an NLTK FreqDist so later tasks can query word counts.",
        ),
    },
    "task_2": {
        "functions": (submission.task_2_most_frequent,),
        "summary": "NLTK sorts the frequency table and returns the twenty largest counts.",
        "steps": (
            "Receive the frequency distribution created in Task 1.",
            "Call most_common(20) to rank words by occurrence count.",
            "Use the returned word-count pairs for both the table and bar chart.",
        ),
    },
    "task_3": {
        "functions": (submission.task_3_frequency_views,),
        "summary": "Each vocabulary item receives a rank and a running percentage of all corpus tokens.",
        "steps": (
            "Walk through every word from most frequent to least frequent.",
            "Add each frequency to a running total.",
            "Convert that total to a percentage for the cumulative-coverage curve.",
        ),
    },
    "task_4": {
        "functions": (submission.task_4_frequency_samples,),
        "summary": "Rare words are selected directly; middle-frequency words are selected around a geometric midpoint.",
        "steps": (
            "Sort ascending by frequency and alphabetically to choose thirty rare words consistently.",
            "Calculate the geometric midpoint between the smallest and largest counts.",
            "Choose the thirty words closest to that midpoint on a logarithmic scale.",
        ),
    },
    "task_5": {
        "functions": (submission.task_5_word_lengths,),
        "summary": "Word counts are regrouped by the number of characters in each distinct word.",
        "steps": (
            "Visit every distinct word and its frequency.",
            "Add its frequency to the total for that character length.",
            "Also count how many distinct word types have each length.",
        ),
    },
    "task_6": {
        "functions": (submission.task_6_modal_words,),
        "summary": "Every post is checked for six modal words, then matching posts are measured.",
        "steps": (
            "Count every occurrence of will, must, might, may, could, and can.",
            "When a post contains a modal, record that post's word and character lengths once.",
            "Return both overall counts and the per-post measurements used by the box plots.",
        ),
    },
    "task_7": {
        "functions": (submission.task_7_brown_stopwords,),
        "summary": "Each Brown sentence is measured in words, characters, and English stopword occurrences.",
        "steps": (
            "Load NLTK's English stopword set and clean each Brown sentence.",
            "Count words that occur in the stopword set.",
            "Store one measurement row per sentence for the density plots.",
        ),
    },
}


def build_code_evidence() -> dict[str, dict[str, object]]:
    """Read the displayed snippets directly from the Moodle submission file."""

    evidence = {}
    for task, guide in CODE_GUIDES.items():
        snippet = "\n\n".join(
            inspect.getsource(function).rstrip()
            for function in guide["functions"]
        )
        evidence[task] = {
            "source": "lab1.py",
            "functions": [function.__name__ for function in guide["functions"]],
            "summary": guide["summary"],
            "steps": guide["steps"],
            "snippet": snippet,
        }
    return evidence


def records_from_pairs(pairs: Iterable[tuple[str, int]]) -> list[dict[str, int | str]]:
    return [{"word": word, "count": int(count)} for word, count in pairs]


def percentile(sorted_values: Sequence[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {
            "minimum": 0,
            "q1": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "q3": 0.0,
            "maximum": 0,
        }
    return {
        "minimum": ordered[0],
        "q1": round(percentile(ordered, 0.25), 2),
        "median": round(percentile(ordered, 0.5), 2),
        "mean": round(statistics.fmean(ordered), 2),
        "q3": round(percentile(ordered, 0.75), 2),
        "maximum": ordered[-1],
    }


def pearson_correlation(xs: Sequence[int], ys: Sequence[int]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return round(numerator / denominator, 4) if denominator else 0.0


def logarithmic_sample(records: Sequence[dict[str, float | int]], limit: int = 500):
    if len(records) <= limit:
        return list(records)
    last = len(records) - 1
    indices = {
        round(math.exp(math.log(last + 1) * step / (limit - 1))) - 1
        for step in range(limit)
    }
    indices.update({0, last})
    return [records[index] for index in sorted(indices)]


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8DEDB", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def save_horizontal_bar(
    records: Sequence[dict[str, int | str]],
    title: str,
    path: Path,
    color: str,
) -> None:
    labels = [str(record["word"]) for record in reversed(records)]
    values = [int(record["count"]) for record in reversed(records)]
    height = max(6.5, len(labels) * 0.31)
    figure, axis = plt.subplots(figsize=(10, height), constrained_layout=True)
    bars = axis.barh(labels, values, color=color, edgecolor="none")
    axis.bar_label(bars, padding=4, fontsize=8)
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Frequency (occurrences)")
    axis.set_ylabel("Word")
    style_axis(axis)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def plot_rank_frequency(rank_records: Sequence[dict[str, float | int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.loglog(
        [record["rank"] for record in rank_records],
        [record["frequency"] for record in rank_records],
        color=COLORS["blue"],
        linewidth=2,
    )
    axis.set_title("NPS Chat rank-frequency distribution", loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Word rank (log scale)")
    axis.set_ylabel("Frequency (log scale)")
    style_axis(axis)
    axis.grid(which="both", color="#D8DEDB", linewidth=0.6, alpha=0.65)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def plot_cumulative_coverage(
    rank_records: Sequence[dict[str, float | int]], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.semilogx(
        [record["rank"] for record in rank_records],
        [record["coverage_percent"] for record in rank_records],
        color=COLORS["coral"],
        linewidth=2.5,
    )
    axis.set_title("Cumulative token coverage by word rank", loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Number of word types included (log scale)")
    axis.set_ylabel("Cumulative coverage (%)")
    axis.set_ylim(0, 101)
    style_axis(axis)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def plot_word_lengths(length_records: Sequence[dict[str, int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    lengths = [record["length"] for record in length_records]
    totals = [record["token_count"] for record in length_records]
    axis.bar(lengths, totals, color=COLORS["teal"], width=0.78)
    axis.plot(lengths, totals, color=COLORS["ink"], linewidth=1.2, marker="o", markersize=3)
    axis.set_title("NPS Chat word length versus frequency", loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Word length (characters)")
    axis.set_ylabel("Total token frequency")
    axis.set_xticks(lengths)
    style_axis(axis)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def plot_modal_frequencies(modal_records: Sequence[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    labels = [str(record["word"]) for record in modal_records]
    values = [int(record["occurrences"]) for record in modal_records]
    bars = axis.bar(labels, values, color=[COLORS["blue"], COLORS["coral"], COLORS["teal"], COLORS["gold"], "#7A5AA6", "#4D7C8A"])
    axis.bar_label(bars, padding=4)
    axis.set_title("Modal-word frequencies in NPS Chat", loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Modal word")
    axis.set_ylabel("Frequency (occurrences)")
    style_axis(axis)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def plot_modal_boxplot(
    values_by_modal: dict[str, list[int]],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    boxplot = axis.boxplot(
        [values_by_modal[modal] for modal in MODALS],
        tick_labels=MODALS,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.8},
    )
    palette = [COLORS["blue"], COLORS["coral"], COLORS["teal"], COLORS["gold"], "#7A5AA6", "#4D7C8A"]
    for patch, color in zip(boxplot["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Modal word")
    axis.set_ylabel(ylabel)
    style_axis(axis)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def plot_stopword_density(
    lengths: Sequence[int],
    stopword_counts: Sequence[int],
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    density = axis.hexbin(
        lengths,
        stopword_counts,
        gridsize=48,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    colorbar = figure.colorbar(density, ax=axis)
    colorbar.set_label("Sentence density (log count)")
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Number of English stopwords")
    style_axis(axis)
    figure.savefig(path, dpi=180, facecolor=COLORS["paper"])
    plt.close(figure)


def analyze_nps_chat(figures_dir: Path, tables_dir: Path) -> dict[str, object]:
    posts, raw_tokens, words, frequencies = submission.task_1_prepare_nps_chat()

    # Task 1: import and prepare NPS Chat.
    task_1 = {
        "corpus": "NPS Chat Corpus",
        "posts": len(posts),
        "raw_tokens": len(raw_tokens),
        "word_tokens": len(words),
        "vocabulary_size": len(frequencies),
        "lexical_diversity": round(len(frequencies) / len(words), 4),
        "normalization": "Lowercase alphabetic words (including internal apostrophes); punctuation and numeric tokens are excluded. Alphabetic chat-event terms such as JOIN and PART remain part of the corpus.",
    }
    print(
        f"[1] NPS Chat loaded: {task_1['posts']:,} posts, "
        f"{task_1['word_tokens']:,} cleaned word tokens."
    )

    # Task 2: twenty most frequent words.
    top_20 = records_from_pairs(submission.task_2_most_frequent(frequencies))
    write_csv(
        tables_dir / "02_top_20_words.csv",
        ("rank", "word", "frequency"),
        ((index, record["word"], record["count"]) for index, record in enumerate(top_20, 1)),
    )
    save_horizontal_bar(
        top_20,
        "Twenty most frequent words in NPS Chat",
        figures_dir / "02_top_20_words.png",
        COLORS["blue"],
    )
    print("[2] Twenty most frequent words calculated and plotted.")

    # Task 3: alternative rank-frequency and cumulative illustrations.
    rank_records = [
        {
            "rank": rank,
            "word": word,
            "frequency": frequency,
            "coverage_percent": round(coverage, 4),
        }
        for rank, word, frequency, coverage
        in submission.task_3_frequency_views(frequencies)
    ]
    write_csv(
        tables_dir / "03_rank_frequency.csv",
        ("rank", "word", "frequency", "cumulative_coverage_percent"),
        (
            (record["rank"], record["word"], record["frequency"], record["coverage_percent"])
            for record in rank_records
        ),
    )
    plot_rank_frequency(rank_records, figures_dir / "03a_zipf_rank_frequency.png")
    plot_cumulative_coverage(rank_records, figures_dir / "03b_cumulative_coverage.png")
    print("[3] Zipf rank-frequency and cumulative-coverage illustrations generated.")

    # Task 4: least- and middle-frequency samples.
    least_pairs, middle_pairs, middle_target = (
        submission.task_4_frequency_samples(frequencies)
    )
    least_30 = records_from_pairs(least_pairs)
    middle_30 = records_from_pairs(middle_pairs)
    write_csv(
        tables_dir / "04a_least_30_words.csv",
        ("word", "frequency"),
        ((record["word"], record["count"]) for record in least_30),
    )
    write_csv(
        tables_dir / "04b_middle_30_words.csv",
        ("word", "frequency", "geometric_midpoint_target"),
        ((record["word"], record["count"], round(middle_target, 4)) for record in middle_30),
    )
    save_horizontal_bar(
        least_30,
        "Thirty least-frequent words in NPS Chat",
        figures_dir / "04a_least_30_words.png",
        COLORS["coral"],
    )
    save_horizontal_bar(
        middle_30,
        "Thirty words near the middle frequency range",
        figures_dir / "04b_middle_30_words.png",
        COLORS["gold"],
    )
    print(
        "[4] Least- and middle-frequency word sets selected, documented, and plotted."
    )

    # Task 5: aggregate token frequency by word length.
    length_records = [
        {
            "length": length,
            "token_count": token_count,
            "type_count": type_count,
        }
        for length, token_count, type_count
        in submission.task_5_word_lengths(frequencies)
    ]
    write_csv(
        tables_dir / "05_word_length_frequency.csv",
        ("word_length", "total_token_frequency", "distinct_word_types"),
        (
            (record["length"], record["token_count"], record["type_count"])
            for record in length_records
        ),
    )
    plot_word_lengths(length_records, figures_dir / "05_word_length_frequency.png")
    print("[5] Word lengths aggregated by token frequency and plotted.")

    # Task 6: modal frequency and lengths of every matching post.
    (
        modal_occurrences,
        modal_word_lengths,
        modal_character_lengths,
        modal_rows,
    ) = submission.task_6_modal_words(posts)
    modal_records: list[dict[str, object]] = []
    for modal in MODALS:
        modal_records.append(
            {
                "word": modal,
                "occurrences": modal_occurrences[modal],
                "posts": len(modal_word_lengths[modal]),
                "word_length": numeric_summary(modal_word_lengths[modal]),
                "character_length": numeric_summary(modal_character_lengths[modal]),
            }
        )
    write_csv(
        tables_dir / "06a_modal_statistics.csv",
        (
            "modal",
            "occurrences",
            "matching_posts",
            "mean_words",
            "median_words",
            "mean_characters",
            "median_characters",
        ),
        (
            (
                record["word"],
                record["occurrences"],
                record["posts"],
                record["word_length"]["mean"],
                record["word_length"]["median"],
                record["character_length"]["mean"],
                record["character_length"]["median"],
            )
            for record in modal_records
        ),
    )
    write_csv(
        tables_dir / "06b_modal_post_lengths.csv",
        ("modal", "post_number", "occurrences_in_post", "words", "characters"),
        modal_rows,
    )
    plot_modal_frequencies(modal_records, figures_dir / "06a_modal_frequencies.png")
    plot_modal_boxplot(
        modal_word_lengths,
        "Lengths of NPS Chat posts containing each modal",
        "Post length (words)",
        figures_dir / "06b_modal_post_length_words.png",
    )
    plot_modal_boxplot(
        modal_character_lengths,
        "Character lengths of posts containing each modal",
        "Post length (characters)",
        figures_dir / "06c_modal_post_length_characters.png",
    )
    print("[6] Modal frequencies and matching-post lengths calculated and plotted.")

    return {
        "task_1": task_1,
        "task_2": {"top_20": top_20},
        "task_3": {
            "rank_frequency": logarithmic_sample(rank_records),
            "full_vocabulary_rows": len(rank_records),
        },
        "task_4": {
            "least_30": least_30,
            "least_method": "Sort by ascending frequency, then alphabetically to resolve ties; take the first 30 words.",
            "middle_30": middle_30,
            "middle_target": round(middle_target, 4),
            "middle_method": "Choose the 30 frequencies nearest the geometric midpoint between the minimum and maximum observed frequencies.",
        },
        "task_5": {"word_lengths": length_records},
        "task_6": {
            "modals": modal_records,
            "sentence_unit": "Each NPS Chat post is treated as a sentence-like unit.",
        },
    }


def analyze_brown(figures_dir: Path, tables_dir: Path) -> dict[str, object]:
    # Task 7: stopword count and word/character length for every Brown sentence.
    english_stopwords, measurement_rows = submission.task_7_brown_stopwords()
    metrics = [
        {
            "sentence": sentence,
            "stopwords": stopword_count,
            "words": word_length,
            "characters": character_length,
        }
        for sentence, stopword_count, word_length, character_length
        in measurement_rows
    ]

    write_csv(
        tables_dir / "07_brown_sentence_metrics.csv",
        ("sentence_number", "stopword_count", "word_length", "character_length"),
        (
            (
                metric["sentence"],
                metric["stopwords"],
                metric["words"],
                metric["characters"],
            )
            for metric in metrics
        ),
    )

    stop_counts = [metric["stopwords"] for metric in metrics]
    word_lengths = [metric["words"] for metric in metrics]
    character_lengths = [metric["characters"] for metric in metrics]
    plot_stopword_density(
        word_lengths,
        stop_counts,
        "Brown: stopwords versus sentence length in words",
        "Sentence length (words)",
        figures_dir / "07a_stopwords_vs_sentence_words.png",
    )
    plot_stopword_density(
        character_lengths,
        stop_counts,
        "Brown: stopwords versus sentence length in characters",
        "Sentence length (characters)",
        figures_dir / "07b_stopwords_vs_sentence_characters.png",
    )

    word_density = Counter((metric["words"], metric["stopwords"]) for metric in metrics)
    character_bin_width = 10
    character_density = Counter(
        (
            (metric["characters"] // character_bin_width) * character_bin_width,
            metric["stopwords"],
        )
        for metric in metrics
    )

    task_7 = {
        "corpus": "Brown Corpus",
        "sentences": len(metrics),
        "stopword_list_size": len(english_stopwords),
        "total_stopword_occurrences": sum(stop_counts),
        "mean_stopwords_per_sentence": round(statistics.fmean(stop_counts), 2),
        "correlation_words": pearson_correlation(word_lengths, stop_counts),
        "correlation_characters": pearson_correlation(character_lengths, stop_counts),
        "word_density": [
            {"length": length, "stopwords": stops, "sentences": count}
            for (length, stops), count in sorted(word_density.items())
        ],
        "character_density": [
            {"length": length, "stopwords": stops, "sentences": count}
            for (length, stops), count in sorted(character_density.items())
        ],
        "character_bin_width": character_bin_width,
    }
    print(
        f"[7] Brown analysis complete: {task_7['sentences']:,} sentences and "
        f"{task_7['total_stopword_occurrences']:,} stopword occurrences."
    )
    return task_7


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def build_report(results: dict[str, object]) -> str:
    task_1 = results["task_1"]
    task_2 = results["task_2"]
    task_4 = results["task_4"]
    task_5 = results["task_5"]
    task_6 = results["task_6"]
    task_7 = results["task_7"]

    lines = [
        "# Lab 1 - Natural Language Processing",
        "",
        f"Generated: {results['meta']['generated_at']}",
        "",
        "## 1. Import and prepare the NPS Chat corpus",
        "",
        f"The corpus contains **{task_1['posts']:,} posts**, **{task_1['raw_tokens']:,} raw tokens**, "
        f"and **{task_1['word_tokens']:,} normalized word tokens** representing "
        f"**{task_1['vocabulary_size']:,} distinct words**.",
        "",
        f"Method: {task_1['normalization']}",
        "",
        "## 2. Twenty most frequent words",
        "",
        *markdown_table(
            ("Rank", "Word", "Frequency"),
            (
                (index, record["word"], record["count"])
                for index, record in enumerate(task_2["top_20"], 1)
            ),
        ),
        "",
        "Figure: `figures/02_top_20_words.png`",
        "",
        "## 3. Alternative graphical illustrations",
        "",
        "The log-log rank-frequency graph tests the Zipf-like shape of the vocabulary. "
        "The cumulative-coverage graph shows how quickly common words account for the corpus tokens.",
        "",
        "Figures: `figures/03a_zipf_rank_frequency.png` and `figures/03b_cumulative_coverage.png`",
        "",
        "## 4. Thirty least-frequent and thirty middle-frequency words",
        "",
        f"Least-frequency method: {task_4['least_method']}",
        "",
        *markdown_table(
            ("Least-frequency word", "Frequency"),
            ((record["word"], record["count"]) for record in task_4["least_30"]),
        ),
        "",
        f"Middle-frequency method: {task_4['middle_method']} The target frequency is "
        f"**{task_4['middle_target']}**.",
        "",
        *markdown_table(
            ("Middle-frequency word", "Frequency"),
            ((record["word"], record["count"]) for record in task_4["middle_30"]),
        ),
        "",
        "Figures: `figures/04a_least_30_words.png` and `figures/04b_middle_30_words.png`",
        "",
        "## 5. Word length versus frequency",
        "",
        "Word frequency is summed for every distinct character length.",
        "",
        *markdown_table(
            ("Length", "Token frequency", "Distinct types"),
            (
                (record["length"], record["token_count"], record["type_count"])
                for record in task_5["word_lengths"]
            ),
        ),
        "",
        "Figure: `figures/05_word_length_frequency.png`",
        "",
        "## 6. Modal words and sentence/post lengths",
        "",
        f"{task_6['sentence_unit']} Modal occurrences count every use; length statistics "
        "include each matching post once for that modal.",
        "",
        *markdown_table(
            (
                "Modal",
                "Occurrences",
                "Posts",
                "Mean words",
                "Median words",
                "Mean characters",
                "Median characters",
            ),
            (
                (
                    record["word"],
                    record["occurrences"],
                    record["posts"],
                    record["word_length"]["mean"],
                    record["word_length"]["median"],
                    record["character_length"]["mean"],
                    record["character_length"]["median"],
                )
                for record in task_6["modals"]
            ),
        ),
        "",
        "Figures: `figures/06a_modal_frequencies.png`, `figures/06b_modal_post_length_words.png`, "
        "and `figures/06c_modal_post_length_characters.png`",
        "",
        "## 7. Brown corpus stopwords versus sentence length",
        "",
        f"The Brown corpus contains **{task_7['sentences']:,} sentences**. Using NLTK's "
        f"**{task_7['stopword_list_size']} English stopwords**, the analysis finds "
        f"**{task_7['total_stopword_occurrences']:,} stopword occurrences** "
        f"({task_7['mean_stopwords_per_sentence']} per sentence on average).",
        "",
        f"Pearson correlation with sentence length in words: **{task_7['correlation_words']}**.",
        "",
        f"Pearson correlation with sentence length in characters: **{task_7['correlation_characters']}**.",
        "",
        "Density plots are used because many sentences share the same values; color records how "
        "frequently each region occurs.",
        "",
        "Figures: `figures/07a_stopwords_vs_sentence_words.png` and "
        "`figures/07b_stopwords_vs_sentence_characters.png`",
        "",
        "## Reproducibility notes",
        "",
        "- Words are lowercased and punctuation-only tokens are excluded.",
        "- Character lengths use the NLTK tokens joined by one space.",
        "- The complete numerical results are stored in `tables/` and `../docs/data/lab_results.json`.",
    ]
    return "\n".join(lines) + "\n"


def validate_results(results: dict[str, object]) -> None:
    assert len(results["task_2"]["top_20"]) == 20
    assert len(results["task_4"]["least_30"]) == 30
    assert len(results["task_4"]["middle_30"]) == 30
    assert [record["word"] for record in results["task_6"]["modals"]] == list(MODALS)
    assert results["task_1"]["word_tokens"] > 0
    assert results["task_7"]["sentences"] > 0
    assert set(results["code"]) == {f"task_{number}" for number in range(1, 8)}


def build_results(figures_dir: Path, tables_dir: Path) -> dict[str, object]:
    """Run the canonical submission analysis and assemble presentation data."""

    nps_results = analyze_nps_chat(figures_dir, tables_dir)
    brown_results = analyze_brown(figures_dir, tables_dir)
    results = {
        "meta": {
            "title": "Lab 1",
            "subject": "Natural Language Processing",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "nltk_version": nltk.__version__,
        },
        **nps_results,
        "task_7": brown_results,
        "code": build_code_evidence(),
    }
    validate_results(results)
    return results


def copy_site_figures(figures_dir: Path) -> None:
    """Copy Python-rendered figures used when interactive charts are unavailable."""

    DEFAULT_SITE_FIGURES.mkdir(parents=True, exist_ok=True)
    for figure in figures_dir.glob("*.png"):
        shutil.copyfile(figure, DEFAULT_SITE_FIGURES / figure.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all NLP Lab 1 analyses, tables, figures, and web data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated report, figures, and tables.",
    )
    parser.add_argument(
        "--site-data",
        type=Path,
        default=DEFAULT_SITE_DATA,
        help="JSON file consumed by the static presentation.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Fail instead of downloading missing NLTK corpora.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    args.site_data.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SITE_REPORT.parent.mkdir(parents=True, exist_ok=True)

    print("NLP Lab 1 analysis")
    submission.ensure_nltk_data(allow_download=not args.skip_download)
    results = build_results(figures_dir, tables_dir)

    with args.site_data.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    report_path = output_dir / "lab1_report.md"
    report_path.write_text(build_report(results), encoding="utf-8")
    shutil.copyfile(report_path, DEFAULT_SITE_REPORT)
    copy_site_figures(figures_dir)

    print(f"Done. Numbered report: {report_path}")
    print(f"      Figures: {figures_dir}")
    print(f"      Tables: {tables_dir}")
    print(f"      Website data: {args.site_data.resolve()}")


if __name__ == "__main__":
    main()
