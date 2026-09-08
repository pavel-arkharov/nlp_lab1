import unittest

from nltk.probability import FreqDist

from lab1_analysis import (
    character_length,
    normalize_words,
    numeric_summary,
    pearson_correlation,
    select_least_frequent,
    select_middle_frequent,
)


class AnalysisHelpersTest(unittest.TestCase):
    def test_normalize_words_excludes_non_words(self):
        tokens = ["Hello", ",", "CAN", "don't", "u42", ":)"]
        self.assertEqual(normalize_words(tokens), ["hello", "can", "don't"])

    def test_character_length_uses_single_spaces(self):
        self.assertEqual(character_length(["A", "short", "post", "."]), 14)

    def test_least_frequency_selection_is_deterministic(self):
        frequencies = FreqDist({"beta": 1, "alpha": 1, "gamma": 2})
        self.assertEqual(
            select_least_frequent(frequencies, 2),
            [("alpha", 1), ("beta", 1)],
        )

    def test_middle_frequency_selection_uses_geometric_target(self):
        frequencies = FreqDist({"one": 1, "ten": 10, "hundred": 100})
        selected, target = select_middle_frequent(frequencies, 1)
        self.assertEqual(selected, [("ten", 10)])
        self.assertEqual(target, 10)

    def test_numeric_summary(self):
        summary = numeric_summary([1, 2, 3, 4, 5])
        self.assertEqual(summary["median"], 3)
        self.assertEqual(summary["mean"], 3)
        self.assertEqual(summary["q1"], 2)
        self.assertEqual(summary["q3"], 4)

    def test_pearson_correlation(self):
        self.assertEqual(pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)


if __name__ == "__main__":
    unittest.main()
