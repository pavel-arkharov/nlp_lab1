import unittest

from nltk.probability import FreqDist

from lab1 import (
    character_length,
    clean_words,
    task_4_frequency_samples,
)
from tools.generate_site_outputs import (
    numeric_summary,
    pearson_correlation,
)


class AnalysisHelpersTest(unittest.TestCase):
    def test_normalize_words_excludes_non_words(self):
        tokens = ["Hello", ",", "CAN", "don't", "u42", ":)"]
        self.assertEqual(clean_words(tokens), ["hello", "can", "don't"])

    def test_character_length_uses_single_spaces(self):
        self.assertEqual(character_length(["A", "short", "post", "."]), 14)

    def test_least_frequency_selection_is_deterministic(self):
        frequencies = FreqDist({"beta": 1, "alpha": 1, "gamma": 2})
        least, _, _ = task_4_frequency_samples(frequencies, 2)
        self.assertEqual(least, [("alpha", 1), ("beta", 1)])

    def test_middle_frequency_selection_uses_geometric_target(self):
        frequencies = FreqDist({"one": 1, "ten": 10, "hundred": 100})
        _, selected, target = task_4_frequency_samples(frequencies, 1)
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
