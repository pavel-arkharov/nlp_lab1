import inspect
import json
import tempfile
import unittest
from pathlib import Path

import lab1
from tools.generate_site_outputs import (
    CODE_GUIDES,
    build_report,
    build_results,
)


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_FIGURES = {
    "02_top_20_words.png",
    "03a_zipf_rank_frequency.png",
    "03b_cumulative_coverage.png",
    "04a_least_30_words.png",
    "04b_middle_30_words.png",
    "05_word_length_frequency.png",
    "06a_modal_frequencies.png",
    "06b_modal_post_length_words.png",
    "06c_modal_post_length_characters.png",
    "07a_stopwords_vs_sentence_words.png",
    "07b_stopwords_vs_sentence_characters.png",
}


class CompleteGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            lab1.ensure_nltk_data(allow_download=False)
        except RuntimeError as error:
            raise unittest.SkipTest(str(error)) from error

        cls.temporary_directory = tempfile.TemporaryDirectory()
        output = Path(cls.temporary_directory.name)
        cls.figures = output / "figures"
        cls.tables = output / "tables"
        cls.figures.mkdir()
        cls.tables.mkdir()
        cls.results = build_results(cls.figures, cls.tables)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temporary_directory"):
            cls.temporary_directory.cleanup()

    def test_complete_analysis_has_all_tasks_and_expected_counts(self):
        self.assertEqual(self.results["task_1"]["posts"], 10_567)
        self.assertEqual(self.results["task_1"]["word_tokens"], 34_461)
        self.assertEqual(len(self.results["task_2"]["top_20"]), 20)
        self.assertEqual(len(self.results["task_4"]["least_30"]), 30)
        self.assertEqual(len(self.results["task_4"]["middle_30"]), 30)
        self.assertEqual(self.results["task_7"]["sentences"], 57_340)

    def test_complete_generation_creates_every_figure_and_table(self):
        self.assertEqual({path.name for path in self.figures.glob("*.png")}, EXPECTED_FIGURES)
        self.assertEqual(len(list(self.tables.glob("*.csv"))), 8)

    def test_report_contains_every_numbered_answer(self):
        report = build_report(self.results)
        for number in range(1, 8):
            self.assertIn(f"## {number}.", report)

    def test_code_evidence_is_read_from_submission_functions(self):
        for task, guide in CODE_GUIDES.items():
            expected = "\n\n".join(
                inspect.getsource(function).rstrip()
                for function in guide["functions"]
            )
            self.assertEqual(self.results["code"][task]["snippet"], expected)


class PublishedArtifactTest(unittest.TestCase):
    def test_committed_json_has_complete_schema(self):
        data = json.loads((ROOT / "docs/data/lab_results.json").read_text())
        self.assertEqual(
            {f"task_{number}" for number in range(1, 8)},
            {key for key in data if key.startswith("task_")},
        )
        self.assertEqual(set(data["code"]), {f"task_{number}" for number in range(1, 8)})
        self.assertAlmostEqual(
            data["task_3"]["rank_frequency"][-1]["coverage_percent"],
            100.0,
            places=3,
        )

    def test_site_has_code_panels_and_static_chart_fallbacks(self):
        html = (ROOT / "docs/index.html").read_text()
        self.assertEqual(html.count('class="implementation"'), 7)
        self.assertEqual(html.count('class="chart-fallback"'), 11)
        for figure in EXPECTED_FIGURES:
            self.assertIn(f'src="figures/{figure}"', html)
            self.assertTrue((ROOT / "docs/figures" / figure).is_file())

    def test_all_chapters_are_collapsed_in_source_html(self):
        html = (ROOT / "docs/index.html").read_text()
        self.assertNotIn('class="task" open', html)
        self.assertNotIn('class="implementation" open', html)


if __name__ == "__main__":
    unittest.main()
