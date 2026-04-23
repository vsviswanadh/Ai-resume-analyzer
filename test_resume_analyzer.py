import unittest

from resume_analyzer import ResumeRAGAnalyzer


class ResumeRAGAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = ResumeRAGAnalyzer(chunk_size=1, top_k=2)
        self.resume = """
        Software Engineer at Acme
        Built ATS optimization tooling for recruiters
        Project Atlas improved deployment speed by 35 percent
        Python, FastAPI, PostgreSQL, Docker
        """

    def test_analyze_returns_question_context_and_suggestions(self) -> None:
        answer = self.analyzer.analyze(
            self.resume,
            "How can I improve this resume for ATS screening?",
        )

        self.assertIn("Question:", answer)
        self.assertIn("Retrieved Resume Context:", answer)
        self.assertIn("Suggested Improvements:", answer)
        self.assertIn("ATS", answer)

    def test_project_question_adds_project_specific_suggestion(self) -> None:
        answer = self.analyzer.analyze(self.resume, "How should I improve project bullets?")
        self.assertIn("For each project, include problem, approach, and measurable outcome.", answer)


    def test_configuration_controls_chunking_and_retrieval(self) -> None:
        analyzer = ResumeRAGAnalyzer(chunk_size=2, top_k=1)
        answer = analyzer.analyze(self.resume, "Tell me about Python and Docker skills")
        context_lines = [
            line for line in answer.splitlines() if line.startswith("- ") and "Suggested Improvements" not in line
        ]
        self.assertEqual(len(context_lines), 1)
        self.assertIn("Python, FastAPI, PostgreSQL, Docker", context_lines[0])

    def test_empty_inputs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.analyzer.analyze("", "question")
        with self.assertRaises(ValueError):
            self.analyzer.analyze(self.resume, "")


if __name__ == "__main__":
    unittest.main()
