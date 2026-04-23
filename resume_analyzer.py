from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, List


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9+#]+")
MAX_SUGGESTIONS = 5


@dataclass
class Chunk:
    text: str
    score: float


class ResumeRAGAnalyzer:
    """Lightweight LLM+RAG style resume analyzer.

    Retrieval picks the most relevant resume chunks for the user question.
    Generation produces concrete improvement suggestions from that context.
    """

    def __init__(self, chunk_size: int = 2, top_k: int = 3):
        self.chunk_size = max(1, chunk_size)
        self.top_k = max(1, top_k)

    def analyze(self, resume_text: str, question: str) -> str:
        if not resume_text.strip():
            raise ValueError("resume_text must not be empty")
        if not question.strip():
            raise ValueError("question must not be empty")

        chunks = self._retrieve(resume_text, question)
        return self._generate_answer(question, chunks)

    def _retrieve(self, resume_text: str, question: str) -> List[Chunk]:
        q_tokens = set(self._tokenize(question))
        scored: List[Chunk] = []
        for block in self._chunks_from_resume(resume_text):
            b_tokens = set(self._tokenize(block))
            overlap = q_tokens.intersection(b_tokens)
            score = len(overlap) / max(len(q_tokens), 1)
            if score > 0:
                scored.append(Chunk(text=block, score=score))

        if not scored:
            # fallback context keeps behavior useful even when overlap is low
            fallback = self._chunks_from_resume(resume_text)[: self.top_k]
            return [Chunk(text=chunk, score=0.0) for chunk in fallback]

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: self.top_k]

    def _generate_answer(self, question: str, chunks: Iterable[Chunk]) -> str:
        context = [c.text for c in chunks]
        context_joined = "\n- ".join(context)

        suggestions = [
            "Quantify impact with metrics (%, $, time saved) for each major achievement.",
            "Use stronger action verbs and keep bullet points concise.",
            "Align skills and project phrasing with keywords in your target role.",
            "Add a short summary highlighting your strongest domain expertise.",
        ]

        question_lower = question.lower()
        if "ats" in question_lower:
            suggestions.insert(0, "Mirror exact job-description keywords to improve ATS matching.")
        if "project" in question_lower:
            suggestions.insert(0, "For each project, include problem, approach, and measurable outcome.")

        suggestion_block = "\n".join(
            f"{i + 1}. {item}" for i, item in enumerate(suggestions[:MAX_SUGGESTIONS])
        )

        return (
            f"Question: {question.strip()}\n\n"
            "Retrieved Resume Context:\n"
            f"- {context_joined}\n\n"
            "Suggested Improvements:\n"
            f"{suggestion_block}"
        )

    def _chunks_from_resume(self, resume_text: str) -> List[str]:
        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
        if not lines:
            return []

        grouped = []
        for chunk_start in range(0, len(lines), self.chunk_size):
            grouped.append(" ".join(lines[chunk_start : chunk_start + self.chunk_size]))
        return grouped

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in _WORD_PATTERN.findall(text)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a resume with a lightweight LLM+RAG workflow")
    parser.add_argument("--resume", required=True, help="Path to a text resume file")
    parser.add_argument("--question", required=True, help="Question to ask about resume improvements")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        with open(args.resume, "r", encoding="utf-8") as handle:
            resume_text = handle.read()
    except FileNotFoundError:
        parser.error(f"resume file not found: {args.resume}")

    analyzer = ResumeRAGAnalyzer()
    print(analyzer.analyze(resume_text=resume_text, question=args.question))


if __name__ == "__main__":
    main()
