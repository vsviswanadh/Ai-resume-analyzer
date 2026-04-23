# Ai-resume-analyzer

A lightweight LLM+RAG-style resume analyzer that reviews a resume and suggests improvements based on a user question.

## Run

```bash
python resume_analyzer.py \
  --resume /path/to/resume.txt \
  --question "How can I improve this resume for ATS and project clarity?"
```

## Test

```bash
python -m unittest -v test_resume_analyzer.py
```
