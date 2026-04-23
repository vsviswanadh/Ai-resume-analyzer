# Ai-resume-analyzer

A lightweight LLM+RAG-style resume analyzer that reviews a resume and suggests improvements based on a user question.

## Run

```bash
python /home/runner/work/Ai-resume-analyzer/Ai-resume-analyzer/resume_analyzer.py \
  --resume /path/to/resume.txt \
  --question "How can I improve this resume for ATS and project clarity?"
```

## Test

```bash
cd /home/runner/work/Ai-resume-analyzer/Ai-resume-analyzer
python -m unittest -v test_resume_analyzer.py
```
