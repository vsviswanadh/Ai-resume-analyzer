import streamlit as st
from transformers import pipeline
from utils.parser import extract_text
from utils.rag import create_index, retrieve

st.title("AI Resume Q&A (Local LLM + RAG)")

resume = st.file_uploader("Upload Resume (PDF)")
question = st.text_input("Ask question about resume")

if resume:
    text = extract_text(resume)
    index, sentences = create_index(text)

    generator = pipeline("text-generation", model="distilgpt2")

    if st.button("Ask"):
        context = retrieve(question, index, sentences)

        prompt = f"""
        Context: {context}

        Question: {question}

        Answer:
        """

        output = generator(prompt, max_length=200, num_return_sequences=1)
        st.write(output[0]['generated_text'])