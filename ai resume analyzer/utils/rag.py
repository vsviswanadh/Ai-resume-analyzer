from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_index(text):
    sentences = text.split('.')
    embeddings = model.encode(sentences)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, sentences

def retrieve(query, index, sentences):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=3)

    results = [sentences[i] for i in I[0]]
    return " ".join(results)