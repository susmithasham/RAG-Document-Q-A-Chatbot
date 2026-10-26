"""
RAG Document Q&A Chatbot
Upload a PDF, ask questions, and get answers based only on that PDF.

Flow: PDF -> text -> chunks -> embeddings -> FAISS index
      question -> embedding -> top matching chunks -> LLM -> answer
"""

import os

import faiss
import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- Settings ----------------
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # overlap so sentences are not cut off badly
TOP_K = 4             # number of chunks sent to the LLM
LLM_MODEL = "gemini-2.5-flash"  # check AI Studio for the current free model name


# ---------------- Step 1: Read the PDF ----------------
def extract_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# ---------------- Step 2: Split into chunks ----------------
def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------- Step 3: Embeddings + FAISS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine similarity
    index.add(embeddings.astype(np.float32))
    return index


# ---------------- Step 4: Retrieve relevant chunks ----------------
def retrieve(question, chunks, index, embedder, k: int = TOP_K):
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    _, ids = index.search(q_emb.astype(np.float32), min(k, len(chunks)))
    return [chunks[i] for i in ids[0]]


# ---------------- Step 5: Ask the LLM ----------------
def ask_llm(question, context_chunks):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set. Set it and restart the app."

    client = genai.Client(api_key=api_key)
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return response.text


# ---------------- Step 6: Streamlit UI ----------------
def main():
    st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
    st.title("📄 RAG Document Q&A Chatbot")
    st.caption("Upload a PDF and ask questions about it.")

    embedder = load_embedder()
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded:
        # Process the PDF only once per uploaded file
        if st.session_state.get("file_name") != uploaded.name:
            with st.spinner("Reading and indexing the PDF..."):
                text = extract_text(uploaded)
                if not text.strip():
                    st.error("No text found. This may be a scanned PDF.")
                    st.stop()
                chunks = split_text(text)
                st.session_state.chunks = chunks
                st.session_state.index = build_index(chunks, embedder)
                st.session_state.file_name = uploaded.name
            st.success(f"Indexed {len(st.session_state.chunks)} chunks.")

        question = st.text_input("Ask a question about the document:")
        if question:
            with st.spinner("Thinking..."):
                top_chunks = retrieve(
                    question, st.session_state.chunks, st.session_state.index, embedder
                )
                answer = ask_llm(question, top_chunks)
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Show retrieved context"):
                for i, c in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(c)


if __name__ == "__main__":
    main()
