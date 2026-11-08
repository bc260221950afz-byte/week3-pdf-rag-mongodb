import os
import streamlit as st
from pymongo import MongoClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(page_title="PDF RAG Assistant (Local MongoDB)", layout="wide")

st.title("📄 Week 3 Mini Project: PDF Question Answering (RAG)")
st.write("Upload a PDF, store chunks & embeddings in **Local MongoDB**, and ask questions!")

# ----------------- LOCAL MONGODB CONNECTION -----------------
@st.cache_resource
def init_mongodb():
    # Local MongoDB connection string
    client = MongoClient("mongodb://localhost:27017/")
    db = client["rag_database"]
    collection = db["pdf_chunks"]
    return collection

collection = init_mongodb()

# ----------------- LOAD EMBEDDING MODEL -----------------
@st.cache_resource
def load_model():
    # Lightweight and fast local embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# ----------------- HELPER FUNCTIONS -----------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def create_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ----------------- SIDEBAR: PDF UPLOAD & PROCESSING -----------------
st.sidebar.header("1. Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process & Save to MongoDB"):
        with st.spinner("Extracting text and generating embeddings..."):
            # 1. Extract text
            raw_text = extract_text_from_pdf(uploaded_file)
            
            # 2. Chunk text
            chunks = create_chunks(raw_text)
            
            # 3. Clear old collection data for fresh upload (Optional)
            collection.delete_many({})
            
            # 4. Generate embeddings and store in Local MongoDB
            documents_to_insert = []
            for idx, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                doc = {
                    "chunk_id": idx,
                    "text": chunk,
                    "embedding": embedding,
                    "filename": uploaded_file.name
                }
                documents_to_insert.append(doc)
            
            if documents_to_insert:
                collection.insert_many(documents_to_insert)
                st.sidebar.success(f"Successfully stored {len(chunks)} chunks in Local MongoDB!")

# ----------------- MAIN UI: Q&A INTERFACE -----------------
st.header("2. Ask Questions from your PDF")

# Optional: OpenAI API Key for LLM Generation (or you can use rule-based/free fallback)
api_key = st.text_input("Enter your OpenAI API Key (Optional for full generation):", type="password")

user_query = st.text_input("Enter your question about the document:")

if st.button("Search & Answer"):
    if not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching MongoDB & Generating Answer..."):
            # Fetch all stored chunks from MongoDB
            all_docs = list(collection.find({}))
            
            if not all_docs:
                st.error("No documents found in MongoDB. Please upload and process a PDF first via the sidebar.")
            else:
                # 1. Embed user query
                query_embedding = embedder.encode(user_query).reshape(1, -1)
                
                # 2. Compute similarity with all stored chunks
                similarities = []
                for doc in all_docs:
                    doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
                    sim = cosine_similarity(query_embedding, doc_embedding)[0][0]
                    similarities.append((sim, doc["text"]))
                
                # Sort by highest similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                # Retrieve Top-3 relevant chunks
                top_k = 3
                retrieved_context = "\n\n".join([item[1] for item in similarities[:top_k]])
                
                st.subheader("🔍 Retrieved Context (From Local MongoDB):")
                st.write(retrieved_context)
                
                # 3. Generate Answer using LLM (if API key provided) or display context directly
                st.subheader("💡 Final Answer:")
                if api_key:
                    try:
                        client = OpenAI(api_key=api_key)
                        prompt = f"Use the following context to answer the question accurately. If the answer is not in the context, say 'I cannot find the answer in the provided document.'\n\nContext:\n{retrieved_context}\n\nQuestion: {user_query}\nAnswer:"
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3
                        )
                        st.success(response.choices.message.content)
                    except Exception as e:
                        st.error(f"Error connecting to OpenAI: {e}")
                else:
                    st.info("API key not provided. Showing top retrieved text chunks above as the direct search result. (Add OpenAI API key above for generative AI answers).")
                    