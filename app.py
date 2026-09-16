import streamlit as st
import tempfile
import os
from rag_engine import RAGPipeline

st.set_page_config(page_title="RAG Doc AI", page_icon="📄", layout="wide")
st.title("📄 PDF Intelligence & Q&A System")

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "processed" not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Chunking, Embedding & Indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            st.session_state.rag.process_pdf(tmp_path)
            os.remove(tmp_path)
            st.session_state.processed = True
            st.success("Document Indexed Successfully!")

st.subheader("Ask Questions About Your Document")
user_query = st.text_input("Enter your question:")

if user_query:
    if not st.session_state.processed:
        st.warning("Please upload and process a PDF in the sidebar first.")
    else:
        with st.spinner("Retrieving relevant context & generating answer..."):
            answer = st.session_state.rag.query(user_query)
            st.markdown("### Answer:")
            st.write(answer)