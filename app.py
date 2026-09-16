import streamlit as st
import tempfile
import os
from rag_engine import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Local Doc AI | RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize RAG Pipeline in session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "processed" not in st.session_state:
    st.session_state.processed = False
if "filename" not in st.session_state:
    st.session_state.filename = None

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.caption("100% Local RAG Engine (Llama 3.2 + FAISS)")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    
    if uploaded_file and st.button("⚡ Process Document", use_container_width=True):
        with st.spinner("Processing PDF, creating chunks & building vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            st.session_state.rag.process_pdf(tmp_path)
            os.remove(tmp_path)
            
            st.session_state.processed = True
            st.session_state.filename = uploaded_file.name
            st.success("Indexing complete!")

    if st.session_state.processed:
        st.divider()
        st.caption("📌 **Active Document:**")
        st.info(st.session_state.filename)

# Main UI
st.title("📄 PDF Document Intelligence System")
st.markdown("Ask natural language questions about your uploaded documents, extract code algorithms, or summarize complex logic.")

if not st.session_state.processed:
    st.info("👈 Please upload and process a PDF file in the sidebar to start asking questions.")
else:
    user_query = st.text_input("💬 Ask a question about your document:", placeholder="e.g., List all process scheduling algorithms in this document.")
    
    if user_query:
        with st.spinner("Analyzing document context..."):
            response = st.session_state.rag.query(user_query)
            
            st.markdown("### 🤖 Answer")
            st.markdown(response)
