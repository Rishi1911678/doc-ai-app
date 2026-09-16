import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_llm():
    # 1. Check Streamlit Cloud Secrets first
    if "GROQ_API_KEY" in st.secrets:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.1
        )
    # 2. Check local environment variables
    elif os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.environ["GROQ_API_KEY"],
            temperature=0.1
        )
    # 3. Fallback to local Ollama on Mac (Import only when running locally)
    else:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="llama3.2", temperature=0.1)
        except ImportError:
            raise ImportError(
                "Neither GROQ_API_KEY was found in secrets/env nor is langchain_ollama available."
            )

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = get_llm()
        self.vector_store = None
        self.rag_chain = None

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def process_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI software developer and document analyzer. "
                       "Use the retrieved context to directly answer the user's question clearly. "
                       "Reformat code snippets into clean markdown code blocks.\n\n"
                       "Context:\n{context}"),
            ("human", "{question}"),
        ])

        self.rag_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str):
        if not self.rag_chain:
            return "Please upload and process a document first."
        return self.rag_chain.invoke(question)