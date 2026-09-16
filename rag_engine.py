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
    # 3. Fallback to local Ollama on Mac
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0.1)