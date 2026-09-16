import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    def __init__(self):
        # 1. Local Free Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # 2. Local Free LLM via Ollama
        self.llm = ChatOllama(model="llama3.2", temperature=0.1)
        self.vector_store = None
        self.rag_chain = None

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def process_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Semantic chunking tuned for code blocks & document summaries
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)

        # Index in local FAISS vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

        # Retrieve top 10 chunks to capture multi-topic PDFs fully
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI technical assistant and code analyzer. "
                       "Use the retrieved document context below to answer the user's question clearly. "
                       "If code snippets are present, reformat them into clean, syntactically correct markdown code blocks.\n\n"
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
