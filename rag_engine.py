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
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOllama(model="llama3.2", temperature=0.1)
        self.vector_store = None
        self.rag_chain = None

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def process_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Larger chunk size so code blocks don't get cut in half
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)

        # Retrieve top 10 chunks instead of 3
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant analyzing an Operating Systems source code file. "
                       "Provide a complete, comprehensive summary of ALL algorithms and programs present in the context. "
                       "Do not skip any topic.\n\n"
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