from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool

pdf_loader=PyPDFLoader("Langchain_Practice/24-Runnable_E2E/RentalConditions.pdf")

pdf_docs=pdf_loader.load()

pdf_doc_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
pdf_doc_splitted=pdf_doc_splitter.split_documents(pdf_docs)

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=384)
chroma_store=Chroma.from_documents(pdf_doc_splitted, openai_embedding)
chromadb_pdf=chroma_store.as_retriever()
pdf_retriever_tool=create_retriever_tool(chromadb_pdf,"tesla_vector_db","search the information related to tesla news")
