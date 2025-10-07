from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()

webload=WebBaseLoader(web_path="https://www.investopedia.com/articles/personal-finance/061915/story-behind-teslas-success.asp")

web_docs=webload.load()

web_doc_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
web_doc_splitted=web_doc_splitter.split_documents(web_docs)

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=384)
chroma_store=Chroma.from_documents(web_doc_splitted, openai_embedding)
chromadb_webbased=chroma_store.as_retriever()
web_retriever_tool=create_retriever_tool(chromadb_webbased,"tesla_vector_db","search the information related to tesla news")
