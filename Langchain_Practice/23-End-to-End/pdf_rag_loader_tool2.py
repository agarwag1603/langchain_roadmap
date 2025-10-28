from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Path relative to this script
pdf_path = Path(__file__).parent / "Policy Copy.pdf"

pypdf_loader=PyPDFLoader(str(pdf_path))
policycondition_loader=pypdf_loader.load()

splitted_file=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
splitted_rental_file= splitted_file.split_documents(policycondition_loader)

embeddings_policy=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

chromavector_db_policy = Chroma.from_documents(splitted_rental_file,embeddings_policy)
chromavector_db_retriever_policy=chromavector_db_policy.as_retriever()

retriver_tool_policy=create_retriever_tool(chromavector_db_retriever_policy,"Insurance_Policy_Retriever", "Insurance Policy retriever tool")