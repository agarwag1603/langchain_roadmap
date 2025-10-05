from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Path relative to this script
pdf_path = Path(__file__).parent / "RentalConditions.pdf"

pypdf_loader=PyPDFLoader(str(pdf_path))
rentalcondition_loader=pypdf_loader.load()

splitted_file=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 50)
splitted_rental_file= splitted_file.split_documents(rentalcondition_loader)

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

chromavector_db = Chroma.from_documents(splitted_rental_file,embeddings)
chromavector_db_retriever=chromavector_db.as_retriever()

retriver_tool=create_retriever_tool(chromavector_db_retriever,"Rental_Car_Retriever", "Rental car information to be pulled from this document")