import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
embeddings=OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPEN_AI_API_KEY,dimensions=1024)

rag_loader=TextLoader("5-Embedding/rag.txt")
rag_documents=rag_loader.load()

rag_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
rag_final_split=rag_splitter.split_documents(rag_documents)

chromadatabase=Chroma.from_documents(rag_final_split,embeddings)

user_query="Experiment with different frameworks (LangChain, AutoGen, CrewAI)"
retrived_llm_output=chromadatabase.similarity_search(user_query)
print(retrived_llm_output)





