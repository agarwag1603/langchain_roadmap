from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

pdf_loader=PyPDFLoader("Langchain_Practice/17-StuffandMapReduceSummary/Speeches.pdf")
pdf_documents=pdf_loader.load()
print(pdf_documents)