from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Set environment variables (if not already set in your shell)
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

st.set_page_config(page_title="PDF talker")

st.title("Upload a PDF")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Get the file name
    file_name = uploaded_file.name
    st.success(f"✅ You uploaded: {file_name}")

attention_document_loader=PyPDFLoader(file_name)
attention_document=attention_document_loader.load()

attention_splitter=RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
attention_splitted_document=attention_splitter.split_documents(attention_document)


gpt_llm = ChatOpenAI(model="gpt-5-mini",api_key=os.environ["OPENAI_API_KEY"])
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",api_key=os.environ["OPENAI_API_KEY"],dimensions=1024)

chromdb=Chroma.from_documents(attention_splitted_document,openai_embedding)
chromdb_retrieved_docs=chromdb.as_retriever(search_kwargs={"k": 3})

prompts = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context
<context>
{context}
</context>

Question: {input}
Answer:        
""")

document_chain = create_stuff_documents_chain(gpt_llm, prompts)

retrieval_chain = create_retrieval_chain(chromdb_retrieved_docs,document_chain)
print(80*"==")
print("Retrieval_chain")
print(retrieval_chain)
print(80*"==")
response=retrieval_chain.invoke({"input":"what is self attention layer used for?"})
print("Retrieval_answer")
print(response['answer'])
print(80*"==")
print("Retrieval_response")
print(response)




