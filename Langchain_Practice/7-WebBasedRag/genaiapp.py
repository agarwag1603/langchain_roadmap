from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough


load_dotenv()

gpt_llm=ChatOpenAI(model="gpt-5-mini")

web_document=WebBaseLoader(web_path="https://weaviate.io/blog/what-is-agentic-rag")
web_document=web_document.load()

web_document_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=30)
final_document=web_document_splitter.split_documents(web_document)

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

faissvectordb=FAISS.from_documents(final_document,openai_embedding)
faiss_retriever=faissvectordb.as_retriever(search_kwargs={"k": 3})

prompts_template=ChatPromptTemplate.from_messages([
    ("system","You an AI agentic assistant and your goal is provide user with agentic information only from the provided context"),
    ("user","Input: {input}\n\n Context:{context}")
    ])

output_parse=StrOutputParser()

chain= ({"input": RunnablePassthrough(), "context":faiss_retriever} | prompts_template | gpt_llm | output_parse)
response=chain.invoke("What is common vanilla RAG like?")
print(response)


