from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT']="RAG Chain"

gpt_llm = ChatOpenAI(model="gpt-4o-mini")


maindir = os.path.dirname(os.path.abspath("__file__"))

pdf_path = os.path.join(maindir, "RentalConditions.pdf")

pdf_loader=PyPDFLoader(pdf_path)
pdf_doc=pdf_loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
splitted_doc=splitter.split_documents(pdf_doc)

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

chromastore=Chroma.from_documents(splitted_doc,embeddings)
chromadb=chromastore.as_retriever(search_kwargs={"k": 3})

prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant who answers user's questions based on the PDF."),
        ("human","Please help in responding to the user query based on the user query: \n {query} \n {context}")
    ]
)

runnable_chain=RunnableParallel({
    "query":RunnablePassthrough(),
    "context": chromadb
}
)

final_chain = runnable_chain | prompt_template | gpt_llm | StrOutputParser()

config = {
    "run_name":"Rag Chain"
}

print(final_chain.invoke("what is the deposit amount for car rental?",config=config))

