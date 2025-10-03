from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

attention_document_loader=PyPDFLoader("Langchain_Practice/8-PDF_Rag/1706.03762v7.pdf")

attention_document=attention_document_loader.load()

attention_splitter=RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
attention_splitted_document=attention_splitter.split_documents(attention_document)


gpt_llm = ChatOpenAI(model="gpt-5-mini")
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)

chromdb=Chroma.from_documents(attention_splitted_document,openai_embedding)
chromdb_retrieved_docs=chromdb.as_retriever(search_kwargs={"k": 3})

prompts = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context
<context>
{context}
</context>

Question: {input}     
""")

document_chain = create_stuff_documents_chain(gpt_llm, prompts)

retrieval_chain = create_retrieval_chain(chromdb_retrieved_docs,document_chain)
print(80*"==")
print("Retrieval_chain")
print(retrieval_chain)
print(80*"==")
response=retrieval_chain.invoke({"input":"what is self attention layer?"})
print("Retrieval_answer")
print(response['answer'])
print(80*"==")
print("Retrieval_response")
print(response)




