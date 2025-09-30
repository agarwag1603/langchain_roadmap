from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from dotenv import load_dotenv


load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", streaming=False)

rental_pdf=PyPDFLoader("14-Tool_calls/RentalConditions.pdf")
loaded_rental_pdf=rental_pdf.load()

splitted_doc=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitted_rental_docs=splitted_doc.split_documents(loaded_rental_pdf)

openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
vectordb=FAISS.from_documents(splitted_rental_docs,openai_embedding)
vectordbfaiss=vectordb.as_retriever()
retriever_tool=create_retriever_tool(vectordbfaiss,"car_assistant_pdf_reader","search any information related to the rental of car") 

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

tools = [arxiv,retriever_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful AI assistant who uses available tools to answer questions, in case you are not sure, do not give answer"),
        ("human","{input}"),
        MessagesPlaceholder("agent_scratchpad"), 
    ]
)

agent = create_openai_tools_agent(gpt_llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input":"what is the sport cricket??"})





