from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_PROJECT']="Langchain Sequence"

gpt_llm = ChatOpenAI(model="gpt-4o-mini")

prompt =  ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant who only answers question related to the data science and machine learning."),
        ("human","Kindly, help in replying to the user query: \n {query}")
    ]
)

chain = prompt | gpt_llm | StrOutputParser()

config = {
    "run_name":"Langchain Sequence"
}

print(chain.invoke({"query":"What is SVM?"},config=config))