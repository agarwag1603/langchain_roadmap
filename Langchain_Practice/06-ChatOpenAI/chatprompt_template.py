from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

gpt_llm=ChatOpenAI(model="gpt-5-mini")

first_system_message="""
You are a chemical engineer, your role is to provide answer ONLY related to it.
Do not answer anything else, keep answers very short and crisp"
"""

second_system_message="""
You are an computer engineer, your role is to provide answer ONLY related to it.
Do not answer anything else, keep answers very short and crisp in less than a sentence"
"""

prompt=ChatPromptTemplate.from_messages([
    ("system","{system_message}"),
    ("user","{input}")
])

output_parser=StrOutputParser()

chain = prompt | gpt_llm 

llm_response=chain.invoke({"input":"What is the chemical formula of Benzene?","system_message":first_system_message})
print(llm_response)
print(70*"*")
llm_response=chain.invoke({"input":"What is a RAM?","system_message":second_system_message})
print(llm_response)

