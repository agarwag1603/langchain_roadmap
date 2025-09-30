from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Set environment variables (if not already set in your shell)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

gpt_llm=ChatOpenAI(model="gpt-5-mini",api_key=os.environ["OPENAI_API_KEY"])

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

chain = prompt | gpt_llm | output_parser

llm_response=chain.invoke({"input":"What is the chemical formula of Benzene?","system_message":first_system_message})
print(llm_response)
print(70*"*")
llm_response=chain.invoke({"input":"What is a RAM?","system_message":second_system_message})
print(llm_response)

