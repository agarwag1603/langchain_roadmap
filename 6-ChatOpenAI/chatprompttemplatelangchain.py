from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Set environment variables (if not already set in your shell)
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

gpt_llm=ChatOpenAI(model="gpt-5-mini",api_key=os.environ["OPENAI_API_KEY"])

prompt=ChatPromptTemplate.from_messages([
    ("system","You are a chemical engineer, your role is to provide answer ONLY related to it, nothing else, keep answers very short and crisp"),
    ("user","{input}")
])

output_parser=StrOutputParser()

chain = prompt | gpt_llm | output_parser
# llm_response=chain.invoke({"input":"What is the elements are there in periodic table?"})
# print(llm_response)

llm_response=chain.invoke({"input":"What is the chemical formula of Chloroform?"})
print(llm_response)

