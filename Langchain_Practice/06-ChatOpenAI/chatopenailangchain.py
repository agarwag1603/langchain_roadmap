import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

gpt_llm=ChatOpenAI(model="gpt-5-mini")

gpt_response=gpt_llm.invoke("Who is Virat Kohli?")
print(gpt_response)
