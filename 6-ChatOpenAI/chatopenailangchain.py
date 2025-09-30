import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Set environment variables (if not already set in your shell)
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

gpt_llm=ChatOpenAI(model="gpt-5-mini",api_key=os.environ["OPENAI_API_KEY"])

gpt_response=gpt_llm.invoke("Who is Virat Kohli?")
print(gpt_response)
