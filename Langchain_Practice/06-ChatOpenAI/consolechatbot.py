from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

while True:
    user_query= input("You: ")
    if user_query=="exit":
        break
    response=gpt_llm.invoke(user_query)
    print("AI:",response.content)