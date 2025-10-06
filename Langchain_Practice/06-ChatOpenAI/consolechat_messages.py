#This is used for logging system, human, AI messages properly in a list as a chatbot history

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

chat_history=[
    SystemMessage(content="You are an AI assistant who is helpful in giving information related to AI engineering")
]

while True:
    user_query= input("You: ")
    chat_history.append(HumanMessage(content=user_query))
    if user_query=="exit":
        break
    response=gpt_llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:",response.content)

print(chat_history)