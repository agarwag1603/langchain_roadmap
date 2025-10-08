#create a chatbot with a chat history in a simple list
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

chat_history=[]

while True:
    user_query= input("You: ")
    chat_history.append(user_query)
    if user_query=="exit":
        break
    response=gpt_llm.invoke(user_query)
    chat_history.append(response.content)
    print("AI:",response.content)

print(chat_history)