from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

message_history=RunnableWithMessageHistory(gpt_llm,get_session_history)

config={"configurable":{"session_id":"1234"}}

response=message_history.invoke([HumanMessage(content="Hi, I am Gaurav, I am working as an automation engineer")],config=config)
print(response)
print(80*"--")
continue_response=message_history.invoke([HumanMessage(content="who am i?")],config=config)
print(continue_response)

