from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


prompt= ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful language assistant. Answer to the best you can in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

gpt_llm = ChatOpenAI(model="gpt-5-mini")

stroutput=StrOutputParser()

chain =  prompt | gpt_llm | stroutput

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

message_history=RunnableWithMessageHistory(chain,get_session_history,input_messages_key="messages")
config = {"configurable":{"session_id":"1234"}}

response=message_history.invoke({'messages':[HumanMessage(content="Hi, I am Gaurav, and I live in Singapore")],
                                 "language":"Hindi"}, config=config)
print(response)
print(80*"--")
response=message_history.invoke({'messages':[HumanMessage(content="Hi, what is my name?")],
                                 "language":"Hindi"}, config=config)
print(response)

