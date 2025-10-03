from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system","You are an expert in {sports}. You should answer related to the topic"),
    ("human","{user_input}")]
    )

chain=chat_prompt | gpt_llm

response =chain.invoke({"sports":"Cricket","user_input":"Who has hit the most number of ODI centuries."})

print(response)