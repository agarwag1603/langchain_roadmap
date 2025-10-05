from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant who only talks about the topic given by the user."),
    ("human", "Provide me information about this topic: {topic}")
])

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = RunnableSequence([
    lambda x: {"topic": x.lower()[::-1]},  # Transform and wrap for prompt
    chat_prompt,
    gpt_llm,
    StrOutputParser()
])

response = chain.invoke("LangChain")
print(response)
