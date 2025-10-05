from langchain.schema.runnable import RunnableMap
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant who only talks about the topic given by the user."),
    ("human", "Provide me the information about this topic: {topic}")
])

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

runnable_map = RunnableMap({
    "topic": lambda x: x[::-1]
})


chain = runnable_map | chat_prompt | gpt_llm | StrOutputParser ()

response=chain.invoke("niahcgnaL")
print(response)