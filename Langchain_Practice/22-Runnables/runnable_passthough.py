from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a {role} assistant, you should only answer related to it."),
        ("human","Please help me with an information related to topic \n\n {topic}")
    ]
)


chain = {"role" : RunnablePassthrough(),"topic":RunnablePassthrough()} | chat_template |  gpt_llm |  StrOutputParser()

response=chain.invoke({"role":"Computer Scientist" , "topic":"Best ever Quantum computers?"})
print(response)

