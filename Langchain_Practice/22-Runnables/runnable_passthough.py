##Runnable passthrough just pass the output. In this program we are calling thellm twice, one for information generation and other for summarization.
##By default in chain the summarization will be an output. Information will be missed, so to have that printed you can use RunnablePassthrough.

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chat_template_topic = ChatPromptTemplate.from_messages(
    [
        ("system","You are a {role} assistant, you should only answer related to it."),
        ("human","Please help me with an information related to topic \n\n {topic}")
    ]
)


topic_runnable = RunnableSequence(chat_template_topic,  gpt_llm , StrOutputParser())


chat_template_summary = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant who summarizes the topic."),
        ("human","Please summarize the topic: \n\n {topic}")
    ]
)

summarize_runnable = RunnableParallel(
    {
        "information":RunnablePassthrough(),
        "summary":RunnableSequence(chat_template_summary,gpt_llm, StrOutputParser())
    }
)

final_runnable= RunnableSequence(topic_runnable, summarize_runnable)

response=final_runnable.invoke({"role":"Computer Scientist" , "topic":"Quantum computers"})
print(response)

