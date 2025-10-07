from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chat_prompt_template =  ChatPromptTemplate(
    [
        ("system","You are an AI system that focus on generating new articles."),
        ("human","Please help in generating a new article on a topic: \n\n {topic}")
    ]
)

parser = StrOutputParser()

sequence_runnable = chat_prompt_template | gpt_llm | parser

def word_count(strvalue):
    return len(strvalue.split())

def word_lower(strvalue):
    return strvalue.lower()

paralle_runnable = RunnableParallel(
    { 
        "article":RunnablePassthrough(),
        "word_count":RunnableLambda(word_count),
        "word_lower":RunnableLambda(word_lower)
     }
)

final_runnable = sequence_runnable | paralle_runnable
print(final_runnable.invoke({"topic":"Black Hole in 30 words"}))