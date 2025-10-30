from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

prompt_template= ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant."),
        ("human","You are supposed to give an interesting fact about the person: {person}")
    ]
)

chain = prompt_template | gpt_llm | StrOutputParser()

article_template= ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant who writes article."),
        ("human","You are supposed to write article in 2 paragraphs for the topic {topic}")
    ]
)

article_chain = {"topic" : chain} | article_template | gpt_llm | StrOutputParser()

print(article_chain.invoke({"person": "Roger Federer"}))