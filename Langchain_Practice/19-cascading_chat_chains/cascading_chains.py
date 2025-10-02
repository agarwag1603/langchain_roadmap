from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

chat_prompts = ChatPromptTemplate.from_messages([
    ("system","You should reply like you are a cosmologist, you should answer ONLY about our solar systems. Say no to the any other questions otherwise"),
    ("human","Tell me 5 lines paragraph about {topic}")
])


output_parser=StrOutputParser()

general_chat_chain= chat_prompts | gpt_llm | output_parser

chat_prompts_summarizer = ChatPromptTemplate.from_messages([
    ("system","You are a summarizer, you should summarize whatever text you get."),
    ("human","Tell me 2 lines summary about {text}")
])

summary_chat_chain = {"text":general_chat_chain} | chat_prompts_summarizer | gpt_llm | output_parser

summary_chain_response=summary_chat_chain.invoke({"planet":"Venus"})
print(summary_chain_response)




