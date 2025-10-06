##Sequential chain where stroutput parser can sent the text directly to chat prompt template

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

chat_prompt_template= ChatPromptTemplate.from_messages([
    ("system","You are a movie critic, who criticizes the movie"),
    ("human","Give me 5 line movie review of the movie: \n\n {movie}")
])

chat_prompt_summary= ChatPromptTemplate.from_messages([
    ("system","You are summarizer assistant"),
    ("human","Summarize the output in 2 lines please : \n\n {text}")
])

chain = chat_prompt_template | gpt_llm | StrOutputParser() | chat_prompt_summary | gpt_llm | StrOutputParser ()
print(chain.invoke({"movie":"Shawshank Redemption"}))

##Sequential chain where stroutput parser can sent the text directly to chat prompt template- Above example with variable


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-5-mini")

chat_prompt_template= ChatPromptTemplate.from_messages([
    ("system","You are a movie critic, who criticizes the movie"),
    ("human","Give me 5 line movie review of the movie: \n\n {movie}")
])

critic_chain=chat_prompt_template | gpt_llm | StrOutputParser() 

chat_prompt_summary= ChatPromptTemplate.from_messages([
    ("system","You are summarizer assistant"),
    ("human","Summarize the output in 2 lines please : \n\n {text}")
])

final_chain = {"text": critic_chain} | chat_prompt_summary | gpt_llm | StrOutputParser() 

print(final_chain.invoke({"movie":"Shawshank Redemption"}))

