from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini", streaming=False)

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results= 3,doc_content_chars_max=300)
wikipedia=WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

tools = [wikipedia,arxiv]

system_template = """
You are a tool call agent who should only return the response from available tool set. 
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("human","Please pull me the information from the internet for {topic}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent_chain =  create_openai_tools_agent(gpt_llm, tools, chat_template)
agent_executor=AgentExecutor(agent=agent_chain, tools=tools, verbose=False)


summary_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("human","Please summarize this topic in 2-3 lines : \n {text}"),
    ]
)

summary_chain = {'text':agent_executor} | summary_template | gpt_llm | StrOutputParser()

response=summary_chain.invoke({'topic':"Tennis"})
print(response)





