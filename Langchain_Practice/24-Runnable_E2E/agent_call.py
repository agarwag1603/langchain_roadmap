import webbased_load,pdf_loader, chat_templates
from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableSequence
from langchain.agents import AgentExecutor,create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

tools = [webbased_load.web_retriever_tool,pdf_loader.pdf_retriever_tool]

gpt_llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

agent = create_openai_tools_agent(gpt_llm,tools,chat_templates.chat_template_rag)

agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

summary_chain = RunnableSequence(
    chat_templates.chat_template_summary | gpt_llm | parser | RunnableLambda(lambda text: text.upper())
)

runnable_parallel = RunnableParallel(
    {
    "answer":RunnablePassthrough(),
    "summary":RunnableSequence(chat_templates.chat_template_summary | gpt_llm | parser),
    "uppersummary":summary_chain
    }
)

final_chain = agent_executor | RunnableLambda(lambda x: {"topic": x})| runnable_parallel

print(final_chain.invoke({"question":"Hello, who are the founders of tesla?"}))

