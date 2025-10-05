# main.py
import streamlit as st
import uuid
from dotenv import load_dotenv
import sqlfile  # Your SQLite helper
import pdf_rag_loader_tool, online_tools, math_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Initialize DB ---
sqlfile.init_db()

# --- Tools ---
tools = [
    online_tools.arxiv_query,
    online_tools.wikipedia_query,
    pdf_rag_loader_tool.retriver_tool,
    math_tool.add,
    math_tool.multiply,
]

# --- LLM ---
gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

# --- Chat Prompt ---
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI agent that calls the right tools for each user query."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Help with providing information about this topic:\n\n{topic}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# --- Agent Executor ---
agent = create_openai_tools_agent(gpt_llm, tools, chat_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Summary Prompt ---
chat_template_summary = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarizes the responses."),
        ("human", "Give the key points here. Make bulletin points:\n\n{text}"),
    ]
)
chain_summary = chat_template_summary | gpt_llm | StrOutputParser()

# --- Session & History ---
def get_session_history_db(session_id: str) -> BaseChatMessageHistory:
    history = InMemoryChatMessageHistory()
    past_msgs = sqlfile.get_conversation(session_id)
    for m in past_msgs:
        if m["role"] == "user":
            history.add_user_message(m["content"])
        else:
            history.add_ai_message(m["content"])
    return history

chat_runnable = RunnableWithMessageHistory(
    agent_executor,
    get_session_history_db,
    input_messages_key="topic",
    history_messages_key="history",
)

# --- Streamlit UI ---
st.title("🤖 Agentic Chatbot with DB Memory + Summary")

# Sidebar: select existing session or create new
sessions = sqlfile.list_sessions()
sessions.append("New Conversation")
selected_session = st.sidebar.selectbox("Select conversation", sessions)

if selected_session == "New Conversation":
    session_id = str(uuid.uuid4())
else:
    session_id = selected_session

# User input
user_query = st.text_input("Your message:")
submit = st.button("Send")
response_placeholder = st.empty()

# --- Helper to extract text from LangChain response ---
def extract_text(resp) -> str:
    if isinstance(resp, dict):
        for key in ["content", "output", "text"]:
            if key in resp:
                return str(resp[key])
        return str(resp)
    return str(resp)

# --- Handle query submission ---
if submit and user_query.strip():
    # Save user message
    sqlfile.add_message(session_id, "user", user_query)

    config = {"configurable": {"session_id": session_id}}

    # --- Agent execution ---
    agent_response = chat_runnable.invoke({"topic": user_query}, config=config)
    agent_text = extract_text(agent_response)
    # response_placeholder.markdown(agent_text)

    # Save agent response
    sqlfile.add_message(session_id, "agent", agent_text)

    # --- Summarize agent response ---
    summary = chain_summary.invoke({"text": agent_text})
    st.markdown(f"**Response:** {summary}")
    sqlfile.add_message(session_id, "summary", summary)
