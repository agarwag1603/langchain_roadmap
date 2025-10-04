from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# --- Prompt ---
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that helps answer users questions."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{Question}")
])

# --- LLM ---
gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Combine prompt + LLM ---
chain = chat_template | gpt_llm

# --- Streamlit UI ---
st.title("ConversationalAI")
user_input = st.text_input("User query")
response_placeholder = st.empty()
full_response = ""

# --- Persist history across reruns using session_state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

def store_chat_history(sessionid: str) -> BaseChatMessageHistory:
    if sessionid not in st.session_state.chat_history:
        st.session_state.chat_history[sessionid] = InMemoryChatMessageHistory()
    return st.session_state.chat_history[sessionid]

# --- Wrap chain with memory ---
chat_runnable = RunnableWithMessageHistory(
    chain,
    store_chat_history,
    input_messages_key="Question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user-123"}}

# --- Stream response ---
if user_input:
    for chunk in chat_runnable.stream({"Question": user_input}, config=config):
        if chunk.content:
            full_response += chunk.content
            response_placeholder.markdown(full_response)
