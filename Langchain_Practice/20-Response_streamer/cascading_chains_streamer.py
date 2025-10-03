from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("🌌 Cosmology Assistant")
topic = st.text_input("Input the topic")

# LLM with streaming enabled
gpt_llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# ----------- First chain (Cosmology response) -----------
chat_prompts = ChatPromptTemplate.from_messages([
    ("system", "You should reply like you are a cosmologist, you should answer ONLY about our universe and space exploration."),
    ("human", "Tell me 5 lines paragraph about: \n\n {topic}")
])

output_parser = StrOutputParser()

streaming_chain = chat_prompts | gpt_llm     # for streaming
general_chat_chain = chat_prompts | gpt_llm | output_parser  # for passing into next step

if topic:
    # ---------- Step 1: General context ----------
    st.subheader(f"General context about {topic}")
    response_placeholder = st.empty()
    full_response = ""

    for chunk in streaming_chain.stream({"topic": topic}):
        if chunk.content:
            full_response += chunk.content
            response_placeholder.markdown(full_response)

    # ---------- Step 2: Summarization ----------
    chat_prompts_summarizer = ChatPromptTemplate.from_messages([
        ("system", "You are a summarizer, you should summarize whatever text you get from the prompt."),
        ("human", "Tell me 2 lines summary about {text}")
    ])

    summarizer_stream_chain = chat_prompts_summarizer | gpt_llm  # stream version
    summarizer_chain = chat_prompts_summarizer | gpt_llm | output_parser

    # Use the *output of general_chat_chain* as input to summarizer
    general_text = general_chat_chain.invoke({"topic": topic})

    st.subheader(f"Summary context about {topic}")
    response_placeholder_2 = st.empty()
    full_response_2 = ""

    for chunk in summarizer_stream_chain.stream({"text": general_text}):
        if chunk.content:
            full_response_2 += chunk.content
            response_placeholder_2.markdown(full_response_2)
