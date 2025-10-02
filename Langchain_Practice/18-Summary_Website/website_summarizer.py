import streamlit as st
from dotenv import load_dotenv
import validators
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

gpt_llm = ChatOpenAI(model="gpt-4o-mini")
prompt_template = """
Provide a summary for the following content in bullet points in less than 200 words.
{text}
"""

prompt= PromptTemplate(input_variables=["text"],template=prompt_template)

header={ "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )}

st.set_page_config(page_title="Summarize Youtube transcript or websites")
st.markdown(
    """
    <style>
    .big-title {
        font-size: 50px !important;
        text-align: left;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='big-title'>Youtube transcript / Websites Summarizer</h1>", unsafe_allow_html=True)

content_url=st.text_input("URL", label_visibility="collapsed")
if st.button("Summarize youtube or website content"):
    if not content_url.strip():
        st.error("Please provide the URL")
    # elif validators.url(content_url):
    #     st.error("Please enter a valid URL")
    else:
        if "youtube.com" in content_url:
            loader=youtube_url=YoutubeLoader.from_youtube_url(content_url)
        else:
            loader=youtube_url=UnstructuredURLLoader(urls=[content_url],ssl_verify=False, headers=header)
        
        data=loader.load()
        splitted_doc=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(data)

        chain = load_summarize_chain(llm=gpt_llm,chain_type="stuff", prompt=prompt, verbose=True)
        response=chain.run(splitted_doc)
        st.success(response)
        
